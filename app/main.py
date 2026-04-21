from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
from typing import Any
from urllib.parse import parse_qsl

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from app.auth.oauth_router import auth_router
from app.envsetup.env_loader import load_env
from app.settings_db.controllers import (
    create_db_connection as create_settings_db_connection,
)
from app.settings_db.controllers import (
    validate_db_connection as validate_settings_db_connection,
)
from app.settings_db.models import DBConnectionCreate
from app.settings_db.router import settings_db_router
from app.utils.db_column_config_view import (
    build_db_column_configuration_payload,
    get_table_configuration_payload,
    save_table_configuration,
)
from app.utils.db_connections import (
    get_table_columns_for_connection,
    get_tables_for_connection,
)
from app.utils.schema_config import SchemaConfigManager
from app.utils.schema_generator import generate_schema_config
from app.utils.schema_summarizer import Summarizer

logger = logging.getLogger(__name__)

load_env()


def configure_logging() -> None:
    """Configure application logging for local runs and Lambda CloudWatch."""
    log_level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    else:
        root_logger.setLevel(log_level)

    logger.setLevel(log_level)


configure_logging()


def get_allowed_origins() -> list[str]:
    """Return normalized CORS origins from the environment."""
    configured_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if not configured_origins:
        return ["*"]

    normalized_origins = [
        origin.strip() for origin in configured_origins.split(",") if origin.strip()
    ]
    return normalized_origins or ["*"]


allowed_origins = get_allowed_origins()

app = FastAPI(
    title="Data Model Insight Accelerator API",
    version="1.0.0",
    description=(
        "FastAPI version of the natural-language-to-SQL orchestrator. "
        "Use `/chat` for questions and `/schema/config` to manage column metadata."
    ),
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(settings_db_router)


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Log uncaught application exceptions for CloudWatch visibility."""
    logger.exception(
        "Unhandled application error on %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(status_code=500, content={"detail": str(error)})


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""

    question: str = Field(default="", description="User question to answer.")
    message: str = Field(
        default="", description="Alternate frontend field for the user question."
    )

    @model_validator(mode="after")
    def normalize_question(self) -> "ChatRequest":
        """Accept either `question` or `message` from the client."""
        normalized_question = (self.question or self.message).strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        self.question = normalized_question
        return self


class ColumnConfigRow(BaseModel):
    """Editable metadata for one database column."""

    column_name: str
    data_type: str = ""
    required: bool = False
    primary_key: bool = False
    display_name: str = ""
    description: str = ""
    include_in_assistant: bool = True


class DBConnectionRequest(DBConnectionCreate):
    """Legacy db-connections request with two-step save confirmation."""

    confirm_save: bool = Field(
        default=False,
        description=(
            "When false, only validate the connection settings. "
            "When true, validate again and persist the connection."
        ),
    )


QUESTION_FIELD_NAMES: tuple[str, ...] = (
    "question",
    "message",
    "query",
    "prompt",
    "input",
    "text",
    "content",
    "user_question",
    "userQuery",
)


def extract_question_from_payload(payload: Any) -> str:
    """Return the first non-empty question string from a request payload."""
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, list):
        for item in reversed(payload):
            extracted_question = extract_question_from_payload(item)
            if extracted_question:
                return extracted_question
        return ""

    if isinstance(payload, dict):
        for field_name in QUESTION_FIELD_NAMES:
            field_value = payload.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                return field_value.strip()

        message_list = payload.get("messages")
        if isinstance(message_list, list):
            extracted_question = extract_question_from_payload(message_list)
            if extracted_question:
                return extracted_question

        for nested_value in payload.values():
            if isinstance(nested_value, (dict, list)):
                extracted_question = extract_question_from_payload(nested_value)
                if extracted_question:
                    return extracted_question

        string_values = [
            value.strip() for value in payload.values() if isinstance(value, str)
        ]
        if len(string_values) == 1 and string_values[0]:
            return string_values[0]

    return ""


async def read_request_payload(request: Request) -> Any:
    """Read JSON, form, or plain-text request bodies safely."""
    content_type = request.headers.get("content-type", "").lower()

    try:
        if "application/json" in content_type:
            return await request.json()

        raw_body = await request.body()
    except (ValueError, UnicodeDecodeError):
        return {}

    if not raw_body:
        return {}

    if "application/x-www-form-urlencoded" in content_type:
        decoded_body = raw_body.decode("utf-8", errors="ignore")
        return {
            key: value
            for key, value in parse_qsl(decoded_body, keep_blank_values=False)
            if value.strip()
        }

    if "multipart/form-data" in content_type:
        with contextlib.suppress(AssertionError, ValueError):
            return dict(await request.form())

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.decode("utf-8", errors="ignore")


def serialize_dataframe(data_frame: pd.DataFrame) -> dict[str, Any]:
    """Convert a DataFrame into a JSON-friendly structure.

    Args:
        data_frame: Pandas DataFrame returned by the orchestrator.

    Returns:
        A dictionary containing rows, columns, and row count.
    """
    safe_frame = data_frame.astype(object).where(data_frame.notna(), None)
    return {
        "columns": safe_frame.columns.tolist(),
        "rows": safe_frame.to_dict(orient="records"),
        "row_count": len(safe_frame.index),
    }


def serialize_figure_to_base64(figure: Any) -> str | None:
    """Encode a Matplotlib figure as a base64 PNG string."""
    if figure is None:
        return None

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def log_captured_orchestrator_output(
    question: str,
    raw_logs: str,
    level: int = logging.INFO,
) -> None:
    """Write captured orchestrator stdout and stderr into application logs.

    Args:
        question: User question tied to the orchestrator run.
        raw_logs: Captured stdout and stderr text.
        level: Logging level used for the message.
    """
    cleaned_logs = raw_logs.strip()
    if not cleaned_logs:
        return

    logger.log(
        level,
        "Orchestrator output for question '%s':\n%s",
        question,
        cleaned_logs,
    )


def build_chat_response(result: Any, logs: str) -> dict[str, Any]:
    """Normalize orchestrator output for API responses."""
    payload: dict[str, Any] = {
        "logs": logs.strip() or None,
        "data": None,
        "chart_image_base64": None,
    }

    is_chart_result = isinstance(result, dict) and result.get("type") == "chart"
    if is_chart_result:
        chart_figure = result.get("figure")
        chart_data = result.get("data")
        payload["answer_type"] = "chart"
        payload["message"] = "Here is your chart."
        payload["chart_image_base64"] = serialize_figure_to_base64(chart_figure)
        if isinstance(chart_data, pd.DataFrame):
            payload["data"] = serialize_dataframe(chart_data)
        else:
            payload["data"] = chart_data
        return payload

    if isinstance(result, pd.DataFrame):
        payload["answer_type"] = "table"
        payload["message"] = "Here are the query results."
        payload["data"] = serialize_dataframe(result)
        return payload

    if isinstance(result, (dict, list)):
        payload["answer_type"] = "json"
        payload["message"] = "Structured response generated."
        payload["data"] = result
        return payload

    payload["answer_type"] = "text"
    payload["message"] = str(result)
    return payload


@app.get("/")
def get_root() -> dict[str, Any]:
    """Return a small API index for quick discovery."""
    return {
        "message": "FastAPI service is running.",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/login",
            "/auth/login",
            "/auth/callback",
            "/auth/session",
            "/settings/db-connections/engines",
            "/settings/db-connections/validate",
            "/settings/db-connections",
            "/chat",
            "/schema/summary",
            "/schema/config",
        ],
    }


@app.get("/health")
def get_health() -> dict[str, str]:
    """Return a simple health response for monitoring."""
    return {"status": "ok"}


@app.get("/schema/summary")
def get_schema_summary() -> dict[str, str]:
    """Return the assistant-facing schema summary."""
    return {"summary": Summarizer.summarize_schema()}


@app.get("/schema/tables")
def get_schema_tables(
    connection_id: str,
    schema_name: str = "public",
) -> dict[str, Any]:
    """Return the list of tables and their count from the target database."""
    return get_tables_for_connection(connection_id, schema_name)


@app.get("/schema/config")
def get_schema_configuration(
    connection_id: str | None = None,
    schema_name: str = "public",
    table_name: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    """Return the full schema plus editable column metadata.

    When connection_id is provided, fetches schema config
    (columns, primary keys, foreign keys) from the target database.
    Otherwise returns locally loaded schema configuration.

    Query params:
        connection_id: Target database connection to query.
        schema_name: Schema inside the target database (default: public).
        table_name: One or more table names to include (repeat for multiple).
            When omitted, all tables in the schema are returned.
    """
    if connection_id:
        if table_name:
            selected_tables = table_name
        else:
            tables_result = get_tables_for_connection(connection_id, schema_name)
            selected_tables = tables_result["table_names"]

        return generate_schema_config(connection_id, selected_tables, schema_name)

    return jsonable_encoder(build_db_column_configuration_payload())


@app.get("/schema/config/{table_name}")
def get_table_configuration(
    table_name: str,
    connection_id: str | None = None,
    schema_name: str = "public",
) -> dict[str, Any]:
    """Return column details for one table.

    When connection_id is provided, fetches columns from the target database.
    Otherwise returns locally loaded column metadata.
    """
    if connection_id:
        return get_table_columns_for_connection(connection_id, table_name, schema_name)

    table_payload = get_table_configuration_payload(table_name)
    if table_payload is None:
        raise HTTPException(status_code=404, detail=f"Unknown table: {table_name}")

    return jsonable_encoder(table_payload)


@app.post("/schema/config/{table_name}")
def save_table_metadata(
    table_name: str,
    rows: list[ColumnConfigRow],
) -> dict[str, Any]:
    """Persist column metadata for one table."""
    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    if table_name not in schema_data.get("tables", {}):
        raise HTTPException(status_code=404, detail=f"Unknown table: {table_name}")

    save_table_configuration(table_name, [row.model_dump() for row in rows])
    updated_payload = get_table_configuration_payload(table_name)
    return jsonable_encoder(updated_payload)


@app.delete("/schema/config/{table_name}")
def reset_table_metadata(table_name: str) -> dict[str, str]:
    """Reset saved metadata for one table."""
    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    if table_name not in schema_data.get("tables", {}):
        raise HTTPException(status_code=404, detail=f"Unknown table: {table_name}")

    SchemaConfigManager.reset_table_configuration(table_name)
    return {"status": "reset", "table_name": table_name}


@app.post("/settings/db-connections", status_code=201)
def create_db_connection(
    payload: DBConnectionRequest,
    response: Response,
) -> dict[str, Any]:
    """Validate first and persist only when the request is confirmed."""
    validation_response = validate_settings_db_connection(payload)
    if not payload.confirm_save:
        response.status_code = 200
        return validation_response

    return create_settings_db_connection(payload)


@app.post("/chat")
async def chat(
    request: Request,
    question: str | None = None,
    message: str | None = None,
    query: str | None = None,
    prompt: str | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    """Run the orchestrator and return a JSON response."""
    normalized_question = (question or message or query or prompt or text or "").strip()
    if not normalized_question:
        payload = await read_request_payload(request)
        normalized_question = extract_question_from_payload(payload)

    if not normalized_question:
        return jsonable_encoder(
            {
                "question": "",
                "answer_type": "text",
                "message": (
                    "Please send a question using `question`, `message`, `query`, "
                    "or `messages[].content`."
                ),
                "data": None,
                "logs": None,
                "chart_image_base64": None,
            }
        )

    try:
        from app.orchestrator.orch_exec import Orchestrator

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = Orchestrator(normalized_question).executor()
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        log_captured_orchestrator_output(
            normalized_question,
            buffer.getvalue(),
            logging.ERROR,
        )
        logger.exception("Chat request failed for question: %s", normalized_question)
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception:
        log_captured_orchestrator_output(
            normalized_question,
            buffer.getvalue(),
            logging.ERROR,
        )
        raise

    log_captured_orchestrator_output(
        normalized_question,
        buffer.getvalue(),
        logging.INFO,
    )

    response_payload = {
        "question": normalized_question,
        **build_chat_response(result, buffer.getvalue()),
    }
    return jsonable_encoder(response_payload)
