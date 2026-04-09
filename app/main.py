from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from envsetup.env_loader import load_env
from utils.db_column_config_view import (
    build_db_column_configuration_payload,
    get_table_configuration_payload,
    save_table_configuration,
)
from utils.schema_config import SchemaConfigManager
from utils.schema_summarizer import Summarizer

load_env()

app = FastAPI(
    title="Data Model Insight Accelerator API",
    version="1.0.0",
    description=(
        "FastAPI version of the natural-language-to-SQL orchestrator. "
        "Use `/chat` for questions and `/schema/config` to manage column metadata."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

        if (
            "application/x-www-form-urlencoded" in content_type
            or "multipart/form-data" in content_type
        ):
            return dict(await request.form())

        raw_body = await request.body()
    except (ValueError, UnicodeDecodeError):
        return {}

    if not raw_body:
        return {}

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


@app.get("/schema/config")
def get_schema_configuration() -> dict[str, Any]:
    """Return the full schema plus editable column metadata."""
    return jsonable_encoder(build_db_column_configuration_payload())


@app.get("/schema/config/{table_name}")
def get_table_configuration(table_name: str) -> dict[str, Any]:
    """Return editable configuration rows for one table."""
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
        from orchestrator.orch_exec import Orchestrator

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = Orchestrator(normalized_question).executor()
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    response_payload = {
        "question": normalized_question,
        **build_chat_response(result, buffer.getvalue()),
    }
    return jsonable_encoder(response_payload)
