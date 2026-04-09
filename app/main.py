from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
from typing import Any
from urllib.parse import parse_qsl, urlencode

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from app.envsetup.env_loader import load_env
from app.utils.db_column_config_view import (
    build_db_column_configuration_payload,
    get_table_configuration_payload,
    save_table_configuration,
)
from app.utils.schema_config import SchemaConfigManager
from app.utils.schema_summarizer import Summarizer

logger = logging.getLogger(__name__)

load_env()


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
    root_path=os.getenv("FASTAPI_ROOT_PATH", "").strip(),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
DEFAULT_OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
DEFAULT_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_OAUTH_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DEFAULT_OAUTH_SCOPE = "openid profile email"
DEFAULT_OAUTH_CALLBACK_PATH = "/auth/callback"


def get_oauth_redirect_uri(request: Request) -> str:
    """Return the configured callback URL for OAuth login."""
    configured_redirect_uri = os.getenv("OAUTH_REDIRECT_URI", "").strip()
    if configured_redirect_uri:
        return configured_redirect_uri

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}{DEFAULT_OAUTH_CALLBACK_PATH}"


def build_oauth_login_payload(request: Request) -> dict[str, Any]:
    """Build a frontend-ready OAuth login URL payload."""
    provider_name = os.getenv("OAUTH_PROVIDER", "oauth").strip() or "oauth"
    authorize_url = (
        os.getenv("OAUTH_AUTHORIZE_URL", DEFAULT_OAUTH_AUTHORIZE_URL).strip()
        or DEFAULT_OAUTH_AUTHORIZE_URL
    )
    scope = os.getenv("OAUTH_SCOPE", DEFAULT_OAUTH_SCOPE).strip() or DEFAULT_OAUTH_SCOPE
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()

    query_params = {
        "client_id": client_id,
        "redirect_uri": get_oauth_redirect_uri(request),
        "response_type": "code",
        "scope": scope,
        "state": os.getenv("OAUTH_STATE", "login").strip() or "login",
    }

    access_type = os.getenv("OAUTH_ACCESS_TYPE", "").strip()
    if access_type:
        query_params["access_type"] = access_type

    include_granted_scopes = os.getenv("OAUTH_INCLUDE_GRANTED_SCOPES", "").strip()
    if include_granted_scopes:
        query_params["include_granted_scopes"] = include_granted_scopes

    return {
        "provider": provider_name,
        "configured": bool(client_id),
        "login_url": f"{authorize_url}?{urlencode(query_params)}",
        "redirect_uri": query_params["redirect_uri"],
        "scopes": scope.split(),
    }


def get_oauth_token_url() -> str:
    """Return the configured OAuth token endpoint."""
    configured_token_url = os.getenv("OAUTH_TOKEN_URL", "").strip()
    return configured_token_url or DEFAULT_OAUTH_TOKEN_URL


def get_oauth_userinfo_url() -> str:
    """Return the configured OAuth userinfo endpoint."""
    configured_userinfo_url = os.getenv("OAUTH_USERINFO_URL", "").strip()
    return configured_userinfo_url or DEFAULT_OAUTH_USERINFO_URL


def can_exchange_oauth_code() -> bool:
    """Return whether the backend can exchange the authorization code."""
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    return bool(client_id and client_secret)


def exchange_oauth_code_for_token(code: str, request: Request) -> dict[str, Any]:
    """Exchange an OAuth authorization code for tokens."""
    client_id = os.getenv("OAUTH_CLIENT_ID", "").strip()
    client_secret = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError(
            "Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET to enable backend authentication."
        )

    token_response = requests.post(
        get_oauth_token_url(),
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": get_oauth_redirect_uri(request),
        },
        timeout=30,
    )
    token_response.raise_for_status()
    token_payload = token_response.json()
    if not isinstance(token_payload, dict):
        raise ValueError("OAuth token response must be a JSON object.")

    return token_payload


def fetch_oauth_user_details(access_token: str) -> dict[str, Any]:
    """Fetch the authenticated user's profile from the OAuth provider."""
    if not access_token.strip():
        return {}

    user_response = requests.get(
        get_oauth_userinfo_url(),
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    user_response.raise_for_status()
    user_payload = user_response.json()
    if not isinstance(user_payload, dict):
        raise ValueError("OAuth user info response must be a JSON object.")

    return user_payload


def build_oauth_token_payload(
    token_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the safe token fields to send back to the frontend."""
    if not token_payload:
        return None

    return {
        "access_token": token_payload.get("access_token"),
        "token_type": token_payload.get("token_type"),
        "expires_in": token_payload.get("expires_in"),
        "scope": token_payload.get("scope"),
        "id_token": token_payload.get("id_token"),
        "refresh_token": token_payload.get("refresh_token"),
    }


async def read_oauth_callback_parameters(request: Request) -> dict[str, str]:
    """Read OAuth callback values from the query string or POST body.

    Args:
        request: Incoming FastAPI request.

    Returns:
        A normalized mapping of callback fields.
    """
    callback_values: dict[str, str] = {
        key: value.strip()
        for key, value in request.query_params.items()
        if isinstance(value, str) and value.strip()
    }

    payload = await read_request_payload(request)
    if not isinstance(payload, dict):
        return callback_values

    for key, value in payload.items():
        if value is None:
            continue

        normalized_value = str(value).strip()
        if normalized_value:
            callback_values[key] = normalized_value

    return callback_values


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
            "/chat",
            "/schema/summary",
            "/schema/config",
        ],
    }


@app.get("/health")
def get_health() -> dict[str, str]:
    """Return a simple health response for monitoring."""
    return {"status": "ok"}


@app.get("/login")
@app.get("/auth/login")
def get_login_url(request: Request) -> dict[str, Any]:
    """Return an OAuth authorization URL for the frontend login flow."""
    return build_oauth_login_payload(request)


@app.get(DEFAULT_OAUTH_CALLBACK_PATH)
@app.post(DEFAULT_OAUTH_CALLBACK_PATH)
@app.get("/login/callback")
@app.post("/login/callback")
async def get_auth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> dict[str, Any]:
    """Authenticate the OAuth callback and return the frontend payload."""
    callback_values = await read_oauth_callback_parameters(request)
    provider_name = os.getenv("OAUTH_PROVIDER", "oauth").strip() or "oauth"
    authorization_code = (code or callback_values.get("code", "")).strip() or None
    callback_state = (state or callback_values.get("state", "")).strip() or None
    callback_error = (error or callback_values.get("error", "")).strip() or None
    callback_error_description = (
        error_description or callback_values.get("error_description", "")
    ).strip() or None

    if callback_error:
        return {
            "status": "error",
            "provider": provider_name,
            "code": None,
            "authorization_code": None,
            "state": callback_state,
            "authenticated": False,
            "user": None,
            "token": None,
            "error": callback_error,
            "error_description": (
                callback_error_description or "OAuth login was not completed."
            ),
            "message": "OAuth callback received with an error.",
        }

    if not authorization_code:
        return {
            "status": "pending",
            "provider": provider_name,
            "code": None,
            "authorization_code": None,
            "state": callback_state,
            "authenticated": False,
            "user": None,
            "token": None,
            "error": None,
            "error_description": None,
            "message": (
                "OAuth callback endpoint is ready to receive the authorization code."
            ),
        }

    token_payload: dict[str, Any] | None = None
    user_payload: dict[str, Any] | None = None
    message = "OAuth callback received successfully."

    if can_exchange_oauth_code():
        try:
            token_payload = exchange_oauth_code_for_token(authorization_code, request)
            access_token = str(token_payload.get("access_token", "")).strip()
            user_payload = fetch_oauth_user_details(access_token)
            message = "OAuth callback received and user details were loaded."
        except (ValueError, requests.RequestException) as request_error:
            logger.warning(
                "OAuth backend authentication could not be completed: %s",
                request_error,
            )
            message = (
                "OAuth callback received successfully, but backend authentication "
                "could not be completed."
            )
            return {
                "status": "success",
                "provider": provider_name,
                "code": authorization_code,
                "authorization_code": authorization_code,
                "state": callback_state,
                "authenticated": False,
                "user": None,
                "token": None,
                "error": "authentication_failed",
                "error_description": (
                    "OAuth callback was received, but user details could not be loaded."
                ),
                "message": message,
            }
    else:
        message = (
            "OAuth callback received, but backend token exchange is not configured yet."
        )

    return {
        "status": "success",
        "provider": provider_name,
        "code": authorization_code,
        "authorization_code": authorization_code,
        "state": callback_state,
        "authenticated": bool(user_payload),
        "user": user_payload,
        "token": build_oauth_token_payload(token_payload),
        "error": None,
        "error_description": None,
        "message": message,
    }


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
