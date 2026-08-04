from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from typing import Any
from urllib.parse import parse_qsl

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from app.api.chat_response import (
    build_chat_response,
    build_persisted_assistant_content,
)
from app.auth.oauth_router import auth_router
from app.envsetup.env_loader import get_llm_model, load_env
from app.settings_db.controllers import (
    create_db_connection as create_settings_db_connection,
)
from app.settings_db.controllers import (
    validate_db_connection as validate_settings_db_connection,
)
from app.settings_db.models import AgentCreate, AgentUpdate, DBConnectionCreate
from app.settings_db.repositories import (
    AgentRepository,
    ColumnDefinitionRepository,
    SelectedTableRepository,
)
from app.settings_db.router import settings_db_router
from app.settings_db.workspace_router import workspace_router
from app.utils.db_column_config_view import (
    get_table_configuration_payload,
    save_table_configuration,
)
from app.utils.db_connections import (
    get_table_columns_for_connection,
    get_tables_for_connection,
)
from app.utils.schema_config import SchemaConfigManager
from app.utils.schema_generator import generate_schema_config

logger = logging.getLogger(__name__)

load_env()


def configure_logging() -> None:
    """Configure application logging for local runs and Lambda CloudWatch.

    Set ``APP_LOG_FORMAT=json`` for one-JSON-object-per-line output, which lets
    CloudWatch Logs Insights filter on request_id / connection_id / thread_id
    directly. Defaults to human-readable.
    """
    from app.utils.logging_context import (
        ContextFilter,
        HumanFormatter,
        JsonFormatter,
    )

    log_level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    use_json = os.getenv("APP_LOG_FORMAT", "").lower() == "json"

    formatter = JsonFormatter() if use_json else HumanFormatter()
    context_filter = ContextFilter()

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        root_logger.addHandler(handler)
    else:
        # Lambda pre-installs a handler; re-dress it rather than adding a
        # second one, which would double every line.
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(context_filter)

    root_logger.setLevel(log_level)
    logger.setLevel(log_level)

    # These are chatty at INFO and drown out application logs.
    for noisy in ("httpx", "httpcore", "urllib3", "botocore", "boto3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
app.include_router(workspace_router)


@app.on_event("shutdown")
def _dispose_pooled_db_engines() -> None:
    """Release pooled DB connections so restarts/reloads are clean."""
    from app.settings_db.gateway import dispose_all_engines
    from app.target_db.gateway import dispose_all_target_engines

    dispose_all_engines()
    dispose_all_target_engines()


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
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID for continuing an existing conversation.",
    )
    connection_id: str | None = Field(
        default=None,
        description="DB connection ID scoping the query.",
    )
    workspace_id: str | None = Field(
        default=None,
        description="Workspace ID for conversation persistence.",
    )

    @model_validator(mode="after")
    def normalize_question(self) -> ChatRequest:
        """Accept either `question` or `message` from the client."""
        normalized_question = (self.question or self.message).strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        self.question = normalized_question
        return self


class ColumnConfigRow(BaseModel):
    """Editable metadata for one database column."""

    # Core fields shared by local config and settings DB
    column_name: str
    data_type: str = ""
    display_name: str = ""
    description: str = ""

    # Local-config-only fields (used for demo/local schema file)
    required: bool = False
    primary_key: bool = False
    include_in_assistant: bool = True

    # Settings-DB-only fields (stored in column_definitions table)
    is_pii: bool = False
    is_filterable: bool = True
    example_values: list[str] = Field(default_factory=list)


class TableConfigRequest(BaseModel):
    """Request body for saving table column configuration."""

    rows: list[ColumnConfigRow]

    # When provided, also saves to settings.selected_tables + column_definitions
    connection_id: str | None = None
    schema_name: str = "public"
    human_description: str | None = None


class TableSelection(BaseModel):
    """One table with an optional column filter."""

    table_name: str
    column_names: list[str] | None = Field(
        default=None,
        description=(
            "Columns the user selected for this table. "
            "When omitted or empty, all columns are included."
        ),
    )

    @model_validator(mode="after")
    def normalize_empty_columns(self) -> TableSelection:
        """Treat an empty column list the same as None (all columns)."""
        if self.column_names is not None and len(self.column_names) == 0:
            self.column_names = None
        return self


class BulkTableConfigRequest(BaseModel):
    """Request body for saving schema configuration.

    Supports four selection modes:
      1. All tables, all columns        → omit `tables`
      2. Specific tables, all columns   → tables: [{table_name: "orders"}, ...]
      3. Specific tables, specific cols  → tables: [{table_name: "orders", column_names: ["id","total"]}, ...]
      4. All tables, specific cols each  → fetch all tables first, then filter cols per entry
    """

    connection_id: str
    schema_name: str = "public"
    tables: list[TableSelection] | None = Field(
        default=None,
        description=(
            "Tables (and optionally columns) the user selected. "
            "When omitted, every table with all columns is saved."
        ),
    )


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
def get_schema_summary(connection_id: str) -> dict[str, str]:
    """Return the assistant-facing schema summary for a connection.

    connection_id is REQUIRED. This previously took no arguments and rendered the
    bundled sample fixture, so it described a movie database no matter which
    connection the caller cared about. The schema now comes only from
    settings.schema_graphs.
    """
    from app.target_db.controllers import load_schema_graph, render_graph_summary

    graph = load_schema_graph(connection_id)
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No schema graph for connection {connection_id}. "
                "Profile the connection first."
            ),
        )
    return {"summary": render_graph_summary(graph)}


@app.get("/schema/tables")
def get_schema_tables(
    connection_id: str,
    schema_name: str = "public",
) -> dict[str, Any]:
    """Return the list of tables and their count from the target database."""
    return get_tables_for_connection(connection_id, schema_name)


class SchemaConfigRequest(BaseModel):
    """Request body for POST /schema/config with full table+column selection.

    Supports the same four scenarios as bulk save:
      1. All tables, all columns        → omit `tables`
      2. Specific tables, all columns   → tables: [{table_name: "x"}, ...]
      3. Specific tables, specific cols  → tables: [{table_name: "x", column_names: ["a","b"]}, ...]
      4. Mixed                           → some entries with column_names, some without
    """

    connection_id: str
    schema_name: str = "public"
    workspace_id: str | None = Field(
        default=None,
        description=(
            "When provided, the generated schema config is saved "
            "into the agent's config JSONB column. If no agent exists "
            "for this workspace, one is created automatically."
        ),
    )
    tables: list[TableSelection] | None = Field(
        default=None,
        description=(
            "Tables (and optionally columns) to include. "
            "When omitted, every table with all columns is returned."
        ),
    )
    profile: bool = Field(
        default=False,
        description=(
            "When true, run Phase 1 statistical data profiling on every "
            "column (cardinality, distinct values, min/max, date ranges)."
        ),
    )
    annotate: bool = Field(
        default=False,
        description=(
            "When true (requires profile=true), run Phase 2 LLM semantic "
            "annotation to generate value descriptions, aliases, and query hints."
        ),
    )
    save_local: bool = Field(
        default=False,
        description=(
            "When true, also persist the (enriched) config to the local "
            "schema_preserve.json file for the existing pipeline."
        ),
    )


@app.post("/schema/config")
def post_schema_configuration(payload: SchemaConfigRequest) -> dict[str, Any]:
    """Return schema config with full table + column selection.

    Supports all four selection modes via the request body.
    """
    if payload.tables:
        selected_tables = [t.table_name for t in payload.tables]
        # Build column filter: only for entries that specify column_names
        column_filter: dict[str, list[str]] = {
            t.table_name: t.column_names
            for t in payload.tables
            if t.column_names is not None
        }
    else:
        tables_result = get_tables_for_connection(
            payload.connection_id, payload.schema_name
        )
        selected_tables = tables_result["table_names"]
        column_filter = {}

    schema_config = generate_schema_config(
        payload.connection_id,
        selected_tables,
        payload.schema_name,
        column_filter=column_filter or None,
    )

    # ── Semantic enrichment (Phase 1 profiling + Phase 2 annotation) ─────
    if payload.profile:
        from app.semantic_catalog.catalog_manager import (
            enrich_schema_config_for_connection,
            save_enriched_config_to_local_file,
        )

        schema_config = enrich_schema_config_for_connection(
            connection_id=payload.connection_id,
            schema_config=schema_config,
            annotate=payload.annotate,
        )
        logger.info(
            "Schema enrichment complete: profiled=%s, annotated=%s",
            schema_config.get("profiled"),
            schema_config.get("annotated"),
        )

        if payload.save_local:
            save_enriched_config_to_local_file(schema_config)

    # Save the generated config into the agent's config JSONB when workspace_id is provided
    if payload.workspace_id:
        agent_repo = AgentRepository()
        agents = agent_repo.list_agents_by_workspace(payload.workspace_id)

        if agents:
            # Update the first (default) agent for this workspace
            agent = agents[0]
            agent_repo.update_agent(
                str(agent["id"]),
                {"config": schema_config},
            )
            schema_config["agent_id"] = str(agent["id"])
        else:
            # Auto-create an agent for this workspace
            from uuid import UUID as _UUID

            new_agent = agent_repo.create_agent(
                {
                    "workspace_id": _UUID(payload.workspace_id),
                    "name": "Data Assistant",
                    "model_version": get_llm_model(),
                    "config": schema_config,
                    "is_public_to_team": True,
                }
            )
            if new_agent:
                schema_config["agent_id"] = str(new_agent["id"])

        schema_config["agent_config_saved"] = True

    return schema_config


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


@app.post("/schema/config/bulk")
def save_all_tables_config(payload: BulkTableConfigRequest) -> dict[str, Any]:
    """Save schema configuration for user-selected tables and columns.

    Selection modes:
      1. All tables, all columns        → omit `tables`
      2. Specific tables, all columns   → tables: [{table_name: "x"}, ...]
      3. Specific tables, specific cols  → tables: [{table_name: "x", column_names: ["a","b"]}, ...]
      4. Mixed                           → some entries with column_names, some without
    """
    # ── Build a mapping: table_name → allowed column names (None = all) ──
    if payload.tables:
        table_column_filter: dict[str, list[str] | None] = {
            t.table_name: t.column_names for t in payload.tables
        }
    else:
        # No selection provided → fetch every table, keep all columns
        tables_result = get_tables_for_connection(
            payload.connection_id, payload.schema_name
        )
        table_column_filter = {name: None for name in tables_result["table_names"]}

    if not table_column_filter:
        return {"saved_tables": 0, "saved_columns": 0, "tables": []}

    selected_table_repo = SelectedTableRepository()
    column_def_repo = ColumnDefinitionRepository()
    summary: list[dict[str, Any]] = []

    for current_table_name, allowed_columns in table_column_filter.items():
        # Fetch all columns from the target DB for this table
        columns_result = get_table_columns_for_connection(
            payload.connection_id,
            current_table_name,
            payload.schema_name,
        )
        raw_columns: list[dict[str, Any]] = columns_result.get("columns", [])

        # Filter to only the columns the user selected (if specified)
        if allowed_columns is not None:
            allowed_set = {c.lower() for c in allowed_columns}
            raw_columns = [
                col for col in raw_columns if col["column_name"].lower() in allowed_set
            ]

        selected_table_row = selected_table_repo.upsert_table(
            connection_id=payload.connection_id,
            schema_name=payload.schema_name,
            table_name=current_table_name,
            human_description=None,
        )
        if selected_table_row is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upsert selected_table for '{current_table_name}'.",
            )

        column_rows = [
            {
                "table_id": selected_table_row["id"],
                "raw_name": col["column_name"],
                "logical_name": None,
                "description": None,
                "data_type": col.get("data_type"),
                "is_pii": False,
                "is_filterable": True,
                "example_values": None,
            }
            for col in raw_columns
        ]
        saved_columns = column_def_repo.replace_columns_for_table(
            table_id=selected_table_row["id"],
            column_rows=column_rows,
        )
        summary.append(
            {
                "table_name": current_table_name,
                "column_count": len(saved_columns),
            }
        )

    return jsonable_encoder(
        {
            "saved_tables": len(summary),
            "saved_columns": sum(t["column_count"] for t in summary),
            "tables": summary,
        }
    )


@app.post("/schema/config/{table_name}")
def save_table_metadata(
    table_name: str,
    payload: TableConfigRequest,
) -> dict[str, Any]:
    """Persist column metadata for one table.

    Always saves to the local schema config file (demo path).
    When connection_id is supplied, also upserts into settings.selected_tables
    and replaces settings.column_definitions for that table.
    """
    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    has_local_table = table_name in schema_data.get("tables", {})

    # ── Local file save (demo path, runs whenever the table exists locally) ──
    if has_local_table:
        save_table_configuration(table_name, [row.model_dump() for row in payload.rows])

    # ── Settings-DB save (runs when caller provides a connection_id) ─────────
    if payload.connection_id:
        selected_table_row = SelectedTableRepository().upsert_table(
            connection_id=payload.connection_id,
            schema_name=payload.schema_name,
            table_name=table_name,
            human_description=payload.human_description,
        )
        if selected_table_row is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to upsert selected_table record.",
            )

        column_rows = [
            {
                "table_id": selected_table_row["id"],
                "raw_name": row.column_name,
                "logical_name": row.display_name or None,
                "description": row.description or None,
                "data_type": row.data_type or None,
                "is_pii": row.is_pii,
                "is_filterable": row.is_filterable,
                "example_values": row.example_values or None,
            }
            for row in payload.rows
        ]
        saved_columns = ColumnDefinitionRepository().replace_columns_for_table(
            table_id=selected_table_row["id"],
            column_rows=column_rows,
        )

        # If there is no local table, return the settings-DB result directly
        if not has_local_table:
            return jsonable_encoder(
                {
                    "table_name": table_name,
                    "selected_table": selected_table_row,
                    "columns": saved_columns,
                }
            )

    if not has_local_table and not payload.connection_id:
        raise HTTPException(status_code=404, detail=f"Unknown table: {table_name}")

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


# ── Agent CRUD ───────────────────────────────────────────────────────────────


@app.get("/settings/llm-providers")
def get_available_llm_providers() -> dict[str, Any]:
    """Return the LLM providers available for selection in the UI."""
    providers = [
        {
            "id": "openai",
            "label": "OpenAI",
            "models": ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
        },
        {
            "id": "azure_openai",
            "label": "Azure OpenAI",
            "models": ["gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
        },
        {
            "id": "claude_merck",
            "label": "Claude (Merck)",
            "models": ["claude-opus-4-6-v1", "claude-sonnet-4-20250514"],
        },
    ]
    return {"providers": providers}


@app.post("/settings/agents", status_code=201)
def create_agent(payload: AgentCreate) -> dict[str, Any]:
    """Create a new agent.

    The config JSONB stores db_connection_id, schema_name, and model params.
    """
    values = {
        "workspace_id": payload.workspace_id,
        "name": payload.name,
        "model_version": payload.model_version,
        "system_instructions": payload.system_instructions,
        "config": payload.config.model_dump(),
        "is_public_to_team": payload.is_public_to_team,
    }
    result = AgentRepository().create_agent(values)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to create agent.")
    return jsonable_encoder(result)


@app.get("/settings/agents/{agent_id}")
def get_agent(agent_id: str) -> dict[str, Any]:
    """Fetch a single agent by ID."""
    result = AgentRepository().get_agent_by_id(agent_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return jsonable_encoder(result)


@app.get("/settings/agents")
def list_agents(workspace_id: str) -> dict[str, Any]:
    """List all agents in a workspace."""
    agents = AgentRepository().list_agents_by_workspace(workspace_id)
    return jsonable_encoder({"agents": agents, "count": len(agents)})


@app.patch("/settings/agents")
def update_agent(workspace_id: str, payload: AgentUpdate) -> dict[str, Any]:
    """Update the agent for a workspace. Only provided fields are changed."""
    agent_repo = AgentRepository()
    agents = agent_repo.list_agents_by_workspace(workspace_id)
    if not agents:
        raise HTTPException(
            status_code=404, detail="No agent found for this workspace."
        )

    agent_id = str(agents[0]["id"])
    updates: dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.model_version is not None:
        updates["model_version"] = payload.model_version
    if payload.system_instructions is not None:
        updates["system_instructions"] = payload.system_instructions
    if payload.config is not None:
        updates["config"] = payload.config.model_dump()
    if payload.is_public_to_team is not None:
        updates["is_public_to_team"] = payload.is_public_to_team

    result = agent_repo.update_agent(agent_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return jsonable_encoder(result)


@app.delete("/settings/agents")
def delete_agent(workspace_id: str) -> dict[str, str]:
    """Delete the agent for a workspace."""
    agent_repo = AgentRepository()
    agents = agent_repo.list_agents_by_workspace(workspace_id)
    if not agents:
        raise HTTPException(
            status_code=404, detail="No agent found for this workspace."
        )

    agent_id = str(agents[0]["id"])
    deleted = agent_repo.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return {"status": "deleted", "agent_id": agent_id}


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
    # Read the body once so we can pull both the question (fallback) and the
    # optional connection_id that scopes the run to a real schema graph.
    body_payload = await read_request_payload(request)
    if not normalized_question:
        normalized_question = extract_question_from_payload(body_payload)

    connection_id: str | None = None
    thread_id_from_client: str | None = None
    workspace_id: str | None = None
    if isinstance(body_payload, dict):
        raw_connection_id = body_payload.get("connection_id")
        if isinstance(raw_connection_id, str) and raw_connection_id.strip():
            connection_id = raw_connection_id.strip()
        raw_thread_id = body_payload.get("thread_id")
        if isinstance(raw_thread_id, str) and raw_thread_id.strip():
            thread_id_from_client = raw_thread_id.strip()
        raw_workspace_id = body_payload.get("workspace_id")
        if isinstance(raw_workspace_id, str) and raw_workspace_id.strip():
            workspace_id = raw_workspace_id.strip()

    # Bind ids to the logging context BEFORE any real work, so every downstream
    # log line — including failures in the early-return paths below — carries
    # them. connection_id in particular: without it in the logs there is no way
    # to tell that a run answered against the wrong database.
    from app.utils.logging_context import bind_log_context, new_request_id

    bind_log_context(
        request_id=new_request_id(),
        connection_id=connection_id or "none",
        workspace_id=workspace_id or "",
    )

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

    import asyncio
    import time
    import uuid

    from app.orchestrator.conversation_manager import ConversationHistoryManager
    from app.orchestrator.orch_graph import GraphOrchestrator

    # Use the client-provided thread_id or generate a new one for HITL
    thread_id = thread_id_from_client or str(uuid.uuid4())
    bind_log_context(thread_id=thread_id)
    logger.info(
        "CHAT — received question=%r thread_id=%s workspace_id=%s connection_id=%s",
        normalized_question[:120],
        thread_id,
        workspace_id,
        connection_id,
    )

    # Retrieve conversation history for an existing thread
    _t0 = time.perf_counter()
    conversation_manager = ConversationHistoryManager()
    prior_history_text: str = ""
    if thread_id_from_client:
        prior_history_text = conversation_manager.format_history_for_context(thread_id)
    logger.info(
        "CHAT — loaded conversation history (%d chars) in %.0f ms",
        len(prior_history_text),
        (time.perf_counter() - _t0) * 1000,
    )

    # Ensure thread row exists before inserting messages.
    _t0 = time.perf_counter()
    if workspace_id:
        conversation_manager.ensure_conversation(
            workspace_id=workspace_id,
            connection_id=connection_id,
            title=normalized_question[:80],
            thread_id=thread_id,
        )

    # Persist the user's message and capture message_id for LLM tracking
    user_message_id = None
    if workspace_id:
        user_msg = conversation_manager.add_message(
            thread_id=thread_id,
            role="user",
            content=normalized_question,
        )
        user_message_id = user_msg.get("id")
    logger.info(
        "CHAT — persisted user message in %.0f ms",
        (time.perf_counter() - _t0) * 1000,
    )

    # Resolve per-workspace LLM provider/model from agent settings
    _t0 = time.perf_counter()
    llm_provider: str | None = None
    llm_model: str | None = None
    if workspace_id:
        try:
            agents = AgentRepository().list_agents_by_workspace(workspace_id)
            if agents:
                agent_config = agents[0].get("config") or {}
                llm_provider = agent_config.get("llm_provider") or None
                llm_model = agents[0].get("model_version") or None
        except Exception:
            logger.debug("Could not load agent config for workspace %s", workspace_id)
    logger.info(
        "CHAT — resolved agent config (provider=%s model=%s) in %.0f ms; starting orchestrator",
        llm_provider,
        llm_model,
        (time.perf_counter() - _t0) * 1000,
    )

    buffer = io.StringIO()
    try:

        def _run_orchestrator() -> Any:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                return GraphOrchestrator(
                    normalized_question,
                    thread_id=thread_id,
                    connection_id=connection_id,
                    conversation_history=prior_history_text,
                    user_message_id=user_message_id,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                ).executor()

        result = await asyncio.to_thread(_run_orchestrator)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        log_captured_orchestrator_output(
            normalized_question,
            buffer.getvalue(),
            logging.ERROR,
        )
        logger.exception("Chat request failed for question: %s", normalized_question)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        ) from error
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
        "thread_id": thread_id,
        **build_chat_response(result, buffer.getvalue(), connection_id),
    }

    # Persist the assistant's reply in a structured format for reload hydration.
    assistant_content = build_persisted_assistant_content(response_payload)
    if workspace_id and assistant_content:
        conversation_manager.add_message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_content,
        )

    return jsonable_encoder(response_payload)


# ============================================================
# Conversation history endpoints
# ============================================================


@app.get("/conversations")
async def list_conversations(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> list[dict[str, Any]]:
    """List all conversation threads for a workspace, newest first."""
    from app.orchestrator.conversation_manager import ConversationHistoryManager

    manager = ConversationHistoryManager()
    return manager.get_workspace_conversations(workspace_id)


@app.get("/conversations/{thread_id}/messages")
async def get_conversation_messages(thread_id: str) -> list[dict[str, Any]]:
    """Retrieve all messages in a conversation thread."""
    from app.orchestrator.conversation_manager import ConversationHistoryManager

    manager = ConversationHistoryManager()
    return manager.get_conversation_history(thread_id)


@app.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str) -> dict[str, Any]:
    """Delete a conversation thread and all its messages."""
    from app.orchestrator.conversation_manager import ConversationHistoryManager

    manager = ConversationHistoryManager()
    deleted = manager.delete_conversation(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "thread_id": thread_id}


# ============================================================
# ### HITL — Resume endpoint for human-in-the-loop responses
# ============================================================


class HitlResumeRequest(BaseModel):
    """### HITL — Request model for resuming a paused graph."""

    thread_id: str
    decision: Any  # Human's choice: string, dict, etc.
    # Optional: lets the resumed response use the same number formatting as the
    # original answer. Absent for older clients, which fall back to auto-detect.
    connection_id: str | None = None


@app.post("/chat/resume")
async def chat_resume(payload: HitlResumeRequest) -> dict[str, Any]:
    """### HITL — Resume a paused orchestrator after human input.

    Called when the /chat endpoint returns a 'pending_hitl' response.
    The frontend sends back the thread_id and the human's decision.
    """
    import asyncio

    from app.orchestrator.orch_graph import GraphOrchestrator

    buffer = io.StringIO()
    try:

        def _resume() -> Any:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                return GraphOrchestrator.resume(payload.thread_id, payload.decision)

        result = await asyncio.to_thread(_resume)
    except Exception as error:
        logger.exception("### HITL — Resume failed for thread: %s", payload.thread_id)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while resuming.",
        ) from error

    response_payload = {
        "thread_id": payload.thread_id,
        **build_chat_response(result, buffer.getvalue(), payload.connection_id),
    }
    return jsonable_encoder(response_payload)
