from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.settings_db.controllers import (
    create_db_connection,
    get_supported_connection_engines,
    list_all_db_connections,
    validate_db_connection,
)
from app.target_db.controllers import get_tables_for_connection
from app.settings_db.models import DBConnectionCreate, DBConnectionValidationRequest
from app.settings_db.repositories import SelectedTableRepository
from pydantic import BaseModel

settings_db_router = APIRouter(
    prefix="/settings/db-connections",
    tags=["settings-db"],
)


@settings_db_router.get("/engines")
def get_db_connection_engines() -> dict[str, list[str]]:
    """Return supported database engines for the settings UI."""
    return {"engines": get_supported_connection_engines()}


@settings_db_router.post("/validate")
def validate_db_connection_settings(
    payload: DBConnectionValidationRequest,
) -> dict[str, Any]:
    """Validate a database connection payload without persisting it."""
    return validate_db_connection(payload)


@settings_db_router.get("")
def get_all_db_connections() -> dict[str, list[dict[str, Any]]]:
    """Get all saved database connections."""
    connections = list_all_db_connections()
    return {"connections": connections}


@settings_db_router.post("")
def save_db_connection(
    payload: DBConnectionCreate,
    skip_validation: bool = Query(default=False, description="Skip connection validation"),
) -> dict[str, Any]:
    """Validate and persist a database connection payload."""
    return create_db_connection(payload, skip_validation)


@settings_db_router.get("/{connection_id}/tables")
def get_connection_tables(
    connection_id: str,
    schema_name: str = Query(default="public", description="Database schema name"),
) -> dict[str, Any]:
    """Get all tables from a database connection."""
    return get_tables_for_connection(connection_id, schema_name)


class SaveSelectedTablesRequest(BaseModel):
    """Request body for saving selected tables."""
    table_names: list[str]
    schema_name: str = "public"


@settings_db_router.post("/{connection_id}/selected-tables")
def save_selected_tables(
    connection_id: str,
    payload: SaveSelectedTablesRequest,
) -> dict[str, Any]:
    """Save selected tables for a connection."""
    repository = SelectedTableRepository()
    saved_tables = repository.bulk_replace_selected_tables(
        connection_id,
        payload.table_names,
        payload.schema_name,
    )
    return {
        "connection_id": connection_id,
        "selected_tables": saved_tables,
        "count": len(saved_tables),
    }


@settings_db_router.get("/{connection_id}/selected-tables")
def get_selected_tables(connection_id: str) -> dict[str, Any]:
    """Get selected tables for a connection."""
    repository = SelectedTableRepository()
    selected_tables = repository.get_selected_tables_by_connection(connection_id)
    return {
        "connection_id": connection_id,
        "selected_tables": selected_tables,
        "count": len(selected_tables),
    }
