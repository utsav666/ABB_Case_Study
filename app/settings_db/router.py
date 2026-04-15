from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.settings_db.controllers import (
    create_db_connection,
    get_supported_connection_engines,
    validate_db_connection,
)
from app.settings_db.models import DBConnectionCreate, DBConnectionValidationRequest

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


@settings_db_router.post("")
def save_db_connection(payload: DBConnectionCreate) -> dict[str, Any]:
    """Validate and persist a database connection payload."""
    return create_db_connection(payload)
