from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.target_db.connection_profiles import (
    build_connect_args,
    build_target_db_url,
    get_validation_query,
    normalize_database_engine,
)

logger = logging.getLogger(__name__)


def validate_target_connection_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a target database connection payload."""
    engine_name = normalize_database_engine(str(payload.get("engine", "")))
    connection_url = build_target_db_url(payload)

    if engine_name == "sqlite":
        database_name = str(payload.get("db_name", "")).strip()
        if database_name != ":memory:":
            database_path = Path(database_name).expanduser()
            if not database_path.is_absolute():
                database_path = Path.cwd() / database_path
            if not database_path.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"SQLite database file not found: {database_path}",
                )

    connect_args = build_connect_args(engine_name)
    # Removed aggressive timeout that was causing AWS rate limiting
    # RDS connections now use default driver timeout (~60s)

    validation_engine = create_engine(
        connection_url,
        # Removed pool_pre_ping to avoid extra SELECT 1 queries
        connect_args=connect_args,
    )
    try:
        with validation_engine.connect() as connection:
            connection.execute(text(get_validation_query(engine_name)))
    except ModuleNotFoundError as error:
        logger.exception("Database driver is missing for engine '%s'", engine_name)
        raise HTTPException(
            status_code=500,
            detail=(
                f"The database driver for '{engine_name}' is not installed in this runtime."
            ),
        ) from error
    except (SQLAlchemyError, ValueError) as error:
        logger.exception(
            "Database connection validation failed for engine '%s'", engine_name
        )
        raise HTTPException(
            status_code=422,
            detail=f"Connection validation failed: {error}",
        ) from error
    finally:
        validation_engine.dispose()

    return {
        "status": "validated",
        "can_save": True,
        "engine": engine_name,
        "message": "Connection validated successfully.",
    }
