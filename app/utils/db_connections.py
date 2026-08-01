from __future__ import annotations

import logging
import os
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from app.utils.query import (
    FETCH_CONNECTION_BY_ID,
    FETCH_TABLE_COLUMNS,
    FETCH_TABLES,
    INSERT_DB_CONNECTION,
)

logger = logging.getLogger(__name__)

ALLOWED_ENGINES = {"postgres", "snowflake", "bigquery"}


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class DBConnectionCreate(BaseModel):
    """Request body for creating a new database connection."""

    workspace_id: UUID = Field(description="UUID of the parent workspace.")
    name: str = Field(min_length=1, description="Display name for this connection.")
    engine: str = Field(
        description="Database engine: postgres, snowflake, or bigquery."
    )
    db_host: str = Field(min_length=1, description="Database host address.")
    db_port: int = Field(default=5432, ge=1, le=65535, description="Database port.")
    db_name: str = Field(min_length=1, description="Database name.")
    db_user: str = Field(min_length=1, description="Database user.")
    db_password: str = Field(min_length=1, description="Database password.")

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "Production Postgres",
                "engine": "postgres",
                "db_host": "db.example.com",
                "db_port": 5432,
                "db_name": "analytics",
                "db_user": "app_user",
                "db_password": "secret",
            }
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_env(value: str) -> str:
    """Strip whitespace and surrounding quotes from an env var value.

    Docker's --env-file does NOT strip surrounding quotes from values,
    unlike python-dotenv. This normalises both cases so the same .env
    file works locally (uvicorn) and inside a Docker container.
    """
    return value.strip().strip('"\'')


def _get_settings_db_url() -> str:
    """Build the settings database connection URL from environment variables."""
    db_host = _strip_env(os.getenv("SETTINGS_DB_HOST", os.getenv("DB_HOST", "")))
    db_port = _strip_env(os.getenv("SETTINGS_DB_PORT", os.getenv("DB_PORT", "5432")))
    db_name = _strip_env(os.getenv("SETTINGS_DB_NAME", os.getenv("DB_NAME", "")))
    db_user = _strip_env(os.getenv("SETTINGS_DB_USER", os.getenv("DB_USER", "")))
    db_password = _strip_env(os.getenv("SETTINGS_DB_PASSWORD", os.getenv("DB_PASSWORD", "")))

    required_vars = {
        "SETTINGS_DB_HOST": db_host,
        "SETTINGS_DB_NAME": db_name,
        "SETTINGS_DB_USER": db_user,
        "SETTINGS_DB_PASSWORD": db_password,
    }
    missing_variables = [name for name, value in required_vars.items() if not value]

    if missing_variables:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_variables)}"
        )

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def _run_settings_query(query, params: dict, *, commit: bool = False) -> list[dict]:
    """Execute a query against the settings database and return mapped rows."""
    settings_engine = create_engine(_get_settings_db_url())
    try:
        with settings_engine.connect() as connection:
            result = connection.execute(query, params)
            if commit:
                connection.commit()
            return [dict(row) for row in result.mappings()]
    except ValueError:
        raise
    except Exception as error:
        logger.exception("Settings DB query failed")
        raise HTTPException(
            status_code=500, detail=f"Database error: {error}"
        ) from error
    finally:
        settings_engine.dispose()


def _run_target_query(target_url: str, query, params: dict) -> list[dict]:
    """Execute a query against a target database and return mapped rows."""
    target_engine = create_engine(target_url)
    try:
        with target_engine.connect() as connection:
            result = connection.execute(query, params)
            return [dict(row) for row in result.mappings()]
    except Exception as error:
        logger.exception("Target DB query failed")
        raise HTTPException(
            status_code=500, detail=f"Could not connect to target database: {error}"
        ) from error
    finally:
        target_engine.dispose()


def _get_connection_by_id(connection_id: str) -> dict:
    """Fetch stored credentials for a connection from settings.db_connections."""
    rows = _run_settings_query(FETCH_CONNECTION_BY_ID, {"connection_id": connection_id})
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"Connection '{connection_id}' not found."
        )
    return rows[0]


def _build_target_db_url(credentials: dict) -> str:
    """Build a SQLAlchemy connection URL from stored credentials."""
    return (
        f"postgresql://{credentials['db_user']}:{credentials['db_password']}"
        f"@{credentials['db_host']}:{credentials['db_port']}/{credentials['db_name']}"
    )


def _connect_target(connection_id: str) -> str:
    """Look up credentials and return the target database URL."""
    credentials = _get_connection_by_id(connection_id)
    return _build_target_db_url(credentials)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def insert_db_connection(payload: DBConnectionCreate) -> dict:
    """Insert a new row into settings.db_connections and return the created record."""
    if payload.engine not in ALLOWED_ENGINES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid engine '{payload.engine}'. "
            f"Must be one of: {', '.join(sorted(ALLOWED_ENGINES))}.",
        )

    rows = _run_settings_query(
        INSERT_DB_CONNECTION,
        {
            "workspace_id": str(payload.workspace_id),
            "name": payload.name,
            "engine": payload.engine,
            "db_host": payload.db_host,
            "db_port": payload.db_port,
            "db_name": payload.db_name,
            "db_user": payload.db_user,
            "db_password": payload.db_password,
        },
        commit=True,
    )

    if not rows:
        raise HTTPException(status_code=500, detail="Insert did not return a row.")

    row = rows[0]
    return {
        "status": "created",
        "connection": {
            "id": str(row["id"]),
            "workspace_id": str(row["workspace_id"]),
            "name": row["name"],
            "engine": row["engine"],
            "db_host": row["db_host"],
            "db_port": row["db_port"],
            "db_name": row["db_name"],
            "db_user": row["db_user"],
            "is_healthy": row["is_healthy"],
            "last_synced_at": str(row["last_synced_at"])
            if row["last_synced_at"]
            else None,
            "created_at": str(row["created_at"]),
        },
    }


def get_tables_for_connection(connection_id: str, schema_name: str = "public") -> dict:
    """Fetch all table names from the target database."""
    target_url = _connect_target(connection_id)
    rows = _run_target_query(target_url, FETCH_TABLES, {"schema_name": schema_name})
    table_names = [row["table_name"] for row in rows]

    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table_names": table_names,
        "table_count": len(table_names),
    }


def get_table_columns_for_connection(
    connection_id: str, table_name: str, schema_name: str = "public"
) -> dict:
    """Fetch column details for a specific table from the target database."""
    target_url = _connect_target(connection_id)
    columns = _run_target_query(
        target_url,
        FETCH_TABLE_COLUMNS,
        {"schema_name": schema_name, "table_name": table_name},
    )

    if not columns:
        raise HTTPException(
            status_code=404,
            detail=f"Table '{table_name}' not found in schema '{schema_name}'.",
        )

    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table_name": table_name,
        "columns": columns,
        "column_count": len(columns),
    }
