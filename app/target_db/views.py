from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from app.target_db.connection_profiles import build_target_db_url as build_database_url

logger = logging.getLogger(__name__)


def build_target_db_url(credentials: dict[str, Any]) -> str:
    """Build a target database URL from stored credentials."""
    return build_database_url(credentials)


def build_tables_response(
    connection_id: str,
    schema_name: str,
    table_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build API payload for table-name metadata."""
    table_names = [row["table_name"] for row in table_rows]
    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table_names": table_names,
        "table_count": len(table_names),
    }


def build_table_columns_response(
    connection_id: str,
    schema_name: str,
    table_name: str,
    columns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build API payload for table column metadata."""
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


def build_schema_config_table_payload(
    schema_name: str,
    table_name: str,
    column_rows: list[dict[str, Any]],
    foreign_key_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build schema payload for one table with PK and FK metadata."""
    if not column_rows:
        logger.warning(
            "Table '%s' not found in schema '%s', skipping.",
            table_name,
            schema_name,
        )
        return None

    columns: dict[str, Any] = {}
    primary_key: list[str] = []
    for column_row in column_rows:
        column_name = column_row["column_name"]
        is_primary_key = bool(column_row["is_pk"])
        columns[column_name] = {
            "type": column_row["data_type"].upper(),
            "notnull": column_row["is_nullable"] == "NO",
            "default": column_row["column_default"],
            "is_pk": is_primary_key,
        }
        if is_primary_key:
            primary_key.append(column_name)

    foreign_keys_by_constraint: dict[str, dict[str, Any]] = {}
    for foreign_key_row in foreign_key_rows:
        constraint_name = foreign_key_row["constraint_name"]
        if constraint_name not in foreign_keys_by_constraint:
            foreign_keys_by_constraint[constraint_name] = {
                "from": [],
                "to_table": f"{schema_name}.{foreign_key_row['to_table']}",
                "to_columns": [],
            }

        foreign_keys_by_constraint[constraint_name]["from"].append(
            foreign_key_row["from_column"]
        )
        foreign_keys_by_constraint[constraint_name]["to_columns"].append(
            foreign_key_row["to_column"]
        )

    return {
        "type": "table",
        "columns": columns,
        "primary_key": primary_key,
        "foreign_keys": list(foreign_keys_by_constraint.values()),
    }


def build_schema_config_response(
    connection_id: str,
    schema_name: str,
    tables: dict[str, Any],
) -> dict[str, Any]:
    """Build API payload for generated schema configuration."""
    if not tables:
        raise HTTPException(
            status_code=404,
            detail="None of the requested tables were found in the target database.",
        )

    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table_count": len(tables),
        "config": {"tables": tables},
    }
