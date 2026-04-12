from __future__ import annotations

from typing import Any

from app.settings_db.controllers import get_settings_connection_credentials
from app.target_db.repositories import TargetMetadataRepository
from app.target_db.views import (
    build_schema_config_response,
    build_schema_config_table_payload,
    build_table_columns_response,
    build_tables_response,
    build_target_db_url,
)


class TargetMetadataController:
    """Controller for target database metadata operations."""

    def __init__(self, repository: TargetMetadataRepository | None = None) -> None:
        self._repository = repository or TargetMetadataRepository()

    def get_tables_for_connection(
        self,
        connection_id: str,
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Fetch table names for a configured connection."""
        target_url = self._resolve_target_url(connection_id)
        table_rows = self._repository.fetch_tables(target_url, schema_name)
        return build_tables_response(connection_id, schema_name, table_rows)

    def get_table_columns_for_connection(
        self,
        connection_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Fetch table columns for a configured connection."""
        target_url = self._resolve_target_url(connection_id)
        columns = self._repository.fetch_table_columns(
            target_url,
            schema_name,
            table_name,
        )
        return build_table_columns_response(
            connection_id,
            schema_name,
            table_name,
            columns,
        )

    def generate_schema_config(
        self,
        connection_id: str,
        table_names: list[str],
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Generate schema configuration for selected tables."""
        target_url = self._resolve_target_url(connection_id)
        tables: dict[str, Any] = {}

        for table_name in table_names:
            column_rows = self._repository.fetch_columns_with_primary_keys(
                target_url,
                schema_name,
                table_name,
            )
            foreign_key_rows = self._repository.fetch_foreign_keys(
                target_url,
                schema_name,
                table_name,
            )
            table_payload = build_schema_config_table_payload(
                schema_name,
                table_name,
                column_rows,
                foreign_key_rows,
            )
            if table_payload is None:
                continue

            qualified_table_name = f"{schema_name}.{table_name}"
            tables[qualified_table_name] = table_payload

        return build_schema_config_response(connection_id, schema_name, tables)

    def get_target_db_url(self, connection_id: str) -> str:
        """Resolve and return a target URL for a stored connection."""
        return self._resolve_target_url(connection_id)

    def _resolve_target_url(self, connection_id: str) -> str:
        """Resolve stored credentials into a SQLAlchemy target URL."""
        credentials = get_settings_connection_credentials(connection_id)
        return build_target_db_url(credentials)


def get_tables_for_connection(connection_id: str, schema_name: str = "public") -> dict:
    """Compatibility wrapper for fetching target tables."""
    return TargetMetadataController().get_tables_for_connection(
        connection_id, schema_name
    )


def get_table_columns_for_connection(
    connection_id: str,
    table_name: str,
    schema_name: str = "public",
) -> dict:
    """Compatibility wrapper for fetching target columns."""
    return TargetMetadataController().get_table_columns_for_connection(
        connection_id,
        table_name,
        schema_name,
    )


def generate_target_schema_config(
    connection_id: str,
    table_names: list[str],
    schema_name: str = "public",
) -> dict:
    """Compatibility wrapper for generating target schema config."""
    return TargetMetadataController().generate_schema_config(
        connection_id,
        table_names,
        schema_name,
    )


def get_target_db_url(connection_id: str) -> str:
    """Compatibility wrapper for resolving target DB URL."""
    return TargetMetadataController().get_target_db_url(connection_id)
