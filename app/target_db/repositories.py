from __future__ import annotations

from app.target_db.gateway import TargetDatabaseGateway
from app.target_db.queries import (
    FETCH_COLUMNS_WITH_PK,
    FETCH_FOREIGN_KEYS,
    FETCH_TABLE_COLUMNS,
    FETCH_TABLES,
)


class TargetMetadataRepository:
    """Repository for information_schema metadata queries."""

    def __init__(self, gateway: TargetDatabaseGateway | None = None) -> None:
        self._gateway = gateway or TargetDatabaseGateway()

    def fetch_tables(self, target_url: str, schema_name: str) -> list[dict]:
        """Fetch table names from a target schema."""
        return self._gateway.execute_query(
            target_url,
            FETCH_TABLES,
            {"schema_name": schema_name},
        )

    def fetch_table_columns(
        self,
        target_url: str,
        schema_name: str,
        table_name: str,
    ) -> list[dict]:
        """Fetch table columns from a target schema."""
        return self._gateway.execute_query(
            target_url,
            FETCH_TABLE_COLUMNS,
            {"schema_name": schema_name, "table_name": table_name},
        )

    def fetch_columns_with_primary_keys(
        self,
        target_url: str,
        schema_name: str,
        table_name: str,
    ) -> list[dict]:
        """Fetch table columns with primary key markers."""
        return self._gateway.execute_query(
            target_url,
            FETCH_COLUMNS_WITH_PK,
            {"schema_name": schema_name, "table_name": table_name},
        )

    def fetch_foreign_keys(
        self,
        target_url: str,
        schema_name: str,
        table_name: str,
    ) -> list[dict]:
        """Fetch foreign key definitions for a table."""
        return self._gateway.execute_query(
            target_url,
            FETCH_FOREIGN_KEYS,
            {"schema_name": schema_name, "table_name": table_name},
        )
