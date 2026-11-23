from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
from sqlalchemy.sql.elements import TextClause

from app.utils import sql_executor
from app.utils.logging_context import bind_log_context


class TestResolveSqlDatabaseUrl:
    def test_prefers_saved_connection_database_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Queries should use the selected connection, not backend defaults."""
        monkeypatch.setenv("DB_NAME", "postgres")
        connection_credentials = {
            "engine": "postgres",
            "db_host": "db.example.test",
            "db_port": 5432,
            "db_name": "pdp",
            "db_user": "target_user",
            "db_password": "target_password",
        }
        monkeypatch.setattr(
            sql_executor,
            "get_settings_connection_credentials",
            lambda connection_id: connection_credentials,
        )
        bind_log_context(connection_id="selected-connection")

        assert sql_executor.resolve_sql_database_url().endswith("/pdp")

    def test_uses_backend_database_name_when_saved_name_is_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing saved name falls back to the backend DB_NAME."""
        monkeypatch.setenv("DB_NAME", "backend_database")
        connection_credentials = {
            "engine": "postgres",
            "db_host": "db.example.test",
            "db_port": 5432,
            "db_user": "target_user",
            "db_password": "target_password",
        }
        monkeypatch.setattr(
            sql_executor,
            "get_settings_connection_credentials",
            lambda connection_id: connection_credentials,
        )
        bind_log_context(connection_id="selected-connection")

        assert sql_executor.resolve_sql_database_url().endswith(
            "/backend_database"
        )

    def test_sql_executor_escapes_literal_like_wildcards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Percent wildcards must stay literal when sent through psycopg2."""
        compiled_queries: list[str] = []

        def capture_query(query: TextClause, engine: object) -> pd.DataFrame:
            compiled_queries.append(
                query.compile(dialect=PGDialect_psycopg2()).string
            )
            return pd.DataFrame()

        monkeypatch.setattr(sql_executor, "_get_engine", lambda database_url: object())
        monkeypatch.setattr(sql_executor.pd, "read_sql_query", capture_query)

        executor = sql_executor.SqlExec()
        executor.sql_executor("SELECT 1 WHERE 'TNGX' ILIKE '%ngx%'")

        assert "ILIKE '%%ngx%%'" in compiled_queries[0]
