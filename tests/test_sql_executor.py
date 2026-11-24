from __future__ import annotations

from typing import Self

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

        assert sql_executor.resolve_sql_database_url().endswith("/backend_database")

    def test_sql_executor_escapes_literal_like_wildcards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Percent wildcards must stay literal when sent through psycopg2."""
        compiled_queries: list[str] = []

        def capture_query(query: TextClause, connection: object) -> pd.DataFrame:
            compiled_queries.append(query.compile(dialect=PGDialect_psycopg2()).string)
            return pd.DataFrame()

        monkeypatch.setattr(
            sql_executor, "_get_engine", lambda database_url: FakeEngine("sqlite")
        )
        monkeypatch.setattr(sql_executor.pd, "read_sql_query", capture_query)

        executor = sql_executor.SqlExec()
        executor.sql_executor("SELECT 1 WHERE 'TNGX' ILIKE '%ngx%'")

        assert "ILIKE '%%ngx%%'" in compiled_queries[0]

    def test_postgres_queries_run_read_only_with_a_statement_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generated SQL must not be able to write, even if the validator is bypassed."""
        engine = FakeEngine("postgresql")
        monkeypatch.setattr(sql_executor, "_get_engine", lambda database_url: engine)
        monkeypatch.setattr(
            sql_executor.pd, "read_sql_query", lambda query, connection: pd.DataFrame()
        )

        sql_executor.SqlExec().sql_executor("SELECT 1", timeout_ms=1234)

        assert engine.connection.driver_sql == [
            "SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = 1234"
        ]

    def test_database_errors_are_returned_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(query: TextClause, connection: object) -> pd.DataFrame:
            raise pd.errors.DatabaseError("relation does not exist")

        monkeypatch.setattr(
            sql_executor, "_get_engine", lambda database_url: FakeEngine("postgresql")
        )
        monkeypatch.setattr(sql_executor.pd, "read_sql_query", fail)

        result = sql_executor.SqlExec().sql_executor("SELECT * FROM missing")

        assert result == "relation does not exist"


class FakeConnection:
    def __init__(self) -> None:
        self.driver_sql: list[str] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def begin(self) -> FakeConnection:
        return self

    def exec_driver_sql(self, statement: str) -> None:
        self.driver_sql.append(statement)


class FakeEngine:
    def __init__(self, dialect_name: str) -> None:
        self.dialect = type("Dialect", (), {"name": dialect_name})()
        self.connection = FakeConnection()

    def connect(self) -> FakeConnection:
        return self.connection
