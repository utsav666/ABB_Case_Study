"""Tests for the read-only SQL guard and the router's decision model.

Generated SQL runs against customer databases, so the validator must reject
anything that is not exactly one read-only query — including writes hidden in
CTEs or SELECT INTO.
"""

from __future__ import annotations

import pytest

from app.router.route_improvisation import RoutingDecision
from app.utils.sql_validator import validator


class TestReadOnlyValidator:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "-- Assumptions: none\nSELECT 1",
            "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
            "SELECT a FROM t UNION ALL SELECT b FROM u",
            "SELECT DISTINCT ON (ticker) ticker FROM t ORDER BY ticker, d DESC",
            "SELECT MAX(v) FILTER (WHERE p = 'x') FROM t",
        ],
    )
    def test_read_queries_pass(self, sql: str) -> None:
        assert validator(sql) is True

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM t",
            "UPDATE t SET a = 1",
            "INSERT INTO t VALUES (1)",
            "DROP TABLE t",
            "SELECT 1; DROP TABLE t",
            "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d",
            "SELECT * INTO new_table FROM t",
            "GRANT ALL ON t TO someone",
            "",
        ],
    )
    def test_writes_and_multiple_statements_are_rejected(self, sql: str) -> None:
        assert validator(sql) is not True

    def test_syntax_errors_are_reported(self) -> None:
        assert isinstance(validator("SELECT FROM WHERE ((("), str)


class TestRoutingDecision:
    def test_legacy_chart_route_becomes_a_query_with_chart_display(self) -> None:
        decision = RoutingDecision(route="sql_and_chart", reasoning="r")

        assert (decision.route, decision.display) == ("sql_query", "chart")

    def test_legacy_route_keeps_an_explicit_display(self) -> None:
        decision = RoutingDecision(
            route="sql_and_chart", display="table", reasoning="r"
        )

        assert decision.display == "table"

    def test_missing_or_unknown_display_is_auto(self) -> None:
        assert RoutingDecision(route="sql_query", reasoning="r").display == "auto"
        assert (
            RoutingDecision(route="follow_up", display="graph", reasoning="r").display
            == "auto"
        )
