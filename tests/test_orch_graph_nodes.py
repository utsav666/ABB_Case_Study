"""Tests for orchestrator graph nodes with the DB and LLMs mocked.

Nodes are the unit of behaviour in the graph, and they are where wrong-database
and silent-empty-result bugs have actually occurred. Each node is a plain
function taking a GraphState dict, so with the DB mocked at
``SqlExec.sql_executor`` (via the shared ``patch_executor`` fixture) and agents
mocked at their call sites, they can be driven directly.

The routing functions need no mocking at all and are the cheapest guard against a
route silently falling through to a path that cannot answer it — which is exactly
what happened to `explanation` before it was wired up.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.orchestrator.orch_graph import (
    build_data_response_node,
    execute_sql_node,
    increment_retry_node,
    route_after_execute,
    route_after_router,
    route_after_validate,
    route_after_verify,
    validate_sql_node,
    verify_result_node,
)

SCHEMA_GRAPH = {
    "nodes": [{"table": "sample", "schema": "public", "columns": []}],
    "edges": [],
}


class TestRouteAfterRouter:
    @pytest.mark.parametrize("route", ["sql_query", "sql_and_chart"])
    def test_sql_routes_go_to_the_table_resolver_path(self, route: str) -> None:
        assert route_after_router({"route": route}) == "sql_path"

    def test_schema_info_has_its_own_path(self) -> None:
        assert route_after_router({"route": "schema_info"}) == "schema_info"

    def test_explanation_has_its_own_path(self) -> None:
        """Regression: `explanation` used to fall through to `general`.

        `general_answer` is an LLM with no database access, so "why was Nov high?"
        was answered by a model that could not see the data.
        """
        assert route_after_router({"route": "explanation"}) == "explanation"

    @pytest.mark.parametrize(
        "route", ["general_answer", "follow_up", "configuration", "", "nonsense"]
    )
    def test_everything_else_falls_back_to_general(self, route: str) -> None:
        assert route_after_router({"route": route}) == "general"

    def test_missing_route_key_does_not_raise(self) -> None:
        assert route_after_router({}) == "general"


class TestRouteAfterValidate:
    def test_clean_sql_proceeds_to_execution(self) -> None:
        assert route_after_validate({"sql_error": ""}) == "execute"

    def test_error_within_retry_budget_retries(self) -> None:
        state = {"sql_error": "syntax error", "retry_count": 0}

        assert route_after_validate(state) == "retry"

    def test_error_at_the_retry_ceiling_gives_up(self) -> None:
        """Without this the graph would loop on an unfixable query."""
        state = {"sql_error": "syntax error", "retry_count": 99}

        assert route_after_validate(state) == "give_up"

    def test_missing_error_key_proceeds(self) -> None:
        assert route_after_validate({}) == "execute"


class TestRouteAfterExecute:
    def test_dataframe_result_is_success(self) -> None:
        state = {"data": pd.DataFrame({"a": [1]})}

        assert route_after_execute(state) == "success"

    def test_list_result_is_success(self) -> None:
        """State stores records as a list for checkpoint serialization."""
        assert route_after_execute({"data": [{"a": 1}]}) == "success"

    def test_empty_dataframe_is_still_success(self) -> None:
        """Zero rows is a valid answer, not an execution failure."""
        assert route_after_execute({"data": pd.DataFrame()}) == "success"

    def test_none_data_within_budget_retries(self) -> None:
        assert route_after_execute({"data": None, "retry_count": 0}) == "retry"

    def test_none_data_at_the_ceiling_gives_up(self) -> None:
        assert route_after_execute({"data": None, "retry_count": 99}) == "give_up"


class TestRouteAfterVerify:
    def test_chart_route_goes_to_the_chart_path(self) -> None:
        assert route_after_verify({"route": "sql_and_chart"}) == "chart"

    @pytest.mark.parametrize("route", ["sql_query", "", "explanation"])
    def test_everything_else_returns_data(self, route: str) -> None:
        assert route_after_verify({"route": route}) == "data_response"


class TestValidateSqlNode:
    def test_valid_sql_clears_the_error(self) -> None:
        result = validate_sql_node({"sql": "SELECT 1 FROM sample"})

        assert result == {"sql_error": ""}

    def test_invalid_sql_reports_a_syntax_error(self) -> None:
        result = validate_sql_node({"sql": "SELECT FROM WHERE ((("})

        assert result["sql_error"].startswith("SQL syntax error")

    def test_missing_sql_is_reported_distinctly(self) -> None:
        """ "No SQL was generated" is a different failure from bad syntax."""
        assert validate_sql_node({"sql": ""}) == {"sql_error": "No SQL was generated."}
        assert validate_sql_node({}) == {"sql_error": "No SQL was generated."}


class TestExecuteSqlNode:
    def test_successful_query_returns_records(self, patch_executor) -> None:
        """Records, not a DataFrame: the checkpointer must serialize the state."""
        patch_executor(pd.DataFrame({"m": ["2016-01"], "v": [10]}))

        result = execute_sql_node({"sql": "SELECT 1"})

        assert result["sql_error"] == ""
        assert result["data"] == [{"m": "2016-01", "v": 10}]

    def test_zero_rows_is_a_success_with_an_empty_list(self, patch_executor) -> None:
        patch_executor(pd.DataFrame())

        result = execute_sql_node({"sql": "SELECT 1 WHERE false"})

        assert result["data"] == []
        assert result["sql_error"] == ""

    def test_executor_error_string_becomes_sql_error(self, patch_executor) -> None:
        patch_executor("relation does not exist")

        result = execute_sql_node({"sql": "SELECT 1 FROM nope"})

        assert result["data"] is None
        assert "relation does not exist" in result["sql_error"]

    def test_the_exact_sql_from_state_is_executed(self, patch_executor) -> None:
        """No rewriting between validation and execution."""
        recorder = patch_executor(pd.DataFrame({"a": [1]}))
        sql = "SELECT a FROM sample WHERE brand_name = 'Brand A'"

        execute_sql_node({"sql": sql})

        assert recorder.queries == [sql]

    def test_an_exception_is_caught_and_surfaced(self, monkeypatch) -> None:
        """A raising driver must not take the whole graph down.

        Patched at `orch_graph.SqlExec`, not the source module: orch_graph did
        `from ... import SqlExec` at import time, so it holds its own binding.
        """

        class Boom:
            def sql_executor(self, _sql: str):
                raise RuntimeError("connection reset")

        monkeypatch.setattr("app.orchestrator.orch_graph.SqlExec", Boom)

        result = execute_sql_node({"sql": "SELECT 1"})

        assert result["data"] is None
        assert "connection reset" in result["sql_error"]


class TestVerifyResultNode:
    def test_returns_no_state_changes(self) -> None:
        """The node only logs warnings; it must not mutate the result."""
        state = {"data": [{"a": 1}], "question": "q"}

        assert verify_result_node(state) == {}

    def test_zero_rows_is_logged_as_a_warning(self, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            verify_result_node({"data": [], "question": "q"})

        assert "0 rows" in caplog.text

    def test_all_null_column_is_logged(self, caplog) -> None:
        import logging

        state = {"data": [{"a": None}, {"a": None}], "question": "q"}

        with caplog.at_level(logging.WARNING):
            verify_result_node(state)

        assert "NULL" in caplog.text

    def test_missing_data_is_a_no_op(self) -> None:
        assert verify_result_node({"question": "q"}) == {}


class TestBuildDataResponseNode:
    def test_wraps_rows_as_a_data_response(self) -> None:
        state = {"data": [{"m": "2016-01", "v": 10}], "sql": "SELECT 1"}

        response = build_data_response_node(state)["final_response"]

        assert response["type"] == "data"
        assert response["data"] == [{"m": "2016-01", "v": 10}]
        assert response["sql"] == "SELECT 1"

    def test_zero_rows_gets_an_explicit_message(self) -> None:
        """Regression: "Here are the query results." with nothing under it reads
        as a broken app. A zero-row result must say so."""
        response = build_data_response_node({"data": [], "sql": "SELECT 1"})[
            "final_response"
        ]

        assert response["type"] == "data"
        assert "no rows" in response["message"].lower()

    def test_non_empty_result_carries_no_override_message(self) -> None:
        """The generic caller-side default is used for a normal result."""
        response = build_data_response_node({"data": [{"a": 1}], "sql": "s"})[
            "final_response"
        ]

        assert "message" not in response

    def test_none_data_becomes_an_error_response(self) -> None:
        state = {"data": None, "sql_error": "syntax error near FROM"}

        response = build_data_response_node(state)["final_response"]

        assert response["type"] == "error"
        assert "syntax error near FROM" in response["message"]

    def test_none_data_without_an_error_still_reports_a_failure(self) -> None:
        response = build_data_response_node({"data": None})["final_response"]

        assert response["type"] == "error"
        assert "Unknown error" in response["message"]


class TestIncrementRetryNode:
    def test_increments_from_absent(self) -> None:
        assert increment_retry_node({}) == {"retry_count": 1}

    def test_increments_an_existing_count(self) -> None:
        assert increment_retry_node({"retry_count": 2}) == {"retry_count": 3}
