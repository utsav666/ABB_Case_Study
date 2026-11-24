"""Tests for the per-connection semantic profile.

The profile is what stops the SQL generator from guessing filter values
('Enterprise Value' instead of 'enterprise_value', '0%' instead of 'Market').
Its facts come from two batched queries per table, so the fake executor below
answers by recognising which of the queries it was given.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.semantic_catalog import connection_profile as cp
from app.target_db.controllers import render_graph_table_structure

NODE = {
    "table": "cap_table",
    "schema": "public",
    "qualified_name": "public.cap_table",
    "columns": [
        {"raw_name": "premium", "data_type": "character varying"},
        {"raw_name": "field_name", "data_type": "character varying"},
        {"raw_name": "period_type", "data_type": "character varying"},
        {"raw_name": "value", "data_type": "numeric"},
        {"raw_name": "payload", "data_type": "jsonb"},
    ],
}


def fake_execute(sql: str) -> pd.DataFrame:
    if sql.startswith("SELECT COUNT(*) AS row_count"):
        return pd.DataFrame(
            [
                {
                    "row_count": 8,
                    "d0": 2,
                    "n0": 0,
                    "len0": 6,
                    "d1": 2,
                    "n1": 0,
                    "len1": 7,
                    "d2": 2,
                    "n2": 0,
                    "len2": 3,
                    "d3": 6,
                    "n3": 0,
                    "lo3": "1",
                    "hi3": "90",
                }
            ]
        )
    if " AS s, " in sql:
        # s=0 is premium (does not separate the variants), s=1 is period_type.
        return pd.DataFrame(
            {
                "s": [0, 0, 0, 0, 1, 1],
                "v": ["Granted", "Granted", "granted", "granted", "Granted", "granted"],
                "o": ["Market", "+25%", "Market", "+25%", "QTD", "YTD"],
                "n": [2, 2, 2, 2, 4, 4],
            }
        )
    return pd.DataFrame(
        {
            "c": [0, 0, 1, 1, 2, 2],
            "v": ["Market", "+25%", "Granted", "granted", "QTD", "YTD"],
            "n": [4, 4, 4, 4, 4, 4],
        }
    )


@pytest.fixture
def table_facts() -> dict:
    return cp._profile_table(NODE, fake_execute)


class TestProfileTable:
    def test_categorical_columns_store_every_real_value(self, table_facts) -> None:
        premium = table_facts["columns"]["premium"]

        assert premium["values"] == [["Market", 4], ["+25%", 4]]
        assert premium["distinct_count"] == 2

    def test_numeric_columns_get_a_range_not_values(self, table_facts) -> None:
        value = table_facts["columns"]["value"]

        assert (value["min"], value["max"]) == ("1", "90")
        assert "values" not in value

    def test_unprofilable_types_are_skipped(self, table_facts) -> None:
        assert "payload" not in table_facts["columns"]

    def test_case_variants_record_the_column_that_separates_them(
        self, table_facts
    ) -> None:
        """'Granted' (QTD) and 'granted' (YTD) are different measures."""
        field_name = table_facts["columns"]["field_name"]

        assert field_name["case_variants"] == [["Granted", "granted"]]
        assert field_name["case_variant_separators"] == [
            {"by": "period_type", "mapping": {"Granted": "QTD", "granted": "YTD"}}
        ]

    def test_failed_stats_query_raises(self) -> None:
        with pytest.raises(RuntimeError):
            cp._profile_table(NODE, lambda sql: "relation does not exist")


class TestIsCategorical:
    @pytest.mark.parametrize(
        ("info", "rows", "expected"),
        [
            ({"family": "text", "distinct_count": 7, "max_length": 6}, 10_000, True),
            ({"family": "text", "distinct_count": 17, "max_length": 30}, 17, True),
            ({"family": "text", "distinct_count": 400, "max_length": 30}, 500, False),
            ({"family": "text", "distinct_count": 5, "max_length": 900}, 100, False),
            ({"family": "identifier", "distinct_count": 3}, 100, False),
            ({"family": "boolean", "distinct_count": 2}, 100, True),
        ],
    )
    def test_classification(self, info: dict, rows: int, expected: bool) -> None:
        assert cp._is_categorical(info, rows) is expected


class TestValidatedInference:
    def test_only_real_columns_and_values_survive(self, table_facts) -> None:
        raw = {
            "tables": [
                {
                    "table": "PUBLIC.CAP_TABLE",
                    "grain": "one row per premium x field",
                    "row_metric_columns": ["field_name", "invented_col"],
                    "scenario_columns": [
                        {"column": "premium", "default": "Market"},
                        {"column": "premium", "default": "0%"},
                    ],
                    "version_columns": [
                        {"column": "nope", "latest_by": "value", "per": []}
                    ],
                    "notes": ["a", "b", "c", "d"],
                },
                {"table": "public.unknown", "grain": "x"},
            ]
        }

        validated = cp._validated_inference(raw, {"public.cap_table": table_facts})

        assert validated == {
            "public.cap_table": {
                "grain": "one row per premium x field",
                "row_metric_columns": ["field_name"],
                "scenario_columns": [{"column": "premium", "default": "Market"}],
                "version_columns": [],
                "notes": ["a", "b", "c"],
            }
        }


class TestLookups:
    @pytest.fixture
    def profile(self, table_facts) -> dict:
        return {
            "version": cp.PROFILE_VERSION,
            "tables": {"public.cap_table": table_facts},
            "inferred": {
                "public.cap_table": {
                    "grain": "one row per premium",
                    "scenario_columns": [{"column": "premium", "default": "Market"}],
                }
            },
        }

    def test_full_value_lists_are_returned_for_bare_table_names(self, profile) -> None:
        assert cp.get_column_values(profile, "cap_table", "premium") == (
            ["Market", "+25%"],
            2,
        )

    def test_unknown_or_unprofiled_columns_return_none(self, profile) -> None:
        assert cp.get_column_values(profile, "cap_table", "value") is None
        assert cp.get_column_values(profile, "other", "premium") is None
        assert cp.get_column_values(None, "cap_table", "premium") is None

    def test_rendered_structure_carries_values_and_table_semantics(
        self, profile
    ) -> None:
        graph = {"nodes": [NODE], "edges": []}

        structure = json.loads(render_graph_table_structure(graph, profile=profile)[0])
        table = structure["cap_table"]

        assert table["grain"] == "one row per premium"
        assert table["scenario_columns"] == [{"column": "premium", "default": "Market"}]
        assert table["columns"]["premium"]["values"] == ["Market", "+25%"]
        assert table["columns"]["field_name"]["distinguished_by"][0]["by"] == (
            "period_type"
        )
        assert table["columns"]["value"]["range"] == ["1", "90"]

    def test_profile_must_cover_every_graph_table(self, profile) -> None:
        graph = {"nodes": [NODE, {**NODE, "qualified_name": "public.other"}]}

        assert cp._profile_matches_graph(profile, {"nodes": [NODE]}) is True
        assert cp._profile_matches_graph(profile, graph) is False


class TestGetConnectionProfile:
    def test_stored_profile_is_used_without_rebuilding(self, monkeypatch) -> None:
        profile = {"version": cp.PROFILE_VERSION, "tables": {"public.cap_table": {}}}
        monkeypatch.setattr(
            cp,
            "refresh_connection_profile",
            lambda *a, **k: pytest.fail("must not rebuild"),
        )

        graph = {"nodes": [NODE], "profile": profile}

        assert cp.get_connection_profile("c1", graph) is profile

    def test_missing_profile_builds_facts_only(self, monkeypatch) -> None:
        calls = []

        def fake_refresh(connection_id, graph, *, infer):
            calls.append(infer)
            return {"version": cp.PROFILE_VERSION, "tables": {"public.cap_table": {}}}

        monkeypatch.setattr(cp, "refresh_connection_profile", fake_refresh)
        monkeypatch.setattr(cp, "_cache", {})

        cp.get_connection_profile("c2", {"nodes": [NODE]})

        assert calls == [False]


class TestLargeSchemaNarrowing:
    def test_large_graphs_send_selected_tables_and_join_neighbours_only(self) -> None:
        nodes = [
            {
                "table": f"t{i}",
                "schema": "s",
                "qualified_name": f"s.t{i}",
                "columns": [],
            }
            for i in range(20)
        ]
        edges = [
            {
                "from_table": "s.t0",
                "from_column": "a",
                "to_table": "s.t1",
                "to_column": "a",
            },
            {
                "from_table": "s.t5",
                "from_column": "b",
                "to_table": "s.t6",
                "to_column": "b",
            },
        ]

        _, tables = render_graph_table_structure(
            {"nodes": nodes, "edges": edges}, selected_tables=["s.t0"]
        )

        assert tables == ["s.t0", "s.t1"]

    def test_small_graphs_are_always_sent_whole(self) -> None:
        nodes = [
            {
                "table": f"t{i}",
                "schema": "s",
                "qualified_name": f"s.t{i}",
                "columns": [],
            }
            for i in range(3)
        ]

        _, tables = render_graph_table_structure(
            {"nodes": nodes, "edges": []}, selected_tables=["s.t0"]
        )

        assert tables == ["s.t0", "s.t1", "s.t2"]


class TestCardinalityResolverUsesProfile:
    @pytest.fixture
    def state(self, table_facts) -> dict:
        profile = {
            "version": cp.PROFILE_VERSION,
            "tables": {"public.cap_table": table_facts},
            "inferred": {},
        }
        return {
            "tables": ["public.cap_table"],
            "schema_graph": {"nodes": [NODE], "edges": [], "profile": profile},
        }

    def test_exact_value_needs_no_database_query(self, state, patch_executor) -> None:
        from app.orchestrator.orch_graph import cardinality_resolver_node

        recorder = patch_executor()
        sql = "SELECT * FROM public.cap_table WHERE premium = 'Market'"

        assert cardinality_resolver_node({**state, "sql": sql}) == {}
        assert recorder.queries == []

    def test_guessed_value_is_resolved_against_profiled_values(
        self, state, patch_executor, monkeypatch
    ) -> None:
        """Regression: '0%' was sent to the DB and returned 0 rows."""
        from app.orchestrator.orch_graph import cardinality_resolver_node

        recorder = patch_executor()
        seen = {}

        def fake_resolve(**kwargs):
            seen.update(kwargs)
            return {
                "confidence": "high",
                "match": "Market",
                "alternatives": [],
                "reasoning": "",
            }

        monkeypatch.setattr(
            "app.orchestrator.orch_graph.resolve_filter_value", fake_resolve
        )
        sql = "SELECT * FROM public.cap_table WHERE premium = '0%'"

        result = cardinality_resolver_node({**state, "sql": sql})

        assert "premium = 'Market'" in result["sql"]
        assert seen["cardinality"] == 2
        assert set(seen["actual_values"]) == {"Market", "+25%"}
        assert recorder.queries == []


def test_long_text_columns_show_only_examples_in_prompts() -> None:
    """Formulas and tags bloat the prompt and are not filter targets."""
    values = [[f"{'x' * 90}{i}", 1] for i in range(40)]
    profile = {
        "tables": {
            "public.cap_table": {
                "columns": {
                    "formula": {
                        "family": "text",
                        "distinct_count": 40,
                        "max_length": 92,
                        "values": values,
                    }
                }
            }
        }
    }

    _, columns = cp.render_table_profile(profile, "public.cap_table")

    assert len(columns["formula"]["example_values"]) == cp.PROMPT_LONG_VALUE_EXAMPLES
    assert "values" not in columns["formula"]
    assert cp.get_column_values(profile, "cap_table", "formula")[1] == 40


def test_query_aliases_are_not_checked_against_the_table(patch_executor) -> None:
    """`rn = 1` filters a ROW_NUMBER() alias, not a column of the table."""
    from app.orchestrator.orch_graph import cardinality_resolver_node

    recorder = patch_executor()
    sql = (
        "SELECT * FROM (SELECT premium, ROW_NUMBER() OVER () AS rn "
        "FROM public.cap_table) t WHERE rn = 1"
    )

    state = {
        "sql": sql,
        "tables": ["public.cap_table"],
        "schema_graph": {"nodes": [NODE], "edges": []},
    }

    assert cardinality_resolver_node(state) == {}
    assert recorder.queries == []
