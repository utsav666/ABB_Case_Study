"""Tests for schema-graph rendering.

These two functions produce the text that goes into every LLM prompt — the
router's summary and the SQL generator's table structure. If a table, column, or
join path is dropped here, the LLM cannot use it and no error is raised: it just
writes SQL against whatever it *was* told about. That silent-degradation property
is why these deserve exact assertions.

Fixture data is the real exported AdventureWorks graph from tests/, so the shapes
match what settings.schema_graphs actually stores.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from app.target_db.controllers import render_graph_summary, render_graph_table_structure

PRODUCTION_GRAPH = json.loads(
    (pathlib.Path(__file__).parent / "production_schema_graph.json").read_text()
)

# A hand-built graph small enough to assert on exhaustively.
SMALL_GRAPH = {
    "nodes": [
        {
            "table": "orders",
            "schema": "sales",
            "qualified_name": "sales.orders",
            "description": "Customer orders",
            "columns": [
                {"raw_name": "id", "data_type": "integer", "logical_name": "Order ID"},
                {
                    "raw_name": "customer_id",
                    "data_type": "integer",
                    "logical_name": "Customer ID",
                },
            ],
        },
        {
            "table": "customers",
            "schema": "sales",
            "qualified_name": "sales.customers",
            "description": None,
            "columns": [
                {"raw_name": "id", "data_type": "integer", "logical_name": None},
            ],
        },
    ],
    "edges": [
        {
            "from_table": "sales.orders",
            "from_column": "customer_id",
            "to_table": "sales.customers",
            "to_column": "id",
        }
    ],
}


class TestRenderGraphSummary:
    def test_renders_qualified_table_with_its_columns(self) -> None:
        summary = render_graph_summary(SMALL_GRAPH)

        assert "sales.orders(id (Order ID), customer_id (Customer ID))" in summary

    def test_column_without_a_logical_name_shows_the_raw_name_alone(self) -> None:
        """No empty parentheses — they waste prompt tokens and read as a bug."""
        summary = render_graph_summary(SMALL_GRAPH)

        assert "sales.customers(id)" in summary
        assert "id ()" not in summary

    def test_foreign_keys_are_appended_once(self) -> None:
        """Join paths are what let the LLM write a correct JOIN."""
        summary = render_graph_summary(SMALL_GRAPH)

        assert summary.count("FK:") == 1
        assert "sales.orders.customer_id→sales.customers.id" in summary

    def test_no_fk_section_when_there_are_no_edges(self) -> None:
        graph = {"nodes": SMALL_GRAPH["nodes"], "edges": []}

        assert "FK:" not in render_graph_summary(graph)

    def test_every_table_in_the_graph_appears(self) -> None:
        """A dropped table is invisible to the LLM and raises no error."""
        summary = render_graph_summary(PRODUCTION_GRAPH)

        for node in PRODUCTION_GRAPH["nodes"]:
            assert node["qualified_name"] in summary, node["qualified_name"]

    def test_every_edge_in_the_graph_appears(self) -> None:
        summary = render_graph_summary(PRODUCTION_GRAPH)

        for edge in PRODUCTION_GRAPH["edges"]:
            expected = (
                f"{edge['from_table']}.{edge['from_column']}→"
                f"{edge['to_table']}.{edge['to_column']}"
            )
            assert expected in summary, expected

    def test_empty_graph_renders_an_empty_string(self) -> None:
        assert render_graph_summary({"nodes": [], "edges": []}) == ""

    def test_missing_keys_are_tolerated(self) -> None:
        assert render_graph_summary({}) == ""

    def test_table_with_no_columns_renders_empty_parentheses(self) -> None:
        graph = {
            "nodes": [{"table": "t", "qualified_name": "s.t", "columns": []}],
            "edges": [],
        }

        assert render_graph_summary(graph) == "s.t()"

    def test_whitespace_only_logical_name_is_treated_as_absent(self) -> None:
        graph = {
            "nodes": [
                {
                    "table": "t",
                    "qualified_name": "s.t",
                    "columns": [{"raw_name": "c", "logical_name": "   "}],
                }
            ],
            "edges": [],
        }

        assert render_graph_summary(graph) == "s.t(c)"


class TestRenderGraphTableStructure:
    def test_returns_json_and_the_qualified_table_list(self) -> None:
        structure, tables = render_graph_table_structure(SMALL_GRAPH)

        assert tables == ["sales.orders", "sales.customers"]
        assert isinstance(structure, str)

    def test_json_is_parseable(self) -> None:
        """It is interpolated into a prompt, but must still be valid JSON."""
        structure, _ = render_graph_table_structure(SMALL_GRAPH)

        assert json.loads(structure)

    def test_tables_are_keyed_by_bare_name_with_schema_inside(self) -> None:
        structure, _ = render_graph_table_structure(SMALL_GRAPH)
        payload = json.loads(structure)

        assert payload["orders"]["schema"] == "sales"
        assert payload["orders"]["qualified_name"] == "sales.orders"

    def test_columns_carry_type_and_logical_name(self) -> None:
        structure, _ = render_graph_table_structure(SMALL_GRAPH)
        columns = json.loads(structure)["orders"]["columns"]

        assert columns["id"] == {"type": "integer", "logical_name": "Order ID"}

    def test_observed_values_are_attached_to_the_matching_column(self) -> None:
        structure, _ = render_graph_table_structure(
            SMALL_GRAPH,
            observed_values={
                "sales.orders": {"customer_id": ["1", "2"]},
            },
        )

        assert json.loads(structure)["orders"]["columns"]["customer_id"][
            "observed_values"
        ] == ["1", "2"]

    def test_relationships_block_lists_every_join(self) -> None:
        """The prompt relies on __relationships__ to know valid joins."""
        structure, _ = render_graph_table_structure(SMALL_GRAPH)
        relationships = json.loads(structure)["__relationships__"]

        assert relationships == [
            {"from": "sales.orders.customer_id", "to": "sales.customers.id"}
        ]

    def test_relationships_block_is_present_even_with_no_edges(self) -> None:
        """A missing key would be a KeyError for a prompt builder expecting it."""
        graph = {"nodes": SMALL_GRAPH["nodes"], "edges": []}

        structure, _ = render_graph_table_structure(graph)

        assert json.loads(structure)["__relationships__"] == []

    def test_relationships_key_is_not_mistaken_for_a_table(self) -> None:
        """__relationships__ shares the payload with real tables, so the table
        LIST must not contain it."""
        _, tables = render_graph_table_structure(SMALL_GRAPH)

        assert "__relationships__" not in tables

    def test_null_description_is_preserved_as_null(self) -> None:
        structure, _ = render_graph_table_structure(SMALL_GRAPH)
        payload = json.loads(structure)

        assert payload["customers"]["description"] is None
        assert payload["orders"]["description"] == "Customer orders"

    def test_every_table_and_column_survives_the_round_trip(self) -> None:
        structure, tables = render_graph_table_structure(PRODUCTION_GRAPH)
        payload = json.loads(structure)

        assert len(tables) == len(PRODUCTION_GRAPH["nodes"])
        for node in PRODUCTION_GRAPH["nodes"]:
            rendered = payload[node["table"]]
            assert len(rendered["columns"]) == len(node["columns"]), node["table"]

    def test_edge_count_is_preserved(self) -> None:
        structure, _ = render_graph_table_structure(PRODUCTION_GRAPH)

        assert len(json.loads(structure)["__relationships__"]) == len(
            PRODUCTION_GRAPH["edges"]
        )

    def test_empty_graph_yields_only_the_relationships_block(self) -> None:
        structure, tables = render_graph_table_structure({"nodes": [], "edges": []})

        assert tables == []
        assert json.loads(structure) == {"__relationships__": []}

    def test_table_missing_qualified_name_raises(self) -> None:
        """Fail loudly: a silently skipped table is worse than an exception,
        because the LLM would write SQL unaware the table exists."""
        graph = {"nodes": [{"table": "t", "columns": []}], "edges": []}

        with pytest.raises(KeyError):
            render_graph_table_structure(graph)
