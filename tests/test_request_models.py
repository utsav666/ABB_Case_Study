"""Tests for API request models and their validators.

These define the wire contract. A validator that normalises inconsistently means
two clients sending semantically identical payloads get different behaviour —
`column_names: []` meaning "no columns" in one path and "all columns" in another
is exactly the kind of ambiguity that produces an empty schema config with no
error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.main import (
    BulkTableConfigRequest,
    ChatRequest,
    ColumnConfigRow,
    HitlResumeRequest,
    SchemaConfigRequest,
    TableSelection,
)


class TestChatRequest:
    def test_accepts_the_question_field(self) -> None:
        assert ChatRequest(question="show me sales").question == "show me sales"

    def test_accepts_message_as_an_alias_for_question(self) -> None:
        """The frontend sends `message`; older clients send `question`."""
        assert ChatRequest(message="show me sales").question == "show me sales"

    def test_question_wins_when_both_are_supplied(self) -> None:
        request = ChatRequest(question="from question", message="from message")

        assert request.question == "from question"

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert ChatRequest(message="  padded  ").question == "padded"

    @pytest.mark.parametrize("value", ["", "   ", "\n\t"])
    def test_blank_questions_are_rejected(self, value: str) -> None:
        """An empty question would reach the router and waste an LLM call."""
        with pytest.raises(ValidationError):
            ChatRequest(question=value)

    def test_both_fields_missing_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_optional_scoping_fields_default_to_none(self) -> None:
        request = ChatRequest(question="q")

        assert request.thread_id is None
        assert request.connection_id is None
        assert request.workspace_id is None

    def test_scoping_fields_are_carried_through(self) -> None:
        request = ChatRequest(
            question="q",
            thread_id="t-1",
            connection_id="c-1",
            workspace_id="w-1",
        )

        assert (request.thread_id, request.connection_id, request.workspace_id) == (
            "t-1",
            "c-1",
            "w-1",
        )


class TestTableSelection:
    def test_table_name_alone_means_all_columns(self) -> None:
        assert TableSelection(table_name="orders").column_names is None

    def test_an_empty_column_list_is_normalised_to_none(self) -> None:
        """Otherwise `[]` would be read as "select zero columns" downstream,
        producing a table entry with no columns instead of all of them."""
        assert TableSelection(table_name="orders", column_names=[]).column_names is None

    def test_a_populated_column_list_is_preserved(self) -> None:
        selection = TableSelection(table_name="orders", column_names=["id", "total"])

        assert selection.column_names == ["id", "total"]

    def test_a_single_column_is_preserved(self) -> None:
        """Guards against a one-element list being mistaken for empty."""
        assert TableSelection(table_name="t", column_names=["id"]).column_names == [
            "id"
        ]

    def test_table_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            TableSelection()


class TestBulkTableConfigRequest:
    def test_omitting_tables_means_every_table(self) -> None:
        """Scenario 1 of four: no selection at all."""
        assert BulkTableConfigRequest(connection_id="c-1").tables is None

    def test_schema_name_defaults_to_public(self) -> None:
        assert BulkTableConfigRequest(connection_id="c-1").schema_name == "public"

    def test_table_entries_are_parsed_into_selections(self) -> None:
        request = BulkTableConfigRequest(
            connection_id="c-1",
            tables=[{"table_name": "orders", "column_names": ["id"]}],
        )

        assert isinstance(request.tables[0], TableSelection)
        assert request.tables[0].column_names == ["id"]

    def test_nested_empty_column_lists_are_normalised(self) -> None:
        """The TableSelection validator must apply through the parent model."""
        request = BulkTableConfigRequest(
            connection_id="c-1", tables=[{"table_name": "orders", "column_names": []}]
        )

        assert request.tables[0].column_names is None

    def test_connection_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            BulkTableConfigRequest()


class TestSchemaConfigRequest:
    def test_defaults(self) -> None:
        request = SchemaConfigRequest(connection_id="c-1")

        assert request.schema_name == "public"
        assert request.workspace_id is None
        assert request.tables is None
        assert request.profile is False

    def test_profiling_is_opt_in(self) -> None:
        """Profiling issues a query per column, so it must not be the default."""
        assert SchemaConfigRequest(connection_id="c-1", profile=True).profile is True

    def test_workspace_id_enables_agent_persistence(self) -> None:
        request = SchemaConfigRequest(connection_id="c-1", workspace_id="w-1")

        assert request.workspace_id == "w-1"

    def test_mixed_table_entries_are_supported(self) -> None:
        """Scenario 4: some entries name columns, some do not."""
        request = SchemaConfigRequest(
            connection_id="c-1",
            tables=[
                {"table_name": "orders", "column_names": ["id"]},
                {"table_name": "customers"},
            ],
        )

        assert request.tables[0].column_names == ["id"]
        assert request.tables[1].column_names is None


class TestColumnConfigRow:
    def test_only_the_column_name_is_required(self) -> None:
        row = ColumnConfigRow(column_name="brand_name")

        assert row.column_name == "brand_name"
        assert row.display_name == ""
        assert row.description == ""

    def test_assistant_inclusion_defaults_to_true(self) -> None:
        """A newly discovered column should be usable without extra config."""
        assert ColumnConfigRow(column_name="c").include_in_assistant is True

    def test_filterable_defaults_to_true_and_pii_to_false(self) -> None:
        row = ColumnConfigRow(column_name="c")

        assert row.is_filterable is True
        assert row.is_pii is False

    def test_example_values_default_to_an_empty_list(self) -> None:
        """A mutable default must not be shared between instances."""
        first = ColumnConfigRow(column_name="a")
        second = ColumnConfigRow(column_name="b")
        first.example_values.append("x")

        assert second.example_values == []

    def test_metadata_fields_are_carried_through(self) -> None:
        row = ColumnConfigRow(
            column_name="email",
            display_name="Email Address",
            description="Customer contact",
            is_pii=True,
            is_filterable=False,
            example_values=["a@b.com"],
        )

        assert row.display_name == "Email Address"
        assert row.is_pii is True
        assert row.is_filterable is False
        assert row.example_values == ["a@b.com"]


class TestHitlResumeRequest:
    def test_thread_id_and_decision_are_required(self) -> None:
        request = HitlResumeRequest(thread_id="t-1", decision="Category A")

        assert request.thread_id == "t-1"
        assert request.decision == "Category A"

    def test_missing_thread_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HitlResumeRequest(decision="x")

    def test_connection_id_is_optional_for_older_clients(self) -> None:
        """Added for number formatting; absent clients must keep working."""
        assert HitlResumeRequest(thread_id="t-1", decision="x").connection_id is None

    def test_connection_id_is_accepted_when_supplied(self) -> None:
        request = HitlResumeRequest(thread_id="t-1", decision="x", connection_id="c-1")

        assert request.connection_id == "c-1"

    def test_decision_accepts_a_non_string_payload(self) -> None:
        """Typed Any: a future HITL type may send a structured decision."""
        request = HitlResumeRequest(thread_id="t-1", decision={"value": "A"})

        assert request.decision == {"value": "A"}
