"""Tests for filter extraction, number formatting, and history serialization.

All three feed the UI's answer-transparency features, and all three had bugs
that produced a plausible-looking but wrong display rather than an error.
"""

from __future__ import annotations

import json

from app.api.chat_response import (
    _humanize_identifier,
    build_applied_filters,
    build_number_format,
    build_persisted_assistant_content,
)
from app.orchestrator.conversation_manager import ConversationHistoryManager

# The connection's display_config: Indian units, reusing the default key names
# with crore/lakh VALUES. Mirrors real stored config.
INDIAN_UNITS = {
    "decimal_places": 2,
    "number_format_enabled": True,
    "number_format_suffixes": {"billion": "Cr", "million": "L", "thousand": "K"},
    "number_format_thresholds": {"billion": 10000000, "million": 100000, "thousand": 1000},
}


class TestBuildAppliedFilters:
    def test_extracts_equality_filter(self) -> None:
        filters = build_applied_filters(
            "SELECT * FROM sample WHERE brand_name = 'Brand A'"
        )

        assert len(filters) == 1
        assert filters[0]["column"] == "brand_name"
        assert filters[0]["value"] == "Brand A"
        assert filters[0]["label"] == "Brand Name"

    def test_extracts_in_clause_as_joined_value(self) -> None:
        filters = build_applied_filters(
            "SELECT * FROM sample WHERE category IN ('Category A', 'Category D')"
        )

        assert filters[0]["operator"] == "IN"
        assert filters[0]["value"] == "Category A, Category D"

    def test_extracts_like_filter(self) -> None:
        filters = build_applied_filters(
            "SELECT * FROM sample WHERE region LIKE 'North%'"
        )

        assert filters[0]["operator"] == "LIKE"
        assert filters[0]["value"] == "North%"

    def test_extracts_extract_date_part_filters(self) -> None:
        """A year filter is what users most want confirmed on a chart."""
        filters = build_applied_filters(
            "SELECT 1 FROM sample WHERE EXTRACT(year FROM date) = 2016"
        )

        labels = {f["label"]: f["value"] for f in filters}
        assert labels["Date Year"] == "2016"

    def test_extracts_between_range(self) -> None:
        filters = build_applied_filters(
            "SELECT 1 FROM sample WHERE date BETWEEN '2016-11-01' AND '2016-11-30'"
        )

        assert filters[0]["operator"] == "between"
        assert filters[0]["value"] == "2016-11-01 → 2016-11-30"

    def test_collapses_two_sided_range_into_one_chip(self) -> None:
        """`date >= X AND date < Y` is the natural way to write a year filter."""
        filters = build_applied_filters(
            "SELECT 1 FROM sample "
            "WHERE date >= '2016-01-01' AND date < '2017-01-01'"
        )

        assert len(filters) == 1
        assert filters[0]["operator"] == "range"
        assert filters[0]["value"] == "2016-01-01 → 2017-01-01"

    def test_keeps_single_sided_comparison(self) -> None:
        filters = build_applied_filters("SELECT 1 FROM sample WHERE list_price > 500")

        assert filters[0]["operator"] == ">"
        assert filters[0]["value"] == "500"

    def test_parses_each_statement_of_multi_statement_sql(self) -> None:
        """The explanation route returns baseline + breakdown queries."""
        sql = (
            "SELECT 1 FROM sample WHERE brand_name = 'Brand A'"
            "\n\n"
            "SELECT 1 FROM sample WHERE EXTRACT(month FROM date) = 11"
        )

        labels = {f["label"] for f in build_applied_filters(sql)}

        assert "Brand Name" in labels
        assert "Date Month" in labels

    def test_dedupes_repeated_filters(self) -> None:
        """Both analysis statements share filters; one chip each."""
        sql = (
            "SELECT 1 FROM sample WHERE brand_name = 'Brand A'"
            "\n\n"
            "SELECT 2 FROM sample WHERE brand_name = 'Brand A'"
        )

        assert len(build_applied_filters(sql)) == 1

    def test_no_where_clause_returns_empty(self) -> None:
        assert build_applied_filters("SELECT SUM(x) FROM sample") == []

    def test_unparseable_sql_returns_empty_rather_than_raising(self) -> None:
        """Display metadata must never break a response."""
        assert build_applied_filters("this is not sql ((") == []

    def test_none_and_non_string_are_safe(self) -> None:
        assert build_applied_filters(None) == []
        assert build_applied_filters(12345) == []


class TestBuildNumberFormat:
    ROWS = {"rows": [{"m": "2016-11", "total_sales": 983041954.21}]}

    def test_uses_configured_thresholds_and_suffixes(self) -> None:
        """983,041,954 / 1e7 = 98.30Cr, matching the chart."""
        result = build_number_format(self.ROWS, INDIAN_UNITS)

        assert result == {"divisor": 10000000.0, "suffix": "Cr", "decimals": 2}

    def test_config_thresholds_are_not_merged_with_defaults(self) -> None:
        """Merging would let million (1e6) outrank the configured crore (1e7)."""
        result = build_number_format(self.ROWS, INDIAN_UNITS)

        assert result["suffix"] != "M"

    def test_picks_lakh_for_mid_range_values(self) -> None:
        result = build_number_format({"rows": [{"v": 350000}]}, INDIAN_UNITS)

        assert result["suffix"] == "L"

    def test_defaults_to_b_m_k_without_config(self) -> None:
        assert build_number_format(self.ROWS, None)["suffix"] == "M"

    def test_respects_explicit_opt_out(self) -> None:
        assert build_number_format(self.ROWS, {"number_format_enabled": False}) is None

    def test_small_values_are_left_unscaled(self) -> None:
        assert build_number_format({"rows": [{"v": 450}]}, INDIAN_UNITS) is None

    def test_accepts_a_flat_record_list(self) -> None:
        """Some paths return records rather than {columns, rows}."""
        assert build_number_format([{"v": 9.9e9}], None)["suffix"] == "B"

    def test_ignores_booleans_when_sizing(self) -> None:
        """bool is an int subclass; a True must not be treated as a magnitude."""
        assert build_number_format({"rows": [{"flag": True}]}, None) is None

    def test_no_numeric_values_returns_none(self) -> None:
        assert build_number_format({"rows": [{"a": "x"}]}, INDIAN_UNITS) is None

    def test_empty_rows_return_none(self) -> None:
        assert build_number_format({"rows": []}, INDIAN_UNITS) is None

    def test_malformed_threshold_entry_is_skipped(self) -> None:
        config = {
            "number_format_thresholds": {"bad": "oops", "billion": 10000000},
            "number_format_suffixes": {"billion": "Cr"},
        }

        assert build_number_format(self.ROWS, config)["suffix"] == "Cr"


class TestHumanizeIdentifier:
    def test_converts_snake_case_to_title_case(self) -> None:
        assert _humanize_identifier("product_category") == "Product Category"

    def test_handles_single_word(self) -> None:
        assert _humanize_identifier("region") == "Region"


class TestPersistedAssistantContent:
    def test_persists_fields_the_frontend_needs_to_rerender(self) -> None:
        """buildAssistantParts re-runs against THIS payload on reload.

        Omitting number_format made a reloaded table fall back to auto-detected
        B/M/K while the live one honoured the connection's Cr config.
        """
        payload = {
            "answer_type": "table",
            "message": "Nov 2016 total sales of 983,041,954.",
            "data": {"columns": ["m"], "rows": [{"m": "2016-11"}]},
            "sql": "SELECT 1",
            "number_format": {"divisor": 1e7, "suffix": "Cr", "decimals": 2},
            "applied_filters": [{"label": "Brand Name", "value": "Brand A"}],
        }

        stored = json.loads(build_persisted_assistant_content(payload))["payload"]

        assert stored["number_format"]["suffix"] == "Cr"
        assert len(stored["applied_filters"]) == 1
        assert stored["message"] == payload["message"]

    def test_falls_back_to_message_when_payload_is_unserialisable(self) -> None:
        class Unserialisable:
            pass

        content = build_persisted_assistant_content(
            {"message": "plain answer", "data": {"bad": {Unserialisable()}}}
        )

        assert "plain answer" in content


class TestHistoryReadableContent:
    def test_extracts_message_and_sql_from_stored_envelope(self) -> None:
        """Raw JSON as "history" wastes the budget and hides the meaning."""
        stored = json.dumps(
            {
                "kind": "assistant_payload_v1",
                "payload": {
                    "message": "Here is your chart.",
                    "data": [{"m": "2016-01", "v": 1}] * 40,
                    "sql": "SELECT 1 FROM sample\n  WHERE brand_name = 'Brand A'",
                },
            }
        )

        result = ConversationHistoryManager._readable_content(stored)

        assert "Here is your chart." in result
        assert "brand_name = 'Brand A'" in result
        # Far smaller than the envelope it came from.
        assert len(result) < len(stored) / 2

    def test_sql_is_whitespace_collapsed(self) -> None:
        stored = json.dumps({"payload": {"message": "ok", "sql": "SELECT\n  1"}})

        assert "[SQL used: SELECT 1]" in ConversationHistoryManager._readable_content(
            stored
        )

    def test_plain_text_passes_through(self) -> None:
        text = "trend of sales for Product A in 2016"

        assert ConversationHistoryManager._readable_content(text) == text

    def test_malformed_json_passes_through(self) -> None:
        assert ConversationHistoryManager._readable_content("{not json") == "{not json"

    def test_none_becomes_empty_string(self) -> None:
        assert ConversationHistoryManager._readable_content(None) == ""
