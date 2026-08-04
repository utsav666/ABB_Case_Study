"""Tests for semantic annotation prompt building and response parsing.

These sit either side of an LLM call. The formatters build the prompt that
teaches the model about each column; the parser turns its reply into catalog
entries. Both degrade silently on failure — a truncated profile or a dropped
column produces a catalog that is merely *less* useful, with no error — so the
contracts are worth pinning.
"""

from __future__ import annotations

import json

from app.semantic_catalog.semantic_annotator import (
    _format_columns_profile,
    _format_columns_schema,
    _parse_annotation_response,
)


class TestFormatColumnsSchema:
    def test_renders_type_and_nullability(self) -> None:
        columns = {"id": {"type": "integer", "notnull": True, "is_pk": True}}

        assert _format_columns_schema(columns) == "  id: integer NOT NULL PK"

    def test_nullable_column_is_marked_null(self) -> None:
        columns = {"note": {"type": "text", "notnull": False}}

        assert _format_columns_schema(columns) == "  note: text NULL"

    def test_primary_key_marker_only_appears_for_a_pk(self) -> None:
        columns = {"a": {"type": "text", "notnull": True, "is_pk": False}}

        assert "PK" not in _format_columns_schema(columns)

    def test_missing_type_falls_back_to_unknown(self) -> None:
        """Better a visible UNKNOWN in the prompt than a bare column name."""
        assert "UNKNOWN" in _format_columns_schema({"mystery": {}})

    def test_one_line_per_column_in_insertion_order(self) -> None:
        columns = {
            "first": {"type": "text"},
            "second": {"type": "integer"},
            "third": {"type": "date"},
        }

        lines = _format_columns_schema(columns).split("\n")

        assert len(lines) == 3
        assert lines[0].strip().startswith("first")
        assert lines[2].strip().startswith("third")

    def test_no_columns_renders_an_empty_string(self) -> None:
        assert _format_columns_schema({}) == ""


class TestFormatColumnsProfile:
    def test_text_profile_includes_cardinality_and_values(self) -> None:
        profiles = {
            "category": {
                "profile_type": "text",
                "cardinality": 3,
                "null_ratio": 0.0,
                "distinct_values": ["A", "B", "C"],
            }
        }

        line = _format_columns_profile(profiles)

        assert "TEXT" in line
        assert "cardinality=3" in line
        assert "['A', 'B', 'C']" in line

    def test_null_ratio_is_rendered_as_a_percentage(self) -> None:
        profiles = {
            "c": {"profile_type": "text", "cardinality": 1, "null_ratio": 0.125}
        }

        assert "nulls=12.5%" in _format_columns_profile(profiles)

    def test_value_list_is_truncated_with_a_remainder_count(self) -> None:
        """Prompt token budget: an unbounded list of values would dominate it."""
        profiles = {
            "c": {
                "profile_type": "text",
                "cardinality": 60,
                "null_ratio": 0.0,
                "distinct_values": [f"v{i}" for i in range(60)],
            }
        }

        line = _format_columns_profile(profiles)

        assert "(+10 more)" in line
        assert "v49" in line
        assert "v50" not in line

    def test_exactly_fifty_values_are_not_truncated(self) -> None:
        profiles = {
            "c": {
                "profile_type": "text",
                "cardinality": 50,
                "null_ratio": 0.0,
                "distinct_values": [f"v{i}" for i in range(50)],
            }
        }

        assert "more)" not in _format_columns_profile(profiles)

    def test_falls_back_to_top_values_when_distinct_values_are_absent(self) -> None:
        """High-cardinality columns are profiled by frequency, not exhaustively."""
        profiles = {
            "c": {
                "profile_type": "text",
                "cardinality": 5000,
                "null_ratio": 0.0,
                "top_values": [{"value": "common"}, {"value": "next"}],
            }
        }

        line = _format_columns_profile(profiles)

        assert "common" in line
        assert "next" in line

    def test_numeric_profile_includes_the_summary_statistics(self) -> None:
        profiles = {
            "amount": {
                "profile_type": "numeric",
                "min_value": 1,
                "max_value": 100,
                "mean_value": 50.5,
                "median_value": 50,
                "null_ratio": 0.01,
            }
        }

        line = _format_columns_profile(profiles)

        assert "NUMERIC" in line
        assert "min=1" in line
        assert "max=100" in line
        assert "mean=50.5" in line

    def test_temporal_profile_includes_range_and_granularity(self) -> None:
        profiles = {
            "order_date": {
                "profile_type": "temporal",
                "min_date": "2015-01-01",
                "max_date": "2016-12-31",
                "granularity": "daily",
                "null_ratio": 0.0,
            }
        }

        line = _format_columns_profile(profiles)

        assert "TEMPORAL" in line
        assert "2015-01-01 → 2016-12-31" in line
        assert "granularity=daily" in line

    def test_unknown_profile_type_still_names_the_column(self) -> None:
        """A skipped column must remain visible in the prompt."""
        line = _format_columns_profile({"blob": {"profile_type": "other"}})

        assert "blob" in line
        assert "(other)" in line

    def test_missing_profile_type_is_reported_as_unknown(self) -> None:
        assert "(unknown)" in _format_columns_profile({"c": {}})

    def test_one_line_per_column(self) -> None:
        profiles = {
            "a": {"profile_type": "text", "cardinality": 1, "null_ratio": 0.0},
            "b": {"profile_type": "numeric", "null_ratio": 0.0},
            "c": {"profile_type": "temporal", "null_ratio": 0.0},
        }

        assert len(_format_columns_profile(profiles).split("\n")) == 3

    def test_no_profiles_renders_an_empty_string(self) -> None:
        assert _format_columns_profile({}) == ""


class TestParseAnnotationResponse:
    def test_parses_a_plain_json_object(self) -> None:
        raw = json.dumps({"category": {"description": "Product grouping"}})

        result = _parse_annotation_response(raw, ["category"])

        assert result["category"]["description"] == "Product grouping"

    def test_strips_markdown_code_fences(self) -> None:
        """LLMs wrap JSON in fences despite instructions not to."""
        raw = '```json\n{"c": {"description": "d"}}\n```'

        assert _parse_annotation_response(raw, ["c"])["c"]["description"] == "d"

    def test_strips_a_bare_fence(self) -> None:
        raw = '```\n{"c": {"description": "d"}}\n```'

        assert _parse_annotation_response(raw, ["c"])["c"]["description"] == "d"

    def test_every_expected_column_gets_an_entry(self) -> None:
        """Callers index by column name; a missing key would be a KeyError."""
        raw = json.dumps({"a": {"description": "x"}})

        result = _parse_annotation_response(raw, ["a", "b", "c"])

        assert set(result) >= {"a", "b", "c"}
        assert result["b"] == {}

    def test_malformed_json_yields_empty_entries_rather_than_raising(self) -> None:
        """Annotation is best-effort enrichment; it must not abort a profiling run."""
        result = _parse_annotation_response("not json at all", ["a", "b"])

        assert result == {"a": {}, "b": {}}

    def test_a_json_list_is_rejected_as_the_wrong_shape(self) -> None:
        result = _parse_annotation_response('[{"a": 1}]', ["a"])

        assert result == {"a": {}}

    def test_a_json_scalar_is_rejected(self) -> None:
        assert _parse_annotation_response('"just a string"', ["a"]) == {"a": {}}

    def test_extra_columns_from_the_llm_are_retained(self) -> None:
        """Dropping them would discard information the model volunteered."""
        raw = json.dumps(
            {"a": {"description": "x"}, "unexpected": {"description": "y"}}
        )

        result = _parse_annotation_response(raw, ["a"])

        assert "unexpected" in result

    def test_empty_response_yields_empty_entries(self) -> None:
        assert _parse_annotation_response("", ["a"]) == {"a": {}}

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        raw = '\n\n  {"c": {"description": "d"}}  \n'

        assert _parse_annotation_response(raw, ["c"])["c"]["description"] == "d"
