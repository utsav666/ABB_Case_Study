"""Tests for column profiling classification and query building.

These are pure functions — no DB — and they decide how each column gets
profiled. Misclassifying a column means the wrong profiling query runs, so the
catalog silently ends up with no distinct values for it, which in turn weakens
the value-grounding the HITL resolver depends on.

Expected values were established by probing the implementation and are asserted
exactly, including the branch-precedence quirks noted below.
"""

from __future__ import annotations

import pytest

from app.semantic_catalog.profiling_queries import (
    HIGH_CARDINALITY_TOP_N,
    LOW_CARDINALITY_THRESHOLD,
    classify_column_type,
    infer_granularity,
    numeric_profile_query,
    temporal_profile_query,
    text_cardinality_query,
    text_distinct_values_query,
    text_sample_values_query,
    text_top_values_query,
)


class TestClassifyColumnType:
    @pytest.mark.parametrize(
        "data_type",
        [
            "text",
            "varchar",
            "VARCHAR(50)",
            "character varying",
            "char(3)",
            "nvarchar(max)",
            "string",
            "  TEXT  ",
        ],
    )
    def test_text_types(self, data_type: str) -> None:
        assert classify_column_type(data_type) == "text"

    @pytest.mark.parametrize(
        "data_type",
        [
            "integer",
            "bigint",
            "smallint",
            "numeric(10,2)",
            "decimal",
            "real",
            "double precision",
            "INTEGER",
        ],
    )
    def test_numeric_types(self, data_type: str) -> None:
        assert classify_column_type(data_type) == "numeric"

    @pytest.mark.parametrize(
        "data_type",
        [
            "date",
            "timestamp",
            "timestamp without time zone",
            "timestamp with time zone",
            "time",
            "TIMESTAMP",
        ],
    )
    def test_temporal_types(self, data_type: str) -> None:
        assert classify_column_type(data_type) == "temporal"

    @pytest.mark.parametrize(
        "data_type", ["boolean", "bytea", "json", "jsonb", "ARRAY", ""]
    )
    def test_unprofilable_types_are_other(self, data_type: str) -> None:
        """'other' means "skip profiling", so anything unrecognised lands here."""
        assert classify_column_type(data_type) == "other"

    def test_uuid_classifies_as_text(self) -> None:
        """Documents real behaviour: uuid is profiled as a text column.

        Reasonable — its values are strings and distinct-value profiling still
        works — but worth pinning, since a UUID column is high-cardinality by
        definition and will always take the top-N sampling path.
        """
        assert classify_column_type("uuid") == "text"

    def test_temporal_is_checked_before_numeric_and_text(self) -> None:
        """Precedence matters: 'timestamp' must not be caught by a text prefix.

        Pinned because reordering the checks in the implementation would be a
        silent behaviour change.
        """
        assert classify_column_type("timestamp without time zone") == "temporal"
        assert classify_column_type("time") == "temporal"

    def test_classification_is_case_and_whitespace_insensitive(self) -> None:
        assert classify_column_type("  VaRcHaR(20)  ") == "text"
        assert classify_column_type("\tBIGINT\n") == "numeric"

    def test_returns_one_of_exactly_four_labels(self) -> None:
        """Callers branch on these strings, so the set must not grow silently."""
        labels = {
            classify_column_type(value)
            for value in ("text", "integer", "date", "boolean", "nonsense")
        }

        assert labels <= {"text", "numeric", "temporal", "other"}


class TestInferGranularity:
    def test_daily_when_days_far_exceed_months(self) -> None:
        """365 days across 12 months is ~30 days/month — clearly daily."""
        assert infer_granularity(365, 12, 1) == "daily"

    def test_monthly_when_days_match_months(self) -> None:
        """One distinct day per month is a monthly series."""
        assert infer_granularity(12, 12, 1) == "monthly"

    def test_irregular_for_a_ratio_between_the_two(self) -> None:
        """2 days per month is neither daily nor monthly."""
        assert infer_granularity(24, 12, 2) == "irregular"

    def test_unknown_when_there_is_no_data(self) -> None:
        assert infer_granularity(0, 0, 0) == "unknown"

    def test_yearly_check_takes_precedence_over_the_day_ratio(self) -> None:
        """KNOWN PRECEDENCE, pinned deliberately.

        30 days in 1 month over 1 year looks daily by ratio, but months (1) <=
        years (1) triggers the yearly branch first. Arguably wrong for a
        one-month sample, and the assertion documents that rather than hiding it.
        """
        assert infer_granularity(30, 1, 1) == "yearly"

    def test_irregular_when_months_exist_without_days(self) -> None:
        assert infer_granularity(0, 12, 1) == "irregular"

    def test_irregular_when_days_exist_without_months(self) -> None:
        assert infer_granularity(100, 0, 0) == "irregular"

    def test_boundary_at_fifteen_days_per_month(self) -> None:
        """> 15 is daily; exactly 15 is not."""
        assert infer_granularity(160, 10, 3) == "daily"  # 16.0/month
        assert infer_granularity(150, 10, 3) == "irregular"  # 15.0/month

    def test_boundary_at_one_and_a_half_days_per_month(self) -> None:
        """> 1.5 is irregular; at or below is monthly."""
        assert infer_granularity(20, 10, 3) == "irregular"  # 2.0/month
        assert infer_granularity(16, 10, 3) == "irregular"  # 1.6/month
        assert infer_granularity(15, 10, 3) == "monthly"  # exactly 1.5, not > 1.5
        assert infer_granularity(10, 10, 3) == "monthly"  # 1.0/month

    def test_never_divides_by_zero_months(self) -> None:
        """max(months, 1) guards the ratio; this must not raise."""
        assert infer_granularity(5, 0, 0) == "irregular"

    @pytest.mark.parametrize(
        ("days", "months", "years"),
        [(365, 12, 1), (12, 12, 1), (0, 0, 0), (30, 1, 1), (5, 0, 0), (0, 12, 1)],
    )
    def test_always_returns_a_known_label(
        self, days: int, months: int, years: int
    ) -> None:
        assert infer_granularity(days, months, years) in {
            "daily",
            "monthly",
            "yearly",
            "sub-daily",
            "irregular",
            "unknown",
        }


class TestQueryBuilders:
    """The builders emit SQL templates, so assert on structure not formatting."""

    def test_cardinality_query_counts_distinct_and_nulls(self) -> None:
        sql = text_cardinality_query().lower()

        assert "count(distinct" in sql
        assert "{col}" in sql
        assert "{table}" in sql

    def test_distinct_values_query_selects_distinct(self) -> None:
        sql = text_distinct_values_query().lower()

        assert "select distinct" in sql
        assert "{col}" in sql

    def test_top_values_query_orders_by_frequency_and_limits(self) -> None:
        """Top-N is only meaningful if it is ordered by count descending."""
        sql = text_top_values_query().lower()

        assert "group by" in sql
        assert "order by" in sql
        assert "desc" in sql
        assert "limit" in sql

    def test_top_values_query_honours_an_explicit_limit(self) -> None:
        sql = text_top_values_query(limit=7)

        assert "7" in sql

    def test_sample_values_query_limits_rows(self) -> None:
        sql = text_sample_values_query().lower()

        assert "limit" in sql

    def test_numeric_profile_query_covers_the_summary_statistics(self) -> None:
        sql = numeric_profile_query().lower()

        for aggregate in ("min(", "max(", "avg("):
            assert aggregate in sql, f"missing {aggregate}"

    def test_temporal_profile_query_counts_each_granularity(self) -> None:
        """infer_granularity consumes these three counts, so all must be present."""
        sql = temporal_profile_query().lower()

        assert "min(" in sql
        assert "max(" in sql
        assert "distinct" in sql

    @pytest.mark.parametrize(
        "builder",
        [
            text_cardinality_query,
            text_distinct_values_query,
            text_top_values_query,
            text_sample_values_query,
            numeric_profile_query,
            temporal_profile_query,
        ],
    )
    def test_every_builder_returns_a_non_empty_string(self, builder) -> None:
        result = builder()

        assert isinstance(result, str)
        assert result.strip()

    @pytest.mark.parametrize(
        "builder",
        [
            text_cardinality_query,
            text_distinct_values_query,
            numeric_profile_query,
            temporal_profile_query,
        ],
    )
    def test_builders_reference_the_table_and_column_placeholders(
        self, builder
    ) -> None:
        """These are .format()-ed by the caller; a missing placeholder means the
        profiler would query the wrong object."""
        sql = builder()

        assert "{table}" in sql
        assert "{col}" in sql


class TestThresholds:
    def test_low_cardinality_threshold_is_a_positive_int(self) -> None:
        assert isinstance(LOW_CARDINALITY_THRESHOLD, int)
        assert LOW_CARDINALITY_THRESHOLD > 0

    def test_top_n_is_a_positive_int(self) -> None:
        assert isinstance(HIGH_CARDINALITY_TOP_N, int)
        assert HIGH_CARDINALITY_TOP_N > 0

    def test_top_n_does_not_exceed_the_low_cardinality_threshold(self) -> None:
        """Sampling more values than the "fetch everything" cutoff would be
        contradictory: below the threshold all values are already fetched."""
        assert HIGH_CARDINALITY_TOP_N <= LOW_CARDINALITY_THRESHOLD
