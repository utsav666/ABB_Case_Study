"""Tests for the explanation route's analysis maths.

These functions are the reason a "why was X high?" answer can be trusted: the
LLM only narrates figures computed here, so a bug in this file produces a
confident, well-written, wrong answer. They are also pure — no DB, no LLM — which
makes them cheap to pin down thoroughly.
"""

from __future__ import annotations

import pandas as pd

from app.agent.analysis_agent import (
    _first_non_numeric_column,
    _first_numeric_column,
    _strip_sql_fences,
    compute_baseline_stats,
    compute_contributions,
)

# A 2016 monthly series where November is the clear peak.
MONTHLY_SALES = pd.DataFrame(
    {
        "month": [
            "2016-01",
            "2016-02",
            "2016-03",
            "2016-04",
            "2016-05",
            "2016-06",
            "2016-07",
            "2016-08",
            "2016-09",
            "2016-10",
            "2016-11",
            "2016-12",
        ],
        "total_sales": [
            64e7,
            63e7,
            68e7,
            69e7,
            60e7,
            81e7,
            86e7,
            74e7,
            66e7,
            68e7,
            96e7,
            65e7,
        ],
    }
)


class TestComputeBaselineStats:
    def test_identifies_focal_period_by_label(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["focal_period"] == "2016-11"
        assert stats["focal_value"] == 96e7
        assert stats["period_count"] == 12
        assert stats["metric_name"] == "total_sales"

    def test_compares_against_mean_of_other_periods(self) -> None:
        """The baseline EXCLUDES the focal period.

        Including it would dilute the very gap being explained.
        """
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        others = [v for v in MONTHLY_SALES["total_sales"] if v != 96e7]
        assert stats["baseline_mean"] == sum(others) / len(others)
        assert stats["difference"] == 96e7 - stats["baseline_mean"]
        assert stats["pct_vs_baseline"] > 0

    def test_ranks_focal_period(self) -> None:
        """rank lets the narrative contradict a wrong premise."""
        assert compute_baseline_stats(MONTHLY_SALES, "2016-11")["rank"] == 1
        # June is the lowest of 12.
        assert compute_baseline_stats(MONTHLY_SALES, "2016-05")["rank"] == 12

    def test_reports_std_devs_to_distinguish_outlier_from_good_month(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["std_devs_from_mean"] > 2

    def test_reports_highest_and_lowest_other_periods(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["max_other_period"] == "2016-07"
        assert stats["max_other_value"] == 86e7
        assert stats["min_other_period"] == "2016-05"
        assert stats["min_other_value"] == 60e7

    def test_substring_match_finds_focal_period(self) -> None:
        """"Nov 2016" style labels must match a "2016-11" row."""
        frame = pd.DataFrame(
            {"m": ["Jan 2016", "Nov 2016"], "v": [10.0, 90.0]},
        )

        assert compute_baseline_stats(frame, "Nov")["focal_period"] == "Nov 2016"

    def test_falls_back_to_max_when_focal_label_absent(self) -> None:
        """A "why is it high" question is usually about the peak."""
        stats = compute_baseline_stats(MONTHLY_SALES, "2017-03")

        assert stats["focal_period"] == "2016-11"

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert compute_baseline_stats(pd.DataFrame(), "2016-11") == {}

    def test_frame_without_numeric_column_returns_empty_dict(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

        assert compute_baseline_stats(frame, "x") == {}

    def test_single_period_omits_comparison_keys(self) -> None:
        """With no peers there is nothing to compare against."""
        frame = pd.DataFrame({"m": ["2016-11"], "v": [96e7]})

        stats = compute_baseline_stats(frame, "2016-11")

        assert stats["focal_value"] == 96e7
        assert "baseline_mean" not in stats
        assert "rank" not in stats

    def test_nulls_are_dropped(self) -> None:
        frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [1.0, None, 3.0]})

        stats = compute_baseline_stats(frame, "c")

        assert stats["focal_value"] == 3.0
        assert stats["period_count"] == 2

    def test_zero_baseline_mean_does_not_raise(self) -> None:
        """Guards against a divide-by-zero in the percentage."""
        frame = pd.DataFrame({"m": ["a", "b"], "v": [5.0, 0.0]})

        stats = compute_baseline_stats(frame, "a")

        assert stats["baseline_mean"] == 0.0
        assert stats["pct_vs_baseline"] is None


class TestComputeContributions:
    BREAKDOWN = pd.DataFrame(
        {
            "category": ["Category D", "Category A", "Category C"],
            "total_sales": [42e7, 21e7, 14e7],
        }
    )

    def test_shares_are_relative_to_the_focal_total(self) -> None:
        """Share-of-total is what explains a gap; ranking alone does not."""
        result = compute_contributions(self.BREAKDOWN, focal_total=96e7)

        assert result[0]["member"] == "Category D"
        assert round(result[0]["share_pct"], 1) == 43.8

    def test_falls_back_to_column_sum_when_no_total_given(self) -> None:
        result = compute_contributions(self.BREAKDOWN)

        assert round(sum(r["share_pct"] for r in result), 2) == 100.0

    def test_sorted_descending_and_limited_to_top_n(self) -> None:
        result = compute_contributions(self.BREAKDOWN, top_n=2)

        assert len(result) == 2
        assert result[0]["value"] > result[1]["value"]

    def test_duplicate_members_are_aggregated(self) -> None:
        frame = pd.DataFrame(
            {"c": ["North", "North", "South"], "v": [10.0, 20.0, 5.0]}
        )

        result = compute_contributions(frame)

        assert len(result) == 2
        assert result[0]["member"] == "North"
        assert result[0]["value"] == 30.0

    def test_empty_frame_returns_empty_list(self) -> None:
        assert compute_contributions(pd.DataFrame()) == []

    def test_zero_total_returns_empty_list(self) -> None:
        """Avoids a divide-by-zero when every value is zero."""
        frame = pd.DataFrame({"c": ["a", "b"], "v": [0.0, 0.0]})

        assert compute_contributions(frame) == []


class TestHelpers:
    def test_first_numeric_column_skips_labels(self) -> None:
        frame = pd.DataFrame({"label": ["a"], "count": [1], "name": ["x"]})

        assert _first_numeric_column(frame) == "count"
        assert _first_non_numeric_column(frame) == "label"

    def test_first_numeric_column_returns_none_when_absent(self) -> None:
        assert _first_numeric_column(pd.DataFrame({"a": ["x"]})) is None

    def test_strips_sql_code_fences(self) -> None:
        assert _strip_sql_fences("```sql\nSELECT 1\n```") == "SELECT 1"
        assert _strip_sql_fences("```\nSELECT 1\n```") == "SELECT 1"

    def test_leaves_unfenced_text_untouched(self) -> None:
        assert _strip_sql_fences("  SELECT 1  ") == "SELECT 1"
