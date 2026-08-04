"""Tests for the explanation route's analysis maths.

These functions are why a "why was X high?" answer can be trusted: the LLM only
narrates figures computed here and is forbidden from recomputing them, so a bug
in this file produces a confident, well-written, wrong answer. They are pure —
no DB, no LLM — so they can be pinned down exactly.

Assertions use exact expected values (derived independently, not copied from the
implementation's output) rather than loose bounds like `> 0`, which would pass
even if the arithmetic were wrong.
"""

from __future__ import annotations

import math
import statistics

import pandas as pd
import pytest

from app.agent.analysis_agent import (
    MAX_ANALYSIS_ROWS,
    _first_non_numeric_column,
    _first_numeric_column,
    _strip_sql_fences,
    compute_baseline_stats,
    compute_contributions,
)

# A 2016 monthly series where November (96e7) is the clear peak and May (60e7)
# the trough. July (86e7) is the second-highest, so max_other_* is unambiguous.
MONTHS = [f"2016-{month:02d}" for month in range(1, 13)]
SALES = [64e7, 63e7, 68e7, 69e7, 60e7, 81e7, 86e7, 74e7, 66e7, 68e7, 96e7, 65e7]
MONTHLY_SALES = pd.DataFrame({"month": MONTHS, "total_sales": SALES})

# Expected values computed from SALES independently of the implementation.
FOCAL = 96e7
OTHERS = [value for value in SALES if value != FOCAL]
BASELINE_MEAN = sum(OTHERS) / len(OTHERS)  # 694_545_454.5454545
DIFFERENCE = FOCAL - BASELINE_MEAN  # 265_454_545.4545455
PCT_VS_BASELINE = DIFFERENCE / BASELINE_MEAN * 100  # 38.219895...
STD_DEVS = DIFFERENCE / statistics.stdev(OTHERS)  # 3.3584967... (ddof=1)


class TestComputeBaselineStats:
    def test_identifies_focal_period_and_metric(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["focal_period"] == "2016-11"
        assert stats["focal_value"] == 96e7
        assert stats["metric_name"] == "total_sales"
        assert stats["period_count"] == 12

    def test_baseline_mean_excludes_the_focal_period(self) -> None:
        """Including the focal value would dilute the gap being explained.

        Pinned to the 11-period mean, so an implementation that averaged all 12
        (735_000_000.0) would fail.
        """
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["baseline_mean"] == pytest.approx(694_545_454.5454545, abs=1e-6)
        assert stats["baseline_mean"] == pytest.approx(BASELINE_MEAN, abs=1e-9)
        # Guard against the all-12 mean specifically.
        assert stats["baseline_mean"] != pytest.approx(sum(SALES) / len(SALES))

    def test_difference_and_percentage_are_exact(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["difference"] == pytest.approx(265_454_545.4545455, abs=1e-6)
        assert stats["pct_vs_baseline"] == pytest.approx(38.219895, abs=1e-6)
        # The percentage must be relative to the baseline, not the focal value.
        assert stats["pct_vs_baseline"] != pytest.approx(DIFFERENCE / FOCAL * 100)

    def test_std_devs_uses_sample_stdev_of_other_periods(self) -> None:
        """Pinned to ddof=1 over the 11 other periods.

        A population stdev (ddof=0) or one including the focal value would give a
        different number, so this catches either substitution.
        """
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["std_devs_from_mean"] == pytest.approx(3.3584968, abs=1e-6)
        assert stats["std_devs_from_mean"] == pytest.approx(STD_DEVS, abs=1e-9)

    def test_rank_is_one_for_the_peak_and_last_for_the_trough(self) -> None:
        """rank is what lets the narrative contradict a wrong premise."""
        assert compute_baseline_stats(MONTHLY_SALES, "2016-11")["rank"] == 1
        assert compute_baseline_stats(MONTHLY_SALES, "2016-05")["rank"] == 12
        assert compute_baseline_stats(MONTHLY_SALES, "2016-07")["rank"] == 2

    def test_rank_counts_strictly_greater_values_only(self) -> None:
        """Tied values must share a rank rather than displacing each other."""
        frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [10.0, 10.0, 5.0]})

        # Two values tie for highest; both are rank 1.
        assert compute_baseline_stats(frame, "a")["rank"] == 1
        assert compute_baseline_stats(frame, "b")["rank"] == 1
        assert compute_baseline_stats(frame, "c")["rank"] == 3

    def test_reports_highest_and_lowest_other_periods_exactly(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        assert stats["max_other_period"] == "2016-07"
        assert stats["max_other_value"] == 86e7
        assert stats["min_other_period"] == "2016-05"
        assert stats["min_other_value"] == 60e7
        # The focal period must never be reported as its own comparator.
        assert stats["max_other_value"] != stats["focal_value"]

    def test_all_numeric_outputs_are_plain_floats(self) -> None:
        """numpy scalars break json.dumps when the payload is persisted."""
        stats = compute_baseline_stats(MONTHLY_SALES, "2016-11")

        for key in (
            "focal_value",
            "baseline_mean",
            "difference",
            "pct_vs_baseline",
            "max_other_value",
            "min_other_value",
            "std_devs_from_mean",
        ):
            assert type(stats[key]) is float, f"{key} is {type(stats[key])}"
        assert type(stats["rank"]) is int
        assert type(stats["period_count"]) is int
        assert type(stats["focal_period"]) is str

    def test_substring_match_is_case_insensitive(self) -> None:
        """ "nov" must find "Nov 2016" — users do not match the stored casing."""
        frame = pd.DataFrame({"m": ["Jan 2016", "Nov 2016"], "v": [10.0, 90.0]})

        assert compute_baseline_stats(frame, "nov")["focal_period"] == "Nov 2016"
        assert compute_baseline_stats(frame, "NOV")["focal_period"] == "Nov 2016"

    def test_substring_match_takes_the_first_match_not_the_max(self) -> None:
        """A named focal period must win over the largest value."""
        frame = pd.DataFrame(
            {"m": ["2016-03", "2016-08"], "v": [10.0, 99.0]},
        )

        stats = compute_baseline_stats(frame, "2016-03")

        assert stats["focal_period"] == "2016-03"
        assert stats["focal_value"] == 10.0
        assert stats["rank"] == 2

    def test_focal_label_is_matched_literally_not_as_a_regex(self) -> None:
        """A label containing regex metacharacters must not raise or mis-match."""
        frame = pd.DataFrame({"m": ["Q1 (est.)", "Q2"], "v": [5.0, 9.0]})

        assert compute_baseline_stats(frame, "Q1 (est.)")["focal_period"] == "Q1 (est.)"

    def test_falls_back_to_max_when_focal_label_absent(self) -> None:
        """A "why is it high" question is usually about the peak."""
        stats = compute_baseline_stats(MONTHLY_SALES, "2017-03")

        assert stats["focal_period"] == "2016-11"
        assert stats["focal_value"] == 96e7

    def test_falls_back_to_max_when_focal_label_is_none(self) -> None:
        stats = compute_baseline_stats(MONTHLY_SALES, None)

        assert stats["focal_period"] == "2016-11"

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert compute_baseline_stats(pd.DataFrame(), "2016-11") == {}

    def test_frame_without_numeric_column_returns_empty_dict(self) -> None:
        frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})

        assert compute_baseline_stats(frame, "x") == {}

    def test_frame_without_label_column_returns_empty_dict(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        assert compute_baseline_stats(frame, "1") == {}

    def test_all_null_metric_returns_empty_dict(self) -> None:
        frame = pd.DataFrame({"m": ["a", "b"], "v": [None, None]})

        assert compute_baseline_stats(frame, "a") == {}

    def test_single_period_omits_every_comparison_key(self) -> None:
        """With no peers there is nothing to compare against.

        Asserting on the exact key set prevents a partially-populated result
        being reported as a comparison.
        """
        frame = pd.DataFrame({"m": ["2016-11"], "v": [96e7]})

        stats = compute_baseline_stats(frame, "2016-11")

        assert stats["focal_value"] == 96e7
        assert stats["period_count"] == 1
        assert set(stats) == {
            "focal_period",
            "focal_value",
            "metric_name",
            "period_count",
        }

    def test_nulls_are_dropped_before_counting_and_ranking(self) -> None:
        frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [1.0, None, 3.0]})

        stats = compute_baseline_stats(frame, "c")

        assert stats["focal_value"] == 3.0
        assert stats["period_count"] == 2  # not 3
        assert stats["baseline_mean"] == 1.0
        assert stats["rank"] == 1

    def test_zero_baseline_mean_yields_none_percentage_not_infinity(self) -> None:
        """Guards the divide-by-zero. None is renderable; inf and nan are not."""
        frame = pd.DataFrame({"m": ["a", "b"], "v": [5.0, 0.0]})

        stats = compute_baseline_stats(frame, "a")

        assert stats["baseline_mean"] == 0.0
        assert stats["difference"] == 5.0
        assert stats["pct_vs_baseline"] is None

    def test_negative_values_are_handled_signed(self) -> None:
        """A loss-making period must not be treated as magnitude."""
        frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [-30.0, 10.0, 20.0]})

        stats = compute_baseline_stats(frame, "a")

        assert stats["focal_value"] == -30.0
        assert stats["baseline_mean"] == 15.0
        assert stats["difference"] == -45.0
        assert stats["pct_vs_baseline"] == pytest.approx(-300.0)
        assert stats["rank"] == 3  # lowest of three

    def test_identical_values_give_zero_difference_and_no_std_devs(self) -> None:
        """Zero variance must not produce a divide-by-zero or nan."""
        frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [10.0, 10.0, 10.0]})

        stats = compute_baseline_stats(frame, "a")

        assert stats["difference"] == 0.0
        assert stats["pct_vs_baseline"] == 0.0
        assert "std_devs_from_mean" not in stats

    def test_integer_metric_column_is_supported(self) -> None:
        """Counts are as valid a metric as currency."""
        frame = pd.DataFrame({"m": ["a", "b"], "orders": [10, 30]})

        stats = compute_baseline_stats(frame, "b")

        assert stats["metric_name"] == "orders"
        assert stats["focal_value"] == 30.0
        assert stats["baseline_mean"] == 10.0


class TestComputeContributions:
    BREAKDOWN = pd.DataFrame(
        {
            "category": ["Category D", "Category A", "Category C"],
            "total_sales": [42e7, 21e7, 14e7],
        }
    )

    def test_shares_are_relative_to_the_supplied_focal_total(self) -> None:
        """Share-of-total explains a gap; ranking alone does not.

        42e7 / 96e7 = 43.75%. If the implementation divided by the column sum
        (77e7) instead, this would read 54.5%.
        """
        result = compute_contributions(self.BREAKDOWN, focal_total=96e7)

        assert result[0]["member"] == "Category D"
        assert result[0]["value"] == 42e7
        assert result[0]["share_pct"] == pytest.approx(43.75, abs=1e-9)
        assert result[1]["share_pct"] == pytest.approx(21.875, abs=1e-9)
        assert result[2]["share_pct"] == pytest.approx(14.583333, abs=1e-6)

    def test_shares_do_not_sum_to_100_when_focal_total_is_larger(self) -> None:
        """The listed members are a subset; the remainder is unexplained."""
        result = compute_contributions(self.BREAKDOWN, focal_total=96e7)

        assert sum(r["share_pct"] for r in result) == pytest.approx(80.208333, abs=1e-6)

    def test_falls_back_to_column_sum_when_no_total_given(self) -> None:
        result = compute_contributions(self.BREAKDOWN)

        assert sum(r["share_pct"] for r in result) == pytest.approx(100.0, abs=1e-9)
        assert result[0]["share_pct"] == pytest.approx(42 / 77 * 100, abs=1e-9)

    def test_sorted_descending_by_value(self) -> None:
        result = compute_contributions(self.BREAKDOWN)

        values = [row["value"] for row in result]
        assert values == sorted(values, reverse=True)
        assert values == [42e7, 21e7, 14e7]

    def test_top_n_truncates_after_sorting_not_before(self) -> None:
        """An unsorted truncation would drop the largest contributor."""
        frame = pd.DataFrame(
            {"c": ["small", "big", "mid"], "v": [1.0, 100.0, 50.0]},
        )

        result = compute_contributions(frame, top_n=2)

        assert [row["member"] for row in result] == ["big", "mid"]

    def test_default_top_n_is_five(self) -> None:
        frame = pd.DataFrame(
            {"c": [f"m{i}" for i in range(9)], "v": [float(9 - i) for i in range(9)]}
        )

        assert len(compute_contributions(frame)) == 5

    def test_duplicate_members_are_summed_before_ranking(self) -> None:
        """Two North rows totalling 30 must outrank a single South row of 25."""
        frame = pd.DataFrame(
            {"c": ["North", "South", "North"], "v": [10.0, 25.0, 20.0]},
        )

        result = compute_contributions(frame)

        assert len(result) == 2
        assert result[0]["member"] == "North"
        assert result[0]["value"] == 30.0
        assert result[1]["member"] == "South"

    def test_member_names_are_strings_and_values_plain_floats(self) -> None:
        """Output must survive json.dumps when the payload is persisted."""
        frame = pd.DataFrame({"c": ["a", "b"], "v": [1, 2]})

        result = compute_contributions(frame)

        assert type(result[0]["member"]) is str
        assert type(result[0]["value"]) is float
        assert type(result[0]["share_pct"]) is float

    def test_all_numeric_columns_yields_no_contributions(self) -> None:
        """Documents a real limitation, so it fails loudly rather than silently.

        The label column is found by "first non-numeric", so a breakdown whose
        dimension is numeric (e.g. `year`) produces nothing. The planner prompt
        asks for a dimension label plus a metric, which avoids this — but if it
        ever returns two numeric columns the caller gets [] rather than a
        mislabelled ranking.
        """
        frame = pd.DataFrame({"year": [2015, 2016], "v": [1, 2]})

        assert compute_contributions(frame) == []

    def test_nulls_are_dropped(self) -> None:
        frame = pd.DataFrame({"c": ["a", "b", "c"], "v": [10.0, None, 30.0]})

        result = compute_contributions(frame)

        assert [row["member"] for row in result] == ["c", "a"]
        assert sum(r["share_pct"] for r in result) == pytest.approx(100.0)

    def test_empty_frame_returns_empty_list(self) -> None:
        assert compute_contributions(pd.DataFrame()) == []

    def test_frame_without_numeric_column_returns_empty_list(self) -> None:
        assert compute_contributions(pd.DataFrame({"a": ["x"], "b": ["y"]})) == []

    def test_zero_total_returns_empty_list_rather_than_nan_shares(self) -> None:
        """Avoids a divide-by-zero producing nan percentages in the narrative."""
        frame = pd.DataFrame({"c": ["a", "b"], "v": [0.0, 0.0]})

        assert compute_contributions(frame) == []

    def test_negative_contributions_keep_their_sign(self) -> None:
        frame = pd.DataFrame({"c": ["gain", "loss"], "v": [100.0, -20.0]})

        result = compute_contributions(frame, focal_total=80.0)

        assert result[0]["value"] == 100.0
        assert result[1]["value"] == -20.0
        assert result[1]["share_pct"] == pytest.approx(-25.0)

    def test_share_is_finite_for_every_row(self) -> None:
        result = compute_contributions(self.BREAKDOWN, focal_total=96e7)

        assert all(math.isfinite(row["share_pct"]) for row in result)


class TestHelpers:
    def test_first_numeric_column_skips_leading_label(self) -> None:
        frame = pd.DataFrame({"label": ["a"], "count": [1], "name": ["x"]})

        assert _first_numeric_column(frame) == "count"
        assert _first_non_numeric_column(frame) == "label"

    def test_first_numeric_column_returns_leftmost_of_several(self) -> None:
        """The planner is told to return label-then-metric, so order matters."""
        frame = pd.DataFrame({"m": ["a"], "sales": [1.0], "orders": [2]})

        assert _first_numeric_column(frame) == "sales"

    def test_bool_columns_are_treated_as_numeric_by_pandas(self) -> None:
        """Documents real behaviour: bool is numeric to is_numeric_dtype.

        Pinned so a future change to the metric-picking rule is a deliberate
        decision rather than a silent one.
        """
        frame = pd.DataFrame({"label": ["a"], "flag": [True], "value": [5.0]})

        assert _first_numeric_column(frame) == "flag"

    def test_returns_none_when_no_column_of_that_kind_exists(self) -> None:
        assert _first_numeric_column(pd.DataFrame({"a": ["x"]})) is None
        assert _first_non_numeric_column(pd.DataFrame({"a": [1]})) is None

    def test_strips_sql_and_bare_code_fences(self) -> None:
        assert _strip_sql_fences("```sql\nSELECT 1\n```") == "SELECT 1"
        assert _strip_sql_fences("```\nSELECT 1\n```") == "SELECT 1"
        assert _strip_sql_fences("```json\n{}\n```") == "{}"

    def test_preserves_inner_newlines_of_multiline_sql(self) -> None:
        fenced = "```sql\nSELECT a,\n       b\nFROM t\n```"

        assert _strip_sql_fences(fenced) == "SELECT a,\n       b\nFROM t"

    def test_leaves_unfenced_text_untouched_apart_from_whitespace(self) -> None:
        assert _strip_sql_fences("  SELECT 1  ") == "SELECT 1"
        assert _strip_sql_fences("SELECT '```'") == "SELECT '```'"

    def test_row_cap_is_an_aggregate_sized_limit(self) -> None:
        """An explanation needs aggregates; a huge result means a wrong query."""
        assert MAX_ANALYSIS_ROWS == 500
