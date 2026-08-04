"""Tests for chart data shaping and display formatting.

These three functions were nested inside ``chart_approval_node`` and therefore
untestable. They decide what a chart actually shows: how a comparison is
reshaped, how the date axis is labelled, and what scale the numbers use.

One finding is pinned rather than fixed — see
``TestDetectAndPivotYoyComparison.test_date_based_pivot_is_unreachable``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.orchestrator.chart_formatting import (
    detect_and_pivot_yoy_comparison,
    detect_date_granularity,
    detect_number_format,
)

INDIAN_UNITS = {
    "decimal_places": 2,
    "number_format_enabled": True,
    "number_format_suffixes": {"billion": "Cr", "million": "L", "thousand": "K"},
    "number_format_thresholds": {
        "billion": 10000000,
        "million": 100000,
        "thousand": 1000,
    },
}


class TestDetectAndPivotYoyComparison:
    def test_non_comparison_question_is_passed_through(self) -> None:
        frame = pd.DataFrame({"month": ["2016-01"], "sales": [10]})

        result, pivoted = detect_and_pivot_yoy_comparison(frame, "total sales in 2016")

        assert pivoted is False
        assert result.equals(frame)

    @pytest.mark.parametrize(
        "question",
        [
            "compare 2015 and 2016",
            "2015 vs 2016 sales",
            "2016 versus 2015",
            "sales compared to last year",
            "previous year comparison",
            "year over year growth",
            "show me YoY sales",
        ],
    )
    def test_comparison_keywords_are_recognised(self, question: str) -> None:
        """Already-pivoted data (2+ numeric columns) is passed through as ready."""
        frame = pd.DataFrame(
            {"month": ["01", "02"], "sales_2015": [10, 20], "sales_2016": [30, 40]}
        )

        _result, pivoted = detect_and_pivot_yoy_comparison(frame, question)

        assert pivoted is True

    def test_keyword_matching_is_case_insensitive(self) -> None:
        frame = pd.DataFrame(
            {"m": ["01"], "sales_2015": [10], "sales_2016": [30]},
        )

        _result, pivoted = detect_and_pivot_yoy_comparison(
            frame, "COMPARE 2015 VS 2016"
        )

        assert pivoted is True

    def test_sql_pivoted_data_is_returned_unchanged(self) -> None:
        """When SQL already produced one column per period, reshaping would break it."""
        frame = pd.DataFrame(
            {"month": ["01", "02"], "sales_2015": [10, 20], "sales_2016": [30, 40]}
        )

        result, pivoted = detect_and_pivot_yoy_comparison(frame, "compare years")

        assert pivoted is True
        assert result.equals(frame)

    def test_two_numeric_columns_without_a_label_column_is_not_pivoted(self) -> None:
        """A pivot needs a categorical axis to plot against."""
        frame = pd.DataFrame({"sales_2015": [10, 20], "sales_2016": [30, 40]})

        _result, pivoted = detect_and_pivot_yoy_comparison(frame, "compare years")

        assert pivoted is False

    def test_date_based_pivot_is_unreachable(self) -> None:
        """KNOWN DEAD CODE, pinned rather than changed.

        chart_formatting.py has an unconditional ``return df, False`` immediately
        after the already-pivoted check, so the ~120 lines of date-based pivoting
        below it never execute. Present in the original nested version too (this
        is not an extraction artefact).

        Consequence: a comparison question only renders as multi-series when the
        SQL generator happened to pivot it. A long result with a real date column
        and two years of data is left unpivoted, which is why some comparison
        questions render as a single flat series.

        Asserting the CURRENT behaviour so that reviving the dead code is a
        deliberate act that shows up as a failing test.
        """
        frame = pd.DataFrame(
            {
                "order_date": pd.to_datetime(
                    ["2015-01-01", "2015-02-01", "2016-01-01", "2016-02-01"]
                ),
                "total_sales": [10, 20, 30, 40],
            }
        )

        result, pivoted = detect_and_pivot_yoy_comparison(frame, "compare 2015 vs 2016")

        assert pivoted is False
        assert list(result.columns) == ["order_date", "total_sales"]

    def test_empty_frame_does_not_raise(self) -> None:
        result, pivoted = detect_and_pivot_yoy_comparison(pd.DataFrame(), "compare")

        assert pivoted is False
        assert result.empty


class TestDetectDateGranularity:
    def test_consecutive_days_are_detected_as_day(self) -> None:
        frame = pd.DataFrame(
            {"d": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-03"])}
        )

        assert detect_date_granularity(frame, "d") == "day"

    def test_month_starts_are_detected_as_month(self) -> None:
        frame = pd.DataFrame(
            {"d": pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01"])}
        )

        assert detect_date_granularity(frame, "d") == "month"

    def test_year_starts_are_detected_as_year(self) -> None:
        frame = pd.DataFrame(
            {"d": pd.to_datetime(["2014-01-01", "2015-01-01", "2016-01-01"])}
        )

        assert detect_date_granularity(frame, "d") == "year"

    def test_weekly_spacing_is_detected_as_week(self) -> None:
        frame = pd.DataFrame(
            {"d": pd.to_datetime(["2016-01-01", "2016-01-08", "2016-01-15"])}
        )

        assert detect_date_granularity(frame, "d") == "week"

    def test_string_dates_are_parsed(self) -> None:
        """SQL results frequently arrive as text."""
        frame = pd.DataFrame({"d": ["2016-01-01", "2016-02-01", "2016-03-01"]})

        assert detect_date_granularity(frame, "d") == "month"

    def test_a_single_row_returns_none(self) -> None:
        """One point has no spacing to infer from."""
        frame = pd.DataFrame({"d": pd.to_datetime(["2016-01-01"])})

        assert detect_date_granularity(frame, "d") is None

    def test_a_non_date_column_returns_none(self) -> None:
        frame = pd.DataFrame({"c": ["Category A", "Category B"]})

        assert detect_date_granularity(frame, "c") is None

    def test_a_missing_column_returns_none_rather_than_raising(self) -> None:
        """Formatting must never break a chart that would otherwise render."""
        frame = pd.DataFrame({"d": pd.to_datetime(["2016-01-01"])})

        assert detect_date_granularity(frame, "does_not_exist") is None

    def test_an_empty_frame_returns_none(self) -> None:
        assert detect_date_granularity(pd.DataFrame({"d": []}), "d") is None


class TestDetectNumberFormat:
    FRAME = pd.DataFrame({"total_sales": [704810000.0, 983041954.21]})

    def test_uses_configured_thresholds_and_suffixes(self) -> None:
        """983,041,954 / 1e7 = 98.30Cr — the connection's configured scale."""
        result = detect_number_format(self.FRAME, ["total_sales"], INDIAN_UNITS)

        assert result == {"suffix": "Cr", "divisor": 10000000, "decimals": 2}

    def test_defaults_to_b_m_k_without_config(self) -> None:
        result = detect_number_format(self.FRAME, ["total_sales"], None)

        assert result["suffix"] == "M"

    def test_respects_an_explicit_opt_out(self) -> None:
        result = detect_number_format(
            self.FRAME, ["total_sales"], {"number_format_enabled": False}
        )

        assert result is None

    def test_scale_comes_from_the_largest_value_across_all_columns(self) -> None:
        """One scale per chart, so the axes stay comparable."""
        frame = pd.DataFrame({"small": [100.0], "large": [983041954.0]})

        result = detect_number_format(frame, ["small", "large"], INDIAN_UNITS)

        assert result["suffix"] == "Cr"

    def test_negative_values_are_sized_by_magnitude(self) -> None:
        frame = pd.DataFrame({"variance": [-983041954.0]})

        result = detect_number_format(frame, ["variance"], INDIAN_UNITS)

        assert result["suffix"] == "Cr"

    def test_small_values_get_a_unit_scale(self) -> None:
        """Not None: the chart still needs a divisor to render with."""
        frame = pd.DataFrame({"count": [5.0, 12.0]})

        result = detect_number_format(frame, ["count"], INDIAN_UNITS)

        assert result == {"suffix": "", "divisor": 1, "decimals": 0}

    def test_columns_absent_from_the_frame_are_skipped(self) -> None:
        result = detect_number_format(self.FRAME, ["total_sales", "missing"], None)

        assert result["suffix"] == "M"

    def test_no_matching_columns_returns_none(self) -> None:
        assert detect_number_format(self.FRAME, ["nope"], None) is None

    def test_empty_column_list_returns_none(self) -> None:
        assert detect_number_format(self.FRAME, [], None) is None

    def test_non_numeric_values_are_coerced_and_ignored(self) -> None:
        """A text column reaching the Y axis must not raise."""
        frame = pd.DataFrame({"c": ["not", "numbers"]})

        assert detect_number_format(frame, ["c"], None) is None

    def test_none_column_names_are_tolerated(self) -> None:
        """y_axis_secondary is None when there is no second axis."""
        result = detect_number_format(self.FRAME, [None, "total_sales"], None)

        assert result["suffix"] == "M"

    def test_decimal_places_come_from_config(self) -> None:
        result = detect_number_format(
            self.FRAME, ["total_sales"], {**INDIAN_UNITS, "decimal_places": 0}
        )

        assert result["decimals"] == 0
