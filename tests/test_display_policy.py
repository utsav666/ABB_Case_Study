"""Tests for result checks and display selection.

These replace "the router said chart, so draw a chart": the display now
follows the user's words and the result's real shape, and results that look
wrong (empty, or the same labels repeated) are regenerated once.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from app.orchestrator.display_policy import (
    decide_display,
    find_result_issue,
    format_single_value,
)
from app.orchestrator.orch_graph import build_data_response_node, sql_assumptions

REPEATED_PERIODS = pd.DataFrame(
    {
        "period": ["q1-2025"] * 3 + ["q2-2025"] * 3,
        "field_value": [1, 2, 3, 4, 5, 6],
        "company_name": list("abcabc"),
    }
)
PER_QUARTER = pd.DataFrame(
    {"period": ["q1-2025", "q2-2025", "q3-2025"], "total_headcount": [10, 12, 15]}
)
ONE_VALUE = pd.DataFrame({"total_headcount": [2309]})
DETAIL = pd.DataFrame({"c1": list(range(100)), "name": [f"n{i}" for i in range(100)]})


class TestDecideDisplay:
    @pytest.mark.parametrize(
        ("intent", "data", "expected"),
        [
            ("auto", REPEATED_PERIODS, "table"),
            ("chart", REPEATED_PERIODS, "table"),
            ("auto", PER_QUARTER, "chart"),
            ("value", ONE_VALUE, "value"),
            ("auto", ONE_VALUE, "value"),
            ("table", PER_QUARTER, "table"),
            ("table", ONE_VALUE, "table"),
            ("auto", DETAIL, "table"),
            ("chart", DETAIL, "chart"),
            ("auto", pd.DataFrame(), "table"),
            ("auto", pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}), "table"),
            ("value", PER_QUARTER, "table"),
            ("nonsense", PER_QUARTER, "chart"),
        ],
    )
    def test_display(self, intent: str, data: pd.DataFrame, expected: str) -> None:
        assert decide_display(intent, data)[0] == expected

    def test_refused_chart_explains_why(self) -> None:
        display, reason = decide_display("chart", REPEATED_PERIODS.to_dict("records"))

        assert display == "table"
        assert "period" in reason

    def test_decimal_values_count_as_numbers(self) -> None:
        records = [
            {"quarter": "q1", "total": Decimal("10.5")},
            {"quarter": "q2", "total": Decimal(12)},
        ]

        assert decide_display("auto", records)[0] == "chart"


class TestFindResultIssue:
    def test_empty_result_is_an_issue(self) -> None:
        assert "0 rows" in find_result_issue([])

    def test_repeated_labels_with_different_numbers_are_an_issue(self) -> None:
        """The Q1 headcount shape: one company twice in a period (two projects)."""
        data = pd.DataFrame(
            {
                "period": ["q1-2025", "q1-2025", "q1-2025"],
                "company_name": ["ORKA", "ORKA", "TNGX"],
                "headcount": [87, 75, 230],
            }
        )

        issue = find_result_issue(data)

        assert "identical" in issue
        assert "'company_name': 'ORKA'" in issue

    @pytest.mark.parametrize(
        "data",
        [PER_QUARTER, ONE_VALUE, pd.DataFrame({"a": ["x", "x"], "b": ["p", "p"]})],
    )
    def test_clean_results_pass(self, data: pd.DataFrame) -> None:
        assert find_result_issue(data) is None


class TestSingleValue:
    def test_integer_with_context(self) -> None:
        data = [{"period": "q4-2027", "total_headcount": 2309.0}]

        assert format_single_value(data) == "**Total headcount:** 2,309 (q4-2027)"

    def test_non_single_values_are_not_formatted(self) -> None:
        assert format_single_value(PER_QUARTER) is None


class TestAssumptions:
    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            (
                "-- Assumptions: latest version; premium = 'Market'\nSELECT 1",
                "latest version; premium = 'Market'",
            ),
            ("-- Assumptions: none\nSELECT 1", ""),
            ("SELECT 1 -- Assumptions: hidden", ""),
            ("", ""),
        ],
    )
    def test_extraction(self, sql: str, expected: str) -> None:
        assert sql_assumptions(sql) == expected

    def test_value_answer_shows_the_number_and_the_assumptions(self) -> None:
        state = {
            "data": [{"total_headcount": 2309}],
            "sql": "-- Assumptions: latest project per company\nSELECT 1",
            "display": "value",
        }

        message = build_data_response_node(state)["final_response"]["message"]

        assert message == (
            "**Total headcount:** 2,309\n\n_Assumptions: latest project per company_"
        )

    def test_refused_chart_is_explained_to_the_user(self) -> None:
        state = {
            "data": [{"a": "x", "v": 1}],
            "sql": "SELECT 1",
            "display_intent": "chart",
            "display": "table",
            "display_reason": "there is only one row",
        }

        message = build_data_response_node(state)["final_response"]["message"]

        assert message == (
            "Showing a table instead of a chart because there is only one row."
        )
