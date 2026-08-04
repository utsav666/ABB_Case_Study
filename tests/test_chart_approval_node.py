"""Tests for chart_approval_node with the chart LLM and schema graph mocked.

This node turns a result set plus an LLM chart suggestion into the chart_config
the frontend renders. It is ~300 lines and was entirely uncovered.

What it gets wrong silently: axis labels falling back to raw column names,
number formatting not reaching the config (so a chart shows 983041954 while the
table beside it shows 98.30Cr), or the pivoted DataFrame not being returned so
the chart plots the unpivoted original.

`decide_chart` is patched on `orch_graph` — it is imported at module level there,
so patching the source module would leave the real function bound.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.orchestrator.orch_graph import chart_approval_node

MONTHLY_ROWS = [
    {"month": "2016-10", "total_sales": 704810000.0},
    {"month": "2016-11", "total_sales": 983041954.0},
]

SCHEMA_GRAPH = {
    "nodes": [
        {
            "table": "sample",
            "schema": "public",
            "qualified_name": "public.sample",
            "columns": [
                {"raw_name": "month", "logical_name": "Month"},
                {"raw_name": "total_sales", "logical_name": "Total Actual Sales"},
            ],
        }
    ],
    "edges": [],
}

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


class Decision:
    """Stands in for the agent's ChartDecision model."""

    def __init__(
        self,
        chart_type: str = "line",
        x_axis: str = "month",
        y_axis: object = "total_sales",
        y_axis_secondary: str | None = None,
    ) -> None:
        self.chart_type = chart_type
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.y_axis_secondary = y_axis_secondary
        self.reasoning = "test decision"


@pytest.fixture
def chart_env(monkeypatch):
    """Patch decide_chart and the display_config lookup; return the state builder."""

    def _setup(
        decision: Decision | None = None,
        *,
        success: bool = True,
        error: str = "chart agent failed",
        display_config: dict | None = None,
    ):
        result = (
            # The node reads result["chart"], not "decision".
            {"success": True, "chart": decision or Decision(), "response": None}
            if success
            else {"success": False, "error": error}
        )
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.decide_chart", lambda *a, **k: result
        )
        monkeypatch.setattr(
            "app.settings_db.repositories.DBConnectionRepository",
            lambda *a, **k: type(
                "Repo",
                (),
                {
                    "get_connection_by_id": lambda _s, _c: {
                        "display_config": display_config
                    }
                },
            )(),
        )

    return _setup


def build_state(**overrides) -> dict:
    state = {
        "question": "trend of sales in 2016",
        "data": MONTHLY_ROWS,
        "schema_graph": SCHEMA_GRAPH,
        "connection_id": "conn-1",
        "sql": "SELECT month, SUM(total_sales) FROM sample GROUP BY 1",
    }
    state.update(overrides)
    return state


class TestChartApprovalNodeBasics:
    def test_missing_data_is_reported_as_an_error(self, chart_env) -> None:
        chart_env()

        result = chart_approval_node(build_state(data=None))

        assert result["final_response"]["type"] == "error"
        assert "No data to chart" in result["final_response"]["message"]

    def test_successful_run_returns_a_chart_response(self, chart_env) -> None:
        chart_env()

        response = chart_approval_node(build_state())["final_response"]

        assert response["type"] == "chart"
        assert response["chart_config"]["chart_type"] == "line"

    def test_axes_come_from_the_llm_decision(self, chart_env) -> None:
        chart_env(Decision(chart_type="bar", x_axis="month", y_axis="total_sales"))

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert config["x_axis"] == "month"
        assert config["y_axis"] == "total_sales"

    def test_sql_is_carried_into_the_response(self, chart_env) -> None:
        """Filter chips are derived from it, so it must survive the chart path."""
        chart_env()

        response = chart_approval_node(build_state())["final_response"]

        assert "FROM sample" in response["sql"]

    def test_original_records_are_returned_when_not_pivoted(self, chart_env) -> None:
        chart_env()

        response = chart_approval_node(build_state())["final_response"]

        assert response["data"] == MONTHLY_ROWS

    def test_hitl_is_not_triggered(self, chart_env) -> None:
        """Chart approval is automatic now; a stray interrupt would hang the graph."""
        chart_env()

        assert chart_approval_node(build_state())["hitl_triggered"] is False

    def test_agent_failure_degrades_to_a_table_not_an_error(self, chart_env) -> None:
        """Good design, pinned: if the chart agent cannot pick a type, the data is
        still returned as a table. Losing the answer entirely would be worse than
        losing the visualisation."""
        chart_env(success=False, error="could not choose a chart type")

        response = chart_approval_node(build_state())["final_response"]

        assert response["type"] == "data"
        assert response["data"] == MONTHLY_ROWS
        assert "FROM sample" in response["sql"]


class TestAxisLabelling:
    def test_logical_names_from_the_graph_become_axis_labels(self, chart_env) -> None:
        """Otherwise the axis reads `total_sales` instead of Total Actual Sales."""
        chart_env()

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert config["labels"]["y_axis_label"] == "Total Actual Sales"
        assert config["labels"]["x_axis_label"] == "Month"

    def test_aggregation_prefixes_are_resolved_to_a_readable_label(
        self, chart_env
    ) -> None:
        """SQL aliases the metric `total_total_sales`; the label must not."""
        chart_env(Decision(y_axis="total_total_sales"))
        rows = [{"month": "2016-11", "total_total_sales": 983041954.0}]

        config = chart_approval_node(build_state(data=rows))["final_response"][
            "chart_config"
        ]

        assert "Total Actual Sales" in config["labels"]["y_axis_label"]

    def test_unknown_columns_get_no_label_override(self, chart_env) -> None:
        """No graph entry means no better name than the raw column."""
        chart_env(Decision(x_axis="mystery", y_axis="total_sales"))
        rows = [{"mystery": "a", "total_sales": 1.0}]

        config = chart_approval_node(build_state(data=rows))["final_response"][
            "chart_config"
        ]

        assert config.get("labels", {}).get("x_axis_label") != "mystery"

    def test_a_missing_schema_graph_does_not_break_the_chart(self, chart_env) -> None:
        """Labels are an enhancement; their absence must not fail the render."""
        chart_env()

        response = chart_approval_node(
            build_state(schema_graph=None, connection_id=None)
        )["final_response"]

        assert response["type"] == "chart"

    def test_a_secondary_axis_label_is_included(self, chart_env) -> None:
        chart_env(
            Decision(
                chart_type="dual_axis_column",
                y_axis="total_sales",
                y_axis_secondary="month",
            )
        )

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert config["y_axis_secondary"] == "month"


class TestNumberFormatting:
    def test_the_connection_scale_reaches_the_chart_config(self, chart_env) -> None:
        """983,041,954 / 1e7 = 98.30Cr — same scale the table uses."""
        chart_env(display_config=INDIAN_UNITS)

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert config["y_axis_format"]["suffix"] == "Cr"
        assert config["y_axis_format"]["divisor"] == 10000000

    def test_defaults_to_b_m_k_without_a_display_config(self, chart_env) -> None:
        chart_env(display_config=None)

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert config["y_axis_format"]["suffix"] == "M"

    def test_formatting_can_be_switched_off(self, chart_env) -> None:
        chart_env(display_config={"number_format_enabled": False})

        config = chart_approval_node(build_state())["final_response"]["chart_config"]

        assert "y_axis_format" not in config

    def test_a_date_axis_gets_a_granularity_format(self, chart_env) -> None:
        """The x-axis label format depends on row spacing."""
        chart_env(Decision(x_axis="order_date", y_axis="total_sales"))
        rows = [
            {"order_date": "2016-01-01", "total_sales": 1.0},
            {"order_date": "2016-02-01", "total_sales": 2.0},
            {"order_date": "2016-03-01", "total_sales": 3.0},
        ]

        config = chart_approval_node(build_state(data=rows, question="monthly trend"))[
            "final_response"
        ]["chart_config"]

        assert config["x_axis_format"] == "month"


class TestMultiSeriesAndPivoting:
    def test_a_list_y_axis_is_preserved_for_multi_series(self, chart_env) -> None:
        chart_env(
            Decision(chart_type="grouped_column", y_axis=["sales_2015", "sales_2016"])
        )
        rows = [{"month": "01", "sales_2015": 10.0, "sales_2016": 30.0}]

        config = chart_approval_node(build_state(data=rows))["final_response"][
            "chart_config"
        ]

        assert config["y_axis"] == ["sales_2015", "sales_2016"]

    def test_sql_pivoted_comparison_data_drives_the_axes(self, chart_env) -> None:
        """When SQL already pivoted, every numeric column becomes a series and the
        LLM's single-column suggestion is overridden."""
        chart_env(Decision(y_axis="sales_2015"))
        rows = [
            {"month": "01", "sales_2015": 10.0, "sales_2016": 30.0},
            {"month": "02", "sales_2015": 20.0, "sales_2016": 40.0},
        ]

        config = chart_approval_node(
            build_state(data=rows, question="compare 2015 vs 2016")
        )["final_response"]["chart_config"]

        assert set(config["y_axis"]) == {"sales_2015", "sales_2016"}

    def test_pivoted_data_returns_the_reshaped_records(self, chart_env) -> None:
        """The chart must plot what the config describes, not the original rows."""
        chart_env(Decision(y_axis="sales_2015"))
        rows = [{"month": "01", "sales_2015": 10.0, "sales_2016": 30.0}]

        response = chart_approval_node(
            build_state(data=rows, question="compare 2015 vs 2016")
        )["final_response"]

        assert response["data"] == rows

    def test_a_dataframe_in_state_is_accepted(self, chart_env) -> None:
        """Some paths keep a DataFrame rather than records."""
        chart_env()

        response = chart_approval_node(build_state(data=pd.DataFrame(MONTHLY_ROWS)))[
            "final_response"
        ]

        assert response["type"] == "chart"

    def test_an_empty_result_degrades_to_a_data_response(self, chart_env) -> None:
        """Zero rows cannot be charted, so the node returns the (empty) data
        rather than an unplottable chart config."""
        chart_env()

        response = chart_approval_node(build_state(data=[]))["final_response"]

        assert response["type"] == "data"
