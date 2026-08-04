"""End-to-end tests for analyze_why with the LLM and DB both mocked.

This is the orchestration that answers "why was X high?": plan two queries, run
them, compute the arithmetic, narrate the result. The individual pieces are
covered elsewhere; what these tests pin is the WIRING — that the planner's SQL
actually gets executed, that the computed evidence (not raw rows) reaches the
synthesis prompt, and that a partial failure degrades instead of aborting.

The LLM is mocked at ``create_chat_llm`` inside analysis_agent, which returns a
stub whose ``.invoke()`` yields queued responses in order: plan first, synthesis
second.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.agent.analysis_agent import analyze_why

SCHEMA_SUMMARY = "public.sample(month, total_sales, brand_name)"

PLAN = {
    "focal_period": "2016-11",
    "metric_description": "total sales",
    "inherited_filters": ["brand_name = 'Brand A'"],
    "baseline_sql": (
        "SELECT to_char(date,'YYYY-MM') AS month, SUM(total_sales) AS total_sales "
        "FROM sample WHERE brand_name = 'Brand A' GROUP BY 1"
    ),
    "breakdown_sql": (
        "SELECT category, SUM(total_sales) AS total_sales FROM sample "
        "WHERE brand_name = 'Brand A' GROUP BY 1"
    ),
}

BASELINE_ROWS = pd.DataFrame(
    {
        "month": ["2016-09", "2016-10", "2016-11"],
        "total_sales": [60e7, 65e7, 96e7],
    }
)
BREAKDOWN_ROWS = pd.DataFrame(
    {"category": ["Category D", "Category A"], "total_sales": [60e7, 36e7]}
)


@pytest.fixture
def analysis_env(monkeypatch, patch_executor, stub_llm):
    """Wire mocked LLM + DB and expose the prompts and SQL that were used."""

    def _setup(
        plan: dict | str = PLAN,
        narrative: str = "November was the peak month.",
        *db_results: object,
    ):
        raw_plan = plan if isinstance(plan, str) else json.dumps(plan)
        llm = stub_llm(raw_plan, narrative)
        monkeypatch.setattr(
            "app.agent.analysis_agent.create_chat_llm", lambda **_kwargs: llm
        )
        recorder = patch_executor(*db_results)
        return llm.prompts, recorder

    return _setup


class TestAnalyzeWhyHappyPath:
    def test_returns_the_narrative_and_evidence(self, analysis_env) -> None:
        analysis_env(PLAN, "November was the peak.", BASELINE_ROWS, BREAKDOWN_ROWS)

        result = analyze_why("why was Nov high?", SCHEMA_SUMMARY)

        assert result["success"] is True
        assert result["answer"] == "November was the peak."
        assert result["evidence"]["baseline"]["focal_period"] == "2016-11"

    def test_the_planner_sql_is_what_actually_executes(self, analysis_env) -> None:
        """The plan is only useful if its queries are the ones run."""
        _prompts, recorder = analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        analyze_why("why?", SCHEMA_SUMMARY)

        assert recorder.queries == [PLAN["baseline_sql"], PLAN["breakdown_sql"]]

    def test_computed_evidence_not_raw_rows_reaches_the_synthesis_prompt(
        self, analysis_env
    ) -> None:
        """The whole safety argument: the LLM narrates figures already computed,
        so it cannot invent or recompute one."""
        prompts, _recorder = analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        analyze_why("why?", SCHEMA_SUMMARY)

        synthesis_prompt = prompts[1]
        assert "baseline_mean" in synthesis_prompt
        assert "std_devs_from_mean" in synthesis_prompt
        assert "contributions" in synthesis_prompt

    def test_conversation_history_reaches_the_planner_prompt(
        self, analysis_env
    ) -> None:
        """Regression: without history the planner dropped the Brand A filter."""
        prompts, _recorder = analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        analyze_why(
            "why was Nov high?",
            SCHEMA_SUMMARY,
            conversation_history="User: trend of sales for Brand A in 2016",
        )

        assert "Brand A" in prompts[0]

    def test_inherited_filters_are_reported_back(self, analysis_env) -> None:
        analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["inherited_filters"] == ["brand_name = 'Brand A'"]
        assert result["metric_description"] == "total sales"

    def test_both_statements_are_returned_for_filter_chips(self, analysis_env) -> None:
        analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["sql"] == [PLAN["baseline_sql"], PLAN["breakdown_sql"]]

    def test_both_llm_calls_are_returned_for_cost_tracking(self, analysis_env) -> None:
        """Untracked calls under-report cost; there must be exactly two."""
        analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert len(result["responses"]) == 2

    def test_period_series_is_included_so_the_ui_can_chart_it(
        self, analysis_env
    ) -> None:
        analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert len(result["evidence"]["period_series"]) == 3

    def test_contributions_are_computed_against_the_focal_total(
        self, analysis_env
    ) -> None:
        analysis_env(PLAN, "ok", BASELINE_ROWS, BREAKDOWN_ROWS)

        contributions = analyze_why("why?", SCHEMA_SUMMARY)["evidence"]["contributions"]

        # 60e7 of a 96e7 focal total.
        assert contributions[0]["member"] == "Category D"
        assert contributions[0]["share_pct"] == pytest.approx(62.5)


class TestAnalyzeWhyFailureModes:
    def test_unparseable_plan_fails_cleanly(self, analysis_env) -> None:
        """A non-JSON plan is a clear failure, not a crash."""
        analysis_env("this is not json", "unused")

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["success"] is False
        assert "plan" in result["error"].lower()

    def test_plan_wrapped_in_code_fences_is_still_accepted(self, analysis_env) -> None:
        """LLMs add fences despite instructions; that must not fail the run."""
        analysis_env(
            f"```json\n{json.dumps(PLAN)}\n```",
            "ok",
            BASELINE_ROWS,
            BREAKDOWN_ROWS,
        )

        assert analyze_why("why?", SCHEMA_SUMMARY)["success"] is True

    def test_baseline_query_failure_still_yields_a_breakdown(
        self, analysis_env
    ) -> None:
        """Partial evidence degrades the answer rather than aborting it."""
        analysis_env(PLAN, "ok", "relation does not exist", BREAKDOWN_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["success"] is True
        assert result["evidence"]["baseline"] == {}
        assert result["evidence"]["contributions"]

    def test_both_queries_failing_is_reported_as_a_failure(self, analysis_env) -> None:
        analysis_env(PLAN, "unused", "error one", "error two")

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["success"] is False
        assert "execute" in result["error"].lower()

    def test_empty_result_sets_are_reported_as_no_figures(self, analysis_env) -> None:
        analysis_env(PLAN, "unused", pd.DataFrame(), pd.DataFrame())

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["success"] is False
        assert "comparable figures" in result["error"]

    def test_an_llm_exception_is_caught_and_returned(self, monkeypatch) -> None:
        """A provider outage must not propagate out of the node."""

        def boom(**_kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr("app.agent.analysis_agent.create_chat_llm", boom)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert result["success"] is False
        assert "provider unavailable" in result["error"]

    def test_missing_breakdown_sql_skips_that_query(self, analysis_env) -> None:
        plan = {**PLAN, "breakdown_sql": ""}
        _prompts, recorder = analysis_env(plan, "ok", BASELINE_ROWS)

        result = analyze_why("why?", SCHEMA_SUMMARY)

        assert recorder.queries == [PLAN["baseline_sql"]]
        assert result["evidence"]["contributions"] == []
