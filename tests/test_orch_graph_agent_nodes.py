"""Tests for graph nodes that delegate to an agent, with the agent mocked.

These nodes are thin adapters: they gather state, call an agent, and shape the
result into a GraphState. The adapter logic is where bugs have actually landed —
a node that forgets to pass the schema graph, or drops conversation history,
produces a confident answer about the wrong data.

Agents are mocked at their import site. Nodes importing inside the function body
(`explanation_node`) are patched at the source module; nodes importing at module
level are patched on `orch_graph` itself, because `from x import y` binds a
separate name.
"""

from __future__ import annotations

from app.orchestrator.orch_graph import (
    explanation_node,
    general_answer_node,
    schema_info_node,
)

SCHEMA_GRAPH = {
    "nodes": [
        {
            "table": "sample",
            "schema": "public",
            "qualified_name": "public.sample",
            "columns": [{"raw_name": "brand_name", "logical_name": "Brand"}],
        }
    ],
    "edges": [],
}


class TestSchemaInfoNode:
    def test_successful_answer_becomes_a_text_response(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.explain_schema",
            lambda *a, **k: {"success": True, "answer": "One table: sample."},
        )

        result = schema_info_node(
            {"question": "what tables?", "schema_graph": SCHEMA_GRAPH}
        )

        assert result["final_response"] == {
            "type": "text",
            "message": "One table: sample.",
        }

    def test_the_connection_graph_is_passed_to_the_agent(self, monkeypatch) -> None:
        """Regression: without this the agent answered from a movie fixture."""
        captured = {}

        def fake_explain(question, **kwargs):
            captured.update(kwargs)
            return {"success": True, "answer": "ok"}

        monkeypatch.setattr("app.orchestrator.orch_graph.explain_schema", fake_explain)

        schema_info_node({"question": "q", "schema_graph": SCHEMA_GRAPH})

        assert captured["schema_graph"] == SCHEMA_GRAPH

    def test_router_reasoning_is_forwarded_as_context(self, monkeypatch) -> None:
        captured = {}

        def fake_explain(question, **kwargs):
            captured.update(kwargs)
            return {"success": True, "answer": "ok"}

        monkeypatch.setattr("app.orchestrator.orch_graph.explain_schema", fake_explain)

        schema_info_node(
            {
                "question": "q",
                "schema_graph": SCHEMA_GRAPH,
                "route_reasoning": "asks about structure",
            }
        )

        assert captured["context"] == "asks about structure"

    def test_provider_and_model_overrides_are_forwarded(self, monkeypatch) -> None:
        """A per-workspace model choice must reach the agent."""
        captured = {}

        def fake_explain(question, **kwargs):
            captured.update(kwargs)
            return {"success": True, "answer": "ok"}

        monkeypatch.setattr("app.orchestrator.orch_graph.explain_schema", fake_explain)

        schema_info_node(
            {
                "question": "q",
                "schema_graph": SCHEMA_GRAPH,
                "llm_provider": "claude_merck",
                "llm_model": "claude-opus-4-6-v1",
            }
        )

        assert captured["provider"] == "claude_merck"
        assert captured["model"] == "claude-opus-4-6-v1"

    def test_failure_becomes_an_error_response(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.explain_schema",
            lambda *a, **k: {"success": False, "error": "no schema graph"},
        )

        result = schema_info_node({"question": "q", "schema_graph": None})

        assert result["final_response"]["type"] == "error"
        assert "no schema graph" in result["final_response"]["message"]


class TestExplanationNode:
    def test_requires_a_connection_graph(self) -> None:
        """No graph means no data to analyse; say so rather than guessing."""
        result = explanation_node(
            {"question": "why was Nov high?", "schema_graph": None}
        )

        assert result["final_response"]["type"] == "error"
        assert "connect" in result["final_response"]["message"].lower()

    def test_successful_analysis_returns_data_with_the_narrative(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.agent.analysis_agent.analyze_why",
            lambda *a, **k: {
                "success": True,
                "answer": "November was 38% above the mean.",
                "evidence": {"period_series": [{"m": "2016-11", "v": 96e7}]},
                "sql": ["SELECT 1", "SELECT 2"],
                "metric_description": "total sales",
                "inherited_filters": ["Brand = Brand A"],
                "responses": [],
            },
        )

        result = explanation_node(
            {"question": "why was Nov high?", "schema_graph": SCHEMA_GRAPH}
        )
        response = result["final_response"]

        assert response["type"] == "data"
        assert response["message"] == "November was 38% above the mean."
        assert response["data"] == [{"m": "2016-11", "v": 96e7}]

    def test_both_analysis_queries_are_joined_into_sql(self, monkeypatch) -> None:
        """Both statements are surfaced so filter chips can describe the scope."""
        monkeypatch.setattr(
            "app.agent.analysis_agent.analyze_why",
            lambda *a, **k: {
                "success": True,
                "answer": "ok",
                "evidence": {},
                "sql": ["SELECT baseline", "SELECT breakdown"],
                "responses": [],
            },
        )

        result = explanation_node({"question": "why?", "schema_graph": SCHEMA_GRAPH})

        assert "SELECT baseline" in result["sql"]
        assert "SELECT breakdown" in result["sql"]

    def test_inherited_filters_are_reported_as_analysis_scope(
        self, monkeypatch
    ) -> None:
        """A follow-up that silently dropped the user's filter looks identical to
        one that kept it, unless the scope is stated."""
        monkeypatch.setattr(
            "app.agent.analysis_agent.analyze_why",
            lambda *a, **k: {
                "success": True,
                "answer": "ok",
                "evidence": {},
                "sql": [],
                "metric_description": "total sales for Brand A",
                "inherited_filters": ["brand_name = 'Brand A'", "year = 2016"],
                "responses": [],
            },
        )

        scope = explanation_node({"question": "why?", "schema_graph": SCHEMA_GRAPH})[
            "final_response"
        ]["analysis_scope"]

        assert scope["inherited_filters"] == ["brand_name = 'Brand A'", "year = 2016"]
        assert scope["metric"] == "total sales for Brand A"

    def test_conversation_history_is_passed_to_the_planner(self, monkeypatch) -> None:
        """Regression: "why was Nov high?" is a follow-up, and without history the
        planner dropped the Brand A filter and explained a different number."""
        captured = {}

        def fake_analyze(question, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "answer": "ok",
                "evidence": {},
                "sql": [],
                "responses": [],
            }

        monkeypatch.setattr("app.agent.analysis_agent.analyze_why", fake_analyze)

        explanation_node(
            {
                "question": "why was Nov high?",
                "schema_graph": SCHEMA_GRAPH,
                "conversation_history": "User: trend of sales for Brand A in 2016",
            }
        )

        assert "Brand A" in captured["conversation_history"]

    def test_a_rendered_schema_summary_is_passed_not_the_raw_graph(
        self, monkeypatch
    ) -> None:
        captured = {}

        def fake_analyze(question, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "answer": "ok",
                "evidence": {},
                "sql": [],
                "responses": [],
            }

        monkeypatch.setattr("app.agent.analysis_agent.analyze_why", fake_analyze)

        explanation_node({"question": "why?", "schema_graph": SCHEMA_GRAPH})

        assert isinstance(captured["schema_summary"], str)
        assert "public.sample" in captured["schema_summary"]

    def test_failure_becomes_an_error_response(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.agent.analysis_agent.analyze_why",
            lambda *a, **k: {
                "success": False,
                "error": "Could not plan the analysis.",
                "responses": [],
            },
        )

        result = explanation_node({"question": "why?", "schema_graph": SCHEMA_GRAPH})

        assert result["final_response"]["type"] == "error"
        assert "Could not plan" in result["final_response"]["message"]

    def test_missing_period_series_yields_an_empty_data_list(self, monkeypatch) -> None:
        """A narrative with no chartable series is still a valid answer."""
        monkeypatch.setattr(
            "app.agent.analysis_agent.analyze_why",
            lambda *a, **k: {
                "success": True,
                "answer": "ok",
                "evidence": {},
                "sql": [],
                "responses": [],
            },
        )

        result = explanation_node({"question": "why?", "schema_graph": SCHEMA_GRAPH})

        assert result["final_response"]["data"] == []


class TestGeneralAnswerNode:
    def test_successful_answer_becomes_a_text_response(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.general_answer",
            lambda *a, **k: {"success": True, "answer": "I can query databases."},
        )

        result = general_answer_node({"question": "what can you do?"})

        assert result["final_response"] == {
            "type": "text",
            "message": "I can query databases.",
        }

    def test_failure_becomes_an_error_response(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.general_answer",
            lambda *a, **k: {"success": False, "error": "LLM unavailable"},
        )

        result = general_answer_node({"question": "hi"})

        assert result["final_response"]["type"] == "error"
        assert "LLM unavailable" in result["final_response"]["message"]

    def test_router_reasoning_is_forwarded(self, monkeypatch) -> None:
        captured = {}

        def fake_general(question, **kwargs):
            captured.update(kwargs)
            return {"success": True, "answer": "ok"}

        monkeypatch.setattr("app.orchestrator.orch_graph.general_answer", fake_general)

        general_answer_node({"question": "hi", "route_reasoning": "not DB related"})

        assert captured["router_reasoning"] == "not DB related"
