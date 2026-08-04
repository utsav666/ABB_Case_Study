"""Endpoint tests via TestClient with the repository layer mocked.

These cover request/response contracts the frontend depends on: status codes,
payload keys, and validation behaviour. A renamed key or a changed status code is
a breaking change the frontend cannot detect until runtime.

Repositories are patched at their import site in the router module, so no DB is
touched — the autouse network guard in conftest would fail the test otherwise.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthAndRoot:
    def test_health_returns_ok(self) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_lists_available_endpoints(self) -> None:
        """The root doubles as lightweight API discovery."""
        response = client.get("/")

        assert response.status_code == 200
        assert "/chat" in response.json()["endpoints"]


class TestDbConnectionEngines:
    def test_lists_the_supported_engines(self) -> None:
        response = client.get("/settings/db-connections/engines")

        assert response.status_code == 200
        assert "postgres" in response.json()["engines"]

    def test_engine_list_has_no_duplicates(self) -> None:
        """Not asserted as sorted: the endpoint returns them in a curated order
        (postgres first, as the primary engine), which the UI relies on."""
        engines = client.get("/settings/db-connections/engines").json()["engines"]

        assert len(engines) == len(set(engines))
        assert engines[0] == "postgres"


class TestChatValidation:
    def test_empty_question_returns_a_helpful_message(self) -> None:
        """A 200 with guidance, not a 422: the frontend renders this as a reply."""
        response = client.post("/chat", json={})

        assert response.status_code == 200
        assert "question" in response.json()["message"].lower()

    def test_whitespace_only_question_is_treated_as_empty(self) -> None:
        response = client.post("/chat", json={"message": "   "})

        assert response.status_code == 200
        assert response.json()["answer_type"] == "text"

    def test_cors_preflight_is_allowed_for_the_dev_origin(self) -> None:
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:3000"
        )


class TestSchemaGraphEndpoint:
    def test_returns_the_stored_graph(self, monkeypatch) -> None:
        graph = {"nodes": [], "edges": []}
        # Patched on the router module: it calls SchemaGraphRepository directly,
        # not through load_schema_graph.
        monkeypatch.setattr(
            "app.settings_db.router.SchemaGraphRepository",
            lambda *a, **k: type(
                "Repo", (), {"get_graph_by_connection": lambda _s, _c: {"graph": graph}}
            )(),
        )

        response = client.get(f"/settings/db-connections/{uuid4()}/schema-graph")

        assert response.status_code == 200

    def test_missing_graph_is_not_a_server_error(self, monkeypatch) -> None:
        """An unprofiled connection is a client-side condition, not a 500."""
        monkeypatch.setattr(
            "app.settings_db.router.SchemaGraphRepository",
            lambda *a, **k: type(
                "Repo", (), {"get_graph_by_connection": lambda _s, _c: None}
            )(),
        )

        response = client.get(f"/settings/db-connections/{uuid4()}/schema-graph")

        assert response.status_code < 500


class TestSchemaSummaryEndpoint:
    def test_connection_id_is_required(self) -> None:
        """It used to take no arguments and render a bundled movie fixture."""
        assert client.get("/schema/summary").status_code == 422

    def test_unprofiled_connection_returns_404(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.target_db.controllers.load_schema_graph", lambda _cid: None
        )

        response = client.get("/schema/summary", params={"connection_id": "x"})

        assert response.status_code == 404

    def test_profiled_connection_returns_a_summary(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.target_db.controllers.load_schema_graph",
            lambda _cid: {
                "nodes": [
                    {
                        "table": "orders",
                        "schema": "sales",
                        "qualified_name": "sales.orders",
                        "columns": [{"raw_name": "id", "logical_name": "Order ID"}],
                    }
                ],
                "edges": [],
            },
        )

        response = client.get("/schema/summary", params={"connection_id": "x"})

        assert response.status_code == 200
        assert "sales.orders" in response.json()["summary"]


class TestSchemaConfigRoutes:
    def test_get_schema_config_is_not_allowed(self) -> None:
        """The read path moved to GET /schema/tables."""
        assert client.get("/schema/config").status_code == 405

    def test_schema_tables_requires_a_connection_id(self) -> None:
        assert client.get("/schema/tables").status_code == 422


class TestChatResumeValidation:
    def test_missing_thread_id_is_rejected(self) -> None:
        assert client.post("/chat/resume", json={"decision": "x"}).status_code == 422

    def test_missing_decision_is_rejected(self) -> None:
        response = client.post("/chat/resume", json={"thread_id": "t-1"})

        assert response.status_code == 422

    def test_connection_id_is_optional_for_backward_compatibility(
        self, monkeypatch
    ) -> None:
        """Older clients do not send it; they must not start failing."""
        monkeypatch.setattr(
            "app.orchestrator.orch_graph.GraphOrchestrator.resume",
            staticmethod(lambda *a, **k: {"type": "text", "message": "resumed"}),
        )

        response = client.post(
            "/chat/resume", json={"thread_id": "t-1", "decision": "Category A"}
        )

        assert response.status_code == 200
        assert response.json()["message"] == "resumed"


class TestChatEndpointResponseShape:
    def test_successful_answer_includes_the_contract_keys(self, monkeypatch) -> None:
        """The frontend reads these by name; a rename breaks rendering silently."""

        class DummyOrchestrator:
            def __init__(self, _question: str, **_kwargs) -> None:
                pass

            def executor(self) -> dict:
                return {
                    "type": "data",
                    "data": [{"m": "2016-11", "v": 983041954.0}],
                    "sql": "SELECT 1 FROM sample WHERE brand_name = 'Brand A'",
                }

        monkeypatch.setattr(
            "app.orchestrator.orch_graph.GraphOrchestrator", DummyOrchestrator
        )

        payload = client.post("/chat", json={"message": "sales by month"}).json()

        for key in ("question", "thread_id", "answer_type", "message", "data", "sql"):
            assert key in payload, key

    def test_applied_filters_describe_the_executed_query(self, monkeypatch) -> None:
        class DummyOrchestrator:
            def __init__(self, _question: str, **_kwargs) -> None:
                pass

            def executor(self) -> dict:
                return {
                    "type": "data",
                    "data": [{"v": 1}],
                    "sql": "SELECT 1 FROM sample WHERE brand_name = 'Brand A'",
                }

        monkeypatch.setattr(
            "app.orchestrator.orch_graph.GraphOrchestrator", DummyOrchestrator
        )

        payload = client.post("/chat", json={"message": "q"}).json()

        assert any(chip["value"] == "Brand A" for chip in payload["applied_filters"])

    def test_thread_id_is_echoed_back_for_hitl_resume(self, monkeypatch) -> None:
        class DummyOrchestrator:
            def __init__(self, _question: str, **_kwargs) -> None:
                pass

            def executor(self) -> dict:
                return {"type": "text", "message": "ok"}

        monkeypatch.setattr(
            "app.orchestrator.orch_graph.GraphOrchestrator", DummyOrchestrator
        )
        # Sending a thread_id makes the handler load prior history, which is a
        # settings-DB read. Stubbed so this stays a contract test.
        monkeypatch.setattr(
            "app.orchestrator.conversation_manager.ConversationHistoryManager."
            "format_history_for_context",
            lambda _self, _thread_id: "",
        )

        payload = client.post(
            "/chat", json={"message": "q", "thread_id": "thread-abc"}
        ).json()

        assert payload["thread_id"] == "thread-abc"

    def test_a_generated_thread_id_is_returned_when_none_is_sent(
        self, monkeypatch
    ) -> None:
        """The client needs an id to resume a HITL interrupt against."""

        class DummyOrchestrator:
            def __init__(self, _question: str, **_kwargs) -> None:
                pass

            def executor(self) -> dict:
                return {"type": "text", "message": "ok"}

        monkeypatch.setattr(
            "app.orchestrator.orch_graph.GraphOrchestrator", DummyOrchestrator
        )

        payload = client.post("/chat", json={"message": "q"}).json()

        assert payload["thread_id"]


class TestErrorHandling:
    def test_unknown_route_returns_404(self) -> None:
        assert client.get("/does-not-exist").status_code == 404

    def test_malformed_json_body_is_rejected(self) -> None:
        response = client.post(
            "/chat",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code < 500

    @pytest.mark.parametrize("method", ["put", "delete", "patch"])
    def test_unsupported_methods_on_chat_are_rejected(self, method: str) -> None:
        response = getattr(client, method)("/chat")

        assert response.status_code == 405
