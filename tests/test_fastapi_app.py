import sys
import types

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schema_config_endpoint_returns_tables() -> None:
    response = client.get("/schema/config")

    assert response.status_code == 200
    payload = response.json()
    assert "tables" in payload


def test_chat_endpoint_supports_cors_preflight() -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_chat_endpoint_accepts_message_payload(monkeypatch) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            return {"echo": self.question}

    fake_module = types.SimpleNamespace(Orchestrator=DummyOrchestrator)
    monkeypatch.setitem(sys.modules, "orchestrator.orch_exec", fake_module)

    response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 200
    assert response.json()["question"] == "show me sales"


def test_chat_endpoint_handles_empty_payload_gracefully() -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 200
    assert "Please send a question" in response.json()["message"]
