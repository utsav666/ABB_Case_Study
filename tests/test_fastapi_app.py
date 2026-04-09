import json
import logging
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
    monkeypatch.setitem(sys.modules, "app.orchestrator.orch_exec", fake_module)

    response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 200
    assert response.json()["question"] == "show me sales"


def test_chat_endpoint_logs_failures(monkeypatch, caplog) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            raise ValueError("demo failure")

    fake_module = types.SimpleNamespace(Orchestrator=DummyOrchestrator)
    monkeypatch.setitem(sys.modules, "app.orchestrator.orch_exec", fake_module)

    with caplog.at_level(logging.ERROR):
        response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 500
    assert response.json()["detail"] == "demo failure"
    assert "Chat request failed" in caplog.text


def test_chat_endpoint_logs_unexpected_failures(monkeypatch, caplog) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            raise RuntimeError("unexpected failure")

    fake_module = types.SimpleNamespace(Orchestrator=DummyOrchestrator)
    monkeypatch.setitem(sys.modules, "app.orchestrator.orch_exec", fake_module)

    error_client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        response = error_client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 500
    assert "Unhandled application error" in caplog.text
    assert "unexpected failure" in caplog.text


def test_chat_endpoint_handles_empty_payload_gracefully() -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 200
    assert "Please send a question" in response.json()["message"]


def test_login_endpoint_returns_oauth_url(monkeypatch) -> None:
    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv(
        "OAUTH_AUTHORIZE_URL",
        "https://accounts.example.com/oauth2/v2/auth",
    )
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
    monkeypatch.setenv("OAUTH_SCOPE", "openid profile email")

    response = client.get("/login")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "oauth"
    assert payload["login_url"].startswith(
        "https://accounts.example.com/oauth2/v2/auth?"
    )
    assert "client_id=demo-client-id" in payload["login_url"]
    assert "response_type=code" in payload["login_url"]


def test_auth_callback_endpoint_returns_frontend_payload() -> None:
    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["code"] == "sample-code"
    assert payload["state"] == "login"


def test_auth_callback_endpoint_accepts_post_payload() -> None:
    response = client.post(
        "/auth/callback",
        data={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["code"] == "sample-code"
    assert payload["state"] == "login"


def test_auth_callback_endpoint_returns_user_details(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(
        url: str,
        data: dict[str, str],
        timeout: int,
    ) -> DummyResponse:
        assert "token" in url
        assert data["code"] == "sample-code"
        return DummyResponse(
            {
                "access_token": "demo-access-token",
                "id_token": "demo-id-token",
                "token_type": "Bearer",
            }
        )

    def fake_get(
        url: str,
        headers: dict[str, str],
        timeout: int,
    ) -> DummyResponse:
        assert "userinfo" in url
        assert headers["Authorization"] == "Bearer demo-access-token"
        return DummyResponse(
            {
                "email": "user@example.com",
                "name": "Demo User",
                "picture": "https://example.com/avatar.png",
            }
        )

    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "demo-client-secret")
    monkeypatch.setenv(
        "OAUTH_TOKEN_URL",
        "https://oauth.example.com/token",
    )
    monkeypatch.setenv(
        "OAUTH_USERINFO_URL",
        "https://oauth.example.com/userinfo",
    )
    monkeypatch.setattr("app.main.requests.post", fake_post)
    monkeypatch.setattr("app.main.requests.get", fake_get)

    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["user"]["email"] == "user@example.com"
    assert payload["user"]["name"] == "Demo User"
    assert payload["token"]["access_token"] == "demo-access-token"


def test_lambda_handler_serves_health_endpoint() -> None:
    from app.lambda_handler import handler

    event = {
        "version": "2.0",
        "routeKey": "GET /health",
        "rawPath": "/health",
        "rawQueryString": "",
        "headers": {
            "host": "lambda.example.com",
            "x-forwarded-proto": "https",
        },
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/health",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            }
        },
        "isBase64Encoded": False,
    }

    response = handler(event, {})

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}
