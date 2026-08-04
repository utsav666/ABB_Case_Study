import json
import logging

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_dummy_graph_orchestrator(dummy_cls):
    """Wrap a simple test double in GraphOrchestrator's real signature.

    The handler constructs GraphOrchestrator with keyword arguments
    (thread_id, connection_id, conversation_history, user_message_id,
    llm_provider, llm_model). Accepting **kwargs keeps these tests focused on
    the endpoint's behaviour rather than the orchestrator's parameter list.
    """

    class DummyGraphOrchestrator:
        def __init__(self, question: str, **_kwargs) -> None:
            self._inner = dummy_cls(question)

        def executor(self):
            return self._inner.executor()

    return DummyGraphOrchestrator


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schema_config_endpoint_is_post_only() -> None:
    """GET /schema/config no longer exists; the route is POST-only.

    The previous version of this test did a GET and asserted 200, which has
    returned 405 since the read path moved to GET /schema/tables (that endpoint
    requires a live connection, so it is not covered here).
    """
    response = client.get("/schema/config")

    assert response.status_code == 405


def test_schema_summary_requires_connection_id() -> None:
    """GET /schema/summary rejects a request with no connection_id.

    The endpoint used to take no arguments and render a bundled fixture, so it
    described a movie database regardless of the caller's connection. Schema now
    comes only from settings.schema_graphs, making connection_id mandatory.
    """
    response = client.get("/schema/summary")

    assert response.status_code == 422


def test_schema_summary_returns_404_for_unprofiled_connection(monkeypatch) -> None:
    """An unknown connection gets a 404, not a fixture-backed answer."""
    monkeypatch.setattr(
        "app.target_db.controllers.load_schema_graph",
        lambda _connection_id: None,
    )

    response = client.get("/schema/summary", params={"connection_id": "missing-id"})

    assert response.status_code == 404
    assert "Profile the connection first" in response.json()["detail"]


def test_schema_summary_renders_connection_graph(monkeypatch) -> None:
    """A profiled connection's summary is rendered from its stored graph."""
    graph = {
        "nodes": [
            {
                "table": "orders",
                "schema": "sales",
                "qualified_name": "sales.orders",
                "columns": [{"raw_name": "id", "logical_name": "Order ID"}],
            }
        ],
        "edges": [],
    }
    monkeypatch.setattr(
        "app.target_db.controllers.load_schema_graph",
        lambda _connection_id: graph,
    )

    response = client.get("/schema/summary", params={"connection_id": "conn-1"})

    assert response.status_code == 200
    summary = response.json()["summary"]
    assert "sales.orders" in summary
    assert "Order ID" in summary


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

    # /chat imports GraphOrchestrator from app.orchestrator.orch_graph; the old
    # stub targeted app.orchestrator.orch_exec, which the handler no longer uses,
    # so the real graph ran and reached the settings DB.
    monkeypatch.setattr(
        "app.orchestrator.orch_graph.GraphOrchestrator",
        _make_dummy_graph_orchestrator(DummyOrchestrator),
    )

    response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 200
    assert response.json()["question"] == "show me sales"


def test_chat_endpoint_logs_failures(monkeypatch, caplog) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            raise ValueError("demo failure")

    # /chat imports GraphOrchestrator from app.orchestrator.orch_graph; the old
    # stub targeted app.orchestrator.orch_exec, which the handler no longer uses,
    # so the real graph ran and reached the settings DB.
    monkeypatch.setattr(
        "app.orchestrator.orch_graph.GraphOrchestrator",
        _make_dummy_graph_orchestrator(DummyOrchestrator),
    )

    with caplog.at_level(logging.ERROR):
        response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 500
    # The endpoint deliberately returns a generic message so internal details
    # are not leaked to clients; the real cause goes to the log, asserted below.
    assert response.json()["detail"] == "An internal error occurred. Please try again."
    assert "demo failure" in caplog.text
    assert "Chat request failed" in caplog.text


def test_chat_endpoint_logs_captured_output_on_failures(monkeypatch, caplog) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            print("step one completed")
            raise ValueError("demo failure")

    # /chat imports GraphOrchestrator from app.orchestrator.orch_graph; the old
    # stub targeted app.orchestrator.orch_exec, which the handler no longer uses,
    # so the real graph ran and reached the settings DB.
    monkeypatch.setattr(
        "app.orchestrator.orch_graph.GraphOrchestrator",
        _make_dummy_graph_orchestrator(DummyOrchestrator),
    )

    with caplog.at_level(logging.INFO):
        response = client.post("/chat", json={"message": "show me sales"})

    assert response.status_code == 500
    assert "step one completed" in caplog.text


def test_chat_endpoint_logs_unexpected_failures(monkeypatch, caplog) -> None:
    class DummyOrchestrator:
        def __init__(self, question: str) -> None:
            self.question = question

        def executor(self) -> dict[str, str]:
            raise RuntimeError("unexpected failure")

    # /chat imports GraphOrchestrator from app.orchestrator.orch_graph; the old
    # stub targeted app.orchestrator.orch_exec, which the handler no longer uses,
    # so the real graph ran and reached the settings DB.
    monkeypatch.setattr(
        "app.orchestrator.orch_graph.GraphOrchestrator",
        _make_dummy_graph_orchestrator(DummyOrchestrator),
    )

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
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "authorization_endpoint": "https://accounts.example.com/oauth2/v2/auth",
            }

    def fake_get(
        url: str,
        timeout: int,
    ) -> DummyResponse:
        assert "openid-configuration" in url
        return DummyResponse()

    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://accounts.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
    monkeypatch.setenv("OAUTH_SCOPE", "openid profile email")
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)

    response = client.get("/login")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "oauth"
    assert payload["login_url"].startswith(
        "https://accounts.example.com/oauth2/v2/auth?"
    )
    assert "client_id=demo-client-id" in payload["login_url"]
    assert "response_type=code" in payload["login_url"]


def test_auth_callback_endpoint_returns_frontend_payload(allow_network) -> None:
    # NOTE: reaches the real identity provider over HTTPS, so it opts out of the
    # network guard rather than silently depending on outbound access in CI.
    # Worth mocking the token/userinfo calls properly.
    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["code"] == "sample-code"
    assert payload["state"] == "login"


def test_auth_callback_endpoint_accepts_post_payload(allow_network) -> None:
    # NOTE: reaches the real identity provider over HTTPS, so it opts out of the
    # network guard rather than silently depending on outbound access in CI.
    # Worth mocking the token/userinfo calls properly.
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
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ) -> DummyResponse:
        if "openid-configuration" in url:
            return DummyResponse(
                {
                    "token_endpoint": "https://oauth.example.com/token",
                    "userinfo_endpoint": "https://oauth.example.com/userinfo",
                }
            )

        assert "userinfo" in url
        assert headers is not None
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
        "OAUTH_WELL_KNOWN_URL",
        "https://oauth.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setattr("app.auth.oauth_service.requests.post", fake_post)
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)
    monkeypatch.setattr(
        "app.auth.oauth_router.persist_oauth_user_details",
        lambda user_payload: {
            **user_payload,
            "id": "11111111-1111-1111-1111-111111111111",
            "full_name": "Demo User",
            "avatar_url": "https://example.com/avatar.png",
            "created_at": "2026-04-12T00:00:00+00:00",
            "updated_at": "2026-04-12T00:00:00+00:00",
            "last_login_at": "2026-04-12T00:00:00+00:00",
            "is_active": True,
        },
    )

    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["user"]["email"] == "user@example.com"
    assert payload["user"]["name"] == "Demo User"
    assert payload["user"]["id"] == "11111111-1111-1111-1111-111111111111"
    assert payload["user"]["avatar_url"] == "https://example.com/avatar.png"
    assert payload["token"]["access_token"] == "demo-access-token"


def test_auth_callback_endpoint_returns_existing_settings_user(monkeypatch) -> None:
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
        return DummyResponse({"access_token": "demo-access-token"})

    def fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ) -> DummyResponse:
        if "openid-configuration" in url:
            return DummyResponse(
                {
                    "token_endpoint": "https://oauth.example.com/token",
                    "userinfo_endpoint": "https://oauth.example.com/userinfo",
                }
            )

        return DummyResponse(
            {
                "email": "user@example.com",
                "name": "Provider Name",
                "picture": "https://example.com/provider-avatar.png",
            }
        )

    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "demo-client-secret")
    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://oauth.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setattr("app.auth.oauth_service.requests.post", fake_post)
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)
    monkeypatch.setattr(
        "app.auth.oauth_router.persist_oauth_user_details",
        lambda user_payload: {
            **user_payload,
            "id": "22222222-2222-2222-2222-222222222222",
            "name": "Existing User",
            "picture": "https://example.com/existing-avatar.png",
            "full_name": "Existing User",
            "avatar_url": "https://example.com/existing-avatar.png",
            "created_at": "2026-04-01T00:00:00+00:00",
            "updated_at": "2026-04-12T00:00:00+00:00",
            "last_login_at": "2026-04-12T00:00:00+00:00",
            "is_active": True,
        },
    )

    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["authenticated"] is True
    assert payload["user"]["id"] == "22222222-2222-2222-2222-222222222222"
    assert payload["user"]["name"] == "Existing User"
    assert payload["user"]["picture"] == "https://example.com/existing-avatar.png"


def test_auth_session_endpoint_validates_existing_token(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ) -> DummyResponse:
        if "openid-configuration" in url:
            return DummyResponse(
                {
                    "userinfo_endpoint": "https://oauth.example.com/userinfo",
                }
            )

        assert "userinfo" in url
        assert headers is not None
        assert headers["Authorization"] == "Bearer demo-access-token"
        return DummyResponse({"email": "user@example.com", "name": "Demo User"})

    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://oauth.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)

    response = client.post(
        "/auth/session",
        json={
            "access_token": "demo-access-token",
            "expires_in": 3600,
            "state": "login",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["authenticated"] is True
    assert payload["user"]["email"] == "user@example.com"
    assert payload["token"]["access_token"] == "demo-access-token"
    assert payload["state"] == "login"


def test_auth_session_endpoint_refreshes_near_expiry_token(monkeypatch) -> None:
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
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "demo-refresh-token"
        return DummyResponse(
            {
                "access_token": "new-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    def fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ) -> DummyResponse:
        if "openid-configuration" in url:
            return DummyResponse(
                {
                    "token_endpoint": "https://oauth.example.com/token",
                    "userinfo_endpoint": "https://oauth.example.com/userinfo",
                }
            )

        assert "userinfo" in url
        assert headers is not None
        assert headers["Authorization"] == "Bearer new-access-token"
        return DummyResponse({"email": "user@example.com", "name": "Demo User"})

    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "demo-client-secret")
    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://oauth.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setattr("app.auth.oauth_service.requests.post", fake_post)
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)

    response = client.post(
        "/auth/session",
        json={
            "access_token": "old-access-token",
            "refresh_token": "demo-refresh-token",
            "expires_in": 10,
            "state": "login",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["authenticated"] is True
    assert payload["token"]["access_token"] == "new-access-token"
    assert payload["token"]["refresh_token"] == "demo-refresh-token"
    assert payload["message"] == "Session token was renewed successfully."


def test_login_endpoint_uses_well_known_authorize_url(monkeypatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                "token_endpoint": "https://issuer.example.com/oauth2/token",
                "userinfo_endpoint": "https://issuer.example.com/oauth2/userinfo",
            }

    def fake_get(
        url: str,
        timeout: int,
    ) -> DummyResponse:
        assert "openid-configuration" in url
        return DummyResponse()

    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://issuer.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)

    response = client.get("/login")

    assert response.status_code == 200
    payload = response.json()
    assert payload["login_url"].startswith(
        "https://issuer.example.com/oauth2/authorize?"
    )


def test_auth_callback_endpoint_uses_well_known_token_and_userinfo(
    monkeypatch, allow_network
) -> None:
    # NOTE: reaches the real identity provider over HTTPS, so it opts out of the
    # network guard rather than silently depending on outbound access in CI.
    # Worth mocking the token/userinfo calls properly.
    class DummyResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ) -> DummyResponse:
        if "openid-configuration" in url:
            return DummyResponse(
                {
                    "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                    "token_endpoint": "https://issuer.example.com/oauth2/token",
                    "userinfo_endpoint": "https://issuer.example.com/oauth2/userinfo",
                }
            )

        assert url == "https://issuer.example.com/oauth2/userinfo"
        assert headers is not None
        assert headers["Authorization"] == "Bearer demo-access-token"
        return DummyResponse({"email": "user@example.com", "name": "Demo User"})

    def fake_post(
        url: str,
        data: dict[str, str],
        timeout: int,
    ) -> DummyResponse:
        assert url == "https://issuer.example.com/oauth2/token"
        assert data["code"] == "sample-code"
        return DummyResponse(
            {
                "access_token": "demo-access-token",
                "id_token": "demo-id-token",
                "token_type": "Bearer",
            }
        )

    monkeypatch.setenv(
        "OAUTH_WELL_KNOWN_URL",
        "https://issuer.example.com/.well-known/openid-configuration",
    )
    monkeypatch.setenv("OAUTH_CLIENT_ID", "demo-client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "demo-client-secret")
    monkeypatch.setattr("app.auth.oauth_service.requests.post", fake_post)
    monkeypatch.setattr("app.auth.oauth_service.requests.get", fake_get)

    response = client.get(
        "/auth/callback",
        params={"code": "sample-code", "state": "login"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["user"]["email"] == "user@example.com"
    assert payload["token"]["access_token"] == "demo-access-token"


def test_lambda_handler_accepts_sparse_api_gateway_v2_events() -> None:
    from app.lambda_handler import handler

    event = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/health",
        "rawQueryString": "",
        "headers": None,
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/health",
            }
        },
        "isBase64Encoded": False,
    }

    response = handler(event, {})

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_lambda_handler_supports_api_gateway_v1_events() -> None:
    from app.lambda_handler import handler

    event = {
        "resource": "/health",
        "path": "/health",
        "httpMethod": "GET",
        "headers": {
            "Host": "lambda.example.com",
            "X-Forwarded-Proto": "https",
        },
        "requestContext": {
            "identity": {
                "sourceIp": "127.0.0.1",
            }
        },
        "isBase64Encoded": False,
    }

    response = handler(event, {})

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


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
