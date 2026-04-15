from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_db_connection_engines_endpoint_returns_supported_engines() -> None:
    response = client.get("/settings/db-connections/engines")

    assert response.status_code == 200
    assert response.json()["engines"] == [
        "postgres",
        "mysql",
        "mariadb",
        "sqlserver",
        "oracle",
        "sqlite",
    ]


def test_db_connection_validate_endpoint_returns_can_save(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.settings_db.router.validate_db_connection",
        lambda payload: {
            "status": "validated",
            "can_save": True,
            "engine": payload.engine,
            "message": "Connection validated successfully.",
        },
    )

    response = client.post(
        "/settings/db-connections/validate",
        json={
            "name": "Warehouse",
            "engine": "postgres",
            "db_host": "db.example.com",
            "db_port": 5432,
            "db_name": "analytics",
            "db_user": "demo",
            "db_password": "secret",
            "db_schema": "public",
            "use_ssl": True,
            "extra_options": "sslmode=require",
        },
    )

    assert response.status_code == 200
    assert response.json()["can_save"] is True
    assert response.json()["engine"] == "postgres"


def test_db_connection_save_endpoint_uses_controller(monkeypatch) -> None:
    workspace_id = uuid4()
    monkeypatch.setattr(
        "app.settings_db.router.create_db_connection",
        lambda payload: {
            "status": "created",
            "connection": {
                "id": str(uuid4()),
                "workspace_id": str(payload.workspace_id),
                "name": payload.name,
                "engine": payload.engine,
                "db_host": payload.db_host,
                "db_port": payload.db_port,
                "db_name": payload.db_name,
                "db_user": payload.db_user,
                "db_schema": payload.db_schema,
                "use_ssl": payload.use_ssl,
                "extra_options": payload.extra_options,
                "is_healthy": True,
                "last_synced_at": None,
                "created_at": "2026-04-13T00:00:00+00:00",
            },
        },
    )

    response = client.post(
        "/settings/db-connections",
        json={
            "workspace_id": str(workspace_id),
            "name": "Warehouse",
            "engine": "sqlite",
            "db_host": "",
            "db_port": 0,
            "db_name": "/tmp/demo.sqlite",
            "db_user": "",
            "db_password": "",
            "db_schema": "main",
            "use_ssl": False,
            "extra_options": "",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "created"
    assert response.json()["connection"]["engine"] == "sqlite"
    assert response.json()["connection"]["workspace_id"] == str(workspace_id)