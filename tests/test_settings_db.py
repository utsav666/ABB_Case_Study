from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.settings_db.config import SettingsDatabaseConfig
from app.settings_db.controllers import DBConnectionController, SettingsUserController
from app.settings_db.models import DBConnectionCreate


class FakeSettingsUserRepository:
    def __init__(self, persisted_row: dict | None) -> None:
        self.persisted_row = persisted_row
        self.calls: list[tuple[str, str | None, str | None]] = []

    def upsert_user(
        self,
        email: str,
        full_name: str | None,
        avatar_url: str | None,
    ) -> dict | None:
        self.calls.append((email, full_name, avatar_url))
        return self.persisted_row


class FakeDBConnectionRepository:
    def __init__(
        self,
        inserted_row: dict | None = None,
        connection_row: dict | None = None,
    ) -> None:
        self.inserted_row = inserted_row
        self.connection_row = connection_row

    def insert_connection(self, payload: DBConnectionCreate) -> dict | None:
        return self.inserted_row

    def get_connection_by_id(self, connection_id: str) -> dict | None:
        return self.connection_row


def test_settings_database_config_uses_fallback_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SETTINGS_DB_HOST", raising=False)
    monkeypatch.delenv("SETTINGS_DB_PORT", raising=False)
    monkeypatch.delenv("SETTINGS_DB_NAME", raising=False)
    monkeypatch.delenv("SETTINGS_DB_USER", raising=False)
    monkeypatch.delenv("SETTINGS_DB_PASSWORD", raising=False)
    monkeypatch.setenv("DB_HOST", "shared-host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "shared-name")
    monkeypatch.setenv("DB_USER", "shared-user")
    monkeypatch.setenv("DB_PASSWORD", "shared-password")

    config = SettingsDatabaseConfig.from_environment()

    assert config.build_connection_url() == (
        "postgresql://shared-user:shared-password@shared-host:5433/shared-name"
    )


def test_settings_user_controller_returns_original_payload_without_email() -> None:
    repository = FakeSettingsUserRepository(persisted_row=None)
    controller = SettingsUserController(repository=repository)
    payload = {"name": "Demo User"}

    result = controller.upsert_oauth_user(payload)

    assert result == payload
    assert repository.calls == []


def test_settings_user_controller_builds_enriched_user_payload() -> None:
    timestamp = datetime(2026, 4, 12, tzinfo=timezone.utc)
    repository = FakeSettingsUserRepository(
        persisted_row={
            "id": uuid4(),
            "email": "user@example.com",
            "full_name": "Persisted User",
            "avatar_url": "https://example.com/avatar.png",
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_login_at": timestamp,
            "is_active": True,
        }
    )
    controller = SettingsUserController(repository=repository)

    result = controller.upsert_oauth_user(
        {
            "email": "USER@example.com ",
            "name": "Provider Name",
            "picture": "https://example.com/provider.png",
        }
    )

    assert repository.calls == [
        (
            "user@example.com",
            "Provider Name",
            "https://example.com/provider.png",
        )
    ]
    assert result["email"] == "user@example.com"
    assert result["name"] == "Persisted User"
    assert result["picture"] == "https://example.com/avatar.png"
    assert result["created_at"] == timestamp.isoformat()
    assert result["is_active"] is True


def test_db_connection_controller_rejects_invalid_engine() -> None:
    controller = DBConnectionController(repository=FakeDBConnectionRepository())
    payload = DBConnectionCreate(
        workspace_id=uuid4(),
        name="Warehouse",
        engine="mysql",
        db_host="db.example.com",
        db_port=5432,
        db_name="analytics",
        db_user="demo",
        db_password="secret",
    )

    with pytest.raises(HTTPException) as error_info:
        controller.create_connection(payload)

    assert error_info.value.status_code == 422
    assert "Invalid engine 'mysql'" in error_info.value.detail


def test_db_connection_controller_serializes_created_connection() -> None:
    timestamp = datetime(2026, 4, 12, tzinfo=timezone.utc)
    workspace_id = uuid4()
    inserted_row = {
        "id": uuid4(),
        "workspace_id": workspace_id,
        "name": "Warehouse",
        "engine": "postgres",
        "db_host": "db.example.com",
        "db_port": 5432,
        "db_name": "analytics",
        "db_user": "demo",
        "is_healthy": True,
        "last_synced_at": None,
        "created_at": timestamp,
    }
    controller = DBConnectionController(
        repository=FakeDBConnectionRepository(inserted_row=inserted_row)
    )
    payload = DBConnectionCreate(
        workspace_id=workspace_id,
        name="Warehouse",
        engine="postgres",
        db_host="db.example.com",
        db_port=5432,
        db_name="analytics",
        db_user="demo",
        db_password="secret",
    )

    result = controller.create_connection(payload)

    assert result["status"] == "created"
    assert result["connection"]["workspace_id"] == str(workspace_id)
    assert result["connection"]["created_at"] == timestamp.isoformat()
    assert result["connection"]["last_synced_at"] is None
