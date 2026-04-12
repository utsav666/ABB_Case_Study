from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from app.settings_db.gateway import SettingsDatabaseGateway
from app.settings_db.models import DBConnectionCreate
from app.settings_db.orm_models import SettingsDBConnectionEntity, SettingsUserEntity


class SettingsUserRepository:
    """Data access for settings.users records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def upsert_user(
        self,
        email: str,
        full_name: str | None,
        avatar_url: str | None,
    ) -> dict[str, Any] | None:
        """Create or update a settings user record.

        Args:
            email: User email.
            full_name: User display name.
            avatar_url: User avatar URL.

        Returns:
            dict[str, Any] | None: Persisted row when available.
        """
        statement = insert(SettingsUserEntity).values(
            email=email,
            full_name=full_name,
            avatar_url=avatar_url,
            last_login_at=func.now(),
        )
        statement = statement.on_conflict_do_update(
            index_elements=[SettingsUserEntity.email],
            set_={
                "full_name": func.coalesce(
                    statement.excluded.full_name,
                    SettingsUserEntity.full_name,
                ),
                "avatar_url": func.coalesce(
                    statement.excluded.avatar_url,
                    SettingsUserEntity.avatar_url,
                ),
                "updated_at": func.now(),
                "last_login_at": func.now(),
                "is_active": True,
            },
        )
        statement = statement.returning(
            SettingsUserEntity.id,
            SettingsUserEntity.email,
            SettingsUserEntity.full_name,
            SettingsUserEntity.avatar_url,
            SettingsUserEntity.created_at,
            SettingsUserEntity.updated_at,
            SettingsUserEntity.last_login_at,
            SettingsUserEntity.is_active,
        )
        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        return dict(row_mapping)


class DBConnectionRepository:
    """Data access for settings.db_connections records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def insert_connection(self, payload: DBConnectionCreate) -> dict[str, Any] | None:
        """Insert a connection record into the settings database.

        Args:
            payload: Connection creation payload.

        Returns:
            dict[str, Any] | None: Persisted row when available.
        """
        statement = insert(SettingsDBConnectionEntity).values(
            workspace_id=payload.workspace_id,
            name=payload.name,
            engine=payload.engine,
            db_host=payload.db_host,
            db_port=payload.db_port,
            db_name=payload.db_name,
            db_user=payload.db_user,
            db_password=payload.db_password,
        )
        statement = statement.returning(
            SettingsDBConnectionEntity.id,
            SettingsDBConnectionEntity.workspace_id,
            SettingsDBConnectionEntity.name,
            SettingsDBConnectionEntity.engine,
            SettingsDBConnectionEntity.db_host,
            SettingsDBConnectionEntity.db_port,
            SettingsDBConnectionEntity.db_name,
            SettingsDBConnectionEntity.db_user,
            SettingsDBConnectionEntity.is_healthy,
            SettingsDBConnectionEntity.last_synced_at,
            SettingsDBConnectionEntity.created_at,
        )
        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        return dict(row_mapping)

    def get_connection_by_id(self, connection_id: str) -> dict[str, Any] | None:
        """Fetch a stored connection by its identifier.

        Args:
            connection_id: Stored connection identifier.

        Returns:
            dict[str, Any] | None: Stored connection row when found.
        """
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return None

        statement = select(
            SettingsDBConnectionEntity.id,
            SettingsDBConnectionEntity.engine,
            SettingsDBConnectionEntity.db_host,
            SettingsDBConnectionEntity.db_port,
            SettingsDBConnectionEntity.db_name,
            SettingsDBConnectionEntity.db_user,
            SettingsDBConnectionEntity.db_password,
        ).where(SettingsDBConnectionEntity.id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()

        if row_mapping is None:
            return None

        return dict(row_mapping)
