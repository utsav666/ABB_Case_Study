from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from app.settings_db.gateway import SettingsDatabaseGateway
from app.settings_db.models import DBConnectionCreate
from app.settings_db.orm_models import (
    AgentEntity,
    ColumnDefinitionEntity,
    SelectedTableEntity,
    SettingsDBConnectionEntity,
    SettingsUserEntity,
)


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
            db_schema=payload.db_schema,
            use_ssl=payload.use_ssl,
            extra_options=payload.extra_options or None,
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
            SettingsDBConnectionEntity.db_schema,
            SettingsDBConnectionEntity.use_ssl,
            SettingsDBConnectionEntity.extra_options,
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
            SettingsDBConnectionEntity.db_schema,
            SettingsDBConnectionEntity.use_ssl,
            SettingsDBConnectionEntity.extra_options,
        ).where(SettingsDBConnectionEntity.id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()

        if row_mapping is None:
            return None

        return dict(row_mapping)


class SelectedTableRepository:
    """Data access for settings.selected_tables records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def upsert_table(
        self,
        connection_id: str,
        schema_name: str,
        table_name: str,
        human_description: str | None,
    ) -> dict[str, Any] | None:
        """Insert or update a selected-table record.

        Args:
            connection_id: The db_connections UUID this table belongs to.
            schema_name: Schema the table lives in.
            table_name: Name of the table.
            human_description: Optional plain-english description.

        Returns:
            dict[str, Any] | None: Persisted row when available.
        """
        parsed_connection_id = UUID(connection_id)

        statement = insert(SelectedTableEntity).values(
            db_connection_id=parsed_connection_id,
            schema_name=schema_name,
            table_name=table_name,
            human_description=human_description,
        )
        statement = statement.on_conflict_do_update(
            constraint="selected_tables_db_connection_id_schema_name_table_name_key",
            set_={"human_description": statement.excluded.human_description},
        )
        statement = statement.returning(
            SelectedTableEntity.id,
            SelectedTableEntity.db_connection_id,
            SelectedTableEntity.schema_name,
            SelectedTableEntity.table_name,
            SelectedTableEntity.human_description,
        )

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        return dict(row_mapping)


class ColumnDefinitionRepository:
    """Data access for settings.column_definitions records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def replace_columns_for_table(
        self,
        table_id: UUID,
        column_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Delete existing columns for a table and insert the new set.

        Args:
            table_id: UUID of the parent selected_tables row.
            column_rows: List of column field dicts to insert.

        Returns:
            list[dict[str, Any]]: All newly inserted column rows.
        """
        delete_statement = delete(ColumnDefinitionEntity).where(
            ColumnDefinitionEntity.table_id == table_id
        )

        if not column_rows:
            with self._gateway.get_session() as session:
                session.execute(delete_statement)
                session.commit()
            return []

        insert_statement = (
            insert(ColumnDefinitionEntity)
            .values(column_rows)
            .returning(
                ColumnDefinitionEntity.id,
                ColumnDefinitionEntity.table_id,
                ColumnDefinitionEntity.raw_name,
                ColumnDefinitionEntity.logical_name,
                ColumnDefinitionEntity.description,
                ColumnDefinitionEntity.data_type,
                ColumnDefinitionEntity.is_pii,
                ColumnDefinitionEntity.is_filterable,
                ColumnDefinitionEntity.example_values,
            )
        )

        with self._gateway.get_session() as session:
            session.execute(delete_statement)
            row_mappings = session.execute(insert_statement).mappings().all()
            session.commit()

        return [dict(row) for row in row_mappings]


class AgentRepository:
    """Data access for settings.agents records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def create_agent(self, values: dict[str, Any]) -> dict[str, Any] | None:
        """Insert a new agent row and return the persisted record."""
        statement = (
            insert(AgentEntity)
            .values(**values)
            .returning(
                AgentEntity.id,
                AgentEntity.workspace_id,
                AgentEntity.name,
                AgentEntity.model_version,
                AgentEntity.system_instructions,
                AgentEntity.config,
                AgentEntity.is_public_to_team,
            )
        )
        with self._gateway.get_session() as session:
            row = session.execute(statement).mappings().first()
            session.commit()
        return dict(row) if row else None

    def get_agent_by_id(self, agent_id: str) -> dict[str, Any] | None:
        """Fetch an agent by its UUID."""
        try:
            parsed_id = UUID(agent_id)
        except ValueError:
            return None

        statement = select(
            AgentEntity.id,
            AgentEntity.workspace_id,
            AgentEntity.name,
            AgentEntity.model_version,
            AgentEntity.system_instructions,
            AgentEntity.config,
            AgentEntity.is_public_to_team,
        ).where(AgentEntity.id == parsed_id)

        with self._gateway.get_session() as session:
            row = session.execute(statement).mappings().first()
        return dict(row) if row else None

    def list_agents_by_workspace(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return all agents belonging to a workspace."""
        parsed_id = UUID(workspace_id)
        statement = select(
            AgentEntity.id,
            AgentEntity.workspace_id,
            AgentEntity.name,
            AgentEntity.model_version,
            AgentEntity.config,
            AgentEntity.is_public_to_team,
        ).where(AgentEntity.workspace_id == parsed_id)

        with self._gateway.get_session() as session:
            rows = session.execute(statement).mappings().all()
        return [dict(r) for r in rows]

    def update_agent(
        self, agent_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update an existing agent's fields."""
        parsed_id = UUID(agent_id)
        # Remove None values so we only update provided fields
        clean = {k: v for k, v in updates.items() if v is not None}
        if not clean:
            return self.get_agent_by_id(agent_id)

        from sqlalchemy import update as sa_update

        statement = (
            sa_update(AgentEntity)
            .where(AgentEntity.id == parsed_id)
            .values(**clean)
            .returning(
                AgentEntity.id,
                AgentEntity.workspace_id,
                AgentEntity.name,
                AgentEntity.model_version,
                AgentEntity.system_instructions,
                AgentEntity.config,
                AgentEntity.is_public_to_team,
            )
        )
        with self._gateway.get_session() as session:
            row = session.execute(statement).mappings().first()
            session.commit()
        return dict(row) if row else None

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID. Returns True if a row was removed."""
        parsed_id = UUID(agent_id)
        statement = delete(AgentEntity).where(AgentEntity.id == parsed_id)
        with self._gateway.get_session() as session:
            result = session.execute(statement)
            session.commit()
        return result.rowcount > 0
