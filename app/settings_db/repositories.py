from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.settings_db.gateway import SettingsDatabaseGateway
from app.settings_db.models import DBConnectionCreate
from app.settings_db.orm_models import (
    AgentEntity,
    ColumnDefinitionEntity,
    ColumnRelationEntity,
    ConversationMessageEntity,
    ConversationThreadEntity,
    LLMCallEntity,
    SchemaGraphEntity,
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

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Get user by email address.

        Args:
            email: User email address.

        Returns:
            dict[str, Any] | None: User record if found, None otherwise.
        """
        statement = select(
            SettingsUserEntity.id,
            SettingsUserEntity.email,
            SettingsUserEntity.full_name,
            SettingsUserEntity.avatar_url,
            SettingsUserEntity.created_at,
            SettingsUserEntity.updated_at,
            SettingsUserEntity.last_login_at,
            SettingsUserEntity.is_active,
        ).where(SettingsUserEntity.email == email)

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()

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
            SettingsDBConnectionEntity.is_active,
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
            SettingsDBConnectionEntity.display_config,
        ).where(SettingsDBConnectionEntity.id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()

        if row_mapping is None:
            return None

        return dict(row_mapping)

    def list_all_connections(self) -> list[dict[str, Any]]:
        """Fetch all stored database connections.

        Returns:
            list[dict[str, Any]]: List of all connection rows.
        """
        # Note: ga_service_account_json is intentionally NOT selected here (it's
        # a secret, like db_password, which is also excluded from listings).
        statement = select(
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
            SettingsDBConnectionEntity.is_healthy,
            SettingsDBConnectionEntity.is_active,
            SettingsDBConnectionEntity.last_synced_at,
            SettingsDBConnectionEntity.created_at,
        ).order_by(SettingsDBConnectionEntity.created_at.desc())

        with self._gateway.get_session() as session:
            row_mappings = session.execute(statement).mappings().all()

        # Convert UUIDs to strings for JSON serialization
        results = []
        for row in row_mappings:
            row_dict = dict(row)
            row_dict["id"] = str(row_dict["id"])
            row_dict["workspace_id"] = str(row_dict["workspace_id"])
            results.append(row_dict)

        return results

    def set_connection_active(
        self, connection_id: str, is_active: bool
    ) -> dict[str, Any] | None:
        """Enable or disable a stored connection.

        Args:
            connection_id: Stored connection identifier.
            is_active: Target active state.

        Returns:
            dict[str, Any] | None: Updated row when found.
        """
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return None

        statement = (
            update(SettingsDBConnectionEntity)
            .where(SettingsDBConnectionEntity.id == parsed_connection_id)
            .values(is_active=is_active)
            .returning(
                SettingsDBConnectionEntity.id,
                SettingsDBConnectionEntity.is_active,
            )
        )

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        return {
            "id": str(row_mapping["id"]),
            "is_active": row_mapping["is_active"],
        }


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

    def get_selected_tables_by_connection(
        self, connection_id: str
    ) -> list[dict[str, Any]]:
        """Fetch all selected tables for a connection.

        Args:
            connection_id: The db_connections UUID.

        Returns:
            list[dict[str, Any]]: List of selected table rows.
        """
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return []

        statement = select(
            SelectedTableEntity.id,
            SelectedTableEntity.db_connection_id,
            SelectedTableEntity.schema_name,
            SelectedTableEntity.table_name,
            SelectedTableEntity.human_description,
        ).where(SelectedTableEntity.db_connection_id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mappings = session.execute(statement).mappings().all()

        return [dict(row) for row in row_mappings]

    def bulk_replace_selected_tables(
        self, connection_id: str, table_names: list[str], schema_name: str = "public"
    ) -> list[dict[str, Any]]:
        """Replace all selected tables for a connection.

        Args:
            connection_id: The db_connections UUID.
            table_names: List of table names to save.
            schema_name: Schema name (default: public).

        Returns:
            list[dict[str, Any]]: List of saved table rows.
        """
        parsed_connection_id = UUID(connection_id)

        # Delete existing selections for this connection
        delete_statement = delete(SelectedTableEntity).where(
            SelectedTableEntity.db_connection_id == parsed_connection_id
        )

        if not table_names:
            with self._gateway.get_session() as session:
                session.execute(delete_statement)
                session.commit()
            return []

        # Insert new selections
        insert_values = [
            {
                "db_connection_id": parsed_connection_id,
                "schema_name": schema_name,
                "table_name": table_name,
                "human_description": None,
            }
            for table_name in table_names
        ]

        insert_statement = (
            insert(SelectedTableEntity)
            .values(insert_values)
            .returning(
                SelectedTableEntity.id,
                SelectedTableEntity.db_connection_id,
                SelectedTableEntity.schema_name,
                SelectedTableEntity.table_name,
                SelectedTableEntity.human_description,
            )
        )

        with self._gateway.get_session() as session:
            session.execute(delete_statement)
            row_mappings = session.execute(insert_statement).mappings().all()
            session.commit()

        return [dict(row) for row in row_mappings]


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

    def get_columns_grouped_by_connection(
        self, connection_id: str
    ) -> list[dict[str, Any]]:
        """Return selected tables with their column definitions for a connection.

        Joins settings.selected_tables to settings.column_definitions so the UI
        can render columns grouped per table in a single call.

        Args:
            connection_id: The db_connections UUID.

        Returns:
            list[dict[str, Any]]: One entry per selected table, each with a
            "columns" list (possibly empty if the async sync hasn't finished).
        """
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return []

        tables_statement = (
            select(
                SelectedTableEntity.id,
                SelectedTableEntity.schema_name,
                SelectedTableEntity.table_name,
                SelectedTableEntity.human_description,
            )
            .where(SelectedTableEntity.db_connection_id == parsed_connection_id)
            .order_by(SelectedTableEntity.table_name)
        )

        columns_statement = (
            select(
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
            .join(
                SelectedTableEntity,
                SelectedTableEntity.id == ColumnDefinitionEntity.table_id,
            )
            .where(SelectedTableEntity.db_connection_id == parsed_connection_id)
            .order_by(ColumnDefinitionEntity.raw_name)
        )

        with self._gateway.get_session() as session:
            table_rows = session.execute(tables_statement).mappings().all()
            column_rows = session.execute(columns_statement).mappings().all()

        # Group columns by their parent table_id
        columns_by_table: dict[str, list[dict[str, Any]]] = {}
        for column in column_rows:
            column_dict = dict(column)
            column_dict["id"] = str(column_dict["id"])
            table_id = str(column_dict["table_id"])
            column_dict["table_id"] = table_id
            columns_by_table.setdefault(table_id, []).append(column_dict)

        results: list[dict[str, Any]] = []
        for table in table_rows:
            table_id = str(table["id"])
            results.append(
                {
                    "table_id": table_id,
                    "schema_name": table["schema_name"],
                    "table_name": table["table_name"],
                    "human_description": table["human_description"],
                    "columns": columns_by_table.get(table_id, []),
                }
            )

        return results

    def update_column_semantics(
        self,
        column_id: UUID,
        logical_name: str | None,
        description: str | None,
    ) -> None:
        """Set logical_name and description for a single column definition."""
        statement = (
            update(ColumnDefinitionEntity)
            .where(ColumnDefinitionEntity.id == column_id)
            .values(
                logical_name=logical_name,
                description=description,
                updated_at=func.now(),
            )
        )
        with self._gateway.get_session() as session:
            session.execute(statement)
            session.commit()

    def update_column_logical_name(
        self,
        column_id: str,
        logical_name: str | None,
    ) -> dict[str, Any] | None:
        """Update only the logical_name of a column and return the updated row."""
        try:
            parsed_column_id = UUID(column_id)
        except ValueError:
            return None

        statement = (
            update(ColumnDefinitionEntity)
            .where(ColumnDefinitionEntity.id == parsed_column_id)
            .values(logical_name=logical_name, updated_at=func.now())
            .returning(
                ColumnDefinitionEntity.id,
                ColumnDefinitionEntity.raw_name,
                ColumnDefinitionEntity.logical_name,
                ColumnDefinitionEntity.description,
                ColumnDefinitionEntity.data_type,
            )
        )

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        row_dict = dict(row_mapping)
        row_dict["id"] = str(row_dict["id"])
        return row_dict


class ColumnRelationRepository:
    """Data access for settings.column_relations records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def replace_relations_for_connection(
        self,
        connection_id: str,
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace all stored relations for a connection.

        Args:
            connection_id: The db_connections UUID.
            relations: List of dicts with keys from_table_id, from_column,
                to_table_id, to_column, constraint_name.

        Returns:
            list[dict[str, Any]]: Inserted relation rows.
        """
        parsed_connection_id = UUID(connection_id)

        delete_statement = delete(ColumnRelationEntity).where(
            ColumnRelationEntity.db_connection_id == parsed_connection_id
        )

        if not relations:
            with self._gateway.get_session() as session:
                session.execute(delete_statement)
                session.commit()
            return []

        insert_values = [
            {
                "db_connection_id": parsed_connection_id,
                "from_table_id": relation["from_table_id"],
                "from_column": relation["from_column"],
                "to_table_id": relation["to_table_id"],
                "to_column": relation["to_column"],
                "constraint_name": relation.get("constraint_name"),
            }
            for relation in relations
        ]

        insert_statement = (
            insert(ColumnRelationEntity)
            .values(insert_values)
            .returning(
                ColumnRelationEntity.id,
                ColumnRelationEntity.from_table_id,
                ColumnRelationEntity.from_column,
                ColumnRelationEntity.to_table_id,
                ColumnRelationEntity.to_column,
                ColumnRelationEntity.constraint_name,
            )
        )

        with self._gateway.get_session() as session:
            session.execute(delete_statement)
            row_mappings = session.execute(insert_statement).mappings().all()
            session.commit()

        return [dict(row) for row in row_mappings]

    def get_relations_by_connection(
        self, connection_id: str
    ) -> list[dict[str, Any]]:
        """Return all stored relations for a connection."""
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return []

        statement = select(
            ColumnRelationEntity.id,
            ColumnRelationEntity.from_table_id,
            ColumnRelationEntity.from_column,
            ColumnRelationEntity.to_table_id,
            ColumnRelationEntity.to_column,
            ColumnRelationEntity.constraint_name,
        ).where(ColumnRelationEntity.db_connection_id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mappings = session.execute(statement).mappings().all()

        results = []
        for row in row_mappings:
            row_dict = dict(row)
            row_dict["id"] = str(row_dict["id"])
            row_dict["from_table_id"] = str(row_dict["from_table_id"])
            row_dict["to_table_id"] = str(row_dict["to_table_id"])
            results.append(row_dict)
        return results


class SchemaGraphRepository:
    """Data access for settings.schema_graphs records."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def upsert_graph(
        self, connection_id: str, graph: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Insert or replace the stored schema graph for a connection."""
        parsed_connection_id = UUID(connection_id)

        statement = insert(SchemaGraphEntity).values(
            db_connection_id=parsed_connection_id,
            graph=graph,
            updated_at=func.now(),
        )
        statement = statement.on_conflict_do_update(
            constraint="schema_graphs_connection_unique",
            set_={
                "graph": statement.excluded.graph,
                "updated_at": func.now(),
            },
        )
        statement = statement.returning(
            SchemaGraphEntity.id,
            SchemaGraphEntity.db_connection_id,
            SchemaGraphEntity.graph,
            SchemaGraphEntity.updated_at,
        )

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()
            session.commit()

        if row_mapping is None:
            return None

        row_dict = dict(row_mapping)
        row_dict["id"] = str(row_dict["id"])
        row_dict["db_connection_id"] = str(row_dict["db_connection_id"])
        return row_dict

    def get_graph_by_connection(self, connection_id: str) -> dict[str, Any] | None:
        """Return the stored schema graph for a connection, if any."""
        try:
            parsed_connection_id = UUID(connection_id)
        except ValueError:
            return None

        statement = select(
            SchemaGraphEntity.id,
            SchemaGraphEntity.db_connection_id,
            SchemaGraphEntity.graph,
            SchemaGraphEntity.updated_at,
        ).where(SchemaGraphEntity.db_connection_id == parsed_connection_id)

        with self._gateway.get_session() as session:
            row_mapping = session.execute(statement).mappings().first()

        if row_mapping is None:
            return None

        row_dict = dict(row_mapping)
        row_dict["id"] = str(row_dict["id"])
        row_dict["db_connection_id"] = str(row_dict["db_connection_id"])
        return row_dict


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


class ConversationRepository:
    """Data access for conversation threads and messages."""

    def __init__(self, gateway: SettingsDatabaseGateway | None = None) -> None:
        self._gateway = gateway or SettingsDatabaseGateway()

    def create_conversation(
        self,
        workspace_id: str,
        connection_id: str | None,
        title: str,
        thread_id: str,
    ) -> dict[str, Any]:
        """Create a new conversation thread.

        Args:
            workspace_id: The workspace UUID string.
            connection_id: Optional DB connection UUID string.
            title: First message text (truncated) used as conversation title.
            thread_id: Caller-provided unique identifier.

        Returns:
            dict with thread_id, title, created_at.
        """
        statement = insert(ConversationThreadEntity).values(
            id=thread_id,
            workspace_id=UUID(workspace_id),
            connection_id=UUID(connection_id) if connection_id else None,
            title=title,
            message_count=0,
        ).returning(
            ConversationThreadEntity.id,
            ConversationThreadEntity.workspace_id,
            ConversationThreadEntity.connection_id,
            ConversationThreadEntity.title,
            ConversationThreadEntity.message_count,
            ConversationThreadEntity.created_at,
            ConversationThreadEntity.updated_at,
        )

        with self._gateway.get_session() as session:
            row = session.execute(statement).mappings().first()
            session.commit()

        if row is None:
            return {}

        result = dict(row)
        result["id"] = str(result["id"])
        result["workspace_id"] = str(result["workspace_id"])
        result["connection_id"] = str(result["connection_id"]) if result["connection_id"] else None
        return result

    def conversation_exists(self, thread_id: str) -> bool:
        """Check whether a conversation thread exists."""
        statement = (
            select(ConversationThreadEntity.id)
            .where(ConversationThreadEntity.id == thread_id)
            .limit(1)
        )

        with self._gateway.get_session() as session:
            row = session.execute(statement).first()

        return row is not None

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        """Append a message to a conversation thread.

        Args:
            thread_id: Conversation thread identifier.
            role: "user" or "assistant".
            content: Message text.

        Returns:
            dict with message id and created_at.
        """
        message_statement = insert(ConversationMessageEntity).values(
            thread_id=thread_id,
            role=role,
            content=content,
        ).returning(
            ConversationMessageEntity.id,
            ConversationMessageEntity.thread_id,
            ConversationMessageEntity.role,
            ConversationMessageEntity.content,
            ConversationMessageEntity.created_at,
        )

        # Increment message_count and update updated_at on the thread
        thread_update = (
            update(ConversationThreadEntity)
            .where(ConversationThreadEntity.id == thread_id)
            .values(
                message_count=ConversationThreadEntity.message_count + 1,
                updated_at=func.now(),
            )
        )

        with self._gateway.get_session() as session:
            row = session.execute(message_statement).mappings().first()
            session.execute(thread_update)
            session.commit()

        if row is None:
            return {}

        result = dict(row)
        result["id"] = str(result["id"])
        return result

    def add_llm_call(
        self,
        message_id: str,
        thread_id: str,
        llm_model: str,
        llm_provider: str,
        llm_purpose: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int | None = None,
        request_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Track an individual LLM API call for a message.

        Args:
            message_id: UUID of the conversation message.
            thread_id: Conversation thread identifier.
            llm_model: LLM model used (e.g., "gpt-4", "gpt-3.5-turbo").
            llm_provider: Provider name (e.g., "openai", "anthropic").
            llm_purpose: Purpose of the call (e.g., "sql-generator", "table-identifier").
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            total_tokens: Total tokens used.
            cost_usd: Cost in USD.
            duration_ms: Duration of the API call in milliseconds.
            request_metadata: Additional metadata (retry attempt, error info, etc.).

        Returns:
            dict with llm_call id and created_at.
        """
        llm_call_statement = insert(LLMCallEntity).values(
            message_id=message_id,
            thread_id=thread_id,
            llm_model=llm_model,
            llm_provider=llm_provider,
            llm_purpose=llm_purpose,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            request_metadata=request_metadata,
        ).returning(
            LLMCallEntity.id,
            LLMCallEntity.created_at,
        )

        with self._gateway.get_session() as session:
            row = session.execute(llm_call_statement).mappings().first()
            session.commit()

        if row is None:
            return {}

        result = dict(row)
        result["id"] = str(result["id"])
        return result

    def get_messages_by_thread(self, thread_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a thread in chronological order.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            list of {role, content, created_at} dicts.
        """
        statement = select(
            ConversationMessageEntity.id,
            ConversationMessageEntity.role,
            ConversationMessageEntity.content,
            ConversationMessageEntity.created_at,
        ).where(
            ConversationMessageEntity.thread_id == thread_id
        ).order_by(ConversationMessageEntity.created_at.asc())

        with self._gateway.get_session() as session:
            rows = session.execute(statement).mappings().all()

        return [
            {
                "id": str(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    def get_conversations_by_workspace(
        self,
        workspace_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List conversation threads for a workspace, newest first.

        Args:
            workspace_id: The workspace UUID string.
            limit: Maximum number of threads to return.

        Returns:
            list of {id, title, message_count, connection_id, created_at, updated_at}.
        """
        statement = select(
            ConversationThreadEntity.id,
            ConversationThreadEntity.connection_id,
            ConversationThreadEntity.title,
            ConversationThreadEntity.message_count,
            ConversationThreadEntity.created_at,
            ConversationThreadEntity.updated_at,
        ).where(
            ConversationThreadEntity.workspace_id == UUID(workspace_id)
        ).order_by(
            ConversationThreadEntity.updated_at.desc()
        ).limit(limit)

        with self._gateway.get_session() as session:
            rows = session.execute(statement).mappings().all()

        return [
            {
                "id": str(row["id"]),
                "connection_id": str(row["connection_id"]) if row["connection_id"] else None,
                "title": row["title"],
                "message_count": row["message_count"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
            for row in rows
        ]

    def delete_conversation(self, thread_id: str) -> bool:
        """Delete a thread and all its messages.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True if the thread was found and deleted.
        """
        messages_delete = delete(ConversationMessageEntity).where(
            ConversationMessageEntity.thread_id == thread_id
        )
        thread_delete = delete(ConversationThreadEntity).where(
            ConversationThreadEntity.id == thread_id
        )

        with self._gateway.get_session() as session:
            session.execute(messages_delete)
            result = session.execute(thread_delete)
            session.commit()

        return result.rowcount > 0
