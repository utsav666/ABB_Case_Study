from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import ARRAY, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON as PostgreSQLJSON
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class SettingsBase(DeclarativeBase):
    """Base declarative model for the settings schema."""


class SettingsUserEntity(SettingsBase):
    """ORM entity for settings.users."""

    __tablename__ = "users"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class SettingsDBConnectionEntity(SettingsBase):
    """ORM entity for settings.db_connections."""

    __tablename__ = "db_connections"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    engine: Mapped[str] = mapped_column(String(50), nullable=False)
    db_host: Mapped[str] = mapped_column(String(255), nullable=False)
    db_port: Mapped[int] = mapped_column(Integer, nullable=False)
    db_name: Mapped[str] = mapped_column(String(255), nullable=False)
    db_user: Mapped[str] = mapped_column(String(255), nullable=False)
    db_password: Mapped[str] = mapped_column(String(255), nullable=False)
    db_schema: Mapped[str | None] = mapped_column(String(255), nullable=True)
    use_ssl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extra_options: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_healthy: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SelectedTableEntity(SettingsBase):
    """ORM entity for settings.selected_tables."""

    __tablename__ = "selected_tables"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    db_connection_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    schema_name: Mapped[str] = mapped_column(Text, nullable=False, default="public")
    table_name: Mapped[str] = mapped_column(Text, nullable=False)
    human_description: Mapped[str | None] = mapped_column(Text, nullable=True)


class ColumnDefinitionEntity(SettingsBase):
    """ORM entity for settings.column_definitions."""

    __tablename__ = "column_definitions"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    table_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    raw_name: Mapped[str] = mapped_column(Text, nullable=False)
    logical_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_pii: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_filterable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    example_values: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )


class AgentEntity(SettingsBase):
    """ORM entity for settings.agents."""

    __tablename__ = "agents"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    workspace_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        server_default="Data Assistant",
    )
    model_version: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        server_default="gpt-4o",
    )
    system_instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(PostgreSQLJSON, nullable=True)
    is_public_to_team: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, server_default="true"
    )
