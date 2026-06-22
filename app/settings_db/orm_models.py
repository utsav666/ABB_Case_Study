from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSON as PostgreSQLJSON
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class SettingsBase(DeclarativeBase):
    """Base declarative model for the settings schema."""


class WorkspaceEntity(SettingsBase):
    """ORM entity for settings.workspaces."""

    __tablename__ = "workspaces"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    owner_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    settings: Mapped[dict | None] = mapped_column(
        PostgreSQLJSON,
        nullable=True,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class SettingsUserEntity(SettingsBase):
    """ORM entity for settings.users."""

    __tablename__ = "users"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
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

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Discriminator: 'database' (SQL) or 'google_analytics' (GA4).
    connection_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'database'"),
        default="database",
    )
    # SQL fields — nullable so non-database connection types can omit them.
    engine: Mapped[str | None] = mapped_column(String(50), nullable=True)
    db_host: Mapped[str | None] = mapped_column(String(255), nullable=True)
    db_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    db_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    db_user: Mapped[str | None] = mapped_column(String(255), nullable=True)
    db_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    db_schema: Mapped[str | None] = mapped_column(String(255), nullable=True)
    use_ssl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    extra_options: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Google Analytics (GA4) fields.
    ga_property_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    ga_service_account_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_healthy: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
        default=True,
    )
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


class ColumnRelationEntity(SettingsBase):
    """ORM entity for settings.column_relations."""

    __tablename__ = "column_relations"
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
    from_table_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    from_column: Mapped[str] = mapped_column(Text, nullable=False)
    to_table_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    to_column: Mapped[str] = mapped_column(Text, nullable=False)
    constraint_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )


class SchemaGraphEntity(SettingsBase):
    """ORM entity for settings.schema_graphs."""

    __tablename__ = "schema_graphs"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    db_connection_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    graph: Mapped[dict | None] = mapped_column(
        PostgreSQLJSON,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now(),
    )


class ConversationThreadEntity(SettingsBase):
    """ORM entity for settings.conversation_threads."""

    __tablename__ = "conversation_threads"
    __table_args__ = {"schema": "settings"}

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    connection_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ConversationMessageEntity(SettingsBase):
    """ORM entity for settings.conversation_messages."""

    __tablename__ = "conversation_messages"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class LLMCallEntity(SettingsBase):
    """ORM entity for settings.llm_calls - tracks individual LLM API calls."""

    __tablename__ = "llm_calls"
    __table_args__ = {"schema": "settings"}

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4(),
    )
    message_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("settings.conversation_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # LLM details
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_purpose: Mapped[str] = mapped_column(String(100), nullable=False)

    # Token usage
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Cost
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Timing
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Metadata (for storing additional context like retry attempts, errors, etc.)
    request_metadata: Mapped[dict | None] = mapped_column(PostgreSQLJSON, nullable=True)


class WorkspaceMemberEntity(SettingsBase):
    """ORM entity for settings.workspace_members."""

    __tablename__ = "workspace_members"
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
    user_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    role: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        server_default=text("'VIEWER'::settings.user_role"),
    )
    joined_at: Mapped[datetime | None] = mapped_column(
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
