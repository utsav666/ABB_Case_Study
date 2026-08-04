from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.target_db.connection_profiles import (
    get_default_port,
    normalize_database_engine,
)


class DBConnectionInput(BaseModel):
    """Shared connection fields from the settings UI."""

    name: str = Field(min_length=1, description="Display name for this connection.")
    engine: str = Field(
        min_length=1, description="Supported database engine identifier."
    )
    db_host: str = Field(default="", description="Database host address.")
    db_port: int | None = Field(
        default=None, ge=0, le=65535, description="Database port."
    )
    db_name: str = Field(min_length=1, description="Database name or SQLite file path.")
    db_user: str = Field(default="", description="Database user.")
    db_password: str = Field(default="", description="Database password.")
    db_schema: str = Field(default="public", description="Default schema name.")
    use_ssl: bool = Field(default=False, description="Whether to require SSL/TLS.")
    extra_options: str = Field(
        default="",
        description="Optional driver-specific key=value pairs, one per line.",
    )

    @model_validator(mode="after")
    def normalize_fields(self) -> DBConnectionInput:
        """Normalize and validate required fields."""
        if not self.engine:
            raise ValueError("Database engine is required.")
        if not self.db_name:
            raise ValueError("Database name is required.")

        normalized_engine = normalize_database_engine(self.engine)
        self.engine = normalized_engine
        self.db_schema = self.db_schema.strip() or "public"
        self.extra_options = self.extra_options.strip()

        if self.db_port is None:
            self.db_port = get_default_port(normalized_engine)

        if normalized_engine == "sqlite":
            self.db_host = ""
            self.db_user = ""
            self.db_password = ""
            return self

        self.db_host = self.db_host.strip()
        self.db_user = self.db_user.strip()
        self.db_password = self.db_password.strip()
        if not self.db_host:
            raise ValueError("Database host is required for this engine.")
        if not self.db_user:
            raise ValueError("Database username is required for this engine.")
        if not self.db_password:
            raise ValueError("Database password is required for this engine.")

        return self


class DBConnectionValidationRequest(DBConnectionInput):
    """Request body for validating a database connection."""


class DBConnectionCreate(DBConnectionInput):
    """Request body for creating a new database connection."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workspace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "Production Postgres",
                "engine": "postgres",
                "db_host": "db.example.com",
                "db_port": 5432,
                "db_name": "analytics",
                "db_user": "app_user",
                "db_password": "secret",
                "db_schema": "public",
                "use_ssl": True,
                "extra_options": "sslmode=require",
            }
        }
    )

    workspace_id: UUID = Field(description="UUID of the parent workspace.")


class AgentConfig(BaseModel):
    """JSONB config stored in settings.agents.config."""

    db_connection_id: str | None = Field(
        default=None,
        description="UUID of the db_connection this agent queries.",
    )
    schema_name: str = Field(
        default="public",
        description="Database schema the agent operates on.",
    )
    llm_provider: str | None = Field(
        default=None,
        description=(
            "LLM provider override for this agent. "
            "Supported: openai, azure_openai, claude_merck. "
            "When None, falls back to the LLM_PROVIDER env var."
        ),
    )
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class AgentCreate(BaseModel):
    """Request body for creating an agent."""

    workspace_id: UUID
    name: str = "Data Assistant"
    model_version: str = "gpt-4o"
    system_instructions: str | None = None
    config: AgentConfig = Field(default_factory=AgentConfig)
    is_public_to_team: bool = True


class AgentUpdate(BaseModel):
    """Request body for updating an agent (all fields optional)."""

    name: str | None = None
    model_version: str | None = None
    system_instructions: str | None = None
    config: AgentConfig | None = None
    is_public_to_team: bool | None = None
