from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.target_db.connection_profiles import (
    get_default_port,
    normalize_database_engine,
)

CONNECTION_TYPE_DATABASE = "database"
CONNECTION_TYPE_GOOGLE_ANALYTICS = "google_analytics"


class DBConnectionInput(BaseModel):
    """Shared connection fields from the settings UI.

    Supports two connection types via the `connection_type` discriminator:
    a SQL database ('database') or Google Analytics ('google_analytics').
    """

    name: str = Field(min_length=1, description="Display name for this connection.")
    connection_type: Literal["database", "google_analytics"] = Field(
        default=CONNECTION_TYPE_DATABASE,
        description="Connection kind: 'database' or 'google_analytics'.",
    )

    # SQL database fields (required only when connection_type == 'database').
    engine: str = Field(default="", description="Supported database engine identifier.")
    db_host: str = Field(default="", description="Database host address.")
    db_port: int | None = Field(
        default=None, ge=0, le=65535, description="Database port."
    )
    db_name: str = Field(default="", description="Database name or SQLite file path.")
    db_user: str = Field(default="", description="Database user.")
    db_password: str = Field(default="", description="Database password.")
    db_schema: str = Field(default="public", description="Default schema name.")
    use_ssl: bool = Field(default=False, description="Whether to require SSL/TLS.")
    extra_options: str = Field(
        default="",
        description="Optional driver-specific key=value pairs, one per line.",
    )

    # Google Analytics (GA4) fields (required only when type == 'google_analytics').
    ga_property_id: str = Field(
        default="", description="GA4 property id, e.g. 'properties/123456789'."
    )
    ga_service_account_json: str = Field(
        default="", description="Google service-account JSON key (full contents)."
    )

    @model_validator(mode="after")
    def normalize_fields(self) -> "DBConnectionInput":
        """Normalize and validate required fields per connection type."""
        if self.connection_type == CONNECTION_TYPE_GOOGLE_ANALYTICS:
            return self._validate_google_analytics()
        return self._validate_database()

    def _validate_google_analytics(self) -> "DBConnectionInput":
        """Validate the Google Analytics branch."""
        self.ga_property_id = self.ga_property_id.strip()
        self.ga_service_account_json = self.ga_service_account_json.strip()
        if not self.ga_property_id:
            raise ValueError("GA4 property ID is required.")
        if not self.ga_service_account_json:
            raise ValueError("Service account JSON is required.")
        # Clear SQL fields so they aren't persisted for a GA connection.
        self.engine = ""
        self.db_host = ""
        self.db_port = None
        self.db_name = ""
        self.db_user = ""
        self.db_password = ""
        return self

    def _validate_database(self) -> "DBConnectionInput":
        """Validate the SQL database branch (original behavior)."""
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
