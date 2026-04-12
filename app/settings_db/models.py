from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DBConnectionCreate(BaseModel):
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
            }
        }
    )

    workspace_id: UUID = Field(description="UUID of the parent workspace.")
    name: str = Field(min_length=1, description="Display name for this connection.")
    engine: str = Field(
        description="Database engine: postgres, snowflake, or bigquery."
    )
    db_host: str = Field(min_length=1, description="Database host address.")
    db_port: int = Field(default=5432, ge=1, le=65535, description="Database port.")
    db_name: str = Field(min_length=1, description="Database name.")
    db_user: str = Field(min_length=1, description="Database user.")
    db_password: str = Field(min_length=1, description="Database password.")
