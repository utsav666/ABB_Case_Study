from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.settings_db.exceptions import (
    SettingsDatabaseConfigurationError,
    SettingsDatabaseQueryError,
)
from app.settings_db.models import DBConnectionCreate, DBConnectionValidationRequest
from app.settings_db.repositories import DBConnectionRepository, SettingsUserRepository
from app.settings_db.views import (
    build_db_connection_response,
    build_db_connection_validation_response,
    build_settings_user_response,
)
from app.target_db.connection_profiles import get_supported_database_engines
from app.target_db.connection_validation import validate_target_connection_settings

ALLOWED_ENGINES = set(get_supported_database_engines())


class SettingsUserController:
    """Controller for settings.users operations."""

    def __init__(self, repository: SettingsUserRepository | None = None) -> None:
        self._repository = repository or SettingsUserRepository()

    def upsert_oauth_user(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        """Create or refresh a settings user from OAuth profile details.

        Args:
            user_payload: OAuth user profile payload.

        Returns:
            dict[str, Any]: User payload enriched with persisted settings user fields.
        """
        email = str(user_payload.get("email", "")).strip().lower()
        if not email:
            return user_payload

        full_name = str(user_payload.get("name", "")).strip() or None
        avatar_url = str(user_payload.get("picture", "")).strip() or None

        try:
            persisted_row = self._repository.upsert_user(email, full_name, avatar_url)
        except SettingsDatabaseConfigurationError:
            raise
        except SettingsDatabaseQueryError as error:
            raise HTTPException(
                status_code=500,
                detail=str(error),
            ) from error

        if persisted_row is None:
            raise ValueError("Settings user upsert did not return a row.")

        return build_settings_user_response(
            user_payload,
            persisted_row,
            full_name,
            avatar_url,
        )


class DBConnectionController:
    """Controller for settings.db_connections operations."""

    def __init__(self, repository: DBConnectionRepository | None = None) -> None:
        self._repository = repository or DBConnectionRepository()

    def create_connection(self, payload: DBConnectionCreate) -> dict[str, Any]:
        """Create a settings DB connection record.

        Args:
            payload: Incoming connection payload.

        Returns:
            dict[str, Any]: Serialized created connection payload.
        """
        self.validate_connection(payload)
        self._validate_engine(payload.engine)
        try:
            persisted_row = self._repository.insert_connection(payload)
        except SettingsDatabaseConfigurationError:
            raise
        except SettingsDatabaseQueryError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        if persisted_row is None:
            raise HTTPException(status_code=500, detail="Insert did not return a row.")

        return build_db_connection_response(persisted_row)

    def validate_connection(
        self,
        payload: DBConnectionValidationRequest | DBConnectionCreate,
    ) -> dict[str, Any]:
        """Validate a connection payload without persisting it."""
        self._validate_engine(payload.engine)
        validation_result = validate_target_connection_settings(payload.model_dump())
        return build_db_connection_validation_response(validation_result)

    def get_connection_credentials(self, connection_id: str) -> dict[str, Any]:
        """Fetch stored credentials for a connection from the settings database.

        Args:
            connection_id: Stored connection identifier.

        Returns:
            dict[str, Any]: Stored connection row.
        """
        try:
            persisted_row = self._repository.get_connection_by_id(connection_id)
        except SettingsDatabaseConfigurationError:
            raise
        except SettingsDatabaseQueryError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        if persisted_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Connection '{connection_id}' not found.",
            )

        return persisted_row

    def _validate_engine(self, engine: str) -> None:
        """Validate the requested engine value.

        Args:
            engine: Requested engine name.
        """
        if engine in ALLOWED_ENGINES:
            return

        allowed_engines = ", ".join(sorted(ALLOWED_ENGINES))
        raise HTTPException(
            status_code=422,
            detail=f"Invalid engine '{engine}'. Must be one of: {allowed_engines}.",
        )


def upsert_oauth_user(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Persist or refresh a settings user from an OAuth payload.

    Args:
        user_payload: OAuth user payload.

    Returns:
        dict[str, Any]: Enriched user payload.
    """
    return SettingsUserController().upsert_oauth_user(user_payload)


def create_db_connection(payload: DBConnectionCreate) -> dict[str, Any]:
    """Create a new settings DB connection record.

    Args:
        payload: Connection creation payload.

    Returns:
        dict[str, Any]: Serialized created connection payload.
    """
    return DBConnectionController().create_connection(payload)


def validate_db_connection(
    payload: DBConnectionValidationRequest | DBConnectionCreate,
) -> dict[str, Any]:
    """Validate a database connection payload."""
    return DBConnectionController().validate_connection(payload)


def get_supported_connection_engines() -> list[str]:
    """Return supported target database engines for the settings UI."""
    return get_supported_database_engines()


def get_settings_connection_credentials(connection_id: str) -> dict[str, Any]:
    """Fetch a stored settings DB connection row.

    Args:
        connection_id: Stored connection identifier.

    Returns:
        dict[str, Any]: Stored connection row.
    """
    return DBConnectionController().get_connection_credentials(connection_id)
