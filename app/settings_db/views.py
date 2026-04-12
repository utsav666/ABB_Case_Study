from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


def format_datetime_value(raw_value: Any) -> str | None:
    """Convert a datetime-like value to a string when available.

    Args:
        raw_value: Value returned from the database.

    Returns:
        str | None: ISO formatted timestamp or None.
    """
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        return raw_value.isoformat()

    return str(raw_value)


def build_settings_user_response(
    user_payload: dict[str, Any],
    persisted_row: Mapping[str, Any],
    fallback_name: str | None,
    fallback_avatar_url: str | None,
) -> dict[str, Any]:
    """Build the API payload for a persisted settings user.

    Args:
        user_payload: Original OAuth payload.
        persisted_row: Persisted database row.
        fallback_name: Provider name when the database has none.
        fallback_avatar_url: Provider avatar URL when the database has none.

    Returns:
        dict[str, Any]: Response payload for the authenticated user.
    """
    persisted_name = persisted_row.get("full_name") or fallback_name
    persisted_avatar_url = persisted_row.get("avatar_url") or fallback_avatar_url

    return {
        **user_payload,
        "id": str(persisted_row["id"]),
        "email": persisted_row["email"],
        "name": persisted_name,
        "picture": persisted_avatar_url,
        "full_name": persisted_row.get("full_name"),
        "avatar_url": persisted_row.get("avatar_url"),
        "created_at": format_datetime_value(persisted_row.get("created_at")),
        "updated_at": format_datetime_value(persisted_row.get("updated_at")),
        "last_login_at": format_datetime_value(persisted_row.get("last_login_at")),
        "is_active": persisted_row.get("is_active"),
    }


def build_db_connection_response(persisted_row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the API payload for a persisted database connection.

    Args:
        persisted_row: Persisted database row.

    Returns:
        dict[str, Any]: Serialized connection payload.
    """
    return {
        "status": "created",
        "connection": {
            "id": str(persisted_row["id"]),
            "workspace_id": str(persisted_row["workspace_id"]),
            "name": persisted_row["name"],
            "engine": persisted_row["engine"],
            "db_host": persisted_row["db_host"],
            "db_port": persisted_row["db_port"],
            "db_name": persisted_row["db_name"],
            "db_user": persisted_row["db_user"],
            "is_healthy": persisted_row["is_healthy"],
            "last_synced_at": format_datetime_value(
                persisted_row.get("last_synced_at")
            ),
            "created_at": format_datetime_value(persisted_row.get("created_at")),
        },
    }
