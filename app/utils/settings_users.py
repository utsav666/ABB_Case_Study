from __future__ import annotations

from typing import Any

from app.settings_db.controllers import upsert_oauth_user as persist_settings_user


def upsert_oauth_user(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Create or refresh a settings user from OAuth profile details.

    Args:
        user_payload: OAuth user profile payload.

    Returns:
        dict[str, Any]: User payload enriched with persisted settings user fields.
    """
    return persist_settings_user(user_payload)
