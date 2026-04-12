from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_OAUTH_SCOPE = "openid profile email"
DEFAULT_OAUTH_CALLBACK_PATH = "/auth/callback"
DEFAULT_OAUTH_SESSION_PATH = "/auth/session"
DEFAULT_OAUTH_SESSION_REFRESH_WINDOW_SECONDS = 300
DEFAULT_OAUTH_WELL_KNOWN_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class OAuthConfig:
    """OAuth configuration loaded from environment variables."""

    provider_name: str
    client_id: str
    client_secret: str
    scope: str
    state: str
    access_type: str
    include_granted_scopes: str
    redirect_uri: str
    well_known_url: str
    refresh_window_seconds: int


def _parse_positive_int(raw_value: str, default_value: int) -> int:
    """Parse a non-negative integer with fallback."""
    try:
        parsed_value = int(raw_value.strip())
    except (AttributeError, ValueError):
        return default_value

    if parsed_value < 0:
        return default_value

    return parsed_value


def get_oauth_config() -> OAuthConfig:
    """Return OAuth settings from environment variables."""
    return OAuthConfig(
        provider_name=os.getenv("OAUTH_PROVIDER", "oauth").strip() or "oauth",
        client_id=os.getenv("OAUTH_CLIENT_ID", "").strip(),
        client_secret=os.getenv("OAUTH_CLIENT_SECRET", "").strip(),
        scope=os.getenv("OAUTH_SCOPE", DEFAULT_OAUTH_SCOPE).strip()
        or DEFAULT_OAUTH_SCOPE,
        state=os.getenv("OAUTH_STATE", "login").strip() or "login",
        access_type=os.getenv("OAUTH_ACCESS_TYPE", "").strip(),
        include_granted_scopes=os.getenv(
            "OAUTH_INCLUDE_GRANTED_SCOPES",
            "",
        ).strip(),
        redirect_uri=os.getenv("OAUTH_REDIRECT_URI", "").strip(),
        well_known_url=os.getenv("OAUTH_WELL_KNOWN_URL", "").strip(),
        refresh_window_seconds=_parse_positive_int(
            os.getenv(
                "OAUTH_SESSION_REFRESH_WINDOW_SECONDS",
                str(DEFAULT_OAUTH_SESSION_REFRESH_WINDOW_SECONDS),
            ),
            DEFAULT_OAUTH_SESSION_REFRESH_WINDOW_SECONDS,
        ),
    )