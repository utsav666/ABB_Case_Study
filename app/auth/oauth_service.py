from __future__ import annotations

import base64
import contextlib
import json
import logging
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode

import requests
from fastapi import HTTPException, Request

from app.auth.oauth_config import (
    DEFAULT_OAUTH_CALLBACK_PATH,
    DEFAULT_OAUTH_SESSION_REFRESH_WINDOW_SECONDS,
    get_oauth_config,
)
from app.settings_db.exceptions import SettingsDatabaseConfigurationError
from app.utils.settings_users import upsert_oauth_user

DEFAULT_OAUTH_SESSION_PATH = "/auth/session"

logger = logging.getLogger(__name__)

_WELL_KNOWN_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def get_oauth_http_timeout_seconds() -> tuple[float, float]:
    """Return connect and read timeouts for OAuth HTTP requests."""
    oauth_config = get_oauth_config()
    return (
        oauth_config.http_connect_timeout_seconds,
        oauth_config.http_read_timeout_seconds,
    )


def _log_timed_request(method: str, url: str, started_at: float) -> None:
    """Log request duration for OAuth network calls."""
    duration_ms = (time.monotonic() - started_at) * 1000
    logger.info("OAuth %s %s completed in %.2f ms", method, url, duration_ms)


def get_oauth_well_known_url() -> str:
    """Return configured OpenID discovery URL from environment settings."""
    return get_oauth_config().well_known_url


def get_oauth_well_known_payload() -> dict[str, Any]:
    """Return provider metadata from OpenID discovery endpoint."""
    well_known_url = get_oauth_well_known_url()
    if not well_known_url:
        return {}

    cached_payload = _WELL_KNOWN_CACHE.get(well_known_url)
    current_time = time.monotonic()
    cache_ttl_seconds = get_oauth_config().well_known_cache_ttl_seconds
    if cached_payload is not None:
        cached_until, payload = cached_payload
        if current_time < cached_until:
            return payload

    timeout_seconds = get_oauth_http_timeout_seconds()
    started_at = time.monotonic()

    try:
        metadata_response = requests.get(
            well_known_url,
            timeout=timeout_seconds,
        )
        _log_timed_request("GET", well_known_url, started_at)
        metadata_response.raise_for_status()
        metadata_payload = metadata_response.json()
    except (requests.RequestException, ValueError):
        _log_timed_request("GET", well_known_url, started_at)
        return {}

    if not isinstance(metadata_payload, dict):
        return {}

    _WELL_KNOWN_CACHE[well_known_url] = (
        current_time + cache_ttl_seconds,
        metadata_payload,
    )

    return metadata_payload


def get_well_known_endpoint(endpoint_name: str) -> str:
    """Read endpoint URL from OpenID discovery metadata."""
    metadata_payload = get_oauth_well_known_payload()
    endpoint_value = metadata_payload.get(endpoint_name)
    if not isinstance(endpoint_value, str):
        return ""

    return endpoint_value.strip()


def get_oauth_redirect_uri(request: Request) -> str:
    """Return the configured callback URL for OAuth login."""
    configured_redirect_uri = get_oauth_config().redirect_uri
    if configured_redirect_uri:
        return configured_redirect_uri

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}{DEFAULT_OAUTH_CALLBACK_PATH}"


def build_oauth_login_payload(request: Request) -> dict[str, Any]:
    """Build a frontend-ready OAuth login URL payload."""
    oauth_config = get_oauth_config()
    provider_name = oauth_config.provider_name
    authorize_url = get_oauth_authorize_url()
    scope = oauth_config.scope
    client_id = oauth_config.client_id

    query_params: dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": get_oauth_redirect_uri(request),
        "response_type": "code",
        "scope": scope,
        "state": oauth_config.state,
    }

    access_type = oauth_config.access_type
    if access_type:
        query_params["access_type"] = access_type

    include_granted_scopes = oauth_config.include_granted_scopes
    if include_granted_scopes:
        query_params["include_granted_scopes"] = include_granted_scopes

    login_url = ""
    if authorize_url:
        login_url = f"{authorize_url}?{urlencode(query_params)}"

    return {
        "provider": provider_name,
        "configured": bool(client_id and authorize_url),
        "login_url": login_url,
        "redirect_uri": query_params["redirect_uri"],
        "scopes": scope.split(),
    }


def get_oauth_provider_name() -> str:
    """Return provider name used by auth responses."""
    return get_oauth_config().provider_name


def get_oauth_token_url() -> str:
    """Return the OAuth token endpoint from OpenID discovery."""
    return get_well_known_endpoint("token_endpoint")


def get_oauth_userinfo_url() -> str:
    """Return the OAuth userinfo endpoint from OpenID discovery."""
    return get_well_known_endpoint("userinfo_endpoint")


def get_oauth_authorize_url() -> str:
    """Return the OAuth authorization endpoint from OpenID discovery."""
    return get_well_known_endpoint("authorization_endpoint")


def can_exchange_oauth_code() -> bool:
    """Return whether the backend can exchange the authorization code."""
    oauth_config = get_oauth_config()
    client_id = oauth_config.client_id
    client_secret = oauth_config.client_secret
    return bool(client_id and client_secret)


def exchange_oauth_code_for_token(code: str, request: Request) -> dict[str, Any]:
    """Exchange an OAuth authorization code for tokens."""
    oauth_config = get_oauth_config()
    client_id = oauth_config.client_id
    client_secret = oauth_config.client_secret
    if not client_id or not client_secret:
        raise ValueError(
            "Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET to enable backend authentication."
        )

    token_url = get_oauth_token_url()
    if not token_url:
        raise ValueError("Set OAUTH_WELL_KNOWN_URL to enable OAuth token exchange.")

    timeout_seconds = get_oauth_http_timeout_seconds()
    started_at = time.monotonic()

    try:
        token_response = requests.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": get_oauth_redirect_uri(request),
            },
            timeout=timeout_seconds,
        )
    finally:
        _log_timed_request("POST", token_url, started_at)
    token_response.raise_for_status()
    token_payload = token_response.json()
    if not isinstance(token_payload, dict):
        raise ValueError("OAuth token response must be a JSON object.")

    return token_payload


def refresh_oauth_token(refresh_token: str, request: Request) -> dict[str, Any]:
    """Exchange a refresh token for a new access token payload."""
    oauth_config = get_oauth_config()
    client_id = oauth_config.client_id
    client_secret = oauth_config.client_secret
    if not client_id or not client_secret:
        raise ValueError(
            "Set OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET to enable token refresh."
        )

    token_url = get_oauth_token_url()
    if not token_url:
        raise ValueError("Set OAUTH_WELL_KNOWN_URL to enable token refresh.")

    timeout_seconds = get_oauth_http_timeout_seconds()
    started_at = time.monotonic()

    try:
        token_response = requests.post(
            token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "redirect_uri": get_oauth_redirect_uri(request),
            },
            timeout=timeout_seconds,
        )
    finally:
        _log_timed_request("POST", token_url, started_at)
    token_response.raise_for_status()
    token_payload = token_response.json()
    if not isinstance(token_payload, dict):
        raise ValueError("OAuth refresh response must be a JSON object.")

    return token_payload


def fetch_oauth_user_details(access_token: str) -> dict[str, Any]:
    """Fetch the authenticated user's profile from the OAuth provider."""
    if not access_token.strip():
        return {}

    userinfo_url = get_oauth_userinfo_url()
    if not userinfo_url:
        raise ValueError("Set OAUTH_WELL_KNOWN_URL to enable user profile fetch.")

    timeout_seconds = get_oauth_http_timeout_seconds()
    started_at = time.monotonic()

    try:
        user_response = requests.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=timeout_seconds,
        )
    finally:
        _log_timed_request("GET", userinfo_url, started_at)
    user_response.raise_for_status()
    user_payload = user_response.json()
    if not isinstance(user_payload, dict):
        raise ValueError("OAuth user info response must be a JSON object.")

    return user_payload


def persist_oauth_user_details(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Persist an OAuth user into settings.users and return the enriched payload.

    Args:
        user_payload: OAuth user profile payload.

    Returns:
        dict[str, Any]: Persisted user payload.
    """
    try:
        return upsert_oauth_user(user_payload)
    except (HTTPException, ModuleNotFoundError, SettingsDatabaseConfigurationError):
        logger.warning(
            "Skipping OAuth user persistence because the settings database is unavailable."
        )
        return user_payload


def build_oauth_token_payload(
    token_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the safe token fields to send back to the frontend."""
    if not token_payload:
        return None

    return {
        "access_token": token_payload.get("access_token"),
        "token_type": token_payload.get("token_type"),
        "expires_in": token_payload.get("expires_in"),
        "expires_at": token_payload.get("expires_at"),
        "scope": token_payload.get("scope"),
        "id_token": token_payload.get("id_token"),
        "refresh_token": token_payload.get("refresh_token"),
    }


def build_oauth_authentication_response(
    provider_name: str,
    message: str,
    *,
    status: str = "success",
    code: str | None = None,
    authorization_code: str | None = None,
    state: str | None = None,
    authenticated: bool = False,
    user_payload: dict[str, Any] | None = None,
    token_payload: dict[str, Any] | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> dict[str, Any]:
    """Build a consistent auth response payload across callback/session endpoints."""
    access_token = None
    if token_payload:
        access_token = token_payload.get("access_token")

    return {
        "status": status,
        "provider": provider_name,
        "code": code,
        "authorization_code": authorization_code,
        "state": state,
        "authenticated": authenticated,
        "user": user_payload,
        "token": build_oauth_token_payload(token_payload),
        "error": error,
        "error_description": error_description,
        "message": message,
        "access_token": access_token,
    }


def decode_jwt_payload(token_value: str) -> dict[str, Any]:
    """Decode JWT payload without signature verification to read expiry metadata."""
    token_parts = token_value.split(".")
    if len(token_parts) != 3:
        return {}

    payload_segment = token_parts[1]
    missing_padding = (-len(payload_segment)) % 4
    padded_segment = payload_segment + ("=" * missing_padding)

    try:
        decoded_payload = base64.urlsafe_b64decode(padded_segment.encode("utf-8"))
        parsed_payload = json.loads(decoded_payload.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}

    if not isinstance(parsed_payload, dict):
        return {}

    return parsed_payload


def parse_epoch_seconds(raw_value: Any) -> int | None:
    """Parse a UNIX epoch timestamp from a value when possible."""
    if raw_value is None:
        return None

    with contextlib.suppress(TypeError, ValueError):
        parsed_value = int(float(str(raw_value).strip()))
        if parsed_value > 0:
            return parsed_value

    return None


def parse_duration_seconds(raw_value: Any) -> int | None:
    """Parse a duration in seconds from a value when possible."""
    if raw_value is None:
        return None

    with contextlib.suppress(TypeError, ValueError):
        parsed_value = int(float(str(raw_value).strip()))
        if parsed_value >= 0:
            return parsed_value

    return None


def get_seconds_until_expiry(
    access_token: str,
    expires_at: Any,
    expires_in: Any,
) -> int | None:
    """Return remaining token lifetime in seconds when expiry metadata is available."""
    now_epoch_seconds = int(time.time())

    expiry_epoch_seconds = parse_epoch_seconds(expires_at)
    if expiry_epoch_seconds is not None:
        return max(0, expiry_epoch_seconds - now_epoch_seconds)

    remaining_duration_seconds = parse_duration_seconds(expires_in)
    if remaining_duration_seconds is not None:
        return max(0, remaining_duration_seconds)

    jwt_payload = decode_jwt_payload(access_token)
    jwt_expiry_epoch_seconds = parse_epoch_seconds(jwt_payload.get("exp"))
    if jwt_expiry_epoch_seconds is not None:
        return max(0, jwt_expiry_epoch_seconds - now_epoch_seconds)

    return None


def get_refresh_window_seconds() -> int:
    """Return the threshold window used to refresh near-expiry tokens."""
    refresh_window_seconds = get_oauth_config().refresh_window_seconds
    parsed_window = parse_duration_seconds(str(refresh_window_seconds))
    if parsed_window is None:
        return DEFAULT_OAUTH_SESSION_REFRESH_WINDOW_SECONDS

    return parsed_window


def get_bearer_token(request: Request) -> str | None:
    """Return bearer token from Authorization header if present."""
    authorization_header = request.headers.get("authorization", "").strip()
    if not authorization_header.lower().startswith("bearer "):
        return None

    token_value = authorization_header[7:].strip()
    return token_value or None


def extract_token_field(
    source_payload: dict[str, Any],
    field_name: str,
) -> Any:
    """Read a token field from top-level payload or nested token object."""
    direct_value = source_payload.get(field_name)
    if direct_value not in (None, ""):
        return direct_value

    nested_token = source_payload.get("token")
    if isinstance(nested_token, dict):
        nested_value = nested_token.get(field_name)
        if nested_value not in (None, ""):
            return nested_value

    return None


async def read_request_payload(request: Request) -> Any:
    """Read JSON, form, or plain-text request bodies safely."""
    content_type = request.headers.get("content-type", "").lower()

    try:
        if "application/json" in content_type:
            return await request.json()

        raw_body = await request.body()
    except (ValueError, UnicodeDecodeError):
        return {}

    if not raw_body:
        return {}

    if "application/x-www-form-urlencoded" in content_type:
        decoded_body = raw_body.decode("utf-8", errors="ignore")
        return {
            key: value
            for key, value in parse_qsl(decoded_body, keep_blank_values=False)
            if value.strip()
        }

    if "multipart/form-data" in content_type:
        with contextlib.suppress(AssertionError, ValueError):
            return dict(await request.form())

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body.decode("utf-8", errors="ignore")


async def read_oauth_callback_parameters(request: Request) -> dict[str, str]:
    """Read OAuth callback values from query string or request body."""
    callback_values: dict[str, str] = {
        key: value.strip()
        for key, value in request.query_params.items()
        if isinstance(value, str) and value.strip()
    }

    payload = await read_request_payload(request)
    if not isinstance(payload, dict):
        return callback_values

    for key, value in payload.items():
        if value is None:
            continue

        normalized_value = str(value).strip()
        if normalized_value:
            callback_values[key] = normalized_value

    return callback_values


async def read_auth_session_parameters(request: Request) -> dict[str, Any]:
    """Read /auth/session fields from query params and request body."""
    session_values: dict[str, Any] = dict(request.query_params)
    payload = await read_request_payload(request)
    if isinstance(payload, dict):
        session_values.update(payload)

    return session_values
