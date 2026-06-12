"""Build GA4 Data API clients from stored service-account credentials.

Clients are cached per connection id (the credentials don't change unless the
connection is edited). The GA4 Data API is a Google gRPC service reached over
the network — analogous to the cross-network customer DB connections.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# GA4 read-only scope.
GA_READONLY_SCOPE = "https://www.googleapis.com/auth/analytics.readonly"

# Cache of (connection_id -> BetaAnalyticsDataClient). The client holds the
# resolved credentials; rebuild only when the connection changes.
_CLIENT_CACHE: dict[str, Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def parse_service_account_json(raw: str) -> dict[str, Any]:
    """Parse a service-account JSON key string into a dict.

    Raises:
        ValueError: if the JSON is malformed or missing required fields.
    """
    if not raw or not raw.strip():
        raise ValueError("Service account JSON is required.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Service account JSON is not valid JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise ValueError("Service account JSON must be a JSON object.")

    # Minimal sanity check for a Google service-account key.
    if parsed.get("type") != "service_account":
        raise ValueError(
            "JSON does not look like a Google service-account key "
            "(missing \"type\": \"service_account\")."
        )
    for required in ("client_email", "private_key", "token_uri"):
        if not parsed.get(required):
            raise ValueError(f"Service account JSON is missing '{required}'.")

    return parsed


def build_ga_client(service_account_json: str):
    """Build a BetaAnalyticsDataClient from a service-account JSON string.

    Imported lazily so the rest of the app loads even if google libraries are
    not installed in a given environment.
    """
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.oauth2 import service_account

    info = parse_service_account_json(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=[GA_READONLY_SCOPE]
    )
    return BetaAnalyticsDataClient(credentials=credentials)


def get_ga_client_for_connection(connection_id: str):
    """Return a cached GA4 client for a stored connection.

    Resolves the connection's service-account JSON from the settings DB.
    Returns None if the connection has no GA credentials.
    """
    cached = _CLIENT_CACHE.get(connection_id)
    if cached is not None:
        return cached

    # Imported here to avoid a circular import at module load time.
    from app.settings_db.controllers import get_settings_connection_credentials

    credentials = get_settings_connection_credentials(connection_id)
    service_account_json = credentials.get("ga_service_account_json")
    if not service_account_json:
        logger.warning("Connection %s has no GA service-account JSON", connection_id)
        return None

    with _CLIENT_CACHE_LOCK:
        cached = _CLIENT_CACHE.get(connection_id)
        if cached is None:
            cached = build_ga_client(service_account_json)
            _CLIENT_CACHE[connection_id] = cached
    return cached


def invalidate_ga_client(connection_id: str) -> None:
    """Drop a cached client (call when a connection's credentials change)."""
    with _CLIENT_CACHE_LOCK:
        _CLIENT_CACHE.pop(connection_id, None)


def normalize_property_id(property_id: str) -> str:
    """Return the property id in the API's expected 'properties/<id>' form."""
    pid = (property_id or "").strip()
    if not pid:
        return ""
    if pid.startswith("properties/"):
        return pid
    return f"properties/{pid}"
