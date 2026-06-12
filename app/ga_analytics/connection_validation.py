"""Validate a Google Analytics (GA4) connection payload.

Mirrors app/target_db/connection_validation.py: parse credentials, build a
client, and make one cheap call (get_metadata) to confirm the service account
can read the property — the GA analogue of running SELECT 1.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from fastapi import HTTPException

from app.ga_analytics.ga_client import (
    build_ga_client,
    normalize_property_id,
    parse_service_account_json,
)

logger = logging.getLogger(__name__)


def validate_ga_connection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a GA4 connection without persisting it.

    Returns {"status": "validated", "can_save": True, ...} on success;
    raises HTTPException(422) on any failure.
    """
    property_id = normalize_property_id(str(payload.get("ga_property_id", "")))
    service_account_json = str(payload.get("ga_service_account_json", ""))

    if not property_id:
        raise HTTPException(
            status_code=422,
            detail="GA4 property ID is required (e.g. 'properties/123456789').",
        )

    # Parse credentials first so malformed JSON gives a clear error.
    try:
        info = parse_service_account_json(service_account_json)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Build client + make a cheap metadata call to confirm access.
    try:
        from google.analytics.data_v1beta.types import GetMetadataRequest

        client = build_ga_client(service_account_json)
        client.get_metadata(
            GetMetadataRequest(name=f"{property_id}/metadata")
        )
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - surface the provider error
        logger.warning("GA connection validation failed: %s", error)
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not access the GA4 property. Confirm the property ID is "
                "correct and the service account "
                f"({info.get('client_email')}) has Viewer access. Error: {error}"
            ),
        ) from error

    return {
        "status": "validated",
        "can_save": True,
        "connection_type": "google_analytics",
        "ga_property_id": property_id,
        "message": "Google Analytics connection validated successfully.",
    }
