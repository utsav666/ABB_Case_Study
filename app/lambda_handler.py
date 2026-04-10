from __future__ import annotations

import os
from typing import Any

from mangum import Mangum

from app.main import app

DEFAULT_SOURCE_IP = "0.0.0.0"


def get_api_gateway_base_path() -> str:
    """Return the optional API Gateway stage prefix for FastAPI routes."""
    configured_root_path = os.getenv("FASTAPI_ROOT_PATH", "/").strip()
    return configured_root_path or "/"


mangum_handler = Mangum(
    app,
    lifespan="off",
    api_gateway_base_path=get_api_gateway_base_path(),
)


def should_normalize_api_gateway_event(event: Any) -> bool:
    """Return whether the Lambda payload looks like an API Gateway event."""
    if not isinstance(event, dict):
        return False

    event_version = str(event.get("version", "")).strip()
    return event_version in {"1.0", "2.0"} or "resource" in event or "routeKey" in event


def normalize_api_gateway_event(event: Any) -> dict[str, Any]:
    """Fill in optional API Gateway fields that Mangum expects to exist."""
    normalized_event = dict(event or {})
    if not should_normalize_api_gateway_event(normalized_event):
        return normalized_event

    normalized_event["headers"] = dict(normalized_event.get("headers") or {})
    normalized_event["isBase64Encoded"] = bool(
        normalized_event.get("isBase64Encoded", False)
    )

    request_context = dict(normalized_event.get("requestContext") or {})
    normalized_event["requestContext"] = request_context

    if str(normalized_event.get("version")) == "2.0":
        http_context = dict(request_context.get("http") or {})
        request_path = (
            normalized_event.get("rawPath") or http_context.get("path") or "/"
        )

        http_context["method"] = http_context.get("method") or "GET"
        http_context["path"] = request_path
        http_context["sourceIp"] = http_context.get("sourceIp") or DEFAULT_SOURCE_IP
        http_context["userAgent"] = http_context.get("userAgent") or normalized_event[
            "headers"
        ].get("user-agent", "")

        request_context["http"] = http_context
        normalized_event["rawPath"] = request_path
        normalized_event["rawQueryString"] = (
            normalized_event.get("rawQueryString") or ""
        )
        normalized_event["cookies"] = list(normalized_event.get("cookies") or [])
        return normalized_event

    normalized_event["path"] = normalized_event.get("path") or "/"
    normalized_event["httpMethod"] = normalized_event.get("httpMethod") or "GET"
    normalized_event["multiValueHeaders"] = dict(
        normalized_event.get("multiValueHeaders") or {}
    )
    return normalized_event


def handler(event: Any, context: Any) -> dict[str, Any]:
    """Serve the FastAPI app from AWS Lambda for API Gateway v1 and v2."""
    normalized_event = normalize_api_gateway_event(event)
    return mangum_handler(normalized_event, context)


__all__ = ["handler"]
