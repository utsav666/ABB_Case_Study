from __future__ import annotations

import os

DEFAULT_CHAT_REQUEST_TIMEOUT_SECONDS = 25.0
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 12.0
DEFAULT_DB_CONNECT_TIMEOUT_SECONDS = 5
DEFAULT_DB_STATEMENT_TIMEOUT_MS = 8000


def _parse_positive_float(raw_value: str, default_value: float) -> float:
    """Parse a positive float with fallback."""
    try:
        parsed_value = float(raw_value.strip())
    except (AttributeError, ValueError):
        return default_value

    if parsed_value <= 0:
        return default_value

    return parsed_value


def _parse_positive_int(raw_value: str, default_value: int) -> int:
    """Parse a positive integer with fallback."""
    try:
        parsed_value = int(raw_value.strip())
    except (AttributeError, ValueError):
        return default_value

    if parsed_value <= 0:
        return default_value

    return parsed_value


def get_chat_request_timeout_seconds() -> float:
    """Return max allowed chat processing duration."""
    return _parse_positive_float(
        os.getenv(
            "CHAT_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_CHAT_REQUEST_TIMEOUT_SECONDS),
        ),
        DEFAULT_CHAT_REQUEST_TIMEOUT_SECONDS,
    )


def get_llm_request_timeout_seconds() -> float:
    """Return max allowed duration for each LLM request."""
    return _parse_positive_float(
        os.getenv(
            "LLM_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS),
        ),
        DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    )


def get_db_connect_timeout_seconds() -> int:
    """Return max DB connection setup duration in seconds."""
    return _parse_positive_int(
        os.getenv(
            "DB_CONNECT_TIMEOUT_SECONDS",
            str(DEFAULT_DB_CONNECT_TIMEOUT_SECONDS),
        ),
        DEFAULT_DB_CONNECT_TIMEOUT_SECONDS,
    )


def get_db_statement_timeout_ms() -> int:
    """Return max DB statement duration in milliseconds."""
    return _parse_positive_int(
        os.getenv(
            "DB_STATEMENT_TIMEOUT_MS",
            str(DEFAULT_DB_STATEMENT_TIMEOUT_MS),
        ),
        DEFAULT_DB_STATEMENT_TIMEOUT_MS,
    )
