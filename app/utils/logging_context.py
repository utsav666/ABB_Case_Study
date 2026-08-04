"""Request-scoped logging context and formatters.

Why this exists
---------------
The app logs plenty, but every line was anonymous: with two questions in
flight you could not tell which request a line belonged to, and *no* line
recorded which database connection was in use. That second gap let a real bug
hide — schema questions answered from a bundled movie-database fixture while
the user had a Sales connection selected, with nothing in the logs to show it.

What it provides
----------------
* ``bind_log_context()`` — stash request-scoped ids (thread, connection, …) in
  contextvars so they reach every log record without threading a logger
  through every function signature. contextvars are the right tool here
  because they are per-task: FastAPI runs handlers concurrently, and a module
  global would leak values between requests.
* ``ContextFilter`` — injects those ids onto each LogRecord.
* ``HumanFormatter`` — readable local output with the context inline.
* ``JsonFormatter`` — one JSON object per line for CloudWatch Insights, which
  can then query on ``connection_id`` or ``thread_id`` directly.
* ``log_duration()`` — context manager that logs how long a step took, and
  logs it even when the step raises.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Request-scoped context. Defaults are empty strings (not None) so formatters
# never have to special-case missing values.
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_thread_id: ContextVar[str] = ContextVar("thread_id", default="")
_connection_id: ContextVar[str] = ContextVar("connection_id", default="")
_workspace_id: ContextVar[str] = ContextVar("workspace_id", default="")

# The record attributes we add. Kept in one place so the filter, both
# formatters, and the JSON reserved-key check stay in sync.
CONTEXT_FIELDS = ("request_id", "thread_id", "connection_id", "workspace_id")


def new_request_id() -> str:
    """Return a short id for correlating all logs from one request."""
    return uuid.uuid4().hex[:12]


def bind_log_context(
    *,
    request_id: str | None = None,
    thread_id: str | None = None,
    connection_id: str | None = None,
    workspace_id: str | None = None,
) -> None:
    """Attach ids to the current context; only non-None values are set.

    Partial binding is intentional: the HTTP handler knows the request and
    workspace up front, while the connection may only be resolved later.
    """
    if request_id is not None:
        _request_id.set(request_id)
    if thread_id is not None:
        _thread_id.set(thread_id)
    if connection_id is not None:
        _connection_id.set(connection_id)
    if workspace_id is not None:
        _workspace_id.set(workspace_id)


def get_log_context() -> dict[str, str]:
    """Return the current context as a plain dict."""
    return {
        "request_id": _request_id.get(),
        "thread_id": _thread_id.get(),
        "connection_id": _connection_id.get(),
        "workspace_id": _workspace_id.get(),
    }


class ContextFilter(logging.Filter):
    """Copy the request context onto every LogRecord.

    Implemented as a Filter rather than a LoggerAdapter so it applies to logs
    from every module — including third-party libraries — without each one
    having to opt in.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_log_context().items():
            # Don't clobber a value a caller passed via `extra=`.
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class HumanFormatter(logging.Formatter):
    """Readable single-line output with context appended when present."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Only show ids that are actually set, so local runs stay uncluttered.
        parts = [
            f"{key}={getattr(record, key)}"
            for key in CONTEXT_FIELDS
            if getattr(record, key, "")
        ]
        return f"{base}  [{' '.join(parts)}]" if parts else base


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for CloudWatch Logs Insights."""

    # LogRecord's built-ins; anything else was passed via extra= and is worth
    # emitting as a top-level field.
    _RESERVED = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in CONTEXT_FIELDS:
            value = getattr(record, key, "")
            if value:
                payload[key] = value

        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key not in CONTEXT_FIELDS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # default=str so non-serialisable values (Decimal, datetime) degrade
        # to a string instead of raising inside the logging call.
        return json.dumps(payload, default=str)


@contextmanager
def log_duration(logger: logging.Logger, step: str, **fields: Any):
    """Log how long ``step`` took, including when it raises.

    Timing on the failure path matters as much as on the success path — a slow
    step that then errors is exactly what you want to see in the log.
    """
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "STEP %s failed after %.0f ms",
            step,
            elapsed_ms,
            extra={
                "step": step,
                "duration_ms": round(elapsed_ms, 1),
                "ok": False,
                **fields,
            },
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "STEP %s took %.0f ms",
            step,
            elapsed_ms,
            extra={
                "step": step,
                "duration_ms": round(elapsed_ms, 1),
                "ok": True,
                **fields,
            },
        )
