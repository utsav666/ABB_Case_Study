"""Tests for request-scoped logging context and formatters."""

from __future__ import annotations

import json
import logging

import pytest

from app.utils.logging_context import (
    ContextFilter,
    HumanFormatter,
    JsonFormatter,
    bind_log_context,
    get_log_context,
    log_duration,
    new_request_id,
)


def _make_record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestBindLogContext:
    def test_binds_and_reads_back(self) -> None:
        bind_log_context(request_id="abc123", connection_id="conn-1")

        context = get_log_context()

        assert context["request_id"] == "abc123"
        assert context["connection_id"] == "conn-1"

    def test_partial_binding_leaves_other_fields_intact(self) -> None:
        """The handler binds request/workspace up front and the connection later."""
        bind_log_context(request_id="req-1", thread_id="thread-1")
        bind_log_context(connection_id="conn-2")

        context = get_log_context()

        assert context["request_id"] == "req-1"
        assert context["thread_id"] == "thread-1"
        assert context["connection_id"] == "conn-2"

    def test_new_request_id_is_short_and_unique(self) -> None:
        first, second = new_request_id(), new_request_id()

        assert first != second
        assert len(first) == 12


class TestContextFilter:
    def test_injects_context_onto_record(self) -> None:
        bind_log_context(request_id="req-9", connection_id="conn-9")
        record = _make_record()

        assert ContextFilter().filter(record) is True
        assert record.request_id == "req-9"
        assert record.connection_id == "conn-9"

    def test_does_not_clobber_explicit_extra(self) -> None:
        """A value passed via extra= is more specific than the ambient context."""
        bind_log_context(connection_id="ambient")
        record = _make_record(connection_id="explicit")

        ContextFilter().filter(record)

        assert record.connection_id == "explicit"


class TestHumanFormatter:
    def test_appends_bound_context(self) -> None:
        bind_log_context(request_id="req-1")
        record = _make_record("schema loaded")
        ContextFilter().filter(record)

        output = HumanFormatter().format(record)

        assert "schema loaded" in output
        assert "request_id=req-1" in output

    def test_omits_empty_context_to_keep_local_runs_clean(self) -> None:
        record = _make_record("plain message")
        ContextFilter().filter(record)

        output = HumanFormatter().format(record)

        assert output.endswith("plain message")
        assert "[" not in output


class TestJsonFormatter:
    def test_emits_one_valid_json_object(self) -> None:
        bind_log_context(request_id="req-2", connection_id="conn-2")
        record = _make_record("sql executed")
        ContextFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert payload["message"] == "sql executed"
        assert payload["level"] == "INFO"
        assert payload["request_id"] == "req-2"
        assert payload["connection_id"] == "conn-2"

    def test_promotes_extra_fields_to_top_level(self) -> None:
        """So CloudWatch Insights can filter on row_count directly."""
        record = _make_record("done", row_count=0, duration_ms=12.5)
        ContextFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert payload["row_count"] == 0
        assert payload["duration_ms"] == 12.5

    def test_unset_context_fields_are_omitted(self) -> None:
        record = _make_record("no context")
        ContextFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert "request_id" not in payload

    def test_non_serialisable_values_degrade_to_string(self) -> None:
        """A logging call must never raise because of an odd extra= value."""
        from decimal import Decimal

        record = _make_record("money", amount=Decimal("1.50"))
        ContextFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert payload["amount"] == "1.50"

    def test_includes_exception_text(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="app.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        ContextFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert "ValueError: boom" in payload["exception"]


class TestLogDuration:
    def test_logs_duration_on_success(self, caplog) -> None:
        logger = logging.getLogger("app.test.duration")

        with caplog.at_level(logging.INFO, logger="app.test.duration"):
            with log_duration(logger, "router_llm"):
                pass

        assert "STEP router_llm took" in caplog.text

    def test_logs_duration_when_the_step_raises(self, caplog) -> None:
        """A slow step that then fails is exactly what you want in the log."""
        logger = logging.getLogger("app.test.duration")

        with caplog.at_level(logging.ERROR, logger="app.test.duration"):
            with pytest.raises(ValueError):
                with log_duration(logger, "sql_exec"):
                    raise ValueError("query failed")

        assert "STEP sql_exec failed after" in caplog.text

    def test_reraises_the_original_exception(self) -> None:
        logger = logging.getLogger("app.test.duration")

        with pytest.raises(ValueError, match="original"):
            with log_duration(logger, "step"):
                raise ValueError("original")
