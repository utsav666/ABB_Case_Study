"""Shared test fixtures.

Design rule: **no test may touch a real database or network.** CI has no
settings-DB credentials and no route to a customer DB, so anything that dials
out fails there while passing on a developer machine that happens to have a
live `.env` — the worst possible split. Several tests were failing in CI for
exactly this reason.

Two guards below:

* ``_dummy_db_env`` (autouse) gives every test placeholder credentials, so code
  that only *reads* config succeeds instead of raising
  SettingsDatabaseConfigurationError.
* ``_block_network`` (autouse) fails a test that tries to open a socket, with a
  message naming the address. Without it, a missing mock shows up as a DNS
  timeout minutes later rather than an obvious error.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _dummy_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder DB credentials for every test.

    These are deliberately unroutable. They exist so config *loading* works;
    any attempt to actually connect is caught by ``_block_network``.
    """
    dummy_settings = {
        "SETTINGS_DB_HOST": "localhost",
        "SETTINGS_DB_PORT": "5432",
        "SETTINGS_DB_NAME": "test_settings",
        "SETTINGS_DB_USER": "test_user",
        "SETTINGS_DB_PASSWORD": "test_password",
        # Legacy non-prefixed names read by DBLoader.db_loader().
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_target",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        # Keep tests off any real LLM provider.
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o-mini",
        "OPENAI_API_KEY": "test-key-not-real",
    }
    for key, value in dummy_settings.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test tries to open a real socket.

    Patching ``socket.socket.connect`` catches every driver — psycopg2,
    mysql-connector, httpx — because they all end up here. The error names the
    address so the missing mock is obvious.

    Unix-domain sockets are left alone: pytest's own capture machinery and some
    local tooling use them, and they cannot reach a remote host.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple):
            # RuntimeError, not TypeError: this is a policy violation, not a bad
            # argument type. TRY004 does not apply here.
            raise RuntimeError(  # noqa: TRY004
                f"Test attempted a real network connection to {address}. "
                "Mock the client/repository instead, or pass "
                "skip_validation=True where the code performs a live probe."
            )
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def _reset_log_context() -> None:
    """Clear request-scoped log context between tests.

    contextvars are per-task, and pytest runs every test in the same task, so a
    value bound by one test is still visible to the next. That makes assertions
    about "no context bound" order-dependent.
    """
    from app.utils import logging_context

    for var in (
        logging_context._request_id,
        logging_context._thread_id,
        logging_context._connection_id,
        logging_context._workspace_id,
    ):
        var.set("")


class RecordingExecutor:
    """Stands in for SqlExec, capturing SQL and returning canned results.

    Lives here rather than in one test module so any test needing a mocked target
    DB reuses one implementation. ``results`` is consumed in order, so code that
    issues several queries (e.g. count-then-fetch) can be given a distinct result
    for each; once exhausted it yields an empty DataFrame.

    Capturing the SQL matters as much as the return value: code that produces the
    right answer from a query against the wrong schema is still broken, and that
    is precisely the class of bug this seam exposes.
    """

    def __init__(self, *results: Any) -> None:
        self._results = list(results)
        self.queries: list[str] = []

    def sql_executor(self, sql: str) -> Any:
        self.queries.append(sql)
        if not self._results:
            import pandas as pd

            return pd.DataFrame()
        return self._results.pop(0)

    @property
    def last_query(self) -> str:
        """The most recent SQL, for single-query assertions."""
        return self.queries[-1] if self.queries else ""


@pytest.fixture
def patch_executor(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that installs a RecordingExecutor for SqlExec.

    Usage::

        recorder = patch_executor(pd.DataFrame({"1": [1]}))
        ...
        assert '"production"."workorder"' in recorder.last_query
    """

    def _install(*results: Any) -> RecordingExecutor:
        recorder = RecordingExecutor(*results)

        def factory() -> RecordingExecutor:
            return recorder

        # Patch the source module AND every module that did
        # `from app.utils.sql_executor import SqlExec` at import time: that binds
        # its own name, so patching only the source leaves the real class in
        # place and the test silently hits a real database.
        monkeypatch.setattr("app.utils.sql_executor.SqlExec", factory, raising=True)
        for module_path in (
            "app.orchestrator.orch_graph.SqlExec",
            "app.agent.analysis_agent.SqlExec",
        ):
            monkeypatch.setattr(module_path, factory, raising=False)
        return recorder

    return _install


class FakeResult:
    """Minimal stand-in for a SQLAlchemy Result.

    Supports the chains the repositories actually use:
    ``.mappings().first()``, ``.mappings().all()``, and ``.scalar()``.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def mappings(self) -> FakeResult:
        return self

    def first(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def scalar(self) -> Any:
        if not self._rows:
            return None
        return next(iter(self._rows[0].values()))


class FakeSession:
    """Records executed statements and returns queued FakeResults."""

    def __init__(self, *results: list[dict[str, Any]] | None) -> None:
        self._results = list(results)
        self.statements: list[Any] = []
        self.committed = False
        self.rolled_back = False

    def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> FakeResult:
        self.statements.append(statement)
        rows = self._results.pop(0) if self._results else None
        return FakeResult(rows)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def add(self, _entity: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class FakeGateway:
    """Settings-DB gateway whose get_session() yields a FakeSession.

    The repositories take ``gateway`` as a constructor argument, so no patching
    is needed — pass one of these in. That injection point is what makes the
    settings-DB layer testable at all; it is worth preserving.
    """

    def __init__(self, *results: list[dict[str, Any]] | None) -> None:
        self.session = FakeSession(*results)

    @contextmanager
    def get_session(self) -> Any:
        yield self.session


@pytest.fixture
def fake_gateway():
    """Return a factory for FakeGateway, one queued result set per execute()."""

    def _build(*results: list[dict[str, Any]] | None) -> FakeGateway:
        return FakeGateway(*results)

    return _build


class StubMessage:
    """A LangChain AIMessage stand-in.

    Carries ``usage_metadata`` in the Anthropic shape (input_tokens /
    output_tokens) because that is the path cost tracking actually takes — the
    OpenAI-only extractor was the bug that logged every Claude call as 0 tokens.
    """

    def __init__(
        self, content: str, input_tokens: int = 10, output_tokens: int = 5
    ) -> None:
        self.content = content
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        self.response_metadata = {"model_name": "stub-model"}


class StubLlm:
    """Returns queued responses in order and records the prompts it received.

    Capturing prompts is the point: for an LLM-backed feature, "was the model
    given the right context?" is usually the real assertion — that the schema
    graph was included, that conversation history was carried forward, that the
    figures handed to a synthesis step were already computed.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, messages: Any) -> StubMessage:
        if isinstance(messages, list) and messages:
            last = messages[-1]
            self.prompts.append(
                last.get("content", "") if isinstance(last, dict) else str(last)
            )
        response = self._responses.pop(0) if self._responses else ""
        return StubMessage(response)

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1] if self.prompts else ""


@pytest.fixture
def stub_llm():
    """Return a factory building a StubLlm from queued response strings."""

    def _build(*responses: str) -> StubLlm:
        return StubLlm(list(responses))

    return _build


@pytest.fixture
def allow_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt back into real sockets for a test that genuinely needs them.

    Request alongside the autouse guard: ``def test_x(allow_network): ...``.
    """
    monkeypatch.undo()
