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


@pytest.fixture
def allow_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt back into real sockets for a test that genuinely needs them.

    Request alongside the autouse guard: ``def test_x(allow_network): ...``.
    """
    monkeypatch.undo()
