from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

EngineFactory = Callable[[str], Engine]

# Customer/target databases are dynamic: there can be many distinct URLs across
# workspaces and customers. Connecting to a (often cross-region) target RDS
# costs several seconds for the TCP+SSL+auth handshake, so we pool connections
# per-URL to amortize that cost across requests. The cache is bounded (LRU) so
# it cannot grow without limit, and evicted engines are disposed.
#
# IMPORTANT: we intentionally do NOT set an aggressive connect timeout here.
# A short connect_timeout was previously removed from the connection path
# because it caused AWS RDS connection failures (see connection_validation.py).
_MAX_CACHED_ENGINES = 32
_ENGINE_CACHE: "OrderedDict[str, Engine]" = OrderedDict()
_ENGINE_CACHE_LOCK = threading.Lock()


def _build_pooled_engine(target_url: str) -> Engine:
    """Create a pooled engine for a target database URL.

    pool_pre_ping validates a pooled connection before use so a connection the
    server has already dropped (common with RDS idle timeouts) is transparently
    replaced instead of surfacing as an error. pool_recycle refreshes
    connections periodically as a second line of defense.
    """
    return create_engine(
        target_url,
        pool_size=2,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def dispose_all_target_engines() -> None:
    """Dispose all cached target engines and clear the cache.

    Called on application shutdown so pooled connections are released cleanly
    (otherwise a uvicorn --reload waits indefinitely for open connections).
    """
    with _ENGINE_CACHE_LOCK:
        for engine in _ENGINE_CACHE.values():
            engine.dispose()
        _ENGINE_CACHE.clear()


class TargetDatabaseGateway:
    """Low-level SQL execution against a selected target database."""

    def __init__(self, engine_factory: EngineFactory | None = None) -> None:
        self._engine_factory = engine_factory

    def _get_engine(self, target_url: str) -> Engine:
        """Return a cached, pooled engine for the given target URL."""
        # A custom factory (e.g. in tests) bypasses the shared cache.
        if self._engine_factory is not None:
            return self._engine_factory(target_url)

        cached_engine = _ENGINE_CACHE.get(target_url)
        if cached_engine is not None:
            with _ENGINE_CACHE_LOCK:
                if target_url in _ENGINE_CACHE:
                    _ENGINE_CACHE.move_to_end(target_url)
            return cached_engine

        with _ENGINE_CACHE_LOCK:
            cached_engine = _ENGINE_CACHE.get(target_url)
            if cached_engine is None:
                cached_engine = _build_pooled_engine(target_url)
                _ENGINE_CACHE[target_url] = cached_engine
                _ENGINE_CACHE.move_to_end(target_url)
                # Evict least-recently-used engines beyond the cap.
                while len(_ENGINE_CACHE) > _MAX_CACHED_ENGINES:
                    _, evicted_engine = _ENGINE_CACHE.popitem(last=False)
                    evicted_engine.dispose()
            else:
                _ENGINE_CACHE.move_to_end(target_url)
        return cached_engine

    def execute_query(
        self,
        target_url: str,
        query: Any,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return mapped rows.

        Args:
            target_url: SQLAlchemy target URL.
            query: SQLAlchemy query object.
            params: Bound query parameters.

        Returns:
            list[dict[str, Any]]: Result rows.
        """
        target_engine = self._get_engine(target_url)
        try:
            t_connect = time.monotonic()
            with target_engine.connect() as connection:
                logger.info(
                    "Target DB connect/checkout took %.2fs",
                    time.monotonic() - t_connect,
                )
                t_query = time.monotonic()
                result = connection.execute(query, params)
                rows = [dict(row) for row in result.mappings()]
                logger.info(
                    "Target DB query exec took %.2fs", time.monotonic() - t_query
                )
                return rows
        except (ModuleNotFoundError, SQLAlchemyError) as error:
            logger.exception(
                "Target DB query failed (%s): %s", type(error).__name__, error
            )
            raise HTTPException(
                status_code=500,
                detail=f"Could not connect to target database: {error}",
            ) from error
