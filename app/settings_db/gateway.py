from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.settings_db.config import SettingsDatabaseConfig
from app.settings_db.exceptions import SettingsDatabaseQueryError

logger = logging.getLogger(__name__)

EngineFactory = Callable[[str], Engine]

# Module-level cache of engines keyed by database URL so connection pools are
# reused across requests instead of recreated (and torn down) on every query.
_ENGINE_CACHE: dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def _build_pooled_engine(database_url: str) -> Engine:
    """Create an engine with a connection pool suitable for a web server."""
    return create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def dispose_all_engines() -> None:
    """Dispose all cached engines and clear the cache.

    Called on application shutdown so pooled connections are released cleanly
    (otherwise a uvicorn --reload waits indefinitely for open connections).
    """
    with _ENGINE_CACHE_LOCK:
        for engine in _ENGINE_CACHE.values():
            engine.dispose()
        _ENGINE_CACHE.clear()


class SettingsDatabaseGateway:
    """Low-level SQL execution against the settings database."""

    def __init__(
        self,
        config: SettingsDatabaseConfig | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._config = config or SettingsDatabaseConfig.from_environment()
        self._engine_factory = engine_factory

    def get_database_url(self) -> str:
        """Return the configured database URL.

        Returns:
            str: SQLAlchemy connection URL.
        """
        return self._config.build_connection_url()

    def _get_engine(self) -> Engine:
        """Return a cached, pooled engine for the configured database URL."""
        database_url = self.get_database_url()

        # If a custom factory is provided (e.g. in tests), don't cache.
        if self._engine_factory is not None:
            return self._engine_factory(database_url)

        cached_engine = _ENGINE_CACHE.get(database_url)
        if cached_engine is not None:
            return cached_engine

        with _ENGINE_CACHE_LOCK:
            cached_engine = _ENGINE_CACHE.get(database_url)
            if cached_engine is None:
                cached_engine = _build_pooled_engine(database_url)
                _ENGINE_CACHE[database_url] = cached_engine
        return cached_engine

    @contextmanager
    def get_session(self) -> Any:
        """Yield a SQLAlchemy session for settings DB ORM operations."""
        settings_engine = self._get_engine()
        session_factory = sessionmaker(bind=settings_engine, autoflush=False)
        session = session_factory()
        try:
            yield session
        except (ModuleNotFoundError, SQLAlchemyError) as error:
            session.rollback()
            import traceback
            print("=" * 80)
            print("SETTINGS DB ERROR:")
            print(f"Error type: {type(error).__name__}")
            print(f"Error message: {str(error)}")
            print("Traceback:")
            traceback.print_exc()
            print("=" * 80)
            raise SettingsDatabaseQueryError(
                f"Settings database query failed: {type(error).__name__}: {str(error)}"
            ) from error
        finally:
            session.close()
