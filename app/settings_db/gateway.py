from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.settings_db.config import SettingsDatabaseConfig
from app.settings_db.exceptions import SettingsDatabaseQueryError

EngineFactory = Callable[[str], Engine]


class SettingsDatabaseGateway:
    """Low-level SQL execution against the settings database."""

    def __init__(
        self,
        config: SettingsDatabaseConfig | None = None,
        engine_factory: EngineFactory = create_engine,
    ) -> None:
        self._config = config or SettingsDatabaseConfig.from_environment()
        self._engine_factory = engine_factory

    def get_database_url(self) -> str:
        """Return the configured database URL.

        Returns:
            str: SQLAlchemy connection URL.
        """
        return self._config.build_connection_url()

    @contextmanager
    def get_session(self) -> Any:
        """Yield a SQLAlchemy session for settings DB ORM operations."""
        settings_engine = self._engine_factory(self.get_database_url())
        session_factory = sessionmaker(bind=settings_engine, autoflush=False)
        session = session_factory()
        try:
            yield session
        except (ModuleNotFoundError, SQLAlchemyError) as error:
            session.rollback()
            raise SettingsDatabaseQueryError(
                "Settings database query failed."
            ) from error
        finally:
            session.close()
            settings_engine.dispose()
