from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

EngineFactory = Callable[[str], Engine]


class TargetDatabaseGateway:
    """Low-level SQL execution against a selected target database."""

    def __init__(self, engine_factory: EngineFactory = create_engine) -> None:
        self._engine_factory = engine_factory

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
        target_engine = self._engine_factory(target_url)
        try:
            with target_engine.connect() as connection:
                result = connection.execute(query, params)
                return [dict(row) for row in result.mappings()]
        except (ModuleNotFoundError, SQLAlchemyError) as error:
            raise HTTPException(
                status_code=500,
                detail=f"Could not connect to target database: {error}",
            ) from error
        finally:
            target_engine.dispose()
