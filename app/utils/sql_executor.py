import logging
import os
from threading import Lock

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.sampledata.data_loader import DBLoader
from app.settings_db.controllers import get_settings_connection_credentials
from app.target_db.views import build_target_db_url
from app.utils.logging_context import get_log_context
from app.utils.runtime_limits import get_db_statement_timeout_ms

logger = logging.getLogger(__name__)

# Pooled engines keyed by connection URL. Building an engine and opening a fresh
# Postgres connection per query cost multiple seconds each (the dominant latency
# in a request); a pooled engine pays the handshake once and reuses live,
# health-checked connections thereafter. Keying by URL keeps this correct when
# more than one target database is in play — each gets its own pool.
_ENGINE_CACHE: dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = Lock()


def _get_backend_database_name() -> str:
    """Return the configured backend database name."""
    return (os.getenv("DB_NAME") or "").strip() or (
        os.getenv("SETTINGS_DB_NAME") or ""
    ).strip()


def _get_backend_database_url() -> str:
    """Build the legacy database URL when no target connection is selected."""
    database_details = DBLoader.db_loader()["data_base_detail"]
    return (
        f"postgresql://{database_details['user']}:{database_details['password']}"
        f"@{database_details['host']}:{database_details['port']}"
        f"/{database_details['database']}"
    )


def resolve_sql_database_url(connection_id: str | None = None) -> str:
    """Resolve the SQL URL from the active saved connection or backend defaults.

    Args:
        connection_id: Explicit connection to use. Defaults to the connection
            bound to the current request's logging context.

    Returns:
        str: SQLAlchemy URL for the active database.
    """
    connection_id = connection_id or get_log_context()["connection_id"]
    if not connection_id or connection_id == "none":
        return _get_backend_database_url()

    connection_credentials = dict(get_settings_connection_credentials(connection_id))
    saved_database_name = str(connection_credentials.get("db_name") or "").strip()
    if saved_database_name:
        connection_credentials["db_name"] = saved_database_name
        return build_target_db_url(connection_credentials)

    fallback_database_name = _get_backend_database_name()
    if not fallback_database_name:
        raise ValueError(
            f"Connection '{connection_id}' has no database name, and "
            "DB_NAME/SETTINGS_DB_NAME is not configured."
        )

    logger.warning(
        "Connection %s has no database name; using backend database fallback.",
        connection_id,
    )
    connection_credentials["db_name"] = fallback_database_name
    return build_target_db_url(connection_credentials)


def _get_engine(url: str) -> Engine:
    """Return a pooled SQLAlchemy engine for ``url``, creating it once."""
    engine = _ENGINE_CACHE.get(url)
    if engine is not None:
        return engine
    with _ENGINE_CACHE_LOCK:
        # Re-check inside the lock (another thread may have just created it).
        engine = _ENGINE_CACHE.get(url)
        if engine is None:
            engine = create_engine(
                url,
                pool_size=5,
                max_overflow=10,
                # pool_pre_ping intentionally OFF — a pre-query round-trip to a
                # remote DB costs ~300ms per checkout. pool_recycle handles the
                # common stale-connection case; a dropped connection is a rare
                # one-off error the caller retries.
                pool_recycle=1800,  # recycle connections older than 30 min
            )
            _ENGINE_CACHE[url] = engine
    return engine


class SqlExec:
    def __init__(self, connection_id: str | None = None) -> None:
        self.database_url = resolve_sql_database_url(connection_id)

    def sql_executor(
        self, sql_query: str, timeout_ms: int | None = None
    ) -> pd.DataFrame | str:
        """Run a query in a read-only transaction and return a DataFrame.

        SQL here is LLM-generated, so the database enforces read-only access
        and a statement timeout on PostgreSQL in addition to the validator's
        parse-level checks. Errors are returned as a string, not raised.
        """
        try:
            engine = _get_engine(self.database_url)
            # Reuse a pooled connection; do NOT dispose the engine here — that
            # would tear down the pool and defeat the caching.
            with engine.connect() as connection, connection.begin():
                if engine.dialect.name == "postgresql":
                    statement_timeout_ms = int(
                        timeout_ms or get_db_statement_timeout_ms()
                    )
                    # One round trip for both settings: the target DB is remote.
                    connection.exec_driver_sql(
                        "SET TRANSACTION READ ONLY; "
                        f"SET LOCAL statement_timeout = {statement_timeout_ms}"
                    )
                return pd.read_sql_query(text(sql_query), connection)
        except (
            KeyError,
            SQLAlchemyError,
            ValueError,
            pd.errors.DatabaseError,
        ) as error:
            logger.exception("SQL execution failed")
            return str(error)


# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res)
