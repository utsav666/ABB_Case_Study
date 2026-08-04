import logging
from threading import Lock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.sampledata.data_loader import DBLoader

logger = logging.getLogger(__name__)

# Pooled engines keyed by connection URL. Building an engine and opening a fresh
# Postgres connection per query cost multiple seconds each (the dominant latency
# in a request); a pooled engine pays the handshake once and reuses live,
# health-checked connections thereafter. Keying by URL keeps this correct when
# more than one target database is in play — each gets its own pool.
_ENGINE_CACHE: dict[str, Engine] = {}
_ENGINE_CACHE_LOCK = Lock()


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
    def __init__(self):
        # self.data_path = 'SQL-Agent/sample_data/database_cred.json'
        self.data_path = DBLoader.db_loader()
        # self.connection = sqlite3.connect(self.data_path)
        # self.sql_query = sql_query

    def sql_executor(self, sql_query: str) -> pd.DataFrame | str:
        try:
            db_details = self.data_path["data_base_detail"]
            # Build connection URL with search_path to include both public and other schemas
            url = f"postgresql://{db_details['user']}:{db_details['password']}@{db_details['host']}:{db_details['port']}/{db_details['database']}"
            engine = _get_engine(url)
            # Reuse a pooled connection; do NOT dispose the engine here — that
            # would tear down the pool and defeat the caching.
            df = pd.read_sql_query(sql_query, engine)
            return df
        except (KeyError, SQLAlchemyError, ValueError) as error:
            logger.exception("SQL execution failed")
            return str(error)


# if __name__ == "__main__":
#     user_query = """SELECT * FROM Movie WHERE rating < 3"""
#     sql_exec= SqlExec()
#     res = sql_exec.sql_executor(user_query)
#     print(res)
