from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy.engine import URL

SUPPORTED_DATABASE_ENGINES: tuple[str, ...] = (
    "postgres",
    "mysql",
    "mariadb",
    "sqlserver",
    "oracle",
    "sqlite",
)

ENGINE_ALIASES: dict[str, str] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "mysql": "mysql",
    "mariadb": "mariadb",
    "sqlserver": "sqlserver",
    "sql_server": "sqlserver",
    "mssql": "sqlserver",
    "oracle": "oracle",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
}

DEFAULT_PORTS: dict[str, int] = {
    "postgres": 5432,
    "mysql": 3306,
    "mariadb": 3306,
    "sqlserver": 1433,
    "oracle": 1521,
    "sqlite": 0,
}


def normalize_database_engine(raw_engine: str) -> str:
    """Normalize a database engine name to a supported identifier."""
    normalized_engine = raw_engine.strip().lower().replace(" ", "_")
    return ENGINE_ALIASES.get(normalized_engine, normalized_engine)


def get_supported_database_engines() -> list[str]:
    """Return supported engine names for API responses."""
    return list(SUPPORTED_DATABASE_ENGINES)


def get_default_port(engine: str) -> int:
    """Return the default port for a supported engine."""
    return DEFAULT_PORTS[engine]


def parse_extra_options(raw_options: str | None) -> dict[str, str]:
    """Parse multiline key=value options into a dictionary."""
    if not raw_options:
        return {}

    parsed_options: dict[str, str] = {}
    for raw_line in raw_options.splitlines():
        normalized_line = raw_line.strip()
        if not normalized_line or normalized_line.startswith("#"):
            continue
        if "=" not in normalized_line:
            continue

        option_key, option_value = normalized_line.split("=", 1)
        option_key = option_key.strip()
        option_value = option_value.strip()
        if option_key and option_value:
            parsed_options[option_key] = option_value

    return parsed_options


def get_validation_query(engine: str) -> str:
    """Return a lightweight validation query for a database engine."""
    if engine == "oracle":
        return "SELECT 1 FROM DUAL"

    return "SELECT 1"


def build_connect_args(engine: str) -> dict[str, Any]:
    """Return engine-specific SQLAlchemy connect args."""
    if engine == "sqlite":
        return {"check_same_thread": False}

    return {}


def build_target_db_url(credentials: Mapping[str, Any]) -> str:
    """Build a SQLAlchemy connection URL for a supported engine."""
    engine = normalize_database_engine(str(credentials.get("engine", "")))
    query_options = parse_extra_options(str(credentials.get("extra_options", "") or ""))
    use_ssl = bool(credentials.get("use_ssl"))

    if engine == "postgres":
        if use_ssl and "sslmode" not in query_options:
            query_options["sslmode"] = "require"

        # Use simple string concatenation (matches settings DB approach)
        # This avoids URL.create() password encoding issues
        db_user = str(credentials.get("db_user", ""))
        db_password = str(credentials.get("db_password", ""))
        db_host = str(credentials.get("db_host", ""))
        db_port = int(credentials.get("db_port", 5432) or 5432)
        db_name = str(credentials.get("db_name", ""))

        base_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        # Append query options if present
        if query_options:
            query_string = "&".join(f"{k}={v}" for k, v in query_options.items())
            base_url = f"{base_url}?{query_string}"

        return base_url

    if engine in {"mysql", "mariadb"}:
        if use_ssl and "ssl_disabled" not in query_options:
            query_options["ssl_disabled"] = "false"
        return str(
            URL.create(
                drivername="mysql+mysqlconnector",
                username=str(credentials.get("db_user", "")) or None,
                password=str(credentials.get("db_password", "")) or None,
                host=str(credentials.get("db_host", "")) or None,
                port=int(credentials.get("db_port", 3306) or 3306),
                database=str(credentials.get("db_name", "")) or None,
                query=query_options,
            )
        )

    if engine == "sqlserver":
        if use_ssl and "encrypt" not in query_options:
            query_options["encrypt"] = "yes"
        return str(
            URL.create(
                drivername="mssql+pytds",
                username=str(credentials.get("db_user", "")) or None,
                password=str(credentials.get("db_password", "")) or None,
                host=str(credentials.get("db_host", "")) or None,
                port=int(credentials.get("db_port", 1433) or 1433),
                database=str(credentials.get("db_name", "")) or None,
                query=query_options,
            )
        )

    if engine == "oracle":
        if use_ssl and "protocol" not in query_options:
            query_options["protocol"] = "tcps"
        query_options.setdefault("service_name", str(credentials.get("db_name", "")))
        return str(
            URL.create(
                drivername="oracle+oracledb",
                username=str(credentials.get("db_user", "")) or None,
                password=str(credentials.get("db_password", "")) or None,
                host=str(credentials.get("db_host", "")) or None,
                port=int(credentials.get("db_port", 1521) or 1521),
                query=query_options,
            )
        )

    if engine == "sqlite":
        database_name = str(credentials.get("db_name", "")).strip()
        if database_name == ":memory:":
            return "sqlite:///:memory:"

        database_path = Path(database_name).expanduser()
        if not database_path.is_absolute():
            database_path = Path.cwd() / database_path
        return f"sqlite:///{database_path}"

    raise ValueError(f"Unsupported database engine: {engine}")
