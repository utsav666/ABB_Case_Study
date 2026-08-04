from __future__ import annotations

import os
from dataclasses import dataclass

from app.settings_db.exceptions import SettingsDatabaseConfigurationError

DEFAULT_SETTINGS_DB_PORT = "5432"


@dataclass(frozen=True)
class SettingsDatabaseConfig:
    """Resolved connection details for the settings database."""

    host: str
    port: str
    name: str
    user: str
    password: str

    @classmethod
    def from_environment(cls) -> SettingsDatabaseConfig:
        """Build a config object from environment variables.

        Returns:
            SettingsDatabaseConfig: Validated settings database config.
        """
        config = cls(
            host=os.getenv("SETTINGS_DB_HOST", os.getenv("DB_HOST", "")).strip(),
            port=os.getenv(
                "SETTINGS_DB_PORT",
                os.getenv("DB_PORT", DEFAULT_SETTINGS_DB_PORT),
            ).strip(),
            name=os.getenv("SETTINGS_DB_NAME", os.getenv("DB_NAME", "")).strip(),
            user=os.getenv("SETTINGS_DB_USER", os.getenv("DB_USER", "")).strip(),
            password=os.getenv(
                "SETTINGS_DB_PASSWORD",
                os.getenv("DB_PASSWORD", ""),
            ).strip(),
        )
        missing_variables = config.get_missing_variables()
        if missing_variables:
            missing_names = ", ".join(missing_variables)
            raise SettingsDatabaseConfigurationError(
                f"Missing required environment variables: {missing_names}"
            )

        return config

    def get_missing_variables(self) -> list[str]:
        """Return missing required environment variable names.

        Returns:
            list[str]: Missing environment variable names.
        """
        required_variables = {
            "SETTINGS_DB_HOST": self.host,
            "SETTINGS_DB_NAME": self.name,
            "SETTINGS_DB_USER": self.user,
            "SETTINGS_DB_PASSWORD": self.password,
        }
        return [
            variable_name
            for variable_name, variable_value in required_variables.items()
            if not variable_value
        ]

    def build_connection_url(self) -> str:
        """Return the SQLAlchemy URL for the settings database.

        Returns:
            str: PostgreSQL connection URL.
        """
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.name}"
        )
