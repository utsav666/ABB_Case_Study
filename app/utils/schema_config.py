from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE_PATH = PROJECT_ROOT / "sample_data" / "schema_preserve.json"
COLUMN_CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "db_column_config.json"


class SchemaConfigManager:
    """Load and persist DB column configuration for the API."""

    @staticmethod
    def load_raw_schema() -> dict[str, Any]:
        """Return the unmodified schema JSON from disk, or an empty schema.

        The bundled ``sample_data/schema_preserve.json`` fixture was removed: it
        described a movie database and was silently answering questions asked
        against real connections. Callers here are all no-connection fallback
        paths, so an empty schema is the correct degraded result — raising
        FileNotFoundError turned an endpoint into an unhandled 500.
        """
        if not SCHEMA_FILE_PATH.exists():
            return {"tables": {}}

        with SCHEMA_FILE_PATH.open("r", encoding="utf-8") as schema_file:
            return json.load(schema_file)

    @staticmethod
    def load_column_configuration() -> dict[str, dict[str, dict[str, Any]]]:
        """Return any saved per-table and per-column configuration."""
        if not COLUMN_CONFIG_FILE_PATH.exists():
            return {}

        with COLUMN_CONFIG_FILE_PATH.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    @classmethod
    def get_schema_with_column_configuration(cls) -> dict[str, Any]:
        """Return schema JSON enriched with UI-configured column metadata."""
        schema_data = cls.load_raw_schema()
        saved_configuration = cls.load_column_configuration()

        for table_name, table_details in schema_data.get("tables", {}).items():
            table_configuration = saved_configuration.get(table_name, {})

            for column_name, column_details in table_details.get("columns", {}).items():
                column_configuration = table_configuration.get(column_name, {})
                column_details["display_name"] = column_configuration.get(
                    "display_name", column_name
                )
                column_details["description"] = column_configuration.get(
                    "description", ""
                )
                column_details["include_in_assistant"] = column_configuration.get(
                    "include_in_assistant", True
                )

        return schema_data

    @classmethod
    def get_table_editor_rows(cls, table_name: str) -> list[dict[str, Any]]:
        """Build editable rows for the configuration API payload."""
        schema_data = cls.get_schema_with_column_configuration()
        table_details = schema_data.get("tables", {}).get(table_name, {})
        primary_keys = set(table_details.get("primary_key", []))

        editor_rows: list[dict[str, Any]] = []
        for column_name, column_details in table_details.get("columns", {}).items():
            editor_rows.append(
                {
                    "column_name": column_name,
                    "data_type": column_details.get("type", ""),
                    "required": bool(column_details.get("notnull", False)),
                    "primary_key": column_name in primary_keys,
                    "display_name": column_details.get("display_name", column_name),
                    "description": column_details.get("description", ""),
                    "include_in_assistant": bool(
                        column_details.get("include_in_assistant", True)
                    ),
                }
            )

        return editor_rows

    @classmethod
    def save_table_configuration(
        cls, table_name: str, column_rows: list[dict[Any, Any]]
    ) -> None:
        """Persist a single table's column settings to disk."""
        saved_configuration = cls.load_column_configuration()
        saved_configuration[table_name] = {}

        for row in column_rows:
            column_name = str(row.get("column_name", "")).strip()
            if not column_name:
                continue

            saved_configuration[table_name][column_name] = {
                "display_name": cls._clean_text_value(
                    row.get("display_name"), column_name
                ),
                "description": cls._clean_text_value(row.get("description"), ""),
                "include_in_assistant": cls._clean_bool_value(
                    row.get("include_in_assistant"), True
                ),
            }

        cls._write_configuration(saved_configuration)

    @classmethod
    def reset_table_configuration(cls, table_name: str) -> None:
        """Remove saved settings for one table and restore defaults."""
        saved_configuration = cls.load_column_configuration()

        if table_name in saved_configuration:
            del saved_configuration[table_name]
            cls._write_configuration(saved_configuration)

    @staticmethod
    def _write_configuration(
        saved_configuration: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        """Write the full configuration payload to disk."""
        COLUMN_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with COLUMN_CONFIG_FILE_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(saved_configuration, config_file, indent=2)

    @staticmethod
    def _clean_text_value(value: Any, fallback_value: str) -> str:
        """Return a JSON-safe string for editable text fields."""
        if value is None:
            return fallback_value

        if isinstance(value, float) and math.isnan(value):
            return fallback_value

        cleaned_value = str(value).strip()
        return cleaned_value or fallback_value

    @staticmethod
    def _clean_bool_value(value: Any, fallback_value: bool) -> bool:
        """Return a safe boolean from editable input values."""
        if value is None:
            return fallback_value

        if isinstance(value, float) and math.isnan(value):
            return fallback_value

        return bool(value)
