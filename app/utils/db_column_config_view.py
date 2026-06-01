from __future__ import annotations

from typing import Any

from app.utils.schema_config import SchemaConfigManager
from app.utils.schema_summarizer import Summarizer


def build_db_column_configuration_payload(
    selected_table_name: str | None = None,
) -> dict[str, Any]:
    """Return schema configuration data for the FastAPI layer."""
    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    table_names = sorted(schema_data.get("tables", {}).keys())

    if selected_table_name is None and table_names:
        selected_table_name = table_names[0]

    editor_rows: list[dict[str, Any]] = []
    if selected_table_name in schema_data.get("tables", {}):
        editor_rows = SchemaConfigManager.get_table_editor_rows(selected_table_name)

    return {
        "tables": schema_data.get("tables", {}),
        "table_names": table_names,
        "selected_table": selected_table_name,
        "editor_rows": editor_rows,
        "schema_summary": Summarizer.summarize_schema(),
        "saved_configuration": SchemaConfigManager.load_column_configuration(),
    }


def get_table_configuration_payload(table_name: str) -> dict[str, Any] | None:
    """Return editable column metadata for one table."""
    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    if table_name not in schema_data.get("tables", {}):
        return None

    return {
        "table_name": table_name,
        "columns": SchemaConfigManager.get_table_editor_rows(table_name),
    }


def save_table_configuration(
    table_name: str, column_rows: list[dict[str, Any]]
) -> None:
    """Persist column metadata for one table."""
    SchemaConfigManager.save_table_configuration(table_name, column_rows)
