from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from utils.schema_config import SchemaConfigManager
from utils.schema_summarizer import Summarizer


def render_db_column_configuration_page() -> None:
    """Render the UI used to configure DB columns."""
    st.title("⚙️ DB Column Configuration")
    st.caption(
        "Update friendly names, descriptions, and assistant visibility for database columns."
    )

    schema_data = SchemaConfigManager.get_schema_with_column_configuration()
    table_names = sorted(schema_data.get("tables", {}).keys())

    if not table_names:
        st.warning("No tables were found in the schema file.")
        return

    selected_table_name = str(st.selectbox("Choose a table", table_names))
    editor_rows = SchemaConfigManager.get_table_editor_rows(selected_table_name)
    editor_frame = pd.DataFrame(editor_rows)

    st.subheader(f"Columns in `{selected_table_name}`")
    edited_frame = st.data_editor(
        editor_frame,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "column_name": st.column_config.TextColumn("Column", disabled=True),
            "data_type": st.column_config.TextColumn("Type", disabled=True),
            "required": st.column_config.CheckboxColumn("Required", disabled=True),
            "primary_key": st.column_config.CheckboxColumn(
                "Primary Key", disabled=True
            ),
            "display_name": st.column_config.TextColumn(
                "Display Name",
                help="Friendly name to show in the UI and schema context.",
            ),
            "description": st.column_config.TextColumn(
                "Description",
                help="Business meaning for the column.",
                width="large",
            ),
            "include_in_assistant": st.column_config.CheckboxColumn(
                "Use in Assistant",
                help="Turn off if a column should be hidden from schema reasoning.",
            ),
        },
    )

    left_column, right_column = st.columns(2)

    with left_column:
        if st.button("Save configuration", type="primary", use_container_width=True):
            edited_rows: list[dict[Any, Any]] = [
                dict(row) for row in edited_frame.to_dict(orient="records")
            ]
            SchemaConfigManager.save_table_configuration(
                selected_table_name,
                edited_rows,
            )
            st.success(f"Saved configuration for `{selected_table_name}`.")

    with right_column:
        if st.button("Reset selected table", use_container_width=True):
            SchemaConfigManager.reset_table_configuration(selected_table_name)
            st.success(f"Reset configuration for `{selected_table_name}`.")
            st.rerun()

    with st.expander("Preview assistant schema summary"):
        st.code(Summarizer.summarize_schema())

    with st.expander("Preview saved configuration JSON"):
        st.json(SchemaConfigManager.load_column_configuration())
