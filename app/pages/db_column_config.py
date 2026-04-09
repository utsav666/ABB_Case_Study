from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.db_column_config_view import render_db_column_configuration_page

st.set_page_config(page_title="DB Column Configuration", page_icon="⚙️")
render_db_column_configuration_page()
