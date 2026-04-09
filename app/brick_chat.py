import contextlib
import io
from typing import Any

import streamlit as st
from envsetup.env_loader import load_env
from orchestrator.orch_exec import Orchestrator
from pandas import DataFrame
from utils.db_column_config_view import render_db_column_configuration_page

load_env()

st.set_page_config(page_title="Chat with SQL Orchestrator", page_icon="🧠")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] .stButton > button > div {
            text-align: left !important;
            justify-content: left !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("Navigation")

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Chat"

if st.sidebar.button("💬 Chat", use_container_width=True):
    st.session_state.selected_page = "Chat"

if st.sidebar.button("⚙️ DB Column Config", use_container_width=True):
    st.session_state.selected_page = "DB Column Config"

if st.session_state.selected_page == "DB Column Config":
    render_db_column_configuration_page()
    st.stop()

st.title("🧠 Natural Language → SQL Orchestrator")

if st.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()

# Keep message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg.get("content")
        if isinstance(content, DataFrame):
            st.dataframe(content)
        elif isinstance(content, (dict, list)):
            st.json(content)
        elif content:
            st.write(content)

        history_figure = msg.get("figure")
        if history_figure is not None:
            st.write("### 📊 Chart")
            st.pyplot(history_figure)

        history_data = msg.get("data")
        if history_data is not None:
            st.write("### 📄 Data")
            st.dataframe(history_data)

        history_logs = msg.get("logs")
        if history_logs:
            with st.expander("📝 Logs"):
                st.code(history_logs)

# User input
user_input = st.chat_input("Ask a question about your database...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    orch = Orchestrator(user_input)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result: Any = orch.executor()
    logs = buffer.getvalue()

    assistant_message: dict[str, Any] = {"role": "assistant"}
    is_chart_result = isinstance(result, dict) and result.get("type") == "chart"

    with st.chat_message("assistant"):
        if is_chart_result:
            chart_figure = result.get("figure")
            chart_data = result.get("data")

            assistant_message["content"] = "Here is your chart."
            assistant_message["figure"] = chart_figure
            assistant_message["data"] = chart_data

            if chart_figure is not None:
                st.write("### 📊 Chart")
                st.pyplot(chart_figure)

            if chart_data is not None:
                st.write("### 📄 Data")
                st.dataframe(chart_data)

        elif isinstance(result, DataFrame):
            assistant_message["content"] = "Here are the query results."
            assistant_message["data"] = result
            st.write("### ✅ Result:")
            st.dataframe(result)

        elif isinstance(result, (dict, list)):
            assistant_message["content"] = result
            st.write("### ✅ Result:")
            st.json(result)

        else:
            assistant_message["content"] = str(result)
            st.write("### ✅ Result:")
            st.write(str(result))

        if logs.strip():
            assistant_message["logs"] = logs
            with st.expander("📝 Logs"):
                st.code(logs)

    st.session_state.messages.append(assistant_message)
