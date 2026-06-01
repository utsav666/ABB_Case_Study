from __future__ import annotations

import io
import logging
import uuid
from typing import Any

import streamlit as st
from app.envsetup.env_loader import load_env

# --- Orchestrators: uncomment ONE at a time ---
# from app.orchestrator.orch_exec import Orchestrator
# from app.orchestrator.orch_improve import AgentOrchestrator
from app.orchestrator.orch_graph import GraphOrchestrator
from pandas import DataFrame

# from app.utils.db_column_config_view import render_db_column_configuration_page

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

# if st.session_state.selected_page == "DB Column Config":
#     render_db_column_configuration_page()
#     st.stop()

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

# ### HITL — Handle resumed result after st.rerun()
if (
    "hitl_resume_result" in st.session_state
    and st.session_state.hitl_resume_result is not None
):
    result = st.session_state.hitl_resume_result
    st.session_state.hitl_resume_result = None
    result_type = result.get("type", "") if isinstance(result, dict) else ""
    with st.chat_message("assistant"):
        msg: dict[str, Any] = {"role": "assistant"}
        if result_type == "chart":
            fig = result.get("figure")
            msg["content"] = "Here is your chart."
            msg["figure"] = fig
            if result.get("sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")
            if fig:
                st.pyplot(fig)
            if result.get("data") is not None:
                st.dataframe(result["data"])
        elif result_type == "data":
            msg["content"] = "Here are the query results."
            msg["data"] = result.get("data")
            if result.get("sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")
            st.dataframe(result.get("data"))
        elif result_type == "error":
            msg["content"] = result.get("message", "Error")
            st.error(result.get("message"))
        else:
            msg["content"] = result.get("message", str(result))
            st.write(result.get("message", str(result)))
        st.session_state.messages.append(msg)

# ### HITL — Show pending HITL UI (persists across reruns via session_state)
if "hitl_pending" in st.session_state and st.session_state.hitl_pending is not None:
    pending = st.session_state.hitl_pending
    hitl_type = pending.get("hitl_type", "")
    thread_id = pending.get("thread_id", "")

    with st.chat_message("assistant"):
        st.write("### 🧑 Human Input Needed")
        st.info(pending.get("message", "Please make a selection."))

        if hitl_type == "cardinality_filter":
            suggestions = pending.get("suggestions", [])
            if suggestions:
                choice = st.radio(
                    f"Which value for **{pending.get('column', 'column')}**?",
                    options=suggestions,
                    key="hitl_cardinality_choice",
                )
            else:
                choice = st.text_input(
                    "Type the exact value:",
                    key="hitl_cardinality_input",
                )
            if st.button("✅ Confirm Selection", key="hitl_confirm_card"):
                from app.orchestrator.orch_graph import GraphOrchestrator

                resume_result = GraphOrchestrator.resume(thread_id, choice)
                st.session_state.hitl_pending = None
                st.session_state.hitl_thread_id = str(uuid.uuid4())
                # Check if another HITL is triggered (e.g., chart after cardinality)
                if (
                    isinstance(resume_result, dict)
                    and resume_result.get("type") == "pending_hitl"
                ):
                    st.session_state.hitl_pending = resume_result
                else:
                    st.session_state.hitl_resume_result = resume_result
                st.rerun()

        elif hitl_type == "chart_approval":
            suggested = pending.get("suggested_chart_type", "bar")
            available_charts = pending.get(
                "available_chart_types", ["bar", "line", "scatter", "pie"]
            )
            available_cols = pending.get("available_columns", [])

            st.write(
                f"Suggested: **{suggested}** (x={pending.get('suggested_x_axis')}, y={pending.get('suggested_y_axis')})"
            )
            chart_choice = st.selectbox(
                "Chart type",
                available_charts,
                index=available_charts.index(suggested)
                if suggested in available_charts
                else 0,
                key="hitl_chart_type",
            )
            x_choice = st.selectbox(
                "X axis",
                available_cols,
                index=available_cols.index(pending.get("suggested_x_axis"))
                if pending.get("suggested_x_axis") in available_cols
                else 0,
                key="hitl_x",
            )
            y_choice = st.selectbox(
                "Y axis",
                available_cols,
                index=available_cols.index(pending.get("suggested_y_axis"))
                if pending.get("suggested_y_axis") in available_cols
                else 0,
                key="hitl_y",
            )

            if st.button("✅ Approve Chart", key="hitl_confirm_chart"):
                from app.orchestrator.orch_graph import GraphOrchestrator

                human_decision = {
                    "chart_type": chart_choice,
                    "x_axis": x_choice,
                    "y_axis": y_choice,
                }
                resume_result = GraphOrchestrator.resume(thread_id, human_decision)
                st.session_state.hitl_pending = None
                st.session_state.hitl_thread_id = str(uuid.uuid4())
                st.session_state.hitl_resume_result = resume_result
                st.rerun()

    st.stop()  # Don't show chat input while waiting for HITL

# User input
user_input = st.chat_input("Ask a question about your database...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # --- Orchestrator ---
    ### HITL — thread_id persists across interrupt/resume cycles
    if "hitl_thread_id" not in st.session_state:
        st.session_state.hitl_thread_id = str(uuid.uuid4())
    thread_id = st.session_state.hitl_thread_id
    orch = GraphOrchestrator(user_input, thread_id=thread_id)

    buffer = io.StringIO()
    log_handler = logging.StreamHandler(buffer)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    )
    log_handler.addFilter(lambda record: not record.name.startswith("httpx"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(log_handler)

    try:
        result: Any = orch.executor()
    finally:
        root_logger.removeHandler(log_handler)

    logs = buffer.getvalue()

    assistant_message: dict[str, Any] = {"role": "assistant"}

    with st.chat_message("assistant"):
        # --- New agent always returns dict with "type" key ---
        result_type = result.get("type", "") if isinstance(result, dict) else ""

        # ### HITL — If pending, store in session and stop (UI handled above on rerun)
        if result_type == "pending_hitl":
            st.session_state.hitl_pending = result
            assistant_message["content"] = result.get("message", "I need your input.")
            st.session_state.messages.append(assistant_message)
            st.rerun()

        if result_type == "chart":
            chart_figure = result.get("figure")
            chart_data = result.get("data")

            assistant_message["content"] = result.get("message", "Here is your chart.")
            assistant_message["figure"] = chart_figure
            assistant_message["data"] = chart_data

            if result.get("sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")

            if chart_figure is not None:
                st.write("### 📊 Chart")
                st.pyplot(chart_figure)

            if chart_data is not None:
                st.write("### 📄 Data")
                st.dataframe(chart_data)

        elif result_type == "data":
            assistant_message["content"] = result.get(
                "message", "Here are the query results."
            )
            assistant_message["data"] = result.get("data")

            if result.get("sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")

            st.write("### ✅ Result:")
            st.dataframe(result.get("data"))

        elif result_type == "text":
            assistant_message["content"] = result.get("message", "")
            st.write("### ✅ Result:")
            st.write(result.get("message", ""))

        elif result_type == "error":
            assistant_message["content"] = result.get("message", "An error occurred.")
            st.write("### ❌ Error:")
            st.error(result.get("message", "Unknown error"))

        # --- Old orchestrator fallback (backward compat) ---
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
