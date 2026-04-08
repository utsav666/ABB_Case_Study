import contextlib
import io

import streamlit as st
from envsetup.env_loader import load_env
from orchestrator.orch_exec import Orchestrator

load_env()

st.set_page_config(page_title="Chat with SQL Orchestrator", page_icon="🧠")
st.title("🧠 Natural Language → SQL Orchestrator")

# Keep message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask a question about your database...")

if user_input:
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # ✅ Run Orchestrator
    orch = Orchestrator(user_input)

    # Capture logs (stdout) + returned result
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = orch.executor()  # <-- whatever executor RETURNS
    logs = buffer.getvalue()  # <-- everything executor PRINTED

    # ✅ Display assistant message (both result + logs)
    with st.chat_message("assistant"):
        # ✅ Chart response
        if isinstance(result, dict) and result.get("type") == "chart":
            st.write("### 📊 Chart")
            st.pyplot(result["figure"])
            if "data" in result:
                st.write("### 📄 Data")
                st.dataframe(result["data"])

        # ✅ Display final result
        if "DataFrame" in str(type(result)):
            st.write("### ✅ Result:")
            st.dataframe(result)

        elif isinstance(result, (dict, list)):
            st.write("### ✅ Result:")
            st.json(result)

        elif isinstance(result, str):
            st.write("### ✅ Result:")
            st.write(result)

        # ✅ Display logs
        if logs.strip():
            st.write("### 📝 Logs:")
            st.code(logs)

    # ✅ Save assistant response in message history
    st.session_state.messages.append(
        {"role": "assistant", "content": f"Result:\n{result}\n\nLogs:\n{logs}"}
    )
