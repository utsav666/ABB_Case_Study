import streamlit as st
from orch_app.chat_api import ChatAPI

st.set_page_config(page_title="Chat with OpenAI", page_icon="🤖")
st.title("🤖 Chat with OpenAI")

# Initialize ChatAPI once (loads from .env)
if "chat_api" not in st.session_state:
    st.session_state.chat_api = ChatAPI()

# Store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
st.subheader("Chat History")
for message in st.session_state.messages:
    role = "🧑 You" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# User input (old way, no chat_input)
prompt = st.text_input("Type your message and press Enter:")

if st.button("Send"):
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"**🧑 You:** {prompt}")

        # Get reply from API
        reply = st.session_state.chat_api.chat(st.session_state.messages)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.markdown(f"**🤖 Assistant:** {reply}")
