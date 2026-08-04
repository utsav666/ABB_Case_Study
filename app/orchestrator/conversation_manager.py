"""Conversation history management using LangChain memory."""

import json
from typing import Any

from langchain_classic.memory import ConversationBufferMemory
from sqlalchemy.orm import Session

from app.settings_db.repositories import ConversationRepository
from app.envsetup.env_loader import load_env

load_env()


class ConversationHistoryManager:
    """Manages conversation threads using LangChain memory backed by database.

    Each conversation has a unique thread_id and maintains a list of
    (user message, assistant message) exchanges.
    """

    def __init__(self, session: Session | None = None):
        self.session = session
        self.repo = ConversationRepository(session)

    def create_conversation(
        self,
        workspace_id: str,
        connection_id: str | None,
        title: str,
        thread_id: str,
    ) -> dict[str, Any]:
        """Create a new conversation thread.

        Returns:
            {"thread_id": "...", "title": "...", "created_at": ...}
        """
        return self.repo.create_conversation(
            workspace_id=workspace_id,
            connection_id=connection_id,
            title=title[:80],
            thread_id=thread_id,
        )

    def ensure_conversation(
        self,
        workspace_id: str,
        connection_id: str | None,
        thread_id: str,
        title: str,
    ) -> None:
        """Ensure a thread exists before messages are inserted."""
        if self.repo.conversation_exists(thread_id):
            return

        self.create_conversation(
            workspace_id=workspace_id,
            connection_id=connection_id,
            title=title,
            thread_id=thread_id,
        )

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        """Add a message to the conversation thread.

        Args:
            thread_id: Unique conversation identifier
            role: "user" or "assistant"
            content: Message text

        Returns:
            {"message_id": "...", "timestamp": ...}
        """
        return self.repo.add_message(thread_id, role, content)

    def get_conversation_history(self, thread_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages in a conversation thread.

        Returns:
            [{"role": "user"|"assistant", "content": "..."}, ...]
        """
        return self.repo.get_messages_by_thread(thread_id)

    def get_workspace_conversations(
        self,
        workspace_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve all conversations for a workspace (metadata only).

        Returns:
            [{"thread_id": "...", "title": "...", "created_at": ..., "message_count": ...}, ...]
        """
        return self.repo.get_conversations_by_workspace(workspace_id, limit)

    def delete_conversation(self, thread_id: str) -> bool:
        """Delete a conversation and all its messages."""
        return self.repo.delete_conversation(thread_id)

    def get_langchain_memory(self, thread_id: str) -> ConversationBufferMemory:
        """Get a LangChain ConversationBufferMemory for a thread.

        Useful for integrating with LangChain agents/chains.
        """
        messages = self.get_conversation_history(thread_id)

        memory = ConversationBufferMemory(
            return_messages=True,
            human_prefix="User",
            ai_prefix="Assistant",
        )

        # Load existing messages into memory
        for msg in messages:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

        return memory

    @staticmethod
    def _readable_content(content: Any) -> str:
        """Extract the human-readable text from a stored message.

        Assistant replies are persisted as a JSON envelope
        ``{"kind": "assistant_payload_v1", "payload": {...}}`` so the frontend can
        rehydrate charts and tables on reload. Feeding that raw JSON to an LLM as
        "history" is worse than sending nothing: the model sees serialised data
        rows and base64 blobs instead of what it said, and the token budget is
        spent on punctuation. Pull out the message text, and mention the SQL so a
        follow-up can build on the previous query's filters.
        """
        if not isinstance(content, str):
            return str(content or "")

        stripped = content.strip()
        if not stripped.startswith("{"):
            return stripped  # plain-text (user messages, legacy rows)

        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return stripped

        payload = parsed.get("payload") if isinstance(parsed, dict) else None
        if not isinstance(payload, dict):
            payload = parsed if isinstance(parsed, dict) else {}

        parts: list[str] = []
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            parts.append(message.strip())

        # The prior SQL is the highest-value context for a follow-up: it states
        # the filters the user already established, in unambiguous form.
        sql = payload.get("sql")
        if isinstance(sql, str) and sql.strip():
            parts.append(f"[SQL used: {' '.join(sql.split())}]")

        return "\n".join(parts) if parts else stripped

    def format_history_for_context(self, thread_id: str, max_tokens: int = 2000) -> str:
        """Format conversation history as a text string for LLM context.

        Includes the most recent messages up to max_tokens.
        """
        messages = self.get_conversation_history(thread_id)
        if not messages:
            return ""

        # Build text from most recent messages
        lines = []
        token_count = 0
        for msg in reversed(messages):
            role = "User" if msg["role"] == "user" else "Assistant"
            body = self._readable_content(msg.get("content"))
            if not body:
                continue
            text = f"{role}: {body}"
            tokens = len(text.split())
            if token_count + tokens > max_tokens:
                break
            lines.append(text)
            token_count += tokens

        # Reverse to chronological order
        return "\n\n".join(reversed(lines))
