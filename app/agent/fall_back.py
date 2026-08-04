"""Tool: Answer general (non-DB) questions."""

from __future__ import annotations

import logging
from typing import Any

from app.envsetup.env_loader import load_env
from app.prompt.prompts import fall_back_prompt
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()


def general_answer(
    user_question: str,
    conversation_history: list[dict[str, str]] | None = None,
    router_reasoning: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a general-purpose question not related to the database.

    Returns
    -------
    {"success": True, "answer": "...", "response": <AIMessage>} or {"success": False, "error": "..."}
    """
    try:
        llm = create_chat_llm(
            purpose="general_answer", temperature=0.2, provider=provider, model=model
        )

        messages = [{"role": "system", "content": fall_back_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        if router_reasoning:
            messages.append(
                {"role": "system", "content": f"Router reasoning: {router_reasoning}"}
            )

        messages.append({"role": "user", "content": user_question})

        response = llm.invoke(messages)
        logger.debug("General answer generated")
        return {"success": True, "answer": response.content, "response": response}
    except Exception as e:
        logger.exception("General answer failed")
        return {"success": False, "error": str(e)}


# --- Backward compatibility ---
class GeneralAnswerAgent:
    """Deprecated: use general_answer() directly."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        pass

    def run(self, conversation_history, user_question, router_reasoning) -> str:
        history = conversation_history if isinstance(conversation_history, list) else []
        result = general_answer(user_question, history, router_reasoning)
        if result["success"]:
            return result["answer"]
        raise RuntimeError(result["error"])


# ====== Unit Tests ======
if __name__ == "__main__":
    test_question = "what is love?"
    test_history = [{"role": "user", "content": "hello"}]
    test_reasoning = "User is asking a general knowledge question."

    # Test new function API
    print("--- Testing general_answer() ---")
    result = general_answer(test_question, test_history, test_reasoning)
    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Answer: {result['answer'][:200]}...")
    else:
        print(f"Error: {result['error']}")

    # Test old class API
    print("\n--- Testing GeneralAnswerAgent (old API) ---")
    agent = GeneralAnswerAgent()
    old_result = agent.run(test_history, test_question, test_reasoning)
    print(f"Result: {old_result[:200]}...")
