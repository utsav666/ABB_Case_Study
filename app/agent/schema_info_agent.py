"""Tool: Answer schema-related questions using the schema summary."""

from __future__ import annotations

import logging
from typing import Any

from app.envsetup.env_loader import load_env
from app.prompt.prompts import schema_info_gen_prompt
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()


def explain_schema(
    user_question: str,
    context: str = "",
    provider: str | None = None,
    model: str | None = None,
    schema_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a schema-related question.

    Parameters
    ----------
    schema_graph:
        The connected database's schema graph, from settings.schema_graphs.
        REQUIRED — there is no file-based fallback. A bundled fixture previously
        stood in here, which meant every schema question was answered from a
        movie database regardless of the connection the user had selected.

    Returns
    -------
    {"success": True, "answer": "..."} or {"success": False, "error": "..."}
    """
    try:
        llm = create_chat_llm(
            purpose="schema_info", temperature=0, provider=provider, model=model
        )

        if not schema_graph:
            raise ValueError(
                "A schema graph is required to answer schema questions. Select a "
                "connection and ensure it has been profiled."
            )

        from app.target_db.controllers import render_graph_summary

        schema_summary = render_graph_summary(schema_graph)
        logger.info(
            "SCHEMA INFO — answering from the connection's schema graph (%d tables)",
            len(schema_graph.get("nodes", [])),
            extra={"schema_source": "connection_graph"},
        )

        prompt = schema_info_gen_prompt.format(
            schema_summary=schema_summary,
            user_question=user_question,
            context=context,
        )
        response = llm.invoke(
            [
                {"role": "system", "content": "You are a Schema expert."},
                {"role": "user", "content": prompt},
            ]
        )
        answer = str(response.content).strip()
        logger.debug("Schema info response generated")
        return {"success": True, "answer": answer}
    except Exception as e:
        logger.exception("Schema info failed")
        return {"success": False, "error": str(e)}


# --- Backward compatibility ---
class SchemaInfoAgent:
    """Deprecated: use explain_schema() directly."""

    def __init__(self, user_question: str, context: str = ""):
        self.user_question = user_question
        self.context = context

    def run(self) -> str:
        result = explain_schema(self.user_question, self.context)
        if result["success"]:
            return result["answer"]
        raise RuntimeError(result["error"])


# ====== Unit Tests ======
if __name__ == "__main__":
    test_question = "what are the columns of movie table?"
    test_context = "This only needs schema detail"

    # Test new function API
    print("--- Testing explain_schema() ---")
    result = explain_schema(test_question, test_context)
    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Answer: {result['answer'][:200]}...")
    else:
        print(f"Error: {result['error']}")

    # Test old class API
    print("\n--- Testing SchemaInfoAgent (old API) ---")
    agent = SchemaInfoAgent(test_question, test_context)
    old_result = agent.run()
    print(f"Result: {old_result[:200]}...")
