"""Tool: Identify relevant tables for a user question."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.envsetup.env_loader import load_env
from app.prompt.prompts import table_identify_prompts
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()


class TableIdentification(BaseModel):
    user_question: str = Field(..., description="question asked by the user")
    relevant_tables: list[str] = Field(
        ..., description="Identify the tables absolutely relevant to the user question"
    )


def identify_tables(
    user_question: str,
    schema_summary: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Identify relevant tables for a user question.

    Parameters
    ----------
    user_question: The natural-language question.
    schema_summary: Optional connection-scoped schema summary; falls back to the
        default (sample/config) summary when not provided.

    Returns
    -------
    {"success": True, "tables": [...], "response": <AIMessage>} or {"success": False, "error": "..."}
    """
    try:
        if not schema_summary:
            raise ValueError(
                "schema_summary is required: render it from the connection's "
                "schema graph (settings.schema_graphs). There is no sample-schema "
                "fallback."
            )
        llm = create_chat_llm(
            purpose="table_identification",
            temperature=0,
            provider=provider,
            model=model,
        )
        parser = JsonOutputParser(pydantic_object=TableIdentification)
        prompt = ChatPromptTemplate.from_template(table_identify_prompts)
        formatted = prompt.format_messages(
            schema_summary=schema_summary,
            user_question=user_question,
            format_instructions=parser.get_format_instructions(),
        )
        response = llm.invoke(formatted)
        parsed = parser.parse(str(response.content))
        logger.debug("Identified tables: %s", parsed.get("relevant_tables"))
        return {
            "success": True,
            "tables": parsed.get("relevant_tables", []),
            "response": response,  # Include response for token tracking
        }
    except Exception as e:
        logger.exception("Table identification failed")
        return {"success": False, "error": str(e)}


# --- Backward compatibility ---
class TableIdentiferAgent:
    """Deprecated: use identify_tables() directly."""

    def __init__(self, user_question: str):
        self.user_question = user_question

    def run(self) -> dict[str, Any]:
        result = identify_tables(self.user_question)
        if result["success"]:
            return {"relevant_tables": result["tables"]}
        raise RuntimeError(result["error"])


# ====== Unit Tests ======
if __name__ == "__main__":
    test_question = "list of all actor played in the movie 'Anand'"

    # Test new function API
    print("--- Testing identify_tables() ---")
    result = identify_tables(test_question)
    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"Tables: {result['tables']}")
    else:
        print(f"Error: {result['error']}")

    # Test old class API (backward compat)
    print("\n--- Testing TableIdentiferAgent (old API) ---")
    agent = TableIdentiferAgent(test_question)
    old_result = agent.run()
    print(f"Result: {old_result}")
