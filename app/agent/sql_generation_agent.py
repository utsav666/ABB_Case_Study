"""Tool: Generate SQL from a user question and table structures."""

from __future__ import annotations

import logging
import os
from typing import Any

import openai

from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.prompts import sql_gen_prompts
from app.utils.table_struc_creation import TabStrucCreation

logger = logging.getLogger(__name__)

load_env()

DEFAULT_SQL_MODEL = os.getenv("SQL_GENERATION_MODEL", "gpt-4")

# Estimated USD pricing per 1M tokens (input, output).
MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4": (30.0, 60.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4.1": (10.0, 30.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
}


def _strip_sql_fences(text: str) -> str:
    """Remove surrounding markdown code fences (```sql ... ```) if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _normalize_model_name(model_name: str) -> str:
    """Normalize a model name for pricing lookup."""
    return model_name.strip().lower()


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage fields from an OpenAI response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_cost_summary(model_name: str, token_usage: dict[str, int]) -> dict[str, Any]:
    """Build estimated token cost metadata for a model usage record."""
    normalized_model_name = _normalize_model_name(model_name)
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(normalized_model_name)
    if pricing is None:
        return {
            "model": model_name,
            "pricing_available": False,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    input_price_per_million, output_price_per_million = pricing
    input_cost = (token_usage["prompt_tokens"] / 1_000_000) * input_price_per_million
    output_cost = (token_usage["completion_tokens"] / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "model": model_name,
        "pricing_available": True,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total_cost, 8),
    }


def generate_sql(
    user_question: str,
    table_list: list[str],
    dialect: str = "PostgreSQL",
    error_context: str = "",
    table_structure: str | None = None,
    conversation_history: str = "",
) -> dict[str, Any]:
    """Generate SQL for a user question.

    Parameters
    ----------
    user_question: The natural language question.
    table_list: List of relevant table names.
    dialect: SQL dialect (PostgreSQL, Snowflake, etc.)
    error_context: If retrying, the previous error message to help the LLM fix it.
    table_structure: Optional pre-built table-structure string. When provided,
        it overrides the default sample-data source (TabStrucCreation) — used to
        feed a connection's real schema graph to the generator.
    conversation_history: Prior conversation text for context continuity.

    Returns
    -------
    {"success": True, "sql": "..."} or {"success": False, "error": "..."}
    """
    try:
        openai.api_key = get_openai_api_key()
        if table_structure is None:
            table_structure = TabStrucCreation(table_list).struc_creation()

        prompt = sql_gen_prompts.format(
            table_structure=table_structure,
            user_question=user_question,
            dialect=dialect,
        )

        # Inject prior conversation for continuity (e.g. "same table as before")
        if conversation_history:
            prompt = (
                f"--- CONVERSATION HISTORY ---\n{conversation_history}\n"
                f"--- END HISTORY ---\n\n{prompt}"
            )

        # If retrying after an error, append context so the LLM can self-correct
        if error_context:
            prompt += (
                f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                f"Error: {error_context}\n"
                f"Please fix the SQL query based on this error.\n"
            )

        response = openai.chat.completions.create(
            model=DEFAULT_SQL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a world-class {dialect} SQL generator. "
                        "Return ONLY the SQL query — no explanation, no prose, "
                        "no markdown code fences. If a previous attempt failed, "
                        "return a corrected SQL query, never a description of the error."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        sql = _strip_sql_fences(response.choices[0].message.content.strip())
        token_usage = _extract_token_usage(response)
        cost_summary = _build_cost_summary(DEFAULT_SQL_MODEL, token_usage)
        logger.debug("Generated SQL: %s", sql)
        return {
            "success": True,
            "sql": sql,
            "usage": token_usage,
            "cost": cost_summary,
        }
    except Exception as e:
        logger.exception("SQL generation failed")
        return {"success": False, "error": str(e)}


# --- Backward compatibility ---
class SQLGenerationAgent:
    """Deprecated: use generate_sql() directly."""

    def __init__(
        self, user_question: str, table_list: list[str], dialect: str = "PostgreSQL"
    ):
        self.user_question = user_question
        self.table_list = table_list
        self.dialect = dialect

    def run(self) -> str:
        result = generate_sql(self.user_question, self.table_list, self.dialect)
        if result["success"]:
            return result["sql"]
        raise RuntimeError(result["error"])


# ====== Unit Tests ======
if __name__ == "__main__":
    test_question = "list of all actor played in the movie 'Anand'"
    test_tables = ["Movie", "M_Cast", "Person"]

    # Test new function API
    print("--- Testing generate_sql() ---")
    result = generate_sql(test_question, test_tables)
    print(f"Success: {result['success']}")
    if result["success"]:
        print(f"SQL: {result['sql']}")
    else:
        print(f"Error: {result['error']}")

    # Test with error_context (retry scenario)
    print("\n--- Testing generate_sql() with error_context (retry) ---")
    result_retry = generate_sql(
        test_question,
        test_tables,
        error_context="column 'actor_name' does not exist. Available columns: Name, PID",
    )
    print(f"Success: {result_retry['success']}")
    if result_retry["success"]:
        print(f"SQL: {result_retry['sql']}")

    # Test old class API
    print("\n--- Testing SQLGenerationAgent (old API) ---")
    agent = SQLGenerationAgent(test_question, test_tables)
    old_result = agent.run()
    print(f"Result: {old_result}")
