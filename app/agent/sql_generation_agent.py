"""Tool: Generate SQL from a user question and table structures."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.envsetup.env_loader import get_llm_model, load_env
from app.prompt.prompts import sql_gen_prompts
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()

DEFAULT_SQL_MODEL = os.getenv("SQL_GENERATION_MODEL", "").strip().strip("'\"")

# Estimated USD pricing per 1M tokens (input, output).
# (input, output) USD per 1M tokens.
MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4": (30.0, 60.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4.1": (10.0, 30.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    # Claude. Without these, every Claude call reported
    # pricing_available=False and a $0.00 cost.
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-6-v1": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
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
    """Extract token usage fields from a LangChain AIMessage response.

    Delegates to the shared extractor. This used to read ONLY
    `response_metadata["token_usage"]` — the OpenAI shape — so Anthropic
    responses (which report via `usage_metadata` with input_tokens/
    output_tokens) always fell through to zeros. Result: sql-generator logged
    0 tokens and $0.00 on every Claude call while routing and chart-generator,
    which already used the shared helper, reported correctly.
    """
    from app.utils.llm_tracker import extract_langchain_usage

    return extract_langchain_usage(response)


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
    output_cost = (
        token_usage["completion_tokens"] / 1_000_000
    ) * output_price_per_million
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
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate SQL for a user question.

    Parameters
    ----------
    user_question: The natural language question.
    table_list: List of relevant table names.
    dialect: SQL dialect (PostgreSQL, Snowflake, etc.)
    error_context: If retrying, the previous error message to help the LLM fix it.
    table_structure: The connection's table structure, rendered from its schema
        graph. REQUIRED — there is no sample-data fallback, because generating
        SQL against a bundled fixture produced queries for a database the user
        was not connected to.
    conversation_history: Prior conversation text for context continuity.

    Returns
    -------
    {"success": True, "sql": "..."} or {"success": False, "error": "..."}
    """
    try:
        selected_sql_model = (
            model or DEFAULT_SQL_MODEL or get_llm_model("sql_generation")
        )
        llm = create_chat_llm(
            purpose="sql_generation",
            model=selected_sql_model,
            provider=provider,
            temperature=0,
        )
        if not table_structure:
            raise ValueError(
                "table_structure is required: it must be rendered from the "
                "connection's schema graph (settings.schema_graphs). There is no "
                "sample-schema fallback."
            )

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

        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are a world-class {dialect} SQL generator. "
                        "Return ONLY the SQL query - no explanation, no prose, "
                        "no markdown code fences. If a previous attempt failed, "
                        "return a corrected SQL query, never a description of the error."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        sql = _strip_sql_fences(str(response.content).strip())
        token_usage = _extract_token_usage(response)
        model_from_response = getattr(response, "response_metadata", {}).get(
            "model_name", selected_sql_model
        )
        cost_summary = _build_cost_summary(model_from_response, token_usage)
        logger.debug("Generated SQL: %s", sql)

        # Validate that the response is actually SQL, not an error message
        sql_lower = sql.lower().strip()
        if not sql_lower or any(
            sql_lower.startswith(prefix)
            for prefix in [
                "sorry",
                "i cannot",
                "i can't",
                "unable to",
                "error:",
                "the provided",
                "there is no",
                "no column",
                "not possible",
            ]
        ):
            error_msg = sql if len(sql) < 200 else sql[:200] + "..."
            logger.warning("LLM returned error message instead of SQL: %s", error_msg)
            return {"success": False, "error": f"Cannot generate SQL: {error_msg}"}

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
