"""Tool: Generate SQL from a user question and table structures."""

from __future__ import annotations

import logging
from typing import Any

import openai

from app.envsetup.env_loader import get_openai_api_key, load_env
from app.prompt.prompts import sql_gen_prompts
from app.utils.table_struc_creation import TabStrucCreation

logger = logging.getLogger(__name__)

load_env()


def generate_sql(
    user_question: str,
    table_list: list[str],
    dialect: str = "PostgreSQL",
    error_context: str = "",
    table_structure: str | None = None,
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

        # If retrying after an error, append context so the LLM can self-correct
        if error_context:
            prompt += (
                f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                f"Error: {error_context}\n"
                f"Please fix the SQL query based on this error.\n"
            )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a world-class {dialect} SQL generator.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        sql = response.choices[0].message.content.strip()
        logger.debug("Generated SQL: %s", sql)
        return {"success": True, "sql": sql}
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
