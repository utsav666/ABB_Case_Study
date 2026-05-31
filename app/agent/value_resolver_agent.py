"""
### HITL — Value Resolver Agent
================================
LLM chain that determines if it can confidently resolve a user's filter value
against actual column values. If confidence is low → triggers HITL.

This is a CHAIN (single LLM call with structured output), not a full agent.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.envsetup.env_loader import get_openai_api_key, load_env

logger = logging.getLogger(__name__)

load_env()


# ─── Structured Output ────────────────────────────────────────────────────────


class ValueResolution(BaseModel):
    """### HITL — LLM's confidence in resolving a filter value."""

    confidence: Literal["high", "low"]
    match: str | None  # The resolved value (if confidence is high)
    alternatives: list[str]  # Top candidate matches (shown to human if low)
    reasoning: str  # Why the LLM is confident or not


# ─── Resolution Prompt ────────────────────────────────────────────────────────

VALUE_RESOLVER_PROMPT = """### HITL — Value Resolution Task

You are a data value resolver. The user asked a question that filters on a database column.
Your job is to determine if you can confidently match the user's intended value to an actual value in the column.

**Column:** {column_name}
**Table:** {table_name}
**Total unique values (cardinality):** {cardinality}
**User typed:** "{user_value}"
**Sample of actual values in this column (closest matches first):**
{sample_values}

**Rules:**
- If there is ONE obvious exact or near-exact match → confidence = "high", return it as "match"
- If the user's value is ambiguous, has multiple plausible matches, or no clear match → confidence = "low"
- Always provide top 5 alternatives (closest matches) in "alternatives"
- If cardinality is very low (< 20) and you can see ALL values, be more confident
- If cardinality is very high and you only see a sample, be less confident unless there's an exact match

{format_instructions}
"""


# ─── Main Function ────────────────────────────────────────────────────────────


def resolve_filter_value(
    column_name: str,
    table_name: str,
    user_value: str,
    actual_values: list[str],
    cardinality: int,
) -> dict[str, Any]:
    """### HITL — Ask LLM if it can confidently resolve a filter value.

    Parameters
    ----------
    column_name: The database column being filtered.
    table_name: The table containing the column.
    user_value: What the user typed / what's in the SQL WHERE clause.
    actual_values: Sample of actual values from the column (pre-sorted by fuzzy similarity).
    cardinality: Total number of unique values in this column.

    Returns
    -------
    {"confidence": "high"|"low", "match": "...", "alternatives": [...], "reasoning": "..."}
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=SecretStr(get_openai_api_key()),
        )
        parser = JsonOutputParser(pydantic_object=ValueResolution)

        prompt = ChatPromptTemplate.from_template(VALUE_RESOLVER_PROMPT)

        # Format sample values for display
        sample_display = "\n".join(f"  - {v}" for v in actual_values[:30])

        messages = prompt.format_messages(
            column_name=column_name,
            table_name=table_name,
            cardinality=cardinality,
            user_value=user_value,
            sample_values=sample_display,
            format_instructions=parser.get_format_instructions(),
        )

        response = llm.invoke(messages)
        parsed = parser.parse(str(response.content))
        resolution = ValueResolution(**parsed)

        logger.info(
            "### HITL — Value resolution: column=%s, user='%s', confidence=%s, match=%s",
            column_name,
            user_value,
            resolution.confidence,
            resolution.match,
        )

        return {
            "confidence": resolution.confidence,
            "match": resolution.match,
            "alternatives": resolution.alternatives,
            "reasoning": resolution.reasoning,
        }

    except Exception as e:
        logger.exception("### HITL — Value resolver failed")
        # On failure, default to LOW confidence → triggers HITL (safe fallback)
        return {
            "confidence": "low",
            "match": None,
            "alternatives": actual_values[:5],
            "reasoning": f"Resolution failed: {e}",
        }


# ─── Helper: Get column values from DB ───────────────────────────────────────


def get_column_distinct_values(
    table_name: str, column_name: str, limit: int = 100
) -> tuple[list[str], int]:
    """### HITL — Query DB for distinct values and cardinality of a column.

    Returns
    -------
    (list of distinct values (up to limit), total cardinality)
    """
    from app.utils.sql_executor import SqlExec

    executor = SqlExec()

    # Get cardinality
    # ### HITL — Resolve schema from schema_preserve.json (not hardcoded)
    from app.sampledata.data_loader import DataLoader

    schema_data = DataLoader.load_data()
    table_schema = ""
    if "tables" in schema_data and table_name in schema_data["tables"]:
        table_schema = schema_data["tables"][table_name].get("schema", "")

    qualified_table = f'{table_schema}."{table_name}"' if table_schema else f'"{table_name}"'

    count_sql = f'SELECT COUNT(DISTINCT "{column_name}") as cnt FROM {qualified_table}'
    count_result = executor.sql_executor(count_sql)

    if isinstance(count_result, str):
        logger.warning("### HITL — Cardinality query failed: %s", count_result)
        return [], 0

    cardinality = int(count_result.iloc[0]["cnt"])

    # Get sample distinct values
    values_sql = f'SELECT DISTINCT "{column_name}" FROM {qualified_table} WHERE "{column_name}" IS NOT NULL LIMIT {limit}'
    values_result = executor.sql_executor(values_sql)

    if isinstance(values_result, str):
        logger.warning("### HITL — Values query failed: %s", values_result)
        return [], cardinality

    values = values_result[column_name].astype(str).tolist()

    logger.info(
        "### HITL — Column '%s.%s': cardinality=%d, fetched=%d values",
        table_name,
        column_name,
        cardinality,
        len(values),
    )

    return values, cardinality


def get_fuzzy_matches(
    user_value: str, actual_values: list[str], top_n: int = 20
) -> list[str]:
    """### HITL — Sort actual values by fuzzy similarity to user's input.

    Uses simple containment + prefix matching (no external deps).
    For production, consider rapidfuzz or pg_trgm.
    """
    user_lower = user_value.lower().strip()

    scored: list[tuple[float, str]] = []
    for val in actual_values:
        val_lower = val.lower().strip()

        # Scoring heuristic
        score = 0.0
        if val_lower == user_lower:
            score = 1.0  # Exact match
        elif val_lower.startswith(user_lower) or user_lower.startswith(val_lower):
            score = 0.8  # Prefix match
        elif user_lower in val_lower or val_lower in user_lower:
            score = 0.6  # Containment
        else:
            # Character overlap ratio
            common = set(user_lower) & set(val_lower)
            score = len(common) / max(len(set(user_lower)), len(set(val_lower)), 1) * 0.4

        scored.append((score, val))

    scored.sort(key=lambda x: -x[0])
    return [val for _, val in scored[:top_n]]


# ====== Unit Test ======
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Simulate
    test_values = [
        "ABC Corp Ltd",
        "ABC Corporation",
        "ABC Corp Inc",
        "XYZ Industries",
        "ABCD Holdings",
        "New York Office",
    ]

    result = resolve_filter_value(
        column_name="customer_name",
        table_name="customers",
        user_value="ABC Corp",
        actual_values=get_fuzzy_matches("ABC Corp", test_values),
        cardinality=len(test_values),
    )
    print(f"Confidence: {result['confidence']}")
    print(f"Match: {result['match']}")
    print(f"Alternatives: {result['alternatives']}")
    print(f"Reasoning: {result['reasoning']}")
