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
from pydantic import BaseModel

from app.envsetup.env_loader import load_env
from app.utils.llm_factory import create_chat_llm

logger = logging.getLogger(__name__)

load_env()

CATEGORICAL_COLUMN_NAME_PARTS = (
    "category",
    "field_name",
    "fieldname",
    "kind",
    "metric",
    "status",
    "type",
)
DISCOVERED_VALUE_LIMIT = 100


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
    provider: str | None = None,
    model: str | None = None,
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
        llm = create_chat_llm(
            purpose="value_resolver", temperature=0, provider=provider, model=model
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


def _lookup_schema_for_table(
    table_name: str, schema_graph: dict[str, Any] | None
) -> str | None:
    """Find a bare table's schema in the connection's schema graph.

    The graph comes from the DB (schema_graph table) and is the source of truth.
    Returns None when the table isn't found, so callers can decide the fallback.
    """
    if not schema_graph:
        return None
    for node in schema_graph.get("nodes", []):
        if str(node.get("table", "")).lower() == table_name.lower():
            return node.get("schema") or None
    return None


def _qualify_table(table_name: str, schema_graph: dict[str, Any] | None = None) -> str:
    """### HITL — Build a schema-qualified, quoted table name.

    Accepts "schema.table" or a bare "table". For a bare name the schema is
    resolved from the connection's schema graph in ``settings.schema_graphs``,
    which is the ONLY source of truth — there is no file-based fallback.

    Raises
    ------
    ValueError
        When a bare table cannot be found in the graph. Guessing "public" here
        produced a query against a schema the table does not live in, which
        surfaced later as a confusing "relation does not exist" or, worse, as a
        silently empty result.
    """
    if "." in table_name:
        table_schema, actual_table_name = table_name.split(".", 1)
        return f'"{table_schema}"."{actual_table_name}"'

    table_schema = _lookup_schema_for_table(table_name, schema_graph)
    if not table_schema:
        raise ValueError(
            f"Table {table_name!r} is not in the connection's schema graph. "
            "Ensure the connection has been profiled so settings.schema_graphs "
            "contains it."
        )

    return f'"{table_schema}"."{table_name}"'


def value_exists_exactly(
    table_name: str,
    column_name: str,
    user_value: str,
    schema_graph: dict[str, Any] | None = None,
) -> bool:
    """### HITL — Does the user's value already match a value in this column?

    This is the ONLY signal that decides whether disambiguation is worthwhile,
    and it is deliberately TYPE-AGNOSTIC: the column is cast to text so a year
    (INTEGER), a pincode (INTEGER *or* TEXT), and a country (TEXT) are all
    checked the same way. If an exact match exists there is nothing to
    disambiguate → the caller skips fuzzy/LLM/HITL. A miss is precisely what
    makes resolution meaningful (typo, alias, wrong casing).

    Fails open to False (i.e. keep resolution) on any query error, so we never
    silently skip a genuinely ambiguous filter.
    """
    from app.utils.sql_executor import SqlExec

    value_str = str(user_value).strip()
    if not value_str:
        return False

    try:
        qualified_table = _qualify_table(table_name, schema_graph)
    except ValueError:
        # Unknown table → we cannot verify the value. Fail open (False) so the
        # caller still attempts resolution rather than trusting an unchecked
        # filter value.
        logger.warning(
            "### HITL — Cannot verify %r in %r: table not in schema graph",
            value_str,
            table_name,
        )
        return False

    # Escape single quotes to keep the literal safe.
    safe_value = value_str.replace("'", "''")
    exists_sql = (
        f"SELECT 1 FROM {qualified_table} "
        f"WHERE CAST(\"{column_name}\" AS TEXT) = '{safe_value}' LIMIT 1"
    )

    executor = SqlExec()
    result = executor.sql_executor(exists_sql)

    if isinstance(result, str):
        # Query failed (bad column, DB error, ...) → don't skip resolution.
        logger.warning("### HITL — Existence check failed: %s", result)
        return False

    found = not result.empty
    logger.info(
        "### HITL — Exact-match check '%s.%s' = '%s' → %s",
        table_name,
        column_name,
        value_str,
        "exists" if found else "no match",
    )
    return found


def get_column_distinct_values(
    table_name: str,
    column_name: str,
    limit: int = 100,
    schema_graph: dict[str, Any] | None = None,
) -> tuple[list[str], int]:
    """### HITL — Query DB for distinct values and cardinality of a column.

    Returns
    -------
    (list of distinct values (up to limit), total cardinality)
    """
    from app.utils.sql_executor import SqlExec

    executor = SqlExec()

    # Schema comes from the connection's graph in settings.schema_graphs, the
    # only source of truth. An unknown table means no suggestions to offer, so
    # return empty and let the caller ask the human directly.
    try:
        qualified_table = _qualify_table(table_name, schema_graph)
    except ValueError:
        logger.warning(
            "### HITL — No distinct values for %r: table not in schema graph",
            table_name,
        )
        return [], 0

    count_sql = f'SELECT COUNT(DISTINCT "{column_name}") as cnt FROM {qualified_table}'
    logger.info("### HITL — cardinality query: %s", count_sql)
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


def discover_column_values(
    table_names: list[str],
    schema_graph: dict[str, Any] | None,
) -> dict[str, dict[str, list[str]]]:
    """Fetch actual values for categorical columns before SQL generation."""
    if not schema_graph:
        return {}

    discovered_values: dict[str, dict[str, list[str]]] = {}
    selected_table_names = {table_name.lower() for table_name in table_names}

    for table_node in schema_graph.get("nodes", []):
        qualified_table_name = str(table_node.get("qualified_name", ""))
        table_name = str(table_node.get("table", ""))
        if (
            qualified_table_name.lower() not in selected_table_names
            and table_name.lower() not in selected_table_names
        ):
            continue

        for column in table_node.get("columns", []):
            column_name = str(column.get("raw_name", ""))
            logical_name = str(column.get("logical_name") or "")
            searchable_name = f"{column_name} {logical_name}".lower()
            if not any(part in searchable_name for part in CATEGORICAL_COLUMN_NAME_PARTS):
                continue

            actual_values, _ = get_column_distinct_values(
                qualified_table_name,
                column_name,
                limit=DISCOVERED_VALUE_LIMIT,
                schema_graph=schema_graph,
            )
            if actual_values:
                discovered_values.setdefault(qualified_table_name, {})[
                    column_name
                ] = actual_values

    return discovered_values


def get_fuzzy_matches(
    user_value: str, actual_values: list[str], top_n: int = 20, min_score: float = 0.2
) -> list[str]:
    """### HITL — Sort actual values by fuzzy similarity to user's input.

    Uses simple containment + prefix matching (no external deps).
    For production, consider rapidfuzz or pg_trgm.

    Values scoring below `min_score` are DROPPED rather than ranked. Without a
    floor, the character-overlap fallback gives every value some tiny nonzero
    score, so a value with no real relative (e.g. "Women" against
    "Category A/B/C") still comes back "ranked" — and the top one gets offered
    to the user as though it were a match. An empty list is the honest answer.
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
            score = (
                len(common) / max(len(set(user_lower)), len(set(val_lower)), 1) * 0.4
            )

        if score >= min_score:
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
