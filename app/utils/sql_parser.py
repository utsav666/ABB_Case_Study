"""
### HITL — SQL WHERE Clause Parser
===================================
Extract columns and filter values from SQL WHERE clauses using sqlglot.
Used by the cardinality resolver to decide if HITL is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


@dataclass
class FilterColumn:
    """A column used in a WHERE clause with its filter value."""

    table: str | None  # Table name (may be None if not qualified)
    column: str  # Column name
    operator: str  # =, LIKE, IN, etc.
    filter_value: Any  # The literal value being filtered on


def _build_alias_map(parsed: exp.Expression) -> dict[str, str]:
    """Build a mapping of table alias → real table name from FROM/JOIN clauses."""
    alias_map: dict[str, str] = {}
    for table in parsed.find_all(exp.Table):
        real_name = table.name
        alias = table.alias
        if alias:
            alias_map[alias] = real_name
        # Also map real name to itself (for unaliased references)
        alias_map[real_name] = real_name
    return alias_map


def extract_where_filters(sql: str, dialect: str = "postgres") -> list[FilterColumn]:
    """### HITL — Parse SQL and extract all WHERE clause filter columns + values.

    Parameters
    ----------
    sql: The SQL query to parse.
    dialect: SQL dialect for parsing.

    Returns
    -------
    List of FilterColumn objects representing each filter condition.
    """
    filters: list[FilterColumn] = []

    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
        logger.warning("### HITL — SQL parse failed: %s", e)
        return filters
    except Exception as e:
        logger.warning("### HITL — Unexpected SQL parse error: %s", e)
        return filters

    # Build alias → real table name mapping from FROM/JOIN clauses
    alias_map = _build_alias_map(parsed)

    # Find all WHERE conditions
    where = parsed.find(exp.Where)
    if where is None:
        return filters

    # Walk through all comparison expressions in WHERE
    for condition in where.walk():
        node = condition[0] if isinstance(condition, tuple) else condition

        # Handle: column = 'value'
        if isinstance(node, exp.EQ):
            col_info = _extract_column_and_literal(node)
            if col_info:
                col_info.operator = "="
                filters.append(col_info)

        # Handle: column LIKE '%value%'
        elif isinstance(node, exp.Like):
            col_info = _extract_column_and_literal(node)
            if col_info:
                col_info.operator = "LIKE"
                filters.append(col_info)

        # Handle: column ILIKE '%value%'
        elif isinstance(node, exp.ILike):
            col_info = _extract_column_and_literal(node)
            if col_info:
                col_info.operator = "ILIKE"
                filters.append(col_info)

        # Handle: column IN ('a', 'b', 'c')
        elif isinstance(node, exp.In):
            col_info = _extract_in_values(node)
            if col_info:
                filters.append(col_info)

    # Resolve aliases to real table names so downstream code uses actual table names
    for f in filters:
        if f.table:
            f.table = alias_map.get(f.table, f.table)

    logger.info("### HITL — Extracted %d filter(s) from SQL", len(filters))
    for f in filters:
        logger.debug(
            "  Filter: %s.%s %s %r", f.table, f.column, f.operator, f.filter_value
        )

    return filters


def _extract_column_and_literal(node: exp.Binary) -> FilterColumn | None:
    """Extract column name and literal value from a binary comparison."""
    left = node.left
    right = node.right

    # column = 'literal'
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
        return FilterColumn(
            table=left.table or None,
            column=left.name,
            operator="",
            filter_value=right.this,
        )

    # 'literal' = column (reversed)
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
        return FilterColumn(
            table=right.table or None,
            column=right.name,
            operator="",
            filter_value=left.this,
        )

    return None


def _extract_in_values(node: exp.In) -> FilterColumn | None:
    """Extract column and values from an IN clause."""
    this = node.this
    if not isinstance(this, exp.Column):
        return None

    values = []
    for expr in node.expressions:
        if isinstance(expr, exp.Literal):
            values.append(expr.this)

    if not values:
        return None

    return FilterColumn(
        table=this.table or None,
        column=this.name,
        operator="IN",
        filter_value=values,
    )


def rewrite_sql_filter(
    sql: str, col_filter: FilterColumn, new_value: str, dialect: str = "postgres"
) -> str:
    """### HITL — Rewrite a SQL filter value after human selection.

    Replaces the old filter value in the SQL with the human-chosen value.
    """
    old_value = col_filter.filter_value

    if isinstance(old_value, list):
        # IN clause — replace first match
        old_value = old_value[0] if old_value else ""

    # Simple string replacement (safe for literal values)
    # For production, use sqlglot tree manipulation
    new_sql = sql.replace(str(old_value), new_value)
    logger.info("### HITL — Rewrote SQL filter: '%s' → '%s'", old_value, new_value)
    return new_sql


# ====== Unit Test ======
if __name__ == "__main__":
    print("--- Test 1: Unaliased query (existing behavior) ---")
    test_sql = """
    SELECT * FROM customers
    WHERE customer_name = 'ABC Corp'
    AND region LIKE '%North%'
    AND status IN ('active', 'pending')
    """
    results = extract_where_filters(test_sql)
    for r in results:
        print(f"  {r.table}.{r.column} {r.operator} {r.filter_value!r}")

    print("\n--- Test 2: Aliased multi-join query (was broken) ---")
    test_sql_alias = """
    SELECT m.title FROM movie m
    JOIN m_country mc ON m.mid = mc.mid
    JOIN country c ON mc.cid = c.cid
    WHERE c.name = 'India'
    """
    results2 = extract_where_filters(test_sql_alias)
    for r in results2:
        print(f"  {r.table}.{r.column} {r.operator} {r.filter_value!r}")
    # Should print: country.name = 'India' (not c.name)

    print("\n--- Test 3: Mixed aliased + unaliased ---")
    test_sql_mixed = """
    SELECT m.title, l.name FROM movie m
    JOIN m_language ml ON m.mid = ml.mid
    JOIN language l ON ml.laid = l.laid
    WHERE l.name = 'Hindi' AND m.year > 2010
    """
    results3 = extract_where_filters(test_sql_mixed)
    for r in results3:
        print(f"  {r.table}.{r.column} {r.operator} {r.filter_value!r}")
    # Should print: language.name = 'Hindi' (not l.name)
