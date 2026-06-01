"""Dialect-aware SQL templates for column-level data profiling.

Each helper returns a **raw SQL string** with named placeholders
(:schema_name, :table_name, :column_name) so they can be executed via
SQLAlchemy ``text()`` binds.

Dialect detection is based on the SQLAlchemy engine's ``dialect.name``
attribute (e.g. ``"postgresql"``, ``"snowflake"``, ``"bigquery"``).

Supported column-type families:
  - TEXT / VARCHAR / CHAR  → ``text_profile_query``
  - NUMERIC / INT / FLOAT  → ``numeric_profile_query``
  - DATE / TIMESTAMP       → ``temporal_profile_query``
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Cardinality thresholds (configurable via environment / .env)
# ---------------------------------------------------------------------------

#: Columns with <= this many distinct values will have all values fetched.
LOW_CARDINALITY_THRESHOLD = int(os.getenv("PROFILER_LOW_CARDINALITY_THRESHOLD", "100"))

#: For high-cardinality columns, fetch the top-N most frequent values.
HIGH_CARDINALITY_TOP_N = int(os.getenv("PROFILER_HIGH_CARDINALITY_TOP_N", "50"))

# ---------------------------------------------------------------------------
# Column-type classification helpers
# ---------------------------------------------------------------------------

_TEXT_TYPE_PREFIXES = (
    "text",
    "varchar",
    "char",
    "character",
    "string",
    "nvarchar",
    "nchar",
    "clob",
    "nclob",
    "bpchar",
    "name",
    "citext",
    "uuid",
)

_NUMERIC_TYPE_PREFIXES = (
    "int",
    "integer",
    "bigint",
    "smallint",
    "tinyint",
    "numeric",
    "decimal",
    "float",
    "double",
    "real",
    "number",
    "money",
    "serial",
    "bigserial",
)

_TEMPORAL_TYPE_PREFIXES = (
    "date",
    "timestamp",
    "datetime",
    "time",
    "timestamptz",
    "timestamp with time zone",
    "timestamp without time zone",
)


def classify_column_type(data_type: str) -> str:
    """Return one of ``'text'``, ``'numeric'``, ``'temporal'``, or ``'other'``.

    Classification is based on the lowercase column type string reported by
    ``information_schema.columns.data_type``.
    """
    normalized = data_type.strip().lower()

    if any(normalized.startswith(prefix) for prefix in _TEMPORAL_TYPE_PREFIXES):
        return "temporal"

    if any(normalized.startswith(prefix) for prefix in _NUMERIC_TYPE_PREFIXES):
        return "numeric"

    if any(normalized.startswith(prefix) for prefix in _TEXT_TYPE_PREFIXES):
        return "text"

    # BOOLEAN, BYTEA, JSON, ARRAY, etc. — skip profiling for these.
    return "other"


# ═══════════════════════════════════════════════════════════════════════════
# TEXT column profiling
# ═══════════════════════════════════════════════════════════════════════════


def text_cardinality_query(dialect: str = "postgresql") -> str:
    """Return SQL that fetches cardinality + null ratio for a TEXT column."""
    # Standard SQL — works on Postgres, Snowflake, BigQuery
    return """
        SELECT
            COUNT(*)                                       AS total_rows,
            COUNT(DISTINCT {col})                          AS cardinality,
            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count
        FROM {schema}.{table}
    """


def text_distinct_values_query(dialect: str = "postgresql") -> str:
    """Return SQL that fetches all distinct values for low-cardinality TEXT columns.

    Should only be used when cardinality <= LOW_CARDINALITY_THRESHOLD.
    """
    if dialect == "bigquery":
        return """
            SELECT DISTINCT {col} AS value
            FROM `{schema}.{table}`
            WHERE {col} IS NOT NULL
            ORDER BY {col}
        """
    # Postgres / Snowflake
    return """
        SELECT DISTINCT {col} AS value
        FROM {schema}.{table}
        WHERE {col} IS NOT NULL
        ORDER BY {col}
    """


def text_top_values_query(
    dialect: str = "postgresql",
    limit: int = HIGH_CARDINALITY_TOP_N,
) -> str:
    """Return SQL that fetches the top-N most frequent values for high-cardinality TEXT."""
    if dialect == "bigquery":
        return f"""
            SELECT {"{col}"} AS value, COUNT(*) AS frequency
            FROM `{"{schema}"}.{"{table}"}`
            WHERE {"{col}"} IS NOT NULL
            GROUP BY {"{col}"}
            ORDER BY frequency DESC
            LIMIT {limit}
        """
    # Postgres / Snowflake
    return f"""
        SELECT {{col}} AS value, COUNT(*) AS frequency
        FROM {{schema}}.{{table}}
        WHERE {{col}} IS NOT NULL
        GROUP BY {{col}}
        ORDER BY frequency DESC
        LIMIT {limit}
    """


def text_sample_values_query(
    dialect: str = "postgresql",
    sample_size: int = 10,
) -> str:
    """Return SQL that fetches a small random sample of non-null TEXT values."""
    if dialect in ("postgresql",):
        return f"""
            SELECT value FROM (
                SELECT DISTINCT {{col}} AS value
                FROM {{schema}}.{{table}}
                WHERE {{col}} IS NOT NULL
            ) sub
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """
    if dialect == "snowflake":
        return f"""
            SELECT DISTINCT {{col}} AS value
            FROM {{schema}}.{{table}} SAMPLE ({sample_size} ROWS)
            WHERE {{col}} IS NOT NULL
        """
    # BigQuery
    return f"""
        SELECT DISTINCT {{col}} AS value
        FROM `{{schema}}.{{table}}`
        WHERE {{col}} IS NOT NULL
        ORDER BY RAND()
        LIMIT {sample_size}
    """


# ═══════════════════════════════════════════════════════════════════════════
# NUMERIC column profiling
# ═══════════════════════════════════════════════════════════════════════════


def numeric_profile_query(dialect: str = "postgresql") -> str:
    """Return SQL that computes min, max, avg, median, null ratio for a NUMERIC column."""
    if dialect == "bigquery":
        return """
            SELECT
                COUNT(*)                                       AS total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                MIN({col})                                     AS min_value,
                MAX({col})                                     AS max_value,
                AVG({col})                                     AS mean_value,
                APPROX_QUANTILES({col}, 2)[OFFSET(1)]         AS median_value,
                COUNT(DISTINCT {col})                          AS cardinality
            FROM `{schema}.{table}`
        """
    if dialect == "snowflake":
        return """
            SELECT
                COUNT(*)                                       AS total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                MIN({col})                                     AS min_value,
                MAX({col})                                     AS max_value,
                AVG({col})                                     AS mean_value,
                MEDIAN({col})                                  AS median_value,
                COUNT(DISTINCT {col})                          AS cardinality
            FROM {schema}.{table}
        """
    # PostgreSQL
    return """
        SELECT
            COUNT(*)                                       AS total_rows,
            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
            MIN({col})                                     AS min_value,
            MAX({col})                                     AS max_value,
            AVG({col})                                     AS mean_value,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS median_value,
            COUNT(DISTINCT {col})                          AS cardinality
        FROM {schema}.{table}
    """


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL (DATE / TIMESTAMP) column profiling
# ═══════════════════════════════════════════════════════════════════════════


def temporal_profile_query(dialect: str = "postgresql") -> str:
    """Return SQL that computes date range, granularity hints, and null ratio."""
    if dialect == "bigquery":
        return """
            SELECT
                COUNT(*)                                       AS total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                CAST(MIN({col}) AS STRING)                     AS min_date,
                CAST(MAX({col}) AS STRING)                     AS max_date,
                COUNT(DISTINCT DATE({col}))                    AS distinct_days,
                COUNT(DISTINCT FORMAT_TIMESTAMP('%Y-%m', {col})) AS distinct_months,
                COUNT(DISTINCT FORMAT_TIMESTAMP('%Y', {col}))  AS distinct_years
            FROM `{schema}.{table}`
        """
    if dialect == "snowflake":
        return """
            SELECT
                COUNT(*)                                       AS total_rows,
                SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
                MIN({col})::VARCHAR                            AS min_date,
                MAX({col})::VARCHAR                            AS max_date,
                COUNT(DISTINCT DATE_TRUNC('DAY', {col}))       AS distinct_days,
                COUNT(DISTINCT DATE_TRUNC('MONTH', {col}))     AS distinct_months,
                COUNT(DISTINCT DATE_TRUNC('YEAR', {col}))      AS distinct_years
            FROM {schema}.{table}
        """
    # PostgreSQL
    return """
        SELECT
            COUNT(*)                                       AS total_rows,
            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS null_count,
            MIN({col})::TEXT                               AS min_date,
            MAX({col})::TEXT                               AS max_date,
            COUNT(DISTINCT DATE_TRUNC('day', {col}))       AS distinct_days,
            COUNT(DISTINCT DATE_TRUNC('month', {col}))     AS distinct_months,
            COUNT(DISTINCT DATE_TRUNC('year', {col}))      AS distinct_years
        FROM {schema}.{table}
    """


def infer_granularity(
    distinct_days: int, distinct_months: int, distinct_years: int
) -> str:
    """Infer the most likely time granularity from profiling counts.

    Returns one of: ``'daily'``, ``'monthly'``, ``'yearly'``, ``'sub-daily'``,
    or ``'irregular'``.
    """
    if distinct_days == 0 and distinct_months == 0:
        return "unknown"

    if distinct_years > 0 and distinct_months <= distinct_years:
        return "yearly"

    if distinct_days > 0 and distinct_months > 0:
        # If days ≈ months * 30, likely daily; if days ≈ months, likely monthly
        days_per_month = distinct_days / max(distinct_months, 1)
        if days_per_month > 15:
            return "daily"
        if days_per_month > 1.5:
            return "irregular"
        return "monthly"

    return "irregular"
