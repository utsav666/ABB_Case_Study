"""Phase 1 — Pure-SQL statistical data profiler.

Connects to the target database (via a SQLAlchemy engine or URL) and
profiles every column in the selected tables:

  - **TEXT** columns  → cardinality, distinct values (low-card) or top-N
    frequencies (high-card), sample values.
  - **NUMERIC** columns → min, max, mean, median, cardinality, null ratio.
  - **TEMPORAL** columns → date range, granularity, null ratio.

The output is a ``dict`` that can be merged directly into the
``schema_preserve``-style JSON config under each column key.

Usage::

    from app.semantic_catalog.data_profiler import DataProfiler

    profiler = DataProfiler(engine_or_url)
    profile = profiler.profile_table("public", "sales_sample")
    # profile == {"category": {"cardinality": 12, ...}, "date": {...}, ...}
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.semantic_catalog.profiling_queries import (
    HIGH_CARDINALITY_TOP_N,
    LOW_CARDINALITY_THRESHOLD,
    classify_column_type,
    infer_granularity,
    numeric_profile_query,
    temporal_profile_query,
    text_cardinality_query,
    text_distinct_values_query,
    text_sample_values_query,
    text_top_values_query,
)

logger = logging.getLogger(__name__)


class DataProfiler:
    """Stateless column-level data profiler backed by live SQL queries.

    Parameters
    ----------
    engine_or_url:
        Either a live ``sqlalchemy.engine.Engine`` or a connection-URL string.
        The profiler creates (and disposes) its own engine when a string is
        provided.
    """

    def __init__(self, engine_or_url: Engine | str) -> None:
        if isinstance(engine_or_url, str):
            self._engine: Engine = create_engine(engine_or_url)
            self._owns_engine = True
        else:
            self._engine = engine_or_url
            self._owns_engine = False

        self._dialect: str = self._engine.dialect.name  # e.g. "postgresql"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_table(
        self,
        schema_name: str,
        table_name: str,
        columns: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Profile every column in a table and return per-column metadata.

        Parameters
        ----------
        schema_name:
            Database schema (e.g. ``"public"`` or ``"sales"``).
        table_name:
            Unqualified table name.
        columns:
            Optional pre-fetched column dict from the schema config.  Each key
            is a column name and the value must contain at least
            ``{"type": "<DATA_TYPE>"}``.  When ``None``, the profiler will
            query ``information_schema.columns`` to discover them.

        Returns
        -------
        dict
            ``{column_name: {profile_data...}, ...}``
        """
        if columns is None:
            columns = self._discover_columns(schema_name, table_name)

        profiles: dict[str, dict[str, Any]] = {}

        for col_name, col_meta in columns.items():
            data_type = col_meta.get("type", "")
            col_family = classify_column_type(data_type)

            try:
                if col_family == "text":
                    profiles[col_name] = self._profile_text_column(
                        schema_name,
                        table_name,
                        col_name,
                    )
                elif col_family == "numeric":
                    profiles[col_name] = self._profile_numeric_column(
                        schema_name,
                        table_name,
                        col_name,
                    )
                elif col_family == "temporal":
                    profiles[col_name] = self._profile_temporal_column(
                        schema_name,
                        table_name,
                        col_name,
                    )
                else:
                    profiles[col_name] = {
                        "profile_type": "skipped",
                        "reason": col_family,
                    }
            except Exception:
                logger.exception(
                    "Failed to profile column %s.%s.%s",
                    schema_name,
                    table_name,
                    col_name,
                )
                profiles[col_name] = {"profile_type": "error"}

        return profiles

    def profile_tables(
        self,
        schema_name: str,
        tables: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Profile multiple tables.  ``tables`` mirrors the schema config
        structure: ``{qualified_table_name: {"columns": {...}, ...}}``.

        Returns
        -------
        dict
            ``{qualified_table_name: {col: profile, ...}, ...}``
        """
        result: dict[str, dict[str, dict[str, Any]]] = {}

        for table_name, table_info in tables.items():
            # Schema can be a field inside the table dict or part of a qualified name
            table_schema = table_info.get("schema") or (
                table_name.rsplit(".", 1)[0] if "." in table_name else schema_name
            )
            bare_name = (
                table_name.rsplit(".", 1)[-1] if "." in table_name else table_name
            )

            columns = table_info.get("columns", {})
            logger.info(
                "Profiling table %s.%s (%d columns)…",
                table_schema,
                bare_name,
                len(columns),
            )

            result[table_name] = self.profile_table(
                table_schema,
                bare_name,
                columns,
            )

        return result

    def dispose(self) -> None:
        """Dispose the engine if this profiler created it."""
        if self._owns_engine:
            self._engine.dispose()

    # ------------------------------------------------------------------
    # Column discovery (fallback when caller doesn't pass columns)
    # ------------------------------------------------------------------

    def _discover_columns(
        self,
        schema_name: str,
        table_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Query information_schema to get column names and data types."""
        query = text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = :schema_name
              AND table_name   = :table_name
            ORDER BY ordinal_position
        """)
        with self._engine.connect() as conn:
            rows = conn.execute(
                query,
                {"schema_name": schema_name, "table_name": table_name},
            ).mappings()
            return {
                row["column_name"]: {"type": row["data_type"].upper()} for row in rows
            }

    # ------------------------------------------------------------------
    # TEXT profiling
    # ------------------------------------------------------------------

    def _profile_text_column(
        self,
        schema: str,
        table: str,
        col: str,
    ) -> dict[str, Any]:
        """Profile a TEXT/VARCHAR column."""
        profile: dict[str, Any] = {"profile_type": "text"}

        # Step 1: Get cardinality + null count
        sql = text(
            text_cardinality_query(self._dialect)
            .replace("{col}", f'"{col}"')
            .replace("{schema}", f'"{schema}"')
            .replace("{table}", f'"{table}"')
        )
        with self._engine.connect() as conn:
            row = dict(conn.execute(sql).mappings().first() or {})

        total_rows = int(row.get("total_rows", 0))
        cardinality = int(row.get("cardinality", 0))
        null_count = int(row.get("null_count", 0))

        profile["cardinality"] = cardinality
        profile["total_rows"] = total_rows
        profile["null_count"] = null_count
        profile["null_ratio"] = round(null_count / max(total_rows, 1), 4)

        if cardinality == 0:
            profile["distinct_values"] = []
            return profile

        # Step 2: Fetch values based on cardinality
        if cardinality <= LOW_CARDINALITY_THRESHOLD:
            profile["cardinality_class"] = "low"
            sql = text(
                text_distinct_values_query(self._dialect)
                .replace("{col}", f'"{col}"')
                .replace("{schema}", f'"{schema}"')
                .replace("{table}", f'"{table}"')
            )
            with self._engine.connect() as conn:
                rows = conn.execute(sql).mappings()
                profile["distinct_values"] = [
                    str(r["value"]) for r in rows if r["value"] is not None
                ]
        else:
            profile["cardinality_class"] = "high"
            sql = text(
                text_top_values_query(self._dialect)
                .replace("{col}", f'"{col}"')
                .replace("{schema}", f'"{schema}"')
                .replace("{table}", f'"{table}"')
            )
            with self._engine.connect() as conn:
                rows = list(conn.execute(sql).mappings())
                profile["top_values"] = [
                    {"value": str(r["value"]), "frequency": int(r["frequency"])}
                    for r in rows
                ]

            # Also grab a small random sample
            sql = text(
                text_sample_values_query(self._dialect)
                .replace("{col}", f'"{col}"')
                .replace("{schema}", f'"{schema}"')
                .replace("{table}", f'"{table}"')
            )
            with self._engine.connect() as conn:
                rows = conn.execute(sql).mappings()
                profile["sample_values"] = [
                    str(r["value"]) for r in rows if r["value"] is not None
                ]

        return profile

    # ------------------------------------------------------------------
    # NUMERIC profiling
    # ------------------------------------------------------------------

    def _profile_numeric_column(
        self,
        schema: str,
        table: str,
        col: str,
    ) -> dict[str, Any]:
        """Profile a NUMERIC/INT/FLOAT column."""
        sql = text(
            numeric_profile_query(self._dialect)
            .replace("{col}", f'"{col}"')
            .replace("{schema}", f'"{schema}"')
            .replace("{table}", f'"{table}"')
        )
        with self._engine.connect() as conn:
            row = dict(conn.execute(sql).mappings().first() or {})

        total_rows = int(row.get("total_rows", 0))
        null_count = int(row.get("null_count", 0))

        return {
            "profile_type": "numeric",
            "total_rows": total_rows,
            "null_count": null_count,
            "null_ratio": round(null_count / max(total_rows, 1), 4),
            "cardinality": int(row.get("cardinality", 0)),
            "min_value": _safe_number(row.get("min_value")),
            "max_value": _safe_number(row.get("max_value")),
            "mean_value": _safe_number(row.get("mean_value")),
            "median_value": _safe_number(row.get("median_value")),
        }

    # ------------------------------------------------------------------
    # TEMPORAL profiling
    # ------------------------------------------------------------------

    def _profile_temporal_column(
        self,
        schema: str,
        table: str,
        col: str,
    ) -> dict[str, Any]:
        """Profile a DATE/TIMESTAMP column."""
        sql = text(
            temporal_profile_query(self._dialect)
            .replace("{col}", f'"{col}"')
            .replace("{schema}", f'"{schema}"')
            .replace("{table}", f'"{table}"')
        )
        with self._engine.connect() as conn:
            row = dict(conn.execute(sql).mappings().first() or {})

        total_rows = int(row.get("total_rows", 0))
        null_count = int(row.get("null_count", 0))
        distinct_days = int(row.get("distinct_days", 0))
        distinct_months = int(row.get("distinct_months", 0))
        distinct_years = int(row.get("distinct_years", 0))

        return {
            "profile_type": "temporal",
            "total_rows": total_rows,
            "null_count": null_count,
            "null_ratio": round(null_count / max(total_rows, 1), 4),
            "min_date": row.get("min_date"),
            "max_date": row.get("max_date"),
            "distinct_days": distinct_days,
            "distinct_months": distinct_months,
            "distinct_years": distinct_years,
            "granularity": infer_granularity(
                distinct_days, distinct_months, distinct_years
            ),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_number(value: Any) -> float | int | None:
    """Coerce a DB value to a Python number, or ``None``."""
    if value is None:
        return None
    try:
        f = float(value)
        return int(f) if f == int(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
