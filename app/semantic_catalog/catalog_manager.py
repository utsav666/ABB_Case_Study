"""Catalog Manager — orchestrates profiling + annotation and merges results.

This is the single entry-point that:

  1. Accepts a schema config dict (the same structure ``generate_schema_config``
     returns) or a connection_id + table list.
  2. Runs **Phase 1** — ``DataProfiler`` to gather statistical profiles.
  3. Runs **Phase 2** (optional) — ``SemanticAnnotator`` to generate
     LLM-based descriptions, value meanings, and aliases.
  4. **Merges** the enrichment data back into the schema config under each
     column's dict, producing an enriched config ready to be stored in
     ``agent.config`` JSONB or the local ``schema_preserve.json``.

Usage::

    from app.semantic_catalog.catalog_manager import enrich_schema_config

    # From an existing schema config dict:
    enriched = enrich_schema_config(
        schema_config=existing_config,
        target_db_url="postgresql://...",
        annotate=True,
    )

    # Or from a connection_id (looks up credentials automatically):
    enriched = enrich_schema_config_for_connection(
        connection_id="abc-123",
        schema_config=existing_config,
        annotate=True,
    )
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from app.semantic_catalog.data_profiler import DataProfiler
from app.semantic_catalog.semantic_annotator import SemanticAnnotator

logger = logging.getLogger(__name__)


def enrich_schema_config(
    schema_config: dict[str, Any],
    target_db_url: str,
    *,
    annotate: bool = False,
    annotator_model: str | None = None,
) -> dict[str, Any]:
    """Enrich an existing schema config with profiling and optional annotations.

    Parameters
    ----------
    schema_config:
        A dict in the ``generate_schema_config`` return format::

            {
                "connection_id": "...",
                "schema": "public",
                "table_count": N,
                "config": {
                    "tables": {
                        "public.orders": {
                            "type": "table",
                            "columns": {...},
                            "primary_key": [...],
                            "foreign_keys": [...]
                        },
                        ...
                    }
                }
            }

    target_db_url:
        SQLAlchemy connection URL for the target database.
    annotate:
        When ``True``, also run the LLM semantic annotator (Phase 2).
    annotator_model:
        Optional override for the LLM model used by the annotator.

    Returns
    -------
    dict
        The same ``schema_config`` structure, with each column dict enriched
        with a ``"semantic_profile"`` key.
    """
    enriched = deepcopy(schema_config)
    tables: dict[str, Any] = enriched.get("config", {}).get("tables", {})
    schema_name = enriched.get("schema", "public")

    if not tables:
        logger.warning("No tables found in schema config — nothing to profile.")
        return enriched

    # ── Phase 1: Statistical Profiling ──────────────────────────────────
    profiler = DataProfiler(target_db_url)
    try:
        all_profiles = profiler.profile_tables(schema_name, tables)
    finally:
        profiler.dispose()

    logger.info(
        "Phase 1 complete — profiled %d tables, %d total columns.",
        len(all_profiles),
        sum(len(cols) for cols in all_profiles.values()),
    )

    # ── Phase 2: Semantic Annotation (optional) ─────────────────────────
    all_annotations: dict[str, dict[str, dict[str, Any]]] = {}
    if annotate:
        annotator = SemanticAnnotator(model=annotator_model)
        all_annotations = annotator.annotate_tables(tables, all_profiles)

        logger.info(
            "Phase 2 complete — annotated %d tables.",
            len(all_annotations),
        )

    # ── Merge into schema config ────────────────────────────────────────
    for table_name, table_info in tables.items():
        columns = table_info.get("columns", {})
        table_profiles = all_profiles.get(table_name, {})
        table_annotations = all_annotations.get(table_name, {})

        for col_name, col_meta in columns.items():
            profile = table_profiles.get(col_name, {})
            annotation = table_annotations.get(col_name, {})

            # Build the semantic_profile sub-object
            semantic_profile: dict[str, Any] = {}

            # From Phase 1 — statistical data
            profile_type = profile.get("profile_type")
            if profile_type == "text":
                semantic_profile["cardinality"] = profile.get("cardinality")
                semantic_profile["cardinality_class"] = profile.get("cardinality_class")
                semantic_profile["null_ratio"] = profile.get("null_ratio")
                if "distinct_values" in profile:
                    semantic_profile["distinct_values"] = profile["distinct_values"]
                if "top_values" in profile:
                    semantic_profile["top_values"] = profile["top_values"]
                if "sample_values" in profile:
                    semantic_profile["sample_values"] = profile["sample_values"]

            elif profile_type == "numeric":
                semantic_profile["min_value"] = profile.get("min_value")
                semantic_profile["max_value"] = profile.get("max_value")
                semantic_profile["mean_value"] = profile.get("mean_value")
                semantic_profile["median_value"] = profile.get("median_value")
                semantic_profile["null_ratio"] = profile.get("null_ratio")

            elif profile_type == "temporal":
                semantic_profile["min_date"] = profile.get("min_date")
                semantic_profile["max_date"] = profile.get("max_date")
                semantic_profile["granularity"] = profile.get("granularity")
                semantic_profile["null_ratio"] = profile.get("null_ratio")

            # From Phase 2 — LLM annotation
            if annotation:
                semantic_profile["description"] = annotation.get("description", "")
                semantic_profile["value_descriptions"] = annotation.get(
                    "value_descriptions", {},
                )
                semantic_profile["natural_language_aliases"] = annotation.get(
                    "natural_language_aliases", {},
                )
                semantic_profile["query_hints"] = annotation.get("query_hints", "")

            if semantic_profile:
                col_meta["semantic_profile"] = semantic_profile

    enriched["profiled"] = True
    enriched["annotated"] = annotate

    return enriched


def enrich_schema_config_for_connection(
    connection_id: str,
    schema_config: dict[str, Any],
    *,
    annotate: bool = False,
    annotator_model: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper that resolves a connection_id to a DB URL, then
    delegates to :func:`enrich_schema_config`.

    Parameters
    ----------
    connection_id:
        ID of a stored DB connection in ``settings.db_connections``.
    schema_config:
        Schema config dict (output of ``generate_schema_config``).
    annotate:
        When ``True``, also run LLM semantic annotation.
    annotator_model:
        Optional LLM model override.

    Returns
    -------
    dict
        Enriched schema config.
    """
    from app.utils.db_connections import _connect_target

    target_url = _connect_target(connection_id)

    return enrich_schema_config(
        schema_config=schema_config,
        target_db_url=target_url,
        annotate=annotate,
        annotator_model=annotator_model,
    )


def save_enriched_config_to_local_file(
    enriched_config: dict[str, Any],
) -> None:
    """Persist the enriched schema config to ``schema_preserve.json``.

    Extracts the inner ``config`` dict (which contains the ``tables`` key)
    and writes it to the local schema file so that the existing pipeline
    (Summarizer, TableIdentifier, SQLGenerator) picks up the semantic
    profiles automatically.

    The API uses qualified table keys (``"sales.sample"``) while the local
    file uses bare table names with a ``"schema"`` field inside.  This
    function converts between the two formats automatically.
    """
    import json
    from pathlib import Path

    schema_file = Path(__file__).resolve().parents[2] / "sample_data" / "schema_preserve.json"
    inner_config = enriched_config.get("config", enriched_config)
    schema_name = enriched_config.get("schema", "public")

    # Convert qualified table keys ("sales.sample") to local format
    # where key is bare name ("sample") and "schema" is a field inside.
    tables = inner_config.get("tables", {})
    converted_tables: dict[str, Any] = {}
    for qualified_name, table_info in tables.items():
        if "." in qualified_name:
            table_schema, bare_name = qualified_name.split(".", 1)
        else:
            table_schema = schema_name
            bare_name = qualified_name

        converted_table = dict(table_info)
        converted_table["schema"] = table_schema
        converted_tables[bare_name] = converted_table

    output = {"tables": converted_tables}

    schema_file.parent.mkdir(parents=True, exist_ok=True)
    with schema_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Saved enriched schema config to %s", schema_file)
