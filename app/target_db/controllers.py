from __future__ import annotations

import logging
import time
from typing import Any

from app.settings_db.controllers import get_settings_connection_credentials
from app.target_db.repositories import TargetMetadataRepository

logger = logging.getLogger(__name__)
from app.target_db.views import (
    build_schema_config_response,
    build_schema_config_table_payload,
    build_table_columns_response,
    build_tables_response,
    build_target_db_url,
)


class TargetMetadataController:
    """Controller for target database metadata operations."""

    def __init__(self, repository: TargetMetadataRepository | None = None) -> None:
        self._repository = repository or TargetMetadataRepository()

    def get_tables_for_connection(
        self,
        connection_id: str,
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Fetch table names for a configured connection."""
        logger.info("get_tables_for_connection start: connection_id=%s schema=%s",
                    connection_id, schema_name)

        t0 = time.monotonic()
        target_url = self._resolve_target_url(connection_id)
        logger.info("Resolved target URL in %.2fs", time.monotonic() - t0)

        t1 = time.monotonic()
        table_rows = self._repository.fetch_tables(target_url, schema_name)
        logger.info("fetch_tables returned %d rows in %.2fs",
                    len(table_rows), time.monotonic() - t1)

        return build_tables_response(connection_id, schema_name, table_rows)

    def get_table_columns_for_connection(
        self,
        connection_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Fetch table columns for a configured connection."""
        target_url = self._resolve_target_url(connection_id)
        columns = self._repository.fetch_table_columns(
            target_url,
            schema_name,
            table_name,
        )
        return build_table_columns_response(
            connection_id,
            schema_name,
            table_name,
            columns,
        )

    def generate_schema_config(
        self,
        connection_id: str,
        table_names: list[str],
        schema_name: str = "public",
    ) -> dict[str, Any]:
        """Generate schema configuration for selected tables."""
        target_url = self._resolve_target_url(connection_id)
        tables: dict[str, Any] = {}

        for table_name in table_names:
            column_rows = self._repository.fetch_columns_with_primary_keys(
                target_url,
                schema_name,
                table_name,
            )
            foreign_key_rows = self._repository.fetch_foreign_keys(
                target_url,
                schema_name,
                table_name,
            )
            table_payload = build_schema_config_table_payload(
                schema_name,
                table_name,
                column_rows,
                foreign_key_rows,
            )
            if table_payload is None:
                continue

            qualified_table_name = f"{schema_name}.{table_name}"
            tables[qualified_table_name] = table_payload

        return build_schema_config_response(connection_id, schema_name, tables)

    def get_target_db_url(self, connection_id: str) -> str:
        """Resolve and return a target URL for a stored connection."""
        return self._resolve_target_url(connection_id)

    def _resolve_target_url(self, connection_id: str) -> str:
        """Resolve stored credentials into a SQLAlchemy target URL."""
        credentials = get_settings_connection_credentials(connection_id)
        return build_target_db_url(credentials)


def sync_column_definitions_for_connection(
    connection_id: str,
    schema_name: str = "public",
) -> None:
    """Sync columns, LLM semantics, and relations for a connection's tables.

    Intended to run as a background task after a table selection is saved.
    For each selected table it:
      1. Reads live column metadata from the target (customer) database and
         replaces the rows in settings.column_definitions.
      2. Calls an LLM to populate logical_name + description per column. If two
         columns in the SAME table get the same logical_name, both that name and
         description are left empty for the conflicting columns.
      3. Discovers foreign-key relationships and persists those whose BOTH ends
         are selected tables into settings.column_relations.

    Failures for individual tables are logged and skipped so one bad table does
    not abort the whole sync.
    """
    # Imported here to avoid a circular import at module load time.
    from app.semantic_catalog.column_annotator import ColumnAnnotator
    from app.settings_db.repositories import (
        ColumnDefinitionRepository,
        ColumnRelationRepository,
        SelectedTableRepository,
    )

    logger.info(
        "Starting column sync for connection_id=%s schema=%s",
        connection_id,
        schema_name,
    )

    selected_table_repository = SelectedTableRepository()
    column_repository = ColumnDefinitionRepository()
    relation_repository = ColumnRelationRepository()
    controller = TargetMetadataController()

    selected_tables = selected_table_repository.get_selected_tables_by_connection(
        connection_id
    )

    if not selected_tables:
        logger.info("No selected tables to sync for connection_id=%s", connection_id)
        return

    # Resolve the target URL once and reuse for all tables.
    try:
        target_url = controller.get_target_db_url(connection_id)
    except Exception as error:  # noqa: BLE001 - background task must not crash
        logger.exception(
            "Column sync aborted: could not resolve target URL for %s: %s",
            connection_id,
            error,
        )
        return

    annotator = ColumnAnnotator()

    # Map (schema, table_name) -> selected_tables.id, used for relation lookup.
    table_id_by_name: dict[tuple[str, str], Any] = {}
    # Collect inserted columns per table so we can annotate the whole DB at once.
    # table_key "schema.table" -> {"schema", "table_name", "columns": [inserted rows]}
    inserted_by_table_key: dict[str, dict[str, Any]] = {}

    synced = 0
    for selected_table in selected_tables:
        table_id = selected_table["id"]
        table_name = selected_table["table_name"]
        table_schema = selected_table.get("schema_name") or schema_name
        table_id_by_name[(table_schema, table_name)] = table_id

        try:
            raw_columns = controller._repository.fetch_table_columns(
                target_url,
                table_schema,
                table_name,
            )

            column_rows = [
                {
                    "table_id": table_id,
                    "raw_name": column["column_name"],
                    "data_type": column.get("data_type"),
                }
                for column in raw_columns
                if column.get("column_name")
            ]

            # Persist raw columns first (returns rows with generated ids).
            inserted = column_repository.replace_columns_for_table(
                table_id, column_rows
            )

            inserted_by_table_key[f"{table_schema}.{table_name}"] = {
                "schema": table_schema,
                "table_name": table_name,
                "columns": inserted,
            }

            synced += 1
            logger.info(
                "Synced %d columns for table %s.%s (table_id=%s)",
                len(column_rows),
                table_schema,
                table_name,
                table_id,
            )
        except Exception as error:  # noqa: BLE001 - skip and continue
            logger.exception(
                "Failed to sync columns for table %s.%s: %s",
                table_schema,
                table_name,
                error,
            )

    # 2. Annotate logical_name + description for the ENTIRE database in one
    #    LLM call, then apply same-table conflict blanking before persisting.
    try:
        _annotate_database_and_store(
            annotator,
            column_repository,
            inserted_by_table_key,
        )
    except Exception as error:  # noqa: BLE001 - semantics are best-effort
        logger.exception(
            "DB-level annotation failed for connection_id=%s: %s",
            connection_id,
            error,
        )

    # 3. Discover and persist relations between selected tables.
    try:
        _sync_relations(
            controller,
            relation_repository,
            connection_id,
            target_url,
            selected_tables,
            schema_name,
            table_id_by_name,
        )
    except Exception as error:  # noqa: BLE001 - relations are best-effort
        logger.exception(
            "Relation sync failed for connection_id=%s: %s", connection_id, error
        )

    # 4. Assemble and persist the schema graph (nodes + edges) for the LLM.
    try:
        build_and_store_schema_graph(connection_id)
    except Exception as error:  # noqa: BLE001 - graph build is best-effort
        logger.exception(
            "Schema graph build failed for connection_id=%s: %s",
            connection_id,
            error,
        )

    logger.info(
        "Column sync complete for connection_id=%s: %d/%d tables synced",
        connection_id,
        synced,
        len(selected_tables),
    )


def validate_sql_for_connection(
    connection_id: str,
    sql: str,
) -> dict[str, Any]:
    """Validate SQL against the customer DB using EXPLAIN (no execution).

    Returns {"valid": True} or {"valid": False, "error": "..."}.
    """
    from sqlalchemy import text

    stripped = (sql or "").strip().rstrip(";").strip()
    if not stripped:
        return {"valid": False, "error": "SQL is empty."}

    # Safety: only allow validating read queries. EXPLAIN does not execute the
    # plan, but we still refuse obvious write/DDL statements outright.
    lowered = stripped.lower()
    forbidden = (
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
        "grant ",
        "revoke ",
    )
    if lowered.startswith(forbidden) or any(
        f";{kw}" in lowered.replace(" ", "") for kw in ("insert", "update", "delete")
    ):
        return {
            "valid": False,
            "error": "Only read-only SELECT queries can be validated.",
        }

    controller = TargetMetadataController()
    try:
        target_url = controller.get_target_db_url(connection_id)
    except Exception as error:  # noqa: BLE001
        logger.exception("Could not resolve target URL for validation")
        return {"valid": False, "error": f"Could not resolve connection: {error}"}

    try:
        controller._repository._gateway.execute_query(
            target_url,
            text(f"EXPLAIN {stripped}"),
            {},
        )
        return {"valid": True}
    except Exception as error:  # noqa: BLE001 - surface DB error to the user
        message = str(error)
        # Unwrap the most useful part of SQLAlchemy/psycopg2 errors.
        if hasattr(error, "orig"):
            message = str(getattr(error, "orig"))
        logger.info("SQL validation failed: %s", message)
        return {"valid": False, "error": message}


def generate_sql_from_graph(
    connection_id: str,
    user_question: str,
    dialect: str = "PostgreSQL",
) -> dict[str, Any]:
    """Generate SQL for a question using the connection's stored schema graph.

    Loads the persisted graph, renders it into the table-structure format the
    SQL agent expects, and delegates to generate_sql().

    Returns {"success": True, "sql": ...} or {"success": False, "error": ...}.
    """
    import json

    from app.agent.sql_generation_agent import generate_sql
    from app.settings_db.repositories import SchemaGraphRepository

    graph_record = SchemaGraphRepository().get_graph_by_connection(connection_id)
    if not graph_record or not graph_record.get("graph", {}).get("nodes"):
        return {
            "success": False,
            "error": "No schema graph available. Select tables first to build it.",
        }

    graph = graph_record["graph"]

    # Render the graph as a compact JSON table-structure for the prompt.
    table_structure_payload: dict[str, Any] = {}
    table_list: list[str] = []
    for node in graph.get("nodes", []):
        qualified = node["qualified_name"]
        table_list.append(qualified)
        table_structure_payload[node["table"]] = {
            "schema": node.get("schema"),
            "qualified_name": qualified,
            "description": node.get("description"),
            "columns": {
                col["raw_name"]: {
                    "type": col.get("data_type"),
                    "logical_name": col.get("logical_name"),
                }
                for col in node.get("columns", [])
            },
        }

    # Include relationships so the LLM knows valid joins.
    table_structure_payload["__relationships__"] = [
        {
            "from": f"{edge['from_table']}.{edge['from_column']}",
            "to": f"{edge['to_table']}.{edge['to_column']}",
        }
        for edge in graph.get("edges", [])
    ]

    table_structure = json.dumps(table_structure_payload, indent=2, default=str)

    return generate_sql(
        user_question=user_question,
        table_list=table_list,
        dialect=dialect,
        table_structure=table_structure,
    )


def build_and_store_schema_graph(connection_id: str) -> dict[str, Any]:
    """Assemble a JSON schema graph from stored metadata and persist it.

    The graph is deterministic (no LLM): nodes are the selected tables with
    their columns + logical names; edges are the stored column_relations.
    Shape::

        {
          "connection_id": "...",
          "nodes": [
            {"table_id", "schema", "table", "qualified_name",
             "columns": [{"raw_name", "logical_name", "data_type"}]}
          ],
          "edges": [
            {"from_table", "from_column", "to_table", "to_column",
             "constraint_name"}
          ]
        }

    Returns the persisted graph payload.
    """
    from app.settings_db.repositories import (
        ColumnDefinitionRepository,
        ColumnRelationRepository,
        SchemaGraphRepository,
    )

    column_repository = ColumnDefinitionRepository()
    relation_repository = ColumnRelationRepository()
    graph_repository = SchemaGraphRepository()

    tables = column_repository.get_columns_grouped_by_connection(connection_id)
    relations = relation_repository.get_relations_by_connection(connection_id)

    # table_id -> qualified name, used to translate edge endpoints.
    qualified_by_table_id: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    for table in tables:
        schema = table.get("schema_name") or "public"
        table_name = table["table_name"]
        qualified = f"{schema}.{table_name}"
        qualified_by_table_id[str(table["table_id"])] = qualified
        nodes.append(
            {
                "table_id": str(table["table_id"]),
                "schema": schema,
                "table": table_name,
                "qualified_name": qualified,
                "description": table.get("human_description"),
                "columns": [
                    {
                        "raw_name": col["raw_name"],
                        "logical_name": col.get("logical_name"),
                        "data_type": col.get("data_type"),
                    }
                    for col in table.get("columns", [])
                ],
            }
        )

    edges: list[dict[str, Any]] = []
    for relation in relations:
        from_q = qualified_by_table_id.get(relation["from_table_id"])
        to_q = qualified_by_table_id.get(relation["to_table_id"])
        if not (from_q and to_q):
            continue
        edges.append(
            {
                "from_table": from_q,
                "from_column": relation["from_column"],
                "to_table": to_q,
                "to_column": relation["to_column"],
                "constraint_name": relation.get("constraint_name"),
            }
        )

    graph = {
        "connection_id": connection_id,
        "nodes": nodes,
        "edges": edges,
        "table_count": len(nodes),
        "relation_count": len(edges),
    }

    graph_repository.upsert_graph(connection_id, graph)
    logger.info(
        "Stored schema graph for connection_id=%s: %d nodes, %d edges",
        connection_id,
        len(nodes),
        len(edges),
    )
    return graph


def _annotate_database_and_store(
    annotator: Any,
    column_repository: Any,
    inserted_by_table_key: dict[str, dict[str, Any]],
) -> None:
    """Annotate the whole DB in one LLM call; persist with per-table conflict blanking.

    Conflict rule (per requirement): if two columns in the SAME table receive the
    same logical_name (case-insensitive), both logical_name and description are
    left empty for those columns.
    """
    if not inserted_by_table_key:
        return

    # Build the single DB-level annotation request.
    request_tables = [
        {
            "table_key": table_key,
            "columns": [
                {"raw_name": col["raw_name"], "data_type": col.get("data_type")}
                for col in info["columns"]
            ],
        }
        for table_key, info in inserted_by_table_key.items()
        if info["columns"]
    ]

    annotations_by_table = annotator.annotate_database(request_tables)

    if not annotations_by_table:
        # No LLM output — leave logical_name/description NULL everywhere.
        logger.info("No DB-level annotations returned; leaving semantics empty.")
        return

    for table_key, info in inserted_by_table_key.items():
        inserted_columns = info["columns"]
        annotations = annotations_by_table.get(table_key, {})
        if not inserted_columns:
            continue

        # Detect duplicate logical_names WITHIN this table (case-insensitive).
        name_counts: dict[str, int] = {}
        for col in inserted_columns:
            entry = annotations.get(col["raw_name"]) or {}
            logical = (entry.get("logical_name") or "").strip()
            if logical:
                name_counts[logical.lower()] = name_counts.get(logical.lower(), 0) + 1

        conflicting = {name for name, count in name_counts.items() if count > 1}

        for col in inserted_columns:
            entry = annotations.get(col["raw_name"]) or {}
            logical = (entry.get("logical_name") or "").strip()
            description = (entry.get("description") or "").strip()

            # On conflict within the table, blank BOTH fields for the dupes.
            if logical and logical.lower() in conflicting:
                logger.info(
                    "Conflicting logical_name '%s' in %s — leaving column '%s' empty",
                    logical,
                    table_key,
                    col["raw_name"],
                )
                logical = ""
                description = ""

            column_repository.update_column_semantics(
                col["id"],
                logical or None,
                description or None,
            )


def _sync_relations(
    controller: Any,
    relation_repository: Any,
    connection_id: str,
    target_url: str,
    selected_tables: list[dict[str, Any]],
    default_schema: str,
    table_id_by_name: dict[tuple[str, str], Any],
) -> None:
    """Discover FK relations and store those linking two selected tables."""
    relations: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for selected_table in selected_tables:
        table_name = selected_table["table_name"]
        table_schema = selected_table.get("schema_name") or default_schema
        from_table_id = selected_table["id"]

        fk_rows = controller._repository.fetch_foreign_keys(
            target_url,
            table_schema,
            table_name,
        )

        for fk in fk_rows:
            to_table = fk.get("to_table")
            from_column = fk.get("from_column")
            to_column = fk.get("to_column")
            if not (to_table and from_column and to_column):
                continue

            # Only store relations where the target table is also selected.
            to_table_id = table_id_by_name.get((table_schema, to_table))
            if to_table_id is None:
                logger.info(
                    "Skipping relation %s.%s -> %s (target table not selected)",
                    table_name,
                    from_column,
                    to_table,
                )
                continue

            dedupe_key = (
                str(from_table_id),
                from_column,
                str(to_table_id),
                to_column,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            relations.append(
                {
                    "from_table_id": from_table_id,
                    "from_column": from_column,
                    "to_table_id": to_table_id,
                    "to_column": to_column,
                    "constraint_name": fk.get("constraint_name"),
                }
            )

    relation_repository.replace_relations_for_connection(connection_id, relations)
    logger.info(
        "Persisted %d relation(s) for connection_id=%s", len(relations), connection_id
    )


def get_tables_for_connection(connection_id: str, schema_name: str = "public") -> dict:
    """Compatibility wrapper for fetching target tables."""
    return TargetMetadataController().get_tables_for_connection(
        connection_id, schema_name
    )


def get_table_columns_for_connection(
    connection_id: str,
    table_name: str,
    schema_name: str = "public",
) -> dict:
    """Compatibility wrapper for fetching target columns."""
    return TargetMetadataController().get_table_columns_for_connection(
        connection_id,
        table_name,
        schema_name,
    )


def generate_target_schema_config(
    connection_id: str,
    table_names: list[str],
    schema_name: str = "public",
) -> dict:
    """Compatibility wrapper for generating target schema config."""
    return TargetMetadataController().generate_schema_config(
        connection_id,
        table_names,
        schema_name,
    )


def get_target_db_url(connection_id: str) -> str:
    """Compatibility wrapper for resolving target DB URL."""
    return TargetMetadataController().get_target_db_url(connection_id)
