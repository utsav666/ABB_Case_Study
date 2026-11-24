from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from app.settings_db.controllers import (
    create_db_connection,
    get_supported_connection_engines,
    list_all_db_connections,
    validate_db_connection,
)
from app.settings_db.models import DBConnectionCreate, DBConnectionValidationRequest
from app.settings_db.repositories import (
    ColumnDefinitionRepository,
    ColumnRelationRepository,
    DBConnectionRepository,
    SchemaGraphRepository,
    SelectedTableRepository,
)
from app.target_db.controllers import (
    build_and_store_schema_graph,
    generate_sql_from_graph,
    get_tables_for_connection,
    sync_column_definitions_for_connection,
    validate_sql_for_connection,
)

settings_db_router = APIRouter(
    prefix="/settings/db-connections",
    tags=["settings-db"],
)


@settings_db_router.get("/engines")
def get_db_connection_engines() -> dict[str, list[str]]:
    """Return supported database engines for the settings UI."""
    return {"engines": get_supported_connection_engines()}


@settings_db_router.post("/validate")
def validate_db_connection_settings(
    payload: DBConnectionValidationRequest,
) -> dict[str, Any]:
    """Validate a database connection payload without persisting it."""
    return validate_db_connection(payload)


@settings_db_router.get("")
def get_all_db_connections() -> dict[str, list[dict[str, Any]]]:
    """Get all saved database connections."""
    connections = list_all_db_connections()
    return {"connections": connections}


@settings_db_router.post("")
def save_db_connection(
    payload: DBConnectionCreate,
    skip_validation: bool = Query(
        default=False, description="Skip connection validation"
    ),
) -> dict[str, Any]:
    """Validate and persist a database connection payload."""
    result = create_db_connection(payload, skip_validation)
    return result


class UpdateConnectionActiveRequest(BaseModel):
    """Request body for enabling/disabling a connection."""

    is_active: bool


@settings_db_router.patch("/{connection_id}/active")
def set_connection_active(
    connection_id: str,
    payload: UpdateConnectionActiveRequest,
) -> dict[str, Any]:
    """Enable or disable a saved database connection."""
    repository = DBConnectionRepository()
    updated = repository.set_connection_active(connection_id, payload.is_active)
    if updated is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"connection": updated}


@settings_db_router.get("/{connection_id}/tables")
def get_connection_tables(
    connection_id: str,
    schema_name: str = Query(default="public", description="Database schema name"),
) -> dict[str, Any]:
    """Get all tables from a database connection."""
    return get_tables_for_connection(connection_id, schema_name)


class SaveSelectedTablesRequest(BaseModel):
    """Request body for saving selected tables."""

    table_names: list[str]
    schema_name: str = "public"


@settings_db_router.post("/{connection_id}/selected-tables")
def save_selected_tables(
    connection_id: str,
    payload: SaveSelectedTablesRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Save selected tables for a connection.

    After persisting the selection, schedules an async background job that
    fetches each table's columns from the target database and stores them in
    settings.column_definitions.
    """
    repository = SelectedTableRepository()
    saved_tables = repository.bulk_replace_selected_tables(
        connection_id,
        payload.table_names,
        payload.schema_name,
    )

    # Fire-and-forget: fetch & persist column metadata after responding.
    if saved_tables:
        background_tasks.add_task(
            sync_column_definitions_for_connection,
            connection_id,
            payload.schema_name,
        )

    return {
        "connection_id": connection_id,
        "selected_tables": saved_tables,
        "count": len(saved_tables),
    }


@settings_db_router.get("/{connection_id}/selected-tables")
def get_selected_tables(connection_id: str) -> dict[str, Any]:
    """Get selected tables for a connection."""
    repository = SelectedTableRepository()
    selected_tables = repository.get_selected_tables_by_connection(connection_id)
    return {
        "connection_id": connection_id,
        "selected_tables": selected_tables,
        "count": len(selected_tables),
    }


@settings_db_router.get("/{connection_id}/column-definitions")
def get_column_definitions(connection_id: str) -> dict[str, Any]:
    """Get selected tables with their stored column definitions for a connection."""
    repository = ColumnDefinitionRepository()
    tables = repository.get_columns_grouped_by_connection(connection_id)
    return {
        "connection_id": connection_id,
        "tables": tables,
        "table_count": len(tables),
    }


class UpdateColumnLogicalNameRequest(BaseModel):
    """Request body for editing a column's logical name."""

    logical_name: str | None = None


@settings_db_router.patch("/columns/{column_id}/logical-name")
def update_column_logical_name(
    column_id: str,
    payload: UpdateColumnLogicalNameRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Update a single column's logical name and refresh the schema graph.

    Chat reads the stored graph, not column_definitions, so without the rebuild
    an edited name never reaches the LLM.
    """
    repository = ColumnDefinitionRepository()
    # Normalize empty string to NULL so cleared values are stored as empty.
    logical_name = (payload.logical_name or "").strip() or None
    updated = repository.update_column_logical_name(column_id, logical_name)
    if updated is None:
        raise HTTPException(status_code=404, detail="Column not found")
    connection_id = SelectedTableRepository().get_connection_id_for_table(
        updated["table_id"]
    )
    if connection_id:
        background_tasks.add_task(build_and_store_schema_graph, connection_id)
    return {"column": updated}


@settings_db_router.get("/{connection_id}/relations")
def get_column_relations(connection_id: str) -> dict[str, Any]:
    """Get stored column relationships for a connection."""
    repository = ColumnRelationRepository()
    relations = repository.get_relations_by_connection(connection_id)
    return {
        "connection_id": connection_id,
        "relations": relations,
        "count": len(relations),
    }


@settings_db_router.get("/{connection_id}/schema-graph")
def get_schema_graph(connection_id: str) -> dict[str, Any]:
    """Get the stored schema graph (nodes + edges) for a connection."""
    record = SchemaGraphRepository().get_graph_by_connection(connection_id)
    if record is None:
        return {"connection_id": connection_id, "graph": None}
    return {"connection_id": connection_id, "graph": record["graph"]}


@settings_db_router.get("/{connection_id}/semantic-profile")
def get_semantic_profile(connection_id: str) -> dict[str, Any]:
    """Return the stored semantic profile (facts + inferred table semantics)."""
    record = SchemaGraphRepository().get_graph_by_connection(connection_id)
    profile = ((record or {}).get("graph") or {}).get("profile")
    return {"connection_id": connection_id, "profile": profile}


@settings_db_router.post("/{connection_id}/semantic-profile/refresh", status_code=202)
def refresh_semantic_profile(
    connection_id: str, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Re-profile the live data and re-infer table semantics in the background.

    Run after the data changes shape (new scenarios, versions, metric names).
    Profiling plus inference can take longer than an HTTP request should.
    """
    from app.semantic_catalog.connection_profile import refresh_connection_profile

    if SchemaGraphRepository().get_graph_by_connection(connection_id) is None:
        raise HTTPException(
            status_code=404, detail="No schema graph. Select tables first."
        )
    background_tasks.add_task(refresh_connection_profile, connection_id)
    return {"connection_id": connection_id, "status": "refresh_started"}


class GenerateSqlRequest(BaseModel):
    """Request body for NL->SQL generation."""

    question: str
    dialect: str = "PostgreSQL"


class RelationPayload(BaseModel):
    """A single table relationship payload."""

    from_table_id: str
    from_column: str
    to_table_id: str
    to_column: str
    constraint_name: str | None = None


class ReplaceRelationsRequest(BaseModel):
    """Request body for replacing connection relationships."""

    relations: list[RelationPayload]


@settings_db_router.post("/{connection_id}/generate-sql")
def generate_sql_for_connection(
    connection_id: str,
    payload: GenerateSqlRequest,
) -> dict[str, Any]:
    """Generate SQL for a natural-language question using the schema graph."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    result = generate_sql_from_graph(
        connection_id,
        payload.question.strip(),
        payload.dialect,
    )
    return result


@settings_db_router.put("/{connection_id}/relations")
def replace_column_relations(
    connection_id: str,
    payload: ReplaceRelationsRequest,
) -> dict[str, Any]:
    """Replace stored relationships and rebuild schema graph."""
    repository = ColumnRelationRepository()

    normalized_relations: list[dict[str, Any]] = []
    for relation in payload.relations:
        from_column = relation.from_column.strip()
        to_column = relation.to_column.strip()
        if not from_column or not to_column:
            raise HTTPException(
                status_code=400,
                detail="Both from_column and to_column are required",
            )

        normalized_relations.append(
            {
                "from_table_id": relation.from_table_id,
                "from_column": from_column,
                "to_table_id": relation.to_table_id,
                "to_column": to_column,
                "constraint_name": (relation.constraint_name or "").strip() or None,
            }
        )

    saved_relations = repository.replace_relations_for_connection(
        connection_id,
        normalized_relations,
    )
    build_and_store_schema_graph(connection_id)
    return {
        "connection_id": connection_id,
        "relations": saved_relations,
        "count": len(saved_relations),
    }


class ValidateSqlRequest(BaseModel):
    """Request body for validating SQL via EXPLAIN."""

    sql: str


@settings_db_router.post("/{connection_id}/validate-sql")
def validate_sql(
    connection_id: str,
    payload: ValidateSqlRequest,
) -> dict[str, Any]:
    """Validate user-edited SQL against the customer DB using EXPLAIN."""
    return validate_sql_for_connection(connection_id, payload.sql)
