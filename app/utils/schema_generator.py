from __future__ import annotations

import logging

from fastapi import HTTPException
from sqlalchemy import create_engine

from app.utils.db_connections import _connect_target
from app.utils.query import FETCH_COLUMNS_WITH_PK, FETCH_FOREIGN_KEYS

logger = logging.getLogger(__name__)


def generate_schema_config(
    connection_id: str,
    table_names: list[str],
    schema_name: str = "public",
) -> dict:
    """Build a schema_preserve-style JSON config for the given tables.

    Fetches columns, primary keys, and foreign keys from the target database
    and returns a structured configuration matching the schema_preserve.json format.

    Args:
        connection_id: ID of the stored database connection.
        table_names: Tables to include in the config.
        schema_name: Database schema to query.
    """
    target_url = _connect_target(connection_id)

    try:
        target_engine = create_engine(target_url)
        with target_engine.connect() as connection:
            schema_config: dict = {"tables": {}}

            for table_name in table_names:
                params = {"schema_name": schema_name, "table_name": table_name}

                col_rows = list(
                    connection.execute(FETCH_COLUMNS_WITH_PK, params).mappings()
                )

                if not col_rows:
                    logger.warning(
                        "Table '%s' not found in schema '%s', skipping.",
                        table_name,
                        schema_name,
                    )
                    continue

                columns = {}
                primary_key = []
                for col in col_rows:
                    col_name = col["column_name"]
                    is_not_null = col["is_nullable"] == "NO"
                    is_pk = bool(col["is_pk"])
                    columns[col_name] = {
                        "type": col["data_type"].upper(),
                        "notnull": is_not_null,
                        "default": col["column_default"],
                        "is_pk": is_pk,
                    }
                    if is_pk:
                        primary_key.append(col_name)

                fk_rows = list(
                    connection.execute(FETCH_FOREIGN_KEYS, params).mappings()
                )

                foreign_keys_by_constraint: dict = {}
                for fk_row in fk_rows:
                    constraint = fk_row["constraint_name"]
                    if constraint not in foreign_keys_by_constraint:
                        foreign_keys_by_constraint[constraint] = {
                            "from": [],
                            "to_table": f"{schema_name}.{fk_row['to_table']}",
                            "to_columns": [],
                        }
                    foreign_keys_by_constraint[constraint]["from"].append(
                        fk_row["from_column"]
                    )
                    foreign_keys_by_constraint[constraint]["to_columns"].append(
                        fk_row["to_column"]
                    )

                qualified_table_name = f"{schema_name}.{table_name}"
                schema_config["tables"][qualified_table_name] = {
                    "type": "table",
                    "columns": columns,
                    "primary_key": primary_key,
                    "foreign_keys": list(foreign_keys_by_constraint.values()),
                }

    except Exception as error:
        logger.exception("Failed to generate schema config from target DB")
        raise HTTPException(
            status_code=500, detail=f"Could not generate schema config: {error}"
        ) from error
    finally:
        target_engine.dispose()

    if not schema_config["tables"]:
        raise HTTPException(
            status_code=404,
            detail="None of the requested tables were found in the target database.",
        )

    return {
        "connection_id": connection_id,
        "schema": schema_name,
        "table_count": len(schema_config["tables"]),
        "config": schema_config,
    }
