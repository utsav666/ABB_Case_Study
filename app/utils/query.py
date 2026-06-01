from __future__ import annotations

from sqlalchemy import text

# ---------------------------------------------------------------------------
# Settings DB queries
# ---------------------------------------------------------------------------

INSERT_DB_CONNECTION = text("""
    INSERT INTO settings.db_connections
        (workspace_id, name, engine, db_host, db_port, db_name, db_user, db_password)
    VALUES
        (:workspace_id, :name, :engine, :db_host, :db_port, :db_name, :db_user, :db_password)
    RETURNING id, workspace_id, name, engine, db_host, db_port, db_name, db_user,
              is_healthy, last_synced_at, created_at
""")

FETCH_CONNECTION_BY_ID = text("""
    SELECT id, engine, db_host, db_port, db_name, db_user, db_password
    FROM settings.db_connections
    WHERE id = :connection_id
""")

# ---------------------------------------------------------------------------
# Target DB queries
# ---------------------------------------------------------------------------

FETCH_TABLES = text("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = :schema_name
      AND table_type = 'BASE TABLE'
    ORDER BY table_name
""")

FETCH_TABLE_COLUMNS = text("""
    SELECT
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        c.character_maximum_length,
        CASE WHEN kcu.column_name IS NOT NULL THEN TRUE ELSE FALSE END AS is_primary_key
    FROM information_schema.columns c
    LEFT JOIN information_schema.table_constraints tc
        ON tc.table_schema = c.table_schema
        AND tc.table_name = c.table_name
        AND tc.constraint_type = 'PRIMARY KEY'
    LEFT JOIN information_schema.key_column_usage kcu
        ON kcu.constraint_name = tc.constraint_name
        AND kcu.table_schema = tc.table_schema
        AND kcu.column_name = c.column_name
    WHERE c.table_schema = :schema_name
      AND c.table_name = :table_name
    ORDER BY c.ordinal_position
""")

FETCH_COLUMNS_WITH_PK = text("""
    SELECT
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        CASE WHEN kcu_pk.column_name IS NOT NULL THEN TRUE ELSE FALSE END AS is_pk
    FROM information_schema.columns c
    LEFT JOIN information_schema.table_constraints tc_pk
        ON tc_pk.table_schema = c.table_schema
        AND tc_pk.table_name = c.table_name
        AND tc_pk.constraint_type = 'PRIMARY KEY'
    LEFT JOIN information_schema.key_column_usage kcu_pk
        ON kcu_pk.constraint_name = tc_pk.constraint_name
        AND kcu_pk.table_schema = tc_pk.table_schema
        AND kcu_pk.column_name = c.column_name
    WHERE c.table_schema = :schema_name
      AND c.table_name = :table_name
    ORDER BY c.ordinal_position
""")

FETCH_FOREIGN_KEYS = text("""
    SELECT
        kcu.column_name AS from_column,
        ccu.table_name AS to_table,
        ccu.column_name AS to_column,
        tc.constraint_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON kcu.constraint_name = tc.constraint_name
        AND kcu.table_schema = tc.table_schema
    JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
        AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema = :schema_name
      AND tc.table_name = :table_name
    ORDER BY tc.constraint_name, kcu.ordinal_position
""")
