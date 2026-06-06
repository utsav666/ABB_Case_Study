from __future__ import annotations

from sqlalchemy import text

FETCH_TABLES = text(
    """
    SELECT
        t.table_name,
        (
            SELECT COUNT(*)
            FROM information_schema.columns c
            WHERE c.table_schema = t.table_schema
              AND c.table_name = t.table_name
        ) AS column_count
    FROM information_schema.tables t
    WHERE t.table_schema = :schema_name
      AND t.table_type = 'BASE TABLE'
    ORDER BY t.table_name
    """
)

FETCH_TABLE_COLUMNS = text(
    """
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
    """
)

FETCH_COLUMNS_WITH_PK = text(
    """
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
    """
)

FETCH_FOREIGN_KEYS = text(
    """
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
    """
)
