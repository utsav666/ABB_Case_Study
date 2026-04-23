-- SCHEMA: settings

-- DROP SCHEMA IF EXISTS settings ;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =========================
-- USERS
-- =========================
CREATE TABLE settings.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

-- =========================
-- WORKSPACES
-- =========================
CREATE TABLE settings.workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    owner_id UUID REFERENCES settings.users(id),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- =========================
-- USER ROLE ENUM
-- =========================
CREATE TYPE settings.user_role AS ENUM ('ADMIN', 'ARCHITECT', 'VIEWER');

-- =========================
-- WORKSPACE MEMBERS
-- =========================
CREATE TABLE settings.workspace_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES settings.workspaces(id) ON DELETE CASCADE,
    user_id UUID REFERENCES settings.users(id) ON DELETE CASCADE,
    role settings.user_role DEFAULT 'VIEWER',
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, user_id)
);

-- =========================
-- DB CONNECTIONS
-- =========================
CREATE TABLE settings.db_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES settings.workspaces(id) ON DELETE CASCADE,

    name TEXT NOT NULL,
    engine TEXT NOT NULL,

    db_host TEXT NOT NULL,
    db_port INTEGER NOT NULL DEFAULT 5432,
    db_name TEXT NOT NULL,
    db_user TEXT NOT NULL,
    db_password TEXT NOT NULL,
    db_schema TEXT,
    use_ssl BOOLEAN NOT NULL DEFAULT FALSE,
    extra_options TEXT,

    is_healthy BOOLEAN DEFAULT TRUE,
    last_synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table: settings.column_definitions

-- DROP TABLE IF EXISTS settings.column_definitions;

CREATE TABLE IF NOT EXISTS settings.column_definitions
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    table_id uuid,
    raw_name text COLLATE pg_catalog."default" NOT NULL,
    logical_name text COLLATE pg_catalog."default",
    description text COLLATE pg_catalog."default",
    data_type text COLLATE pg_catalog."default",
    is_pii boolean DEFAULT false,
    is_filterable boolean DEFAULT true,
    example_values text[] COLLATE pg_catalog."default",
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT column_definitions_pkey PRIMARY KEY (id),
    CONSTRAINT column_definitions_table_id_fkey FOREIGN KEY (table_id)
        REFERENCES settings.selected_tables (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS settings.column_definitions
    OWNER to postgres;


-- Table: settings.selected_tables

-- DROP TABLE IF EXISTS settings.selected_tables;

CREATE TABLE IF NOT EXISTS settings.selected_tables
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    db_connection_id uuid,
    schema_name text COLLATE pg_catalog."default" NOT NULL DEFAULT 'public'::text,
    table_name text COLLATE pg_catalog."default" NOT NULL,
    human_description text COLLATE pg_catalog."default",
    CONSTRAINT selected_tables_pkey PRIMARY KEY (id),
    CONSTRAINT selected_tables_db_connection_id_schema_name_table_name_key UNIQUE (db_connection_id, schema_name, table_name),
    CONSTRAINT selected_tables_db_connection_id_fkey FOREIGN KEY (db_connection_id)
        REFERENCES settings.db_connections (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS settings.selected_tables
    OWNER to postgres;


-- Table: settings.agents

-- DROP TABLE IF EXISTS settings.agents;

CREATE TABLE IF NOT EXISTS settings.agents
(
    id uuid NOT NULL DEFAULT uuid_generate_v4(),
    workspace_id uuid,
    name text COLLATE pg_catalog."default" DEFAULT 'Data Assistant'::text,
    model_version text COLLATE pg_catalog."default" DEFAULT 'gpt-4o'::text,
    system_instructions text COLLATE pg_catalog."default",
    config jsonb DEFAULT '{"max_tokens": 1000, "temperature": 0}'::jsonb,
    is_public_to_team boolean DEFAULT true,
    CONSTRAINT agents_pkey PRIMARY KEY (id),
    CONSTRAINT agents_workspace_id_fkey FOREIGN KEY (workspace_id)
        REFERENCES settings.workspaces (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS settings.agents
    OWNER to postgres;    


-- =========================
-- INDEXES
-- =========================
CREATE INDEX idx_workspace_owner 
ON settings.workspaces(owner_id);

CREATE INDEX idx_workspace_members_user 
ON settings.workspace_members(user_id);

CREATE INDEX idx_db_connections_workspace 
ON settings.db_connections(workspace_id);