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

    is_healthy BOOLEAN DEFAULT TRUE,
    last_synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =========================
-- INDEXES
-- =========================
CREATE INDEX idx_workspace_owner 
ON settings.workspaces(owner_id);

CREATE INDEX idx_workspace_members_user 
ON settings.workspace_members(user_id);

CREATE INDEX idx_db_connections_workspace 
ON settings.db_connections(workspace_id);