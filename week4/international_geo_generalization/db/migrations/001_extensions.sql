-- 001_extensions.sql
-- Idempotent: enable required extensions.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version      TEXT PRIMARY KEY,
    applied_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    description  TEXT
);

INSERT INTO schema_migrations (version, description)
VALUES ('001', 'extensions: postgis, pgcrypto, schema_migrations')
ON CONFLICT (version) DO NOTHING;
