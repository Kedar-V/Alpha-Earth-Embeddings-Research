-- Consolidated schema reference (applied via migrations/*.sql).
-- Prefer running scripts/init_db.sh rather than this file alone.
\i migrations/001_extensions.sql
\i migrations/002_core_tables.sql
\i migrations/003_indexes.sql
\i migrations/004_validation.sql
