-- 008_cell_key.sql
-- Stable cell identity for upserts (geometry UNIQUE is brittle).

ALTER TABLE alphaearth_cells
    ADD COLUMN IF NOT EXISTS cell_key TEXT;

UPDATE alphaearth_cells
SET cell_key = metadata->>'pixel_wkt_sha1'
WHERE cell_key IS NULL AND metadata ? 'pixel_wkt_sha1';

CREATE UNIQUE INDEX IF NOT EXISTS uq_alphaearth_cells_city_year_key
    ON alphaearth_cells (city_id, year, cell_key)
    WHERE cell_key IS NOT NULL;

INSERT INTO schema_migrations (version, description)
VALUES ('008', 'alphaearth_cells.cell_key for stable upserts')
ON CONFLICT (version) DO NOTHING;
