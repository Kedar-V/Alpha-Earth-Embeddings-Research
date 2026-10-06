-- 006_drop_aoi_zone_rings.sql
-- AOI is only center + enclosing radius R. No urban/peri-urban/rural rings.

DROP TABLE IF EXISTS city_aoi_zones CASCADE;

DROP FUNCTION IF EXISTS aoi_zone_from_distance(DOUBLE PRECISION, DOUBLE PRECISION, JSONB);

ALTER TABLE cities DROP COLUMN IF EXISTS aoi_zone_spec;

ALTER TABLE alphaearth_cells DROP CONSTRAINT IF EXISTS alphaearth_cells_aoi_zone_chk;
ALTER TABLE alphaearth_cells DROP COLUMN IF EXISTS aoi_zone;
ALTER TABLE alphaearth_cells DROP COLUMN IF EXISTS dist_to_center_m;

ALTER TABLE buildings DROP CONSTRAINT IF EXISTS buildings_aoi_zone_chk;
ALTER TABLE buildings DROP COLUMN IF EXISTS aoi_zone;
ALTER TABLE buildings DROP COLUMN IF EXISTS dist_to_center_m;

DROP INDEX IF EXISTS idx_alphaearth_cells_aoi_zone;
DROP INDEX IF EXISTS idx_buildings_aoi_zone;

INSERT INTO schema_migrations (version, description)
VALUES ('006', 'drop AOI zone rings; keep center + enclosing radius only')
ON CONFLICT (version) DO NOTHING;
