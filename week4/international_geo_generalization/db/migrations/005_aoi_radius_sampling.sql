-- 005_aoi_radius_sampling.sql
-- City AOI = center + radius R (disk), with urban / peri-urban / rural rings.
-- Intentional departure from US downtown bboxes; label/embedding rules stay aligned.

ALTER TABLE cities
    ADD COLUMN IF NOT EXISTS aoi_center           geometry(Point, 4326),
    ADD COLUMN IF NOT EXISTS aoi_radius_m         DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS aoi_disk             geometry(Polygon, 4326),
    ADD COLUMN IF NOT EXISTS aoi_method           TEXT,
    ADD COLUMN IF NOT EXISTS aoi_discovery_radius_m DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS aoi_buffer_m         DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS aoi_building_count   BIGINT,
    ADD COLUMN IF NOT EXISTS aoi_computed_at      TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS aoi_zone_spec        JSONB NOT NULL DEFAULT '{
        "urban_max_frac": 0.35,
        "peri_urban_max_frac": 0.70,
        "rural_max_frac": 1.00,
        "distance_metric": "geodesic_m_from_aoi_center"
    }'::jsonb;

ALTER TABLE cities
    DROP CONSTRAINT IF EXISTS cities_aoi_radius_chk;
ALTER TABLE cities
    ADD CONSTRAINT cities_aoi_radius_chk
    CHECK (aoi_radius_m IS NULL OR aoi_radius_m > 0);

ALTER TABLE cities
    DROP CONSTRAINT IF EXISTS cities_aoi_method_chk;
ALTER TABLE cities
    ADD CONSTRAINT cities_aoi_method_chk
    CHECK (
        aoi_method IS NULL OR aoi_method IN (
            'building_enclosing_radius',
            'manual',
            'admin_boundary_equiv_radius',
            'provisional'
        )
    );

-- Backfill aoi_center from centroid when missing
UPDATE cities
SET aoi_center = centroid
WHERE aoi_center IS NULL AND centroid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_cities_aoi_center_gix ON cities USING GIST (aoi_center);
CREATE INDEX IF NOT EXISTS idx_cities_aoi_disk_gix ON cities USING GIST (aoi_disk);

-- Explicit zone rings per city (materialized for audit / sampling)
CREATE TABLE IF NOT EXISTS city_aoi_zones (
    city_id             TEXT NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
    zone                TEXT NOT NULL,
    r_min_m             DOUBLE PRECISION NOT NULL,
    r_max_m             DOUBLE PRECISION NOT NULL,
    ring_geom           geometry(Polygon, 4326),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city_id, zone),
    CONSTRAINT city_aoi_zones_zone_chk CHECK (
        zone IN ('urban', 'peri_urban', 'rural')
    ),
    CONSTRAINT city_aoi_zones_r_chk CHECK (r_min_m >= 0 AND r_max_m > r_min_m)
);

CREATE INDEX IF NOT EXISTS idx_city_aoi_zones_ring_gix ON city_aoi_zones USING GIST (ring_geom);

-- Optional zone tag on cells / buildings once ingested
ALTER TABLE alphaearth_cells
    ADD COLUMN IF NOT EXISTS aoi_zone TEXT,
    ADD COLUMN IF NOT EXISTS dist_to_center_m DOUBLE PRECISION;
ALTER TABLE alphaearth_cells
    DROP CONSTRAINT IF EXISTS alphaearth_cells_aoi_zone_chk;
ALTER TABLE alphaearth_cells
    ADD CONSTRAINT alphaearth_cells_aoi_zone_chk
    CHECK (aoi_zone IS NULL OR aoi_zone IN ('urban', 'peri_urban', 'rural'));

ALTER TABLE buildings
    ADD COLUMN IF NOT EXISTS aoi_zone TEXT,
    ADD COLUMN IF NOT EXISTS dist_to_center_m DOUBLE PRECISION;
ALTER TABLE buildings
    DROP CONSTRAINT IF EXISTS buildings_aoi_zone_chk;
ALTER TABLE buildings
    ADD CONSTRAINT buildings_aoi_zone_chk
    CHECK (aoi_zone IS NULL OR aoi_zone IN ('urban', 'peri_urban', 'rural'));

CREATE INDEX IF NOT EXISTS idx_alphaearth_cells_aoi_zone ON alphaearth_cells (aoi_zone);
CREATE INDEX IF NOT EXISTS idx_buildings_aoi_zone ON buildings (aoi_zone);

-- Helper: geodesic buffer disk around aoi_center (meters → geography)
CREATE OR REPLACE FUNCTION city_aoi_disk_from_radius(
    center geometry(Point, 4326),
    radius_m DOUBLE PRECISION
) RETURNS geometry(Polygon, 4326)
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT ST_SetSRID(
        ST_Buffer(center::geography, radius_m)::geometry,
        4326
    )::geometry(Polygon, 4326);
$$;

-- Assign zone from distance fraction of R using cities.aoi_zone_spec
CREATE OR REPLACE FUNCTION aoi_zone_from_distance(
    dist_m DOUBLE PRECISION,
    radius_m DOUBLE PRECISION,
    zone_spec JSONB
) RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE
        WHEN dist_m IS NULL OR radius_m IS NULL OR radius_m <= 0 THEN NULL
        WHEN dist_m / radius_m <= COALESCE((zone_spec->>'urban_max_frac')::DOUBLE PRECISION, 0.35)
            THEN 'urban'
        WHEN dist_m / radius_m <= COALESCE((zone_spec->>'peri_urban_max_frac')::DOUBLE PRECISION, 0.70)
            THEN 'peri_urban'
        WHEN dist_m / radius_m <= COALESCE((zone_spec->>'rural_max_frac')::DOUBLE PRECISION, 1.00)
            THEN 'rural'
        ELSE NULL  -- outside AOI
    END;
$$;

INSERT INTO schema_migrations (version, description)
VALUES ('005', 'AOI center+radius sampling with urban/peri-urban/rural zones')
ON CONFLICT (version) DO NOTHING;
