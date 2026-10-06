-- 004_validation.sql
-- Validation helpers. Never silently drop/repair without logging.

CREATE OR REPLACE VIEW v_duplicate_buildings AS
SELECT source, source_feature_id, city_id, COUNT(*) AS n
FROM buildings
GROUP BY 1, 2, 3
HAVING COUNT(*) > 1;

CREATE OR REPLACE VIEW v_duplicate_alphaearth_cells AS
SELECT city_id, year, ST_AsText(geom) AS geom_wkt, COUNT(*) AS n
FROM alphaearth_cells
GROUP BY 1, 2, 3
HAVING COUNT(*) > 1;

CREATE OR REPLACE VIEW v_invalid_geometries AS
SELECT 'buildings' AS entity, building_id::text AS entity_id, ST_IsValidReason(geom) AS reason
FROM buildings WHERE NOT ST_IsValid(geom)
UNION ALL
SELECT 'alphaearth_cells', cell_id::text, ST_IsValidReason(geom)
FROM alphaearth_cells WHERE NOT ST_IsValid(geom)
UNION ALL
SELECT 'cities.urban_boundary', city_id, ST_IsValidReason(urban_boundary)
FROM cities WHERE urban_boundary IS NOT NULL AND NOT ST_IsValid(urban_boundary);

CREATE OR REPLACE VIEW v_orphan_intersections AS
SELECT cbi.cell_id, cbi.building_id
FROM cell_building_intersections cbi
LEFT JOIN alphaearth_cells c ON c.cell_id = cbi.cell_id
LEFT JOIN buildings b ON b.building_id = cbi.building_id
WHERE c.cell_id IS NULL OR b.building_id IS NULL;

CREATE OR REPLACE VIEW v_null_embeddings AS
SELECT cell_id, city_id, year
FROM alphaearth_cells
WHERE a00 IS NULL OR a01 IS NULL OR a02 IS NULL OR a03 IS NULL
   OR a04 IS NULL OR a05 IS NULL OR a06 IS NULL OR a07 IS NULL
   OR a08 IS NULL OR a09 IS NULL OR a10 IS NULL OR a11 IS NULL
   OR a12 IS NULL OR a13 IS NULL OR a14 IS NULL OR a15 IS NULL
   OR a16 IS NULL OR a17 IS NULL OR a18 IS NULL OR a19 IS NULL
   OR a20 IS NULL OR a21 IS NULL OR a22 IS NULL OR a23 IS NULL
   OR a24 IS NULL OR a25 IS NULL OR a26 IS NULL OR a27 IS NULL
   OR a28 IS NULL OR a29 IS NULL OR a30 IS NULL OR a31 IS NULL
   OR a32 IS NULL OR a33 IS NULL OR a34 IS NULL OR a35 IS NULL
   OR a36 IS NULL OR a37 IS NULL OR a38 IS NULL OR a39 IS NULL
   OR a40 IS NULL OR a41 IS NULL OR a42 IS NULL OR a43 IS NULL
   OR a44 IS NULL OR a45 IS NULL OR a46 IS NULL OR a47 IS NULL
   OR a48 IS NULL OR a49 IS NULL OR a50 IS NULL OR a51 IS NULL
   OR a52 IS NULL OR a53 IS NULL OR a54 IS NULL OR a55 IS NULL
   OR a56 IS NULL OR a57 IS NULL OR a58 IS NULL OR a59 IS NULL
   OR a60 IS NULL OR a61 IS NULL OR a62 IS NULL OR a63 IS NULL;

CREATE OR REPLACE VIEW v_inconsistent_srids AS
SELECT 'buildings' AS entity, building_id::text AS entity_id, ST_SRID(geom) AS srid
FROM buildings WHERE ST_SRID(geom) <> 4326
UNION ALL
SELECT 'alphaearth_cells', cell_id::text, ST_SRID(geom)
FROM alphaearth_cells WHERE ST_SRID(geom) <> 4326
UNION ALL
SELECT 'cities.centroid', city_id, ST_SRID(centroid)
FROM cities WHERE centroid IS NOT NULL AND ST_SRID(centroid) <> 4326;

CREATE OR REPLACE FUNCTION run_data_quality_checks()
RETURNS TABLE(check_name TEXT, n_issues BIGINT)
LANGUAGE sql
AS $$
    SELECT 'duplicate_buildings'::text, COUNT(*) FROM v_duplicate_buildings
    UNION ALL
    SELECT 'duplicate_alphaearth_cells', COUNT(*) FROM v_duplicate_alphaearth_cells
    UNION ALL
    SELECT 'invalid_geometries', COUNT(*) FROM v_invalid_geometries
    UNION ALL
    SELECT 'orphan_intersections', COUNT(*) FROM v_orphan_intersections
    UNION ALL
    SELECT 'null_embeddings', COUNT(*) FROM v_null_embeddings
    UNION ALL
    SELECT 'inconsistent_srids', COUNT(*) FROM v_inconsistent_srids
    UNION ALL
    SELECT 'missing_city_country', COUNT(*)
    FROM cities c LEFT JOIN countries co ON co.country_id = c.country_id
    WHERE co.country_id IS NULL
    UNION ALL
    SELECT 'impossible_latlon_cities', COUNT(*)
    FROM cities
    WHERE latitude IS NOT NULL AND (latitude < -90 OR latitude > 90
       OR longitude < -180 OR longitude > 180)
    UNION ALL
    SELECT 'unexpected_labels', COUNT(*)
    FROM buildings
    WHERE harmonized_label IS NOT NULL
      AND harmonized_label NOT IN ('residential','non_residential','mixed','unknown','empty');
$$;

INSERT INTO schema_migrations (version, description)
VALUES ('004', 'validation views + run_data_quality_checks()')
ON CONFLICT (version) DO NOTHING;
