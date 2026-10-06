-- 003_indexes.sql
-- Spatial GiST + conventional B-tree indexes for million-row workloads.

-- Countries / cities
CREATE INDEX IF NOT EXISTS idx_cities_country_id ON cities (country_id);
CREATE INDEX IF NOT EXISTS idx_cities_inclusion_status ON cities (inclusion_status);
CREATE INDEX IF NOT EXISTS idx_cities_selection_role ON cities (selection_role);
CREATE INDEX IF NOT EXISTS idx_cities_centroid_gix ON cities USING GIST (centroid);
CREATE INDEX IF NOT EXISTS idx_cities_urban_boundary_gix ON cities USING GIST (urban_boundary);

CREATE INDEX IF NOT EXISTS idx_city_candidates_country_id ON city_candidates (country_id);
CREATE INDEX IF NOT EXISTS idx_city_candidates_selected ON city_candidates (selected);
CREATE INDEX IF NOT EXISTS idx_city_candidates_centroid_gix ON city_candidates USING GIST (centroid);

-- Buildings
CREATE INDEX IF NOT EXISTS idx_buildings_city_id ON buildings (city_id);
CREATE INDEX IF NOT EXISTS idx_buildings_source ON buildings (source);
CREATE INDEX IF NOT EXISTS idx_buildings_harmonized_label ON buildings (harmonized_label);
CREATE INDEX IF NOT EXISTS idx_buildings_source_feature_id ON buildings (source_feature_id);
CREATE INDEX IF NOT EXISTS idx_buildings_geom_gix ON buildings USING GIST (geom);

-- AlphaEarth cells
CREATE INDEX IF NOT EXISTS idx_alphaearth_cells_city_id ON alphaearth_cells (city_id);
CREATE INDEX IF NOT EXISTS idx_alphaearth_cells_year ON alphaearth_cells (year);
CREATE INDEX IF NOT EXISTS idx_alphaearth_cells_city_year ON alphaearth_cells (city_id, year);
CREATE INDEX IF NOT EXISTS idx_alphaearth_cells_geom_gix ON alphaearth_cells USING GIST (geom);

-- Intersections / labels
CREATE INDEX IF NOT EXISTS idx_cbi_building_id ON cell_building_intersections (building_id);
CREATE INDEX IF NOT EXISTS idx_cell_labels_label_version ON cell_labels (label_version);
CREATE INDEX IF NOT EXISTS idx_cell_labels_binary_label ON cell_labels (binary_label);

-- Morphology / experiments
CREATE INDEX IF NOT EXISTS idx_morphology_city_id ON morphology_features (city_id);
CREATE INDEX IF NOT EXISTS idx_experiment_splits_experiment_id ON experiment_splits (experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_splits_city_id ON experiment_splits (city_id);
CREATE INDEX IF NOT EXISTS idx_experiment_splits_country_id ON experiment_splits (country_id);
CREATE INDEX IF NOT EXISTS idx_experiment_splits_run_id ON experiment_splits (run_id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_experiment_id ON experiment_runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_runs_status ON experiment_runs (status);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_run_id ON experiment_metrics (run_id);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_metric ON experiment_metrics (metric);
CREATE INDEX IF NOT EXISTS idx_experiment_metrics_source_target
    ON experiment_metrics (source_geography, target_geography);

CREATE INDEX IF NOT EXISTS idx_countries_continent ON countries (continent);

INSERT INTO schema_migrations (version, description)
VALUES ('003', 'spatial GiST + conventional indexes')
ON CONFLICT (version) DO NOTHING;
