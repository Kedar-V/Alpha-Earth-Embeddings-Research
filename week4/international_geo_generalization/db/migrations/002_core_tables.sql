-- 002_core_tables.sql
-- Core normalized entities for the international AlphaEarth study.

CREATE TABLE IF NOT EXISTS data_sources (
    source_id           BIGSERIAL PRIMARY KEY,
    source_name         TEXT NOT NULL,
    source_version      TEXT,
    source_year         INTEGER,
    acquisition_date    DATE,
    processing_version  TEXT,
    label_mapping_version TEXT,
    uri                 TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_name, source_version, processing_version)
);

CREATE TABLE IF NOT EXISTS countries (
    country_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    iso2                CHAR(2) NOT NULL UNIQUE,
    iso3                CHAR(3) NOT NULL UNIQUE,
    continent           TEXT NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT countries_continent_chk CHECK (
        continent IN (
            'Africa', 'Antarctica', 'Asia', 'Europe',
            'North America', 'Oceania', 'South America'
        )
    )
);

CREATE TABLE IF NOT EXISTS cities (
    city_id             TEXT PRIMARY KEY,
    country_id          TEXT NOT NULL REFERENCES countries(country_id),
    city_name           TEXT NOT NULL,
    admin_region        TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    centroid            geometry(Point, 4326),
    urban_boundary      geometry(MultiPolygon, 4326),
    selection_role      TEXT,
    selection_reason    TEXT,
    morphology_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    environment_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_status      TEXT NOT NULL DEFAULT 'unknown',
    inclusion_status    TEXT NOT NULL DEFAULT 'candidate',
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT cities_lat_chk CHECK (latitude IS NULL OR (latitude BETWEEN -90 AND 90)),
    CONSTRAINT cities_lon_chk CHECK (longitude IS NULL OR (longitude BETWEEN -180 AND 180)),
    CONSTRAINT cities_inclusion_chk CHECK (
        inclusion_status IN ('candidate', 'selected', 'excluded', 'deferred')
    )
);

CREATE TABLE IF NOT EXISTS city_candidates (
    candidate_id        BIGSERIAL PRIMARY KEY,
    country_id          TEXT NOT NULL REFERENCES countries(country_id),
    city_name           TEXT NOT NULL,
    admin_region        TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    centroid            geometry(Point, 4326),
    morphology_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    geographic_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_scores      JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    selected            BOOLEAN NOT NULL DEFAULT FALSE,
    rejection_reason    TEXT,
    selected_city_id    TEXT REFERENCES cities(city_id),
    selection_rank      INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (country_id, city_name, admin_region),
    CONSTRAINT city_candidates_lat_chk CHECK (latitude IS NULL OR (latitude BETWEEN -90 AND 90)),
    CONSTRAINT city_candidates_lon_chk CHECK (longitude IS NULL OR (longitude BETWEEN -180 AND 180))
);

CREATE TABLE IF NOT EXISTS buildings (
    building_id         BIGSERIAL PRIMARY KEY,
    city_id             TEXT NOT NULL REFERENCES cities(city_id),
    source              TEXT NOT NULL,
    source_feature_id   TEXT NOT NULL,
    geom                geometry(Geometry, 4326) NOT NULL,
    building_type       TEXT,
    subtype             TEXT,
    original_source_label TEXT,
    harmonized_label    TEXT,
    label_confidence    DOUBLE PRECISION,
    source_version      TEXT,
    source_date         DATE,
    source_id           BIGINT REFERENCES data_sources(source_id),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_feature_id, city_id),
    CONSTRAINT buildings_label_chk CHECK (
        harmonized_label IS NULL OR harmonized_label IN (
            'residential', 'non_residential', 'mixed', 'unknown', 'empty'
        )
    )
);

-- AlphaEarth 64-D embedding as explicit numeric columns a00..a63
CREATE TABLE IF NOT EXISTS alphaearth_cells (
    cell_id             BIGSERIAL PRIMARY KEY,
    city_id             TEXT NOT NULL REFERENCES cities(city_id),
    geom                geometry(Polygon, 4326) NOT NULL,
    year                INTEGER NOT NULL,
    source              TEXT NOT NULL DEFAULT 'alphaearth',
    source_version      TEXT,
    source_id           BIGINT REFERENCES data_sources(source_id),
    a00 DOUBLE PRECISION, a01 DOUBLE PRECISION, a02 DOUBLE PRECISION, a03 DOUBLE PRECISION,
    a04 DOUBLE PRECISION, a05 DOUBLE PRECISION, a06 DOUBLE PRECISION, a07 DOUBLE PRECISION,
    a08 DOUBLE PRECISION, a09 DOUBLE PRECISION, a10 DOUBLE PRECISION, a11 DOUBLE PRECISION,
    a12 DOUBLE PRECISION, a13 DOUBLE PRECISION, a14 DOUBLE PRECISION, a15 DOUBLE PRECISION,
    a16 DOUBLE PRECISION, a17 DOUBLE PRECISION, a18 DOUBLE PRECISION, a19 DOUBLE PRECISION,
    a20 DOUBLE PRECISION, a21 DOUBLE PRECISION, a22 DOUBLE PRECISION, a23 DOUBLE PRECISION,
    a24 DOUBLE PRECISION, a25 DOUBLE PRECISION, a26 DOUBLE PRECISION, a27 DOUBLE PRECISION,
    a28 DOUBLE PRECISION, a29 DOUBLE PRECISION, a30 DOUBLE PRECISION, a31 DOUBLE PRECISION,
    a32 DOUBLE PRECISION, a33 DOUBLE PRECISION, a34 DOUBLE PRECISION, a35 DOUBLE PRECISION,
    a36 DOUBLE PRECISION, a37 DOUBLE PRECISION, a38 DOUBLE PRECISION, a39 DOUBLE PRECISION,
    a40 DOUBLE PRECISION, a41 DOUBLE PRECISION, a42 DOUBLE PRECISION, a43 DOUBLE PRECISION,
    a44 DOUBLE PRECISION, a45 DOUBLE PRECISION, a46 DOUBLE PRECISION, a47 DOUBLE PRECISION,
    a48 DOUBLE PRECISION, a49 DOUBLE PRECISION, a50 DOUBLE PRECISION, a51 DOUBLE PRECISION,
    a52 DOUBLE PRECISION, a53 DOUBLE PRECISION, a54 DOUBLE PRECISION, a55 DOUBLE PRECISION,
    a56 DOUBLE PRECISION, a57 DOUBLE PRECISION, a58 DOUBLE PRECISION, a59 DOUBLE PRECISION,
    a60 DOUBLE PRECISION, a61 DOUBLE PRECISION, a62 DOUBLE PRECISION, a63 DOUBLE PRECISION,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (city_id, year, geom)
);

CREATE TABLE IF NOT EXISTS cell_building_intersections (
    cell_id             BIGINT NOT NULL REFERENCES alphaearth_cells(cell_id) ON DELETE CASCADE,
    building_id         BIGINT NOT NULL REFERENCES buildings(building_id) ON DELETE CASCADE,
    intersection_area   DOUBLE PRECISION NOT NULL,
    building_fraction   DOUBLE PRECISION,
    cell_fraction       DOUBLE PRECISION,
    frac_res_contrib    DOUBLE PRECISION,
    frac_nres_contrib   DOUBLE PRECISION,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_id, building_id),
    CONSTRAINT cbi_area_chk CHECK (intersection_area >= 0)
);

CREATE TABLE IF NOT EXISTS cell_labels (
    cell_id             BIGINT NOT NULL REFERENCES alphaearth_cells(cell_id) ON DELETE CASCADE,
    label_version       TEXT NOT NULL,
    frac_res            DOUBLE PRECISION NOT NULL DEFAULT 0,
    frac_nres           DOUBLE PRECISION NOT NULL DEFAULT 0,
    frac_empty          DOUBLE PRECISION NOT NULL DEFAULT 0,
    binary_label        SMALLINT,
    purity_status       TEXT,
    ambiguity_status    TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_id, label_version),
    CONSTRAINT cell_labels_frac_sum_chk CHECK (
        frac_res >= 0 AND frac_nres >= 0 AND frac_empty >= 0
        AND frac_res + frac_nres + frac_empty <= 1.000001
    ),
    CONSTRAINT cell_labels_binary_chk CHECK (
        binary_label IS NULL OR binary_label IN (0, 1)
    )
);

CREATE TABLE IF NOT EXISTS morphology_features (
    morphology_id       BIGSERIAL PRIMARY KEY,
    city_id             TEXT NOT NULL REFERENCES cities(city_id),
    feature_version     TEXT NOT NULL,
    building_density    DOUBLE PRECISION,
    footprint_mean_m2   DOUBLE PRECISION,
    footprint_median_m2 DOUBLE PRECISION,
    footprint_p90_m2    DOUBLE PRECISION,
    built_up_density    DOUBLE PRECISION,
    building_height_mean DOUBLE PRECISION,
    building_height_p90  DOUBLE PRECISION,
    road_density        DOUBLE PRECISION,
    vegetation_fraction DOUBLE PRECISION,
    built_fraction      DOUBLE PRECISION,
    compactness         DOUBLE PRECISION,
    features            JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_id           BIGINT REFERENCES data_sources(source_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (city_id, feature_version)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id              TEXT PRIMARY KEY,
    experiment_name     TEXT NOT NULL,
    experiment_id       TEXT NOT NULL,
    git_commit          TEXT,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    seed                INTEGER,
    representation      TEXT,
    model               TEXT,
    source_geographies  JSONB NOT NULL DEFAULT '[]'::jsonb,
    target_geographies  JSONB NOT NULL DEFAULT '[]'::jsonb,
    status              TEXT NOT NULL DEFAULT 'pending',
    output_directory    TEXT,
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT experiment_runs_status_chk CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    )
);

CREATE TABLE IF NOT EXISTS experiment_splits (
    split_id            BIGSERIAL PRIMARY KEY,
    experiment_id       TEXT NOT NULL,
    run_id              TEXT REFERENCES experiment_runs(run_id),
    sample_id           TEXT,
    city_id             TEXT REFERENCES cities(city_id),
    country_id          TEXT REFERENCES countries(country_id),
    split               TEXT NOT NULL,
    fold                INTEGER,
    seed                INTEGER,
    split_strategy      TEXT NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT experiment_splits_split_chk CHECK (
        split IN ('train', 'validation', 'test', 'heldout')
    )
);

CREATE TABLE IF NOT EXISTS experiment_metrics (
    metric_id           BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    fold                INTEGER,
    source_geography    TEXT,
    target_geography    TEXT,
    metric              TEXT NOT NULL,
    value               DOUBLE PRECISION NOT NULL,
    dimensionality      INTEGER,
    representation      TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_extracts (
    extract_id          BIGSERIAL PRIMARY KEY,
    experiment_id       TEXT NOT NULL,
    extract_path        TEXT NOT NULL,
    sql_query           TEXT,
    extraction_script   TEXT,
    database_snapshot   TEXT,
    label_version       TEXT,
    selected_countries  JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_cities     JSONB NOT NULL DEFAULT '[]'::jsonb,
    row_count           BIGINT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    event_id            BIGSERIAL PRIMARY KEY,
    entity_type         TEXT NOT NULL,
    entity_id           TEXT,
    check_name          TEXT NOT NULL,
    severity            TEXT NOT NULL DEFAULT 'warning',
    detail              TEXT,
    action_taken        TEXT NOT NULL DEFAULT 'logged',
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT dqe_severity_chk CHECK (severity IN ('info', 'warning', 'error')),
    CONSTRAINT dqe_action_chk CHECK (
        action_taken IN ('logged', 'repaired', 'dropped', 'quarantined')
    )
);

INSERT INTO schema_migrations (version, description)
VALUES ('002', 'core tables: countries..experiment_metrics + provenance')
ON CONFLICT (version) DO NOTHING;
