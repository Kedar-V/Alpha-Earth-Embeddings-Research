-- 007_week4_audit_sidecar_tables.sql
-- Extra week4-parity fields live in sidecar tables; core raw tables stay lean.

-- Track-A / resi_def_v1 audit attributes for each building
CREATE TABLE IF NOT EXISTS building_label_audit (
    building_id         BIGINT PRIMARY KEY REFERENCES buildings(building_id) ON DELETE CASCADE,
    label_definition    TEXT NOT NULL DEFAULT 'resi_def_v1',
    overture_class      TEXT,
    overture_subtype    TEXT,
    brief_category      TEXT,
    study_label         TEXT,
    y_res               SMALLINT,
    height_m            DOUBLE PRECISION,
    num_floors          INTEGER,
    area_m2_approx      DOUBLE PRECISION,
    source_primary      TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT building_label_audit_y_res_chk CHECK (y_res IS NULL OR y_res IN (0, 1)),
    CONSTRAINT building_label_audit_study_label_chk CHECK (
        study_label IS NULL OR study_label IN (
            'single_family', 'multi_family',
            'education', 'medical', 'retail', 'office', 'hotel',
            'industrial', 'religious', 'civic'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_building_label_audit_overture_class
    ON building_label_audit (overture_class);
CREATE INDEX IF NOT EXISTS idx_building_label_audit_y_res
    ON building_label_audit (y_res);
CREATE INDEX IF NOT EXISTS idx_building_label_audit_label_definition
    ON building_label_audit (label_definition);

-- Per-cell spatial metrics used by week4 analysis / area math
CREATE TABLE IF NOT EXISTS cell_spatial_metrics (
    cell_id             BIGINT PRIMARY KEY REFERENCES alphaearth_cells(cell_id) ON DELETE CASCADE,
    pixel_lon           DOUBLE PRECISION NOT NULL,
    pixel_lat           DOUBLE PRECISION NOT NULL,
    pixel_size_m        DOUBLE PRECISION NOT NULL DEFAULT 10,
    pixel_area_m2       DOUBLE PRECISION,
    area_crs            TEXT,
    utm_x               DOUBLE PRECISION,
    utm_y               DOUBLE PRECISION,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT cell_spatial_metrics_lat_chk CHECK (pixel_lat BETWEEN -90 AND 90),
    CONSTRAINT cell_spatial_metrics_lon_chk CHECK (pixel_lon BETWEEN -180 AND 180),
    CONSTRAINT cell_spatial_metrics_size_chk CHECK (pixel_size_m > 0)
);

CREATE INDEX IF NOT EXISTS idx_cell_spatial_metrics_lonlat
    ON cell_spatial_metrics (pixel_lon, pixel_lat);

-- Area / full-fraction / building-count audit beside versioned cell_labels
CREATE TABLE IF NOT EXISTS cell_label_metrics (
    cell_id             BIGINT NOT NULL REFERENCES alphaearth_cells(cell_id) ON DELETE CASCADE,
    label_version       TEXT NOT NULL,
    area_res_m2         DOUBLE PRECISION,
    area_nres_m2        DOUBLE PRECISION,
    frac_res_full       DOUBLE PRECISION,
    frac_nres_full      DOUBLE PRECISION,
    frac_empty_full     DOUBLE PRECISION,
    n_res_buildings     INTEGER,
    n_nres_buildings    INTEGER,
    n_buildings         INTEGER,
    area_crs            TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cell_id, label_version),
    CONSTRAINT cell_label_metrics_fk
        FOREIGN KEY (cell_id, label_version)
        REFERENCES cell_labels (cell_id, label_version)
        ON DELETE CASCADE,
    CONSTRAINT cell_label_metrics_counts_chk CHECK (
        (n_buildings IS NULL OR n_buildings >= 0)
        AND (n_res_buildings IS NULL OR n_res_buildings >= 0)
        AND (n_nres_buildings IS NULL OR n_nres_buildings >= 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_cell_label_metrics_label_version
    ON cell_label_metrics (label_version);

INSERT INTO schema_migrations (version, description)
VALUES ('007', 'week4 audit sidecar tables: building_label_audit, cell_spatial_metrics, cell_label_metrics')
ON CONFLICT (version) DO NOTHING;
