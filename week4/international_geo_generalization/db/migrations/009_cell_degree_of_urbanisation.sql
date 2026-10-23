-- 009_cell_degree_of_urbanisation.sql
-- Per-cell GHSL GHS-SMOD Degree of Urbanisation. Stratifier only — not building GT.

CREATE TABLE IF NOT EXISTS cell_degree_of_urbanisation (
    cell_id             BIGINT PRIMARY KEY REFERENCES alphaearth_cells(cell_id) ON DELETE CASCADE,
    smod_code           SMALLINT NOT NULL,
    smod_label          TEXT NOT NULL,
    dou                 TEXT,
    epoch               INTEGER NOT NULL,
    product             TEXT NOT NULL,
    source_id           BIGINT REFERENCES data_sources(source_id),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT cell_dou_label_chk CHECK (
        smod_label IN (
            'nodata', 'water',
            'very_low_density_rural', 'low_density_rural', 'rural_cluster',
            'suburban_peri_urban', 'semi_dense_urban_cluster',
            'dense_urban_cluster', 'urban_centre',
            'unclassified'
        )
    ),
    CONSTRAINT cell_dou_rollup_chk CHECK (
        dou IS NULL OR dou IN ('urban', 'peri_urban', 'rural')
    )
);

CREATE INDEX IF NOT EXISTS idx_cell_dou_smod_code
    ON cell_degree_of_urbanisation (smod_code);
CREATE INDEX IF NOT EXISTS idx_cell_dou_dou
    ON cell_degree_of_urbanisation (dou);
CREATE INDEX IF NOT EXISTS idx_cell_dou_smod_label
    ON cell_degree_of_urbanisation (smod_label);

INSERT INTO schema_migrations (version, description)
VALUES ('009', 'cell_degree_of_urbanisation: GHS-SMOD label per AlphaEarth cell')
ON CONFLICT (version) DO NOTHING;
