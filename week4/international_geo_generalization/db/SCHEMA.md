# Schema overview — international AlphaEarth study

PostgreSQL + PostGIS is the **source of truth**. CSV/Parquet/JSON files are exports or disposable experiment caches.

## Hierarchy

```text
continent
  └── countries
        ├── city_candidates          (full selection pool)
        └── cities                   (selected / included)
              ├── aoi_center + aoi_radius_m → aoi_disk
              ├── buildings
              │     └── building_label_audit      (Track-A / y_res sidecar)
              ├── alphaearth_cells
              │     ├── cell_spatial_metrics      (lon/lat/size/area_crs)
              │     ├── cell_degree_of_urbanisation  (GHS-SMOD code + dou rollup)
              │     ├── cell_building_intersections
              │     └── cell_labels (versioned)
              │           └── cell_label_metrics  (areas, frac_*_full, counts)
              └── morphology_features (versioned)
```

AOI contract (international): disk from city center with radius R that covers
all discovered buildings for the city. No zone rings. Week4 audit fields live
in sidecar tables — see `docs/DATASET_CONTRACT.md`.

Experiment lineage (orthogonal, references geography):

```text
experiment_runs
  ├── experiment_splits
  └── experiment_metrics

experiment_extracts   (versioned caches generated FROM SQL)
data_sources          (provenance for every import)
data_quality_events   (logged repairs / quarantines — never silent drops)
```

## Entity notes

| Table | Role |
|-------|------|
| `countries` | Fixed 30-country study frame (`iso2`/`iso3`, continent) |
| `city_candidates` | Entire candidate pool used during selection (winners + rejects) |
| `cities` | Canonical city records with centroid / optional urban boundary |
| `buildings` | Source footprints + original/harmonized labels + `source_feature_id` |
| `alphaearth_cells` | Cell geometry + year + explicit `a00`…`a63` embedding columns |
| `cell_building_intersections` | Area/fraction join so labels can be recomputed offline |
| `cell_labels` | Versioned cell ground truth (`label_version` is part of PK) |
| `cell_degree_of_urbanisation` | GHS-SMOD L2 class at the cell center, plus the urban / peri-urban / rural rollup. Stratifier only — not residential ground truth |
| `morphology_features` | Versioned city morphology descriptors |
| `experiment_splits` | Geographic train/val/test assignments for leakage audits |
| `experiment_runs` / `experiment_metrics` | Run metadata + SQL-reconstructable transfer matrices |
| `experiment_extracts` | Manifest for NumPy/Parquet caches derived from the DB |

## Geometry / SRID

All geometries use **EPSG:4326**. Spatial GiST indexes exist on:

- `cities.centroid`, `cities.urban_boundary`
- `city_candidates.centroid`
- `buildings.geom`
- `alphaearth_cells.geom`

## Provenance rule

Every imported dataset should attach a `data_sources` row (`source_name`, `source_version`, `acquisition_date`, `processing_version`, `label_mapping_version`) and reference it from buildings/cells/morphology where applicable.

## Label versioning

`cell_labels` is keyed by `(cell_id, label_version)`. Never overwrite an old definition in place — insert a new version.
