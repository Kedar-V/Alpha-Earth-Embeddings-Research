# International dataset contract (parity + AOI)

Locked sampling / labeling rules for the international AlphaEarth study.

## Keep parity with the 10 US downtown study

| Component | Locked value | US reference |
|-----------|--------------|--------------|
| Buildings | Overture Buildings (same Track-A class allowlist) | `scripts/build_building_type_dataset.py` |
| Label map | `resi_def_v1` | `docs/residential_definition_v1.md` |
| Cells | AlphaEarth **10 m**, year **2024**, **64-D** `a00…a63` | `building_bbox_grid` / week4 pixel GT |
| Cell GT | coverage fractions `frac_res` / `frac_nres` / `frac_empty` | `docs/week4/pixel_eval_gt_design.md` |
| Binary analysis set | pure exclusive cells (`resi_def_v1_pure_exclusive`) | `week4_pixel_gt_balanced` |
| Balance | per-city 1:1 undersample of majority class (seed recorded) | seed 42 in US run |

Do **not** silently switch to τ=0.7 hard labels for the main transfer tables. If τ labels are needed, store them under a separate `label_version`.

The core DB stores **raw / full** lineage. Balanced analysis tables are derived later.

---

## Table split: core raw vs week4 audit sidecars

### Core raw (source of truth)

| Table | Role |
|-------|------|
| `countries` / `cities` / `city_candidates` | geography + selection |
| `buildings` | footprints + harmonized label + source ids |
| `alphaearth_cells` | cell geom, year, `a00…a63` |
| `cell_building_intersections` | recomputable cell↔building areas |
| `cell_labels` | versioned `frac_res` / `frac_nres` / `frac_empty` / `binary_label` / purity |
| `data_sources` | provenance |
| `experiment_*` | runs, splits, metrics, extract manifests |

### Additional week4-parity info (separate sidecar tables)

These hold fields needed to audit against `datasets/week4_pixel_gt*` without bloating core rows.

#### 1. `building_label_audit`

Track-A / `resi_def_v1` audit attributes per building.

| Column | Week4 analogue | Why |
|--------|----------------|-----|
| `overture_class` | `class` | Overture class before mapping |
| `overture_subtype` | `subtype` | source subtype |
| `brief_category` | `brief_category` | study coarse category |
| `study_label` | `label` | single_family / office / … |
| `y_res` | `y_res` | 1=RES, 0=NRES |
| `height_m` / `num_floors` | `height` / `num_floors` | optional morphology |
| `area_m2_approx` | `area_m2_approx` | footprint area audit |
| `source_primary` | `source_primary` | e.g. OpenStreetMap |
| `label_definition` | `label_definition` | default `resi_def_v1` |

PK: `building_id` → `buildings`.

#### 2. `cell_spatial_metrics`

Per-cell geometry helpers used by week4 explain / area math.

| Column | Week4 analogue | Why |
|--------|----------------|-----|
| `pixel_lon` / `pixel_lat` | same | metro centroids, geo-distance matrices |
| `pixel_size_m` | same | locked 10 |
| `pixel_area_m2` | same | denominator for fractions |
| `area_crs` | `crs` | local UTM used for area (e.g. `EPSG:32618`) |
| `utm_x` / `utm_y` | same | optional projected cell center |

PK: `cell_id` → `alphaearth_cells`.

#### 3. `cell_label_metrics`

Area and count audit beside each `(cell_id, label_version)` in `cell_labels`.

| Column | Week4 analogue | Why |
|--------|----------------|-----|
| `area_res_m2` / `area_nres_m2` | same | absolute intersection areas |
| `frac_res_full` / `frac_nres_full` / `frac_empty_full` | same | unclipped / pre-balance fractions |
| `n_res_buildings` / `n_nres_buildings` / `n_buildings` | same | building counts in cell |
| `area_crs` | `crs` | CRS used when areas were computed |

PK: `(cell_id, label_version)` → `cell_labels`.

`cell_labels` keeps the canonical clipped fractions + binary/purity flags used for filtering.  
`cell_label_metrics` keeps the extra numeric audit trail from the US pixel table.

### Deferred (not in this migration)

| Future table / extract | Role |
|------------------------|------|
| Balanced analysis membership | `stratum`, `balance_n_per_class`, `y` after per-city 1:1 undersample — create later from raw `cell_labels` |
| File extracts | NumPy/Parquet via `experiment_extracts` |

---

## US metros are in the international run

The same 10 week4 US metros are first-class `cities` under `country_id='USA'`:

| `city_id` | week4 `region` |
|-----------|----------------|
| `USA_ATLANTA` | `Atlanta_SE` |
| `USA_BOSTON` | `Boston_NE` |
| `USA_CHICAGO` | `Chicago_MW` |
| `USA_DENVER` | `Denver_W` |
| `USA_HOUSTON` | `Houston_S` |
| `USA_LOS_ANGELES` | `LA_W` |
| `USA_NEW_YORK` | `NYC_NE` |
| `USA_PHOENIX` | `Phoenix_SW` |
| `USA_SAN_FRANCISCO` | `SF_W` |
| `USA_SEATTLE` | `Seattle_NW` |

Mapping: `cities.metadata.week4_region_id`. Legacy downtown bboxes stay in metadata for reference only.

International AOI for these cities is **center + enclosing R**, not the week4 downtown bbox — pixel sets will **not** be identical.

---

## Audit standard (USA)

We do **not** require identical samples to week4.

We do require:

1. **Structure match** — USA uses the same tables as every other country (core + sidecars).
2. **Definition match** — same Overture Track-A allowlist, `resi_def_v1`, 10 m / 2024 / 64-D, pure-exclusive binary rule, per-city balance (when extract is built).
3. **Analysis match** — week4-style analysis on USA cities from this international extract should reproduce week4 scientific conclusions, not exact cell IDs or downtown-bbox counts.

If USA analysis diverges sharply from week4 under the same model protocol, treat that as a pipeline bug until explained by the AOI change (center+R vs downtown box).

### Reconstructing a week4-like pixel row

```sql
SELECT
  c.cell_id,
  ci.city_id,
  ci.metadata->>'week4_region_id' AS week4_region_id,
  sm.pixel_lon, sm.pixel_lat, sm.pixel_size_m, sm.pixel_area_m2, sm.area_crs,
  cl.frac_res, cl.frac_nres, cl.frac_empty,
  cm.frac_res_full, cm.frac_nres_full, cm.frac_empty_full,
  cm.area_res_m2, cm.area_nres_m2,
  cm.n_res_buildings, cm.n_nres_buildings, cm.n_buildings,
  cl.label_version,
  c.a00, c.a01  -- … a63
FROM alphaearth_cells c
JOIN cities ci ON ci.city_id = c.city_id
JOIN cell_labels cl ON cl.cell_id = c.cell_id
LEFT JOIN cell_spatial_metrics sm ON sm.cell_id = c.cell_id
LEFT JOIN cell_label_metrics cm
  ON cm.cell_id = cl.cell_id AND cm.label_version = cl.label_version
WHERE ci.country_id = 'USA';
```

---

## Intentional change: AOI geometry

US week4 used compact **downtown lon/lat bboxes**.

International study (including USA cities) uses:

```text
AOI = disk(center = city center, radius = R)
```

where **R is the smallest radius (plus optional buffer) that covers all discovered buildings for that city**.

No urban / peri-urban / rural zone rings.

### How R is determined

1. Fix `aoi_center` = city center (WGS84 point; usually `cities.centroid`).
2. Choose a generous discovery window and pull Overture buildings that intersect it.
3. Compute  
   `R_raw = max geodesic distance(aoi_center → building)`  
   over those buildings.
4. Set `R = R_raw + aoi_buffer_m` (buffer default 0).
5. Persist `aoi_center`, `aoi_radius_m`, `aoi_disk`, `aoi_method='building_enclosing_radius'`, `aoi_building_count`.

Provisional R is allowed before buildings exist; recompute before final experiments.

### Sampling inside the disk

1. Discover Track-A buildings in a generous disk (up to ~30 km); persist full footprints.
2. For AlphaEarth download, **stratified area-balance** by distance ring (e.g. 5 km bins): within each ring, thin the majority class by footprint area to ≈ minority area × headroom (~1.10). Then cap each city to the **same** footprint budget = week4 total Track-A footprint / 10 metros (≈ 24.60 km²; `--target-area-m2`). Rings are sampling strata only — not AOI zones.
3. Embed / label cells for the sampled set; fill sidecars (`building_label_audit`, `cell_spatial_metrics`, `cell_label_metrics`).
4. Later: pure-exclusive filter + per-city 1:1 **pixel-count** undersample; register extracts in `experiment_extracts`. Expand the sample later by raising headroom/rings without re-pulling Overture.

See package [`README.md`](../README.md) § Dataset creation for rationale and cost notes.

**Speedups:** see [`README.md`](../README.md) § Concurrency & batching — within-tile AEF shards, stratified area-balance, buildings/cells `COPY`, parallel label writers.

---

## Lineage

```text
country
  → city (aoi_center, R, aoi_disk; USA ↔ week4 region ids)
    → buildings
    │    └── building_label_audit   (Track-A / y_res sidecar)
    → alphaearth_cells (a00…a63)
    │    └── cell_spatial_metrics   (lon/lat/size/area_crs)
    → cell_building_intersections
    → cell_labels (versioned fractions / binary / purity)
         └── cell_label_metrics     (areas, frac_*_full, counts)
              → balanced extract (later)
                   → experiment_splits / experiment_metrics
```

## Non-goals

- Do not treat GHSL/DoU as building GT.
- Do not use downtown-only boxes as the international default AOI.
- Do not invent urban/peri-urban/rural rings unless explicitly re-requested.
- Do not require identical US pixel IDs vs week4 downtown extracts.
- Do not stuff week4 audit columns into core `buildings` / `alphaearth_cells` / `cell_labels` — use sidecars.
- Do not overwrite label versions; add new ones.
