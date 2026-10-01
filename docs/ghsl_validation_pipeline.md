# Goal

Implement a publication-quality validation pipeline for comparing our **building-level residential vs non-residential classifier** against **GHSL GHS-BUILT-C R2023A**.

**EDA first (before trusting the scoreboard):** run [`notebooks/10_ghsl_built_c_validation_eda.ipynb`](../notebooks/10_ghsl_built_c_validation_eda.ipynb) — product codes, GT audit, spatial grid rules, functional vs end-to-end benchmarks, disagreement maps. Artifacts land in `outputs/ghsl_eda/`. (Notebook 09 is GHS-SMOD Degree of Urbanization, a different GHSL product.)

The key requirement is that GHSL and our model must be evaluated:

* on exactly the same ground-truth locations
* using the same labels
* using the same spatial support
* with the same metrics
* using geographically independent test AOIs
* without modifying or resampling GHSL in a way that changes its predictions

We need two evaluations:

1. **10 m pixel-level comparison** directly comparable to GHSL.
2. **Building-level comparison** matching our actual prediction task.

---

# 1. Expected inputs

Design the validation code around these inputs.

## Ground-truth buildings

`data/validation/gt_buildings.parquet`

Required columns:

```text
building_id
aoi_id
geometry
label
confidence
source
```

Label values:

```text
RES
NRES
MIXED
UNKNOWN
```

Only `RES` and `NRES` are used in the primary binary benchmark.

`MIXED` and `UNKNOWN` must never silently be converted to RES or NRES.

Geometry CRS can vary at ingestion, but convert internally as needed.

---

## Our model predictions

`data/predictions/model_predictions.parquet`

Required:

```text
building_id
prediction
p_res
p_nres
```

where:

```text
prediction ∈ {RES, NRES}
```

Probabilities are optional for classification metrics but required for ROC-AUC / PR-AUC and calibration analysis.

---

## GHSL raster

Use:

```text
GHS-BUILT-C_FUN
2018
10 m
World Mollweide / ESRI:54009
```

Expected values:

```text
0   non-built
1   residential
2   non-residential
255 nodata
```

Never bilinearly interpolate this raster.

It is categorical.

---

# 2. Repository structure

Create:

```text
validation/
├── config.yaml
├── README.md
├── schemas.py
├── io.py
├── spatial.py
├── ghsl.py
├── rasterize.py
├── building_eval.py
├── pixel_eval.py
├── metrics.py
├── bootstrap.py
├── plots.py
├── report.py
└── run_validation.py

tests/
├── test_ghsl.py
├── test_rasterize.py
├── test_metrics.py
├── test_building_eval.py
├── test_pixel_eval.py
└── test_bootstrap.py
```

Outputs:

```text
outputs/validation/
├── summary.json
├── comparison_table.csv
├── comparison_table.md
├── per_aoi_metrics.csv
├── per_building_predictions.parquet
├── per_pixel_predictions.parquet
├── bootstrap_results.parquet
├── disagreement_buildings.gpkg
└── figures/
```

---

# 3. Configuration

All validation behavior should be controlled by `config.yaml`.

Example:

```yaml
seed: 42

labels:
  residential: RES
  non_residential: NRES
  exclude:
    - MIXED
    - UNKNOWN

ghsl:
  raster_path: data/ghsl/GHS_BUILT_C_FUN_2018.tif
  residential_value: 1
  non_residential_value: 2
  nonbuilt_value: 0
  nodata_value: 255

pixel_validation:
  enabled: true
  minimum_building_fraction: 0.5

building_validation:
  enabled: true
  aggregation: area_weighted_majority

bootstrap:
  iterations: 2000
  cluster: aoi_id
  confidence_level: 0.95

metrics:
  - accuracy
  - balanced_accuracy
  - macro_f1
  - residential_f1
  - non_residential_f1
  - residential_iou
  - non_residential_iou
  - mean_iou
  - precision
  - recall
```

Do not hard-code experimental parameters.

---

# 4. Validate the ground-truth dataset first

Before scoring either model, generate a validation-data audit.

Check:

```text
number of AOIs
number of buildings
RES count
NRES count
MIXED count
UNKNOWN count
class balance
buildings per AOI
buildings per country/region if metadata exists
geometry validity
duplicate building IDs
duplicate geometries
```

Fail loudly for:

* missing labels
* invalid geometries
* duplicate `building_id`
* model predictions missing for unexpected numbers of buildings
* CRS problems

Generate a dataset summary before computing model metrics.

---

# 5. Enforce spatial independence

Validation must operate by AOI, never by random individual-building split.

Check whether any training data intersect validation AOIs if training geometries are available.

If training footprints/AOIs are supplied:

```text
training geometries
        ↓
buffer validation AOIs by 5–10 km
        ↓
check intersection
        ↓
flag leakage
```

Create:

```python
validate_spatial_independence(...)
```

Return:

```text
validation AOI
nearest training sample distance
number of training samples within exclusion buffer
pass/fail
```

The validation pipeline should refuse to call the benchmark "held-out" if spatial leakage is detected.

---

# 6. Reproject ground truth to GHSL coordinates

For pixel-level evaluation, transform geometries to:

```text
ESRI:54009
```

Do not construct an independent 10 m grid.

Read the GHSL raster's exact:

```text
CRS
transform
width
height
pixel size
pixel origin
```

and use this grid as the canonical pixel grid.

Our predictions must be rasterized directly onto the GHSL grid.

This avoids half-pixel shifts and reprojection artifacts.

---

# 7. Pixel-level ground truth

Rasterize validated building polygons onto the exact GHSL grid.

For each GHSL pixel, calculate overlap with ground-truth building polygons.

Only include pixels with sufficient building coverage.

Default:

```text
building footprint overlap >= 50% of pixel area
```

Make threshold configurable.

If one pixel overlaps multiple labeled buildings:

1. calculate area contributed by each building
2. select label with largest footprint area
3. record overlap fractions
4. optionally flag near-ties as ambiguous

Exclude:

```text
MIXED
UNKNOWN
ambiguous overlaps
```

Output:

```text
pixel_id
aoi_id
row
col
x
y
gt_label
gt_building_fraction
```

---

# 8. Rasterize our predictions

Our model predicts one class per building.

Rasterize predictions using exactly the same procedure and exact same GHSL grid.

For each evaluation pixel:

```text
model_prediction = prediction of building contributing
largest footprint area to that pixel
```

Never interpolate predictions.

Output:

```text
pixel_id
model_prediction
model_p_res
model_p_nres
```

---

# 9. Extract GHSL predictions

For every ground-truth evaluation pixel, directly sample the corresponding GHSL raster cell.

Map:

```text
1 -> RES
2 -> NRES
0 -> NONBUILT
255 -> NODATA
```

Do not resample GHSL.

Do not move pixel centers.

Do not replace `0` with RES.

---

# 10. Run TWO GHSL pixel evaluations

This distinction is important.

## A. Functional classification benchmark

Question:

> When GHSL provides a RES/NRES classification, how accurately does it distinguish the two?

Evaluation set:

```text
GT ∈ {RES, NRES}
AND
GHSL ∈ {RES, NRES}
AND
our model prediction exists
```

Evaluate both GHSL and our model on this **identical subset**.

Report:

```text
N evaluated
GHSL coverage
Macro-F1
mIoU
Balanced Accuracy
class-specific precision/recall/F1
```

This should be the primary apples-to-apples **functional classification** comparison.

---

## B. End-to-end benchmark

Question:

> For known ground-truth buildings, how often can the entire system provide the correct functional class?

Include all valid GT RES/NRES pixels.

GHSL values:

```text
0
255
```

are treated as abstentions/missing predictions rather than silently discarded.

Report separately:

```text
coverage
selective accuracy among covered samples
overall error including abstentions
```

For example:

```text
GHSL:
coverage = 91.4%
Macro-F1 on covered pixels = ...
effective accuracy on all GT pixels = ...

Our model:
coverage = 99.8%
Macro-F1 on cover
```
