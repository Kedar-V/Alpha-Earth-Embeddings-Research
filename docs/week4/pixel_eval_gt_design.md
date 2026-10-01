# Week 4 — Pixel fraction GT (polygon → 10 m)

**Status:** locked for dataset build  
**Data:** [`datasets/week4_pixel_gt/`](../../datasets/week4_pixel_gt/)  
**Builder:** [`scripts/build_week4_pixel_gt.py`](../../scripts/build_week4_pixel_gt.py)  
**Label definition:** `resi_def_v1_pixel_frac_v1`

## Goal

Rasterize Track-A Overture building polygons onto unique **10 m AlphaEarth** cells and store **coverage fractions only** for model evaluation.

## Sources

| Path | Role |
|---|---|
| `datasets/building_bbox_grid/buildings.parquet` | Track-A footprints + `y_res` (unstratified natural mix) |
| `datasets/building_bbox_grid/building_cells.parquet` | AE 10 m cells + `A00`–`A63` |

**Sampling protocol:** polygon GT is **not** stratified by class or `y_res`. Take all Track-A typed Overture buildings in each AOI (optional random `max_per_region` cap still unstratified). RES dominates by count (~90% in these downtown boxes), matching global built-up trends. Class balance for training happens later by **undersampling RES pixels**, not by oversampling rare NRES polygons.

Unlabeled footprints are excluded (`y_res` must be 0 or 1). Building class → `y_res` follows [`resi_def_v1`](../residential_definition_v1.md).

## Balanced training subset

[`scripts/build_week4_pixel_gt_balanced.py`](../../scripts/build_week4_pixel_gt_balanced.py) keeps pure cells only and, per metro, undersamples the majority class (typically RES) to `min(n_res, n_nres)`.

## Fraction definitions

For each unique AE cell `c`:

```text
frac_res(c)   = area( union(RES footprints)  ∩ c ) / area(c)
frac_nres(c)  = area( union(NRES footprints) ∩ c ) / area(c)
frac_empty(c) = clip(1 − frac_res − frac_nres, 0, 1)
```

- RES = footprints with `y_res == 1`; NRES = `y_res == 0`.
- Same-class overlaps use **union** (no double-count).
- RES∩NRES overlap can make `frac_res + frac_nres > 1` before empty is clipped.
- No hard RES/NRES/mixed label and no τ threshold in this dataset.

Inclusion rule: cell must intersect ≥1 Track-A footprint (so pure empty cells outside footprints are not sampled here). `frac_empty` is still written for every included cell.

## Schema (core)

`pixel_id`, `region`, `pixel_lon`, `pixel_lat`, `pixel_wkt`, `frac_res`, `frac_nres`, `frac_empty`, `n_res_buildings`, `n_nres_buildings`, `building_ids`, `A00`…`A63`, `label_definition`.

Rounded `frac_*` at 2 decimals for display; `*_full` keep full precision.

## Scoring contract (downstream)

Models predict `{frac_res, frac_nres, frac_empty}` (or a subset) keyed by `pixel_id`. Primary metrics: MAE / RMSE per fraction under leave-one-region-out.

Scorer: [`scripts/eval_week4_pixel.py`](../../scripts/eval_week4_pixel.py)  
Week 3 rescore: `outputs/week4_pixel_eval/week3_rescore_v1/`  
Notebook: [`notebooks/week4/01_pixel_model_eval.ipynb`](../../notebooks/week4/01_pixel_model_eval.ipynb)
