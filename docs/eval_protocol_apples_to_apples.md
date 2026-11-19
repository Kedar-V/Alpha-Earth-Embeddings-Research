# Eval protocol — apples-to-apples comparison

**Status:** locked for Sections 3–5 and future AE vs GHSL heads  
**Last updated:** 2026-11-13

This doc defines how we split train vs eval so new methods can be compared fairly against AE Ridge, AE PLS, AE XGB, AE rings, and GHS-BUILT-S.

---

## 1. One-sentence rule

**Training may use any soft-covered Track-A data; headline metrics must be scored only on the locked Section 3/4 holdout IDs.**

Do not rebuild the eval set when you change features or the head. Change predictions; keep IDs, soft GT, and baseline columns frozen.

---

## 2. Locked holdout cohorts (do not rebuild)

| Unit | Path | n | Key columns |
|---|---|---:|---|
| Buildings | `outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet` | **71,497** | `id`, `region`, `y_res` |
| 100 m tiles | `outputs/week7_section3_xgb_mae/ridge_section3_tile_ids.parquet` | **25,220** | `tile_id`, `region`, `tile_y` |

These IDs are the **exact** rows used on the Week 6 Section 3 Ridge board and reused by Section 4 (XGB) and Section 5 (rings).

### Building cohort construction (historical — frozen)

1. Overture Track-A footprints (`resi_def_v1`)
2. Soft cell fractions inside the polygon (`soft_gt_*`)
3. Shared coverage: soft ∩ AE ∩ GHSL ∩ PLS
4. Balance **1:1 on `y_res` per metro**
5. Freeze resulting building `id`s → cohort parquet above

Canonical artifact for soft GT + baseline preds:

- `outputs/week6_overture_polygon_mae/20261110_041557/artifacts/per_building_with_pls.parquet`

### Tile cohort construction (historical — frozen)

1. Start from unbalanced soft-covered buildings
2. Soft-argmax **1:1 cells** per metro
3. Mean-pool into UTM **100 m** tiles (`min ≥ 10` cells)
4. Soft-argmax **1:1 tiles** per metro
5. Freeze resulting `tile_id`s → cohort parquet above

Canonical artifact:

- `outputs/week6_overture_polygon_mae_100m/20261110_062739/artifacts/tiles_100m.parquet`

---

## 3. Training vs evaluation (explicit split)

```text
┌─────────────────────────────────────────────────────────────┐
│ TRAIN (flexible)                                            │
│  · Any soft-covered Track-A buildings / tiles               │
│  · May be larger than the holdout (e.g. week8 n≈86,158)     │
│  · May add features (rings, neighbors, multi-scale, …)      │
│  · Must still respect spatial CV (see §4)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ EVAL (locked — report only these)                           │
│  · Join model preds onto locked cohort IDs                  │
│  · Soft GT + GHSL / Ridge / PLS columns from week6 artifacts│
│  · Same metric: 0.5*(MAE_res + MAE_nres)                    │
└─────────────────────────────────────────────────────────────┘
```

### Allowed

- Train on all soft-covered buildings, then score the 71,497 holdout IDs
- Train a new head (Ridge, XGB, MLP, …) on new features; compare on the same IDs
- Use leave-one-region-out (LORO) so each metro’s OOF preds are fair

### Not allowed (breaks comparison)

- Resampling / rebalancing the eval set for a new method
- Dropping buildings/tiles that the new model fails to cover (report coverage; do not silently shrink n)
- Scoring on train-only rows and calling it Section 3/4/5
- Changing soft GT definition mid-comparison without a new named protocol version
- Mixing hard labels / IoU / F1 into the headline MAE table without labeling them as side metrics

---

## 4. Spatial CV (LORO)

Default head training for metro-level USA Track-A:

| Item | Rule |
|---|---|
| Split | **Leave-one-region-out** over the 10 metros |
| OOF | Predictions on the held-out metro only |
| Leakage | No cells/buildings/tiles from the holdout metro in that fold’s train set |
| Aggregation | Building = mean of `inside_polygon` cells (unless the method is explicitly footprint-level features); tile = mean-pool of assigned cells |

Metros (canvas order):

`Boston_NE`, `NYC_NE`, `Atlanta_SE`, `Houston_S`, `Chicago_MW`, `Phoenix_SW`, `Denver_W`, `LA_W`, `SF_W`, `Seattle_NW`

If a method cannot do LORO (e.g. a global GHSL product), say so explicitly and still score the locked IDs.

---

## 5. Headline metric

```text
mae_mean_res_nres = 0.5 * (MAE(soft_gt_frac_res, pred_res) + MAE(soft_gt_frac_nres, pred_nres))
```

- Lower is better
- Pooled over the locked cohort, and also reported **per metro**
- Clip continuous preds to `[0, 1]` before MAE unless the method is calibrated differently (document if not)

Side metrics (F1, IoU, hard argmax accuracy) are fine in appendices; they are **not** the Section 3–5 headline.

---

## 6. Baseline columns to keep beside any new method

On the locked cohort, always show at least:

| Column family | Role |
|---|---|
| `soft_gt_frac_res` / `soft_gt_frac_nres` | Labels |
| `soft_gt_frac_empty` (optional side target) | Mean week4 `frac_empty_full` over inside-polygon AE cells; see §10 |
| `ghsl_frac_*` | GHS-BUILT-S zonal mean |
| `ae_frac_*` / `ridge_frac_*` | AE Ridge (Section 3) |
| `pls_frac_*` | AE PLS |
| `xgb_frac_*` | AE XGB (Section 4) |
| new method preds | e.g. `ring_frac_*` (Section 5), `s2_frac_*` (Section 6), `s2e_frac_*` (Section 8) |

Headline metric stays `0.5*(MAE_res+MAE_nres)`. Empty MAE is a side metric only unless a new protocol version promotes it.

Join new preds by `id` (buildings) or `tile_id` (tiles). Do not recompute GHSL/Ridge/PLS just to refresh the board unless you are deliberately versioning those baselines.

---

## 7. Checklist for a new experiment

Copy this into the run README / `summary.json`:

- [ ] Eval IDs = locked cohort parquets (assert `len(scored) == 71497` / `25220`)
- [ ] Soft GT taken from week6 artifacts (not regenerated)
- [ ] Train set documented (n, filters, whether ⊇ holdout)
- [ ] Spatial split = LORO (or justified exception)
- [ ] Metric = `0.5*(MAE_res+MAE_nres)` on soft fractions
- [ ] Coverage: % of holdout IDs with finite preds (target ≥ 99%)
- [ ] Baselines joined unchanged for the board
- [ ] Seed / hyperparams / feature dim written to `summary.json`
- [ ] Checkpoints + OOF preds + metrics CSV saved under `outputs/<run>/`

---

## 8. What “train-where-you-score” means here

It does **not** mean “train only on the holdout IDs.”

It means: train a head whose **targets and spatial units match the scoreboard** (building soft fractions and/or 100 m tile soft fractions), using whatever eligible training rows you want, then **publish metrics only on the locked IDs**.

Example (Week 8 rings):

| Stage | n | Notes |
|---|---:|---|
| Train buildings | ~86,158 | all `soft_gt_covered` |
| Eval buildings | 71,497 | locked Section 3/4 IDs |
| Train tiles | ~28,405 | available soft tile rows |
| Eval tiles | 25,220 | locked Section 3/4 IDs |

---

## 9. When you *may* define a new protocol

Create a **new named section / version** (do not silently overwrite Section 3) if you change any of:

- Geography (add/remove metros, leave USA Track-A)
- Label definition (soft fractions, hard classes, different GT source)
- Balance rule (`y_res` 1:1, soft-argmax, none)
- Spatial unit (building / 100 m / something else)
- Coverage intersection (e.g. drop PLS requirement)

New protocol checklist:

1. Write a short design note under `docs/`
2. Freeze new ID lists as versioned parquets
3. Re-score **all** baselines on the new IDs before claiming a win
4. Keep Section 3/4/5 boards intact for historical comparison

---

## 10. Quick pointers

| Need | Location |
|---|---|
| Building holdout IDs | `outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet` |
| Tile holdout IDs | `outputs/week7_section3_xgb_mae/ridge_section3_tile_ids.parquet` |
| Soft GT + empty (buildings) | `outputs/_shared/soft_gt_frac_empty_by_building.parquet` (also week11 run artifacts) |
| Empty cell definition | `docs/week4/pixel_eval_gt_design.md` — `frac_empty = clip(1−frac_res−frac_nres, 0, 1)` |
| Section 3 design | `docs/superpowers/specs/2026-11-12-week7-xgb-section3-baseline-design.md` |
| Week 7 eval script | `scripts/eval_week7_section3_xgb_mae.py` |
| Week 8 rings (train wide / eval locked) | `scripts/run_week8_ae_rings_loro.py` |
| Week 9 AE+S2 | `scripts/run_week9_ae_s2_nres_loro.py` |
| Week 11 AE+S2+empty head | `scripts/run_week11_ae_s2_empty_head_loro.py` |
| Canvas Sections 3+ | `notebooks/ghsl_built_s_vs_ae/AlphaEarth Weeks 1–5 Canvas.html` |
