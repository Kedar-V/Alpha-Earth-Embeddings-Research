---
name: balanced_eval
description: >-
  Enforces the locked apples-to-apples eval protocol for AE vs GHSL heads
  (Sections 3–5): train freely on soft-covered Track-A data, score only on
  frozen building/tile holdout IDs with LORO and mae_mean_res_nres. Use when
  training, evaluating, or comparing Ridge/PLS/XGB/rings/MLP/GHSL methods,
  rebuilding boards, choosing train/eval splits, or when the user mentions
  balanced_eval, locked holdout, Section 3/4/5, apples-to-apples, or soft GT MAE.
---

# Balanced eval (apples-to-apples)

Canonical protocol: `docs/eval_protocol_apples_to_apples.md` (locked 2026-11-13).  
Follow this skill whenever adding or comparing a method against AE Ridge, AE PLS, AE XGB, AE rings, or GHS-BUILT-S.

## One-sentence rule

**Training may use any soft-covered Track-A data; headline metrics must be scored only on the locked Section 3/4 holdout IDs.**

Do not rebuild the eval set when you change features or the head. Change predictions; keep IDs, soft GT, and baseline columns frozen.

## Locked holdout cohorts (do not rebuild)

| Unit | Path | n | Key columns |
|---|---|---:|---|
| Buildings | `outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet` | **71,497** | `id`, `region`, `y_res` |
| 100 m tiles | `outputs/week7_section3_xgb_mae/ridge_section3_tile_ids.parquet` | **25,220** | `tile_id`, `region`, `tile_y` |

Soft GT + baseline preds (buildings):

- `outputs/week6_overture_polygon_mae/20261110_041557/artifacts/per_building_with_pls.parquet`

Tiles:

- `outputs/week6_overture_polygon_mae_100m/20261110_062739/artifacts/tiles_100m.parquet`

Assert after scoring: `len(scored) == 71497` (buildings) or `25220` (tiles).

## Train vs eval

```text
TRAIN (flexible)
  · Any soft-covered Track-A buildings / tiles
  · May be larger than the holdout (e.g. week8 n≈86,158)
  · May add features (rings, neighbors, multi-scale, …)
  · Must still respect LORO (§ Spatial CV)

EVAL (locked — report only these)
  · Join model preds onto locked cohort IDs
  · Soft GT + GHSL / Ridge / PLS columns from week6 artifacts
  · Metric: 0.5*(MAE_res + MAE_nres)
```

### Allowed

- Train on all soft-covered buildings, then score the 71,497 holdout IDs
- Train a new head on new features; compare on the same IDs
- Use LORO so each metro’s OOF preds are fair

### Not allowed

- Resampling / rebalancing the eval set for a new method
- Dropping holdout rows the new model fails to cover (report coverage; do not shrink n)
- Scoring on train-only rows and calling it Section 3/4/5
- Changing soft GT mid-comparison without a new named protocol version
- Mixing hard labels / IoU / F1 into the headline MAE table (side metrics only)

## Spatial CV (LORO)

Default for metro-level USA Track-A:

| Item | Rule |
|---|---|
| Split | Leave-one-region-out over the 10 metros |
| OOF | Predictions on the held-out metro only |
| Leakage | No cells/buildings/tiles from the holdout metro in that fold’s train set |
| Aggregation | Building = mean of `inside_polygon` cells (unless footprint-level features); tile = mean-pool of assigned cells |

Metros (canvas order):

`Boston_NE`, `NYC_NE`, `Atlanta_SE`, `Houston_S`, `Chicago_MW`, `Phoenix_SW`, `Denver_W`, `LA_W`, `SF_W`, `Seattle_NW`

If a method cannot do LORO (e.g. global GHSL), say so explicitly and still score the locked IDs.

## Headline metric

```text
mae_mean_res_nres = 0.5 * (MAE(soft_gt_frac_res, pred_res) + MAE(soft_gt_frac_nres, pred_nres))
```

- Lower is better; report pooled **and** per metro
- Clip continuous preds to `[0, 1]` before MAE unless calibrated differently (document if not)
- F1 / IoU / hard argmax = appendix only, not the Section 3–5 headline

## Baseline columns on every board

Join new preds by `id` (buildings) or `tile_id` (tiles). Keep at least:

| Column family | Role |
|---|---|
| `soft_gt_frac_res` / `soft_gt_frac_nres` | Labels |
| `ghsl_frac_*` | GHS-BUILT-S zonal mean |
| `ae_frac_*` / `ridge_frac_*` | AE Ridge (Section 3) |
| `pls_frac_*` | AE PLS |
| `xgb_frac_*` | AE XGB (Section 4) |
| new method preds | e.g. `ring_frac_*` (Section 5) |

Do not recompute GHSL/Ridge/PLS just to refresh the board unless deliberately versioning those baselines.

## Checklist (copy into run README / `summary.json`)

- [ ] Eval IDs = locked cohort parquets (assert `len(scored) == 71497` / `25220`)
- [ ] Soft GT taken from week6 artifacts (not regenerated)
- [ ] Train set documented (n, filters, whether ⊇ holdout)
- [ ] Spatial split = LORO (or justified exception)
- [ ] Metric = `0.5*(MAE_res+MAE_nres)` on soft fractions
- [ ] Coverage: % of holdout IDs with finite preds (target ≥ 99%)
- [ ] Baselines joined unchanged for the board
- [ ] Seed / hyperparams / feature dim written to `summary.json`
- [ ] Checkpoints + OOF preds + metrics CSV saved under `outputs/<run>/`

## “Train-where-you-score”

Does **not** mean train only on holdout IDs.  
Means: train a head whose targets/units match the scoreboard, on any eligible rows, then **publish metrics only on locked IDs**.

Example (Week 8 rings): train buildings ~86,158 → eval 71,497; train tiles ~28,405 → eval 25,220.

## New protocol (only when needed)

Do **not** silently overwrite Section 3. Create a new named section/version if you change geography, label definition, balance rule, spatial unit, or coverage intersection.

1. Design note under `docs/`
2. Freeze new ID lists as versioned parquets
3. Re-score **all** baselines on the new IDs before claiming a win
4. Keep Section 3/4/5 boards intact

## Quick pointers

| Need | Location |
|---|---|
| Building holdout IDs | `outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet` |
| Tile holdout IDs | `outputs/week7_section3_xgb_mae/ridge_section3_tile_ids.parquet` |
| Full protocol | `docs/eval_protocol_apples_to_apples.md` |
| Section 3 design | `docs/superpowers/specs/2026-11-12-week7-xgb-section3-baseline-design.md` |
| Week 7 eval script | `scripts/eval_week7_section3_xgb_mae.py` |
| Week 8 rings (train wide / eval locked) | `scripts/run_week8_ae_rings_loro.py` |
| Canvas Sections 3–5 | `notebooks/ghsl_built_s_vs_ae/AlphaEarth Weeks 1–5 Canvas.html` |
