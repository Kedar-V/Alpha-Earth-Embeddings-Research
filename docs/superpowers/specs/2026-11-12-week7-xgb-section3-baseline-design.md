# Week 7 — XGBoost head swap on Section 3 baseline

**Date:** 2026-11-12  
**Status:** draft for review  
**Approved approach:** #1 drop-in OOF swap (user 2026-11-12)

## Goal

Replace the **AE Ridge** soft-fraction head with **XGBoost multi-output regression**, then re-score the locked **Section 3 / Week 6** Overture-polygon fraction MAE board (buildings + 100 m tiles). Keep AE PLS and GHS-BUILT-S unchanged so the canvas comparison stays apples-to-apples.

Document Section 3 as the project’s **AE vs GHSL baseline** in READMEs.

## Non-goals

- Hyperparameter sweep / Optuna (fixed defaults for v1)
- Retraining PLS or GHSL
- Changing soft-argmax / 1:1 balance / min-cells rules
- Places / SearchDestinations / national GT expansion
- Binary F1 scoreboard as the Week 7 headline (optional side metric only)

## Locked Section 3 baseline (reference)

From the Week 6 canvas / eval scripts:

| Piece | Choice |
|---|---|
| Spatial support | Overture Track-A footprints (`resi_def_v1`) |
| Building GT | Soft cell fractions inside polygon; buildings balanced **1:1 on `y_res`** |
| Building cohort | soft ∩ Ridge ∩ PLS ∩ BUILT-S covered · **n ≈ 71,497** |
| 100 m path | Unbalanced soft-covered buildings → soft-argmax 1:1 **cells** per metro → mean-pool UTM 100 m (`min ≥ 10`) → soft-argmax 1:1 **tiles** · **n ≈ 25,220** |
| AE Ridge | Mean week4 Ridge LORO OOF on `inside_polygon` cells |
| AE PLS | Mean week4 PLS LORO OOF (same aggregation) |
| GHS-BUILT-S | Building-zonal mean (polygon / Mollweide path) |
| Metric | `mae_mean_res_nres = 0.5*(MAE_res + MAE_nres)` (lower better) |

Canonical runs today:

- Building soft+balance: `outputs/week6_overture_polygon_mae/20261110_041557/` (and successors)
- 100 m soft-argmax: `outputs/week6_overture_polygon_mae_100m/` (min≥10 cell+tile protocol)

## Design

### 1. Train XGBoost LORO OOF (Ridge drop-in)

Mirror `scripts/train_week4_pixel_balanced_loro.py`:

| Item | Value |
|---|---|
| Panel | `datasets/week4_pixel_gt_balanced/pixels.parquet` |
| Features | `A00`–`A63` |
| Targets | `(frac_res_full, frac_nres_full, frac_empty_full)` |
| Splits | Leave-one-region-out (10 metros) |
| Model | `XGBRegressor` multi-output (3 targets), **GPU hist** when CUDA available |
| Defaults (v1) | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, `tree_method=hist`, `device=cuda:{fold_gpu}` |
| Predictions | Clip to `[0, 1]`; also write full-GT projection like Ridge (`oof_predictions_full_gt.parquet`) for polygon join |
| Concurrency | One process per fold, mapped across all visible GPUs (`cuda:i % n_gpus`) |
| Outputs | `outputs/week7_xgb_loro/<timestamp>/` — fold checkpoints, OOF parquet, fold MAE/F1 side metrics, `progress.json`, `summary.json` |

Dependency: install `xgboost` into `.venv` (not present today).

### 2. Section 3 rescore with `ae_xgb`

Extend or wrap the Week 6 MAE pipelines so the AE source can be:

- `ae_ridge` → existing `week4_pixel_balanced_train/oof_predictions_full_gt.parquet` (kept as reference bar)
- `ae_xgb` → new Week 7 OOF
- `ae_pls` / `ghsl_built_s` unchanged

Headline figure / table models: **AE XGB · AE Ridge · AE PLS · GHS-BUILT-S**.

**Eval sets are frozen to the Ridge Section 3 board (not rebuilt):**

1. Buildings: exact IDs with `shared_soft ∩ pls_covered` on the week6 `per_building_with_pls` table → **n=71,497**
2. 100 m: exact `tile_id`s from `tiles_100m.parquet` → **n=25,220**

Only XGB predictions are recomputed; Ridge/PLS/GHSL/soft GT columns stay from week6 artifacts.

Outputs: `outputs/week7_section3_xgb_mae/<timestamp>/` with metrics CSV, per-region bars, PNGs, manifest.

### 3. Notebooks

```text
notebooks/week7/
  README.md
  00_briefing.ipynb      # protocol: Section 3 baseline + XGB swap
  01_xgb_vs_section3.ipynb  # results from outputs/ (embedded figures)
```

### 4. README baseline lock

Update to state explicitly:

> **Baseline (locked):** Week 6 Section 3 — fraction MAE in Overture polygon space (AE Ridge / AE PLS / GHS-BUILT-S; building n≈71k; 100 m tiles n≈25k).  
> **Week 7:** replace AE Ridge with XGBoost under the same protocol.

Touch:

- `README.md` (project root — baseline callout)
- `notebooks/week6/README.md` (mark Section 3 as baseline; point to Week 7)
- `notebooks/week7/README.md` (new)
- Optionally one line in `notebooks/ghsl_built_s_vs_ae/README.md` if it still describes an older baseline

## Success criteria

1. XGB LORO OOF exists and joins ≥99% of the Ridge full-GT pixel keys used by Section 3.
2. Building + 100 m MAE tables include `ae_xgb` beside Ridge / PLS / BUILT-S on the same cohorts.
3. `notebooks/week7/` shows results without retrain.
4. Root + week6 READMEs name Section 3 as the locked baseline.

## Implementation touchpoints

| File | Action |
|---|---|
| `scripts/train_week7_xgb_loro.py` | create |
| `scripts/submit_week7_xgb_loro.sh` | create |
| `scripts/eval_week7_section3_xgb_mae.py` (+ submit) | create (or thin wrapper over week6 evals) |
| `notebooks/week7/*` | create |
| `README.md`, `notebooks/week6/README.md` | update baseline language |
| `.venv` | `pip install xgboost` |

## Spec self-review

- No placeholders for paths or metrics definitions.
- Scope matches approved approach #1 (drop-in OOF; no sweep).
- Section 3 cohort rules copied from existing week6 scripts / canvas (min≥10 for tiles).
- Ambiguity resolved: Ridge retained as reference bar; XGB is the new AE head under test.
