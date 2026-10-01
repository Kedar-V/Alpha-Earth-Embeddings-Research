# Week 3 — Pixel fraction regression (Stage 3)

**Status:** implemented 2026-09-24  
**Notebook:** [`notebooks/week3/03_pixel_fraction_regression.ipynb`](../../notebooks/week3/03_pixel_fraction_regression.ipynb)  
**Data:** [`datasets/week3_pixel_gt/pixels.parquet`](../../datasets/week3_pixel_gt/pixels.parquet)  
**Outputs:** `outputs/week3_fraction_regression/`  
**Features:** AlphaEarth `A00`–`A63` **only**

## Goal

1. **Regress** per-pixel `frac_res` and `frac_nres` (Stage 1 coverage fractions).
2. **Majority head** on predictions → `residential` / `non_residential` / `mixed`.

Train on **pure** Stage-1 pixels only (mixed dropped — fair vs Stage-2 LR). Each LORO train fold is **SMOTE**-balanced (RES≈NRES). Training targets are hard 0/1 fractions from the binary label; reported preds rounded to 2 decimals.

## Model

- **X:** 64-D AE embedding (**CPU only**)
- **Net:** `SMOTE` + `StandardScaler` + `MultiOutputRegressor(Ridge)`
- **Loss / fit:** least-squares Ridge on hard `(y_res, 1−y_res)`; preds clipped to `[0, 1]`
- **Protocol:** leave-one-region-out; CPU serial folds (`OPENBLAS_NUM_THREADS=1`)

## Majority head

```text
pred_res > pred_nres  → residential
pred_nres > pred_res  → non_residential
else                  → mixed
```

## Pipeline

```mermaid
flowchart LR
  AE["A00-A63"] --> MLP["Ridge regressor CPU"]
  MLP --> Fracs["pred frac_res / frac_nres"]
  Fracs --> Maj["majority head"]
  Maj --> Cls["RES / NRES / mixed"]
```

## Metrics

- Regression: MAE / RMSE per target (OOF + per-region)
- Classification: majority vs Stage-1 `label` (3-class + pure binary)
- Compare pure-binary macro-F1 to Stage-2 pixel LR (`outputs/week3_pixel_ae/`)

## Artifacts

| Path | Contents |
|---|---|
| `loro_regression_metrics.csv` | Per-region MAE/RMSE |
| `loro_majority_metrics.csv` | Per-region class F1 |
| `oof_predictions.parquet` | True/pred fracs + majority label |
| `summary.json` | Means, GPU count, hyperparams |
| `figures/` | Scatter, confusion, vs Stage-2 |
