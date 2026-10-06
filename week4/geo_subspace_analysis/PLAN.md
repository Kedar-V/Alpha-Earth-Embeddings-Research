# Geographic subspace analysis — execution plan

Question: are the residential vs non-residential subspaces of frozen 64-D AlphaEarth aligned across the ten US metros, and does that alignment track cross-metro transfer?

This does not retrain the week-4 ViT. Stages 1–4 use logistic regression, PCA, and PLS on the same balanced pure 10 m cells as `outputs/week4_pls_loro`. Stage 5 (a learned projection) starts only after those results are in.

## Reused data and code

| Item | Path |
|---|---|
| Balanced pure cells, `resi_def_v1`, seed 42 | `datasets/week4_pixel_gt_balanced/pixels.parquet` |
| Cached `X` (n=1,751,814, A00–A63), `y_bin`, `regions` | `outputs/week4_pls_loro/_cache/` (read only) |
| LORO definition | train = all metros except the held-out metro; scaler, PCA, and PLS fit on train only |
| Existing 64-D LORO reference | `outputs/week4_pls_10m_vs_100m/sweep_summary.csv` (macro-F1 0.779) |
| v1–v4 3D tile embeddings | `outputs/week4_vit_100m_3d*/` (100 m tiles, separate from the 10 m matrix) |
| Fast PLS idea | Gram NIPALS in `scripts/train_week4_pls_loro.py` (reimplemented here on CPU) |

New files live under `week4/geo_subspace_analysis/`. Nothing under `outputs/week4_*` is rewritten.

## Stages

| Stage | Experiment | Depends on | CPU / GPU | Concurrent with |
|---|---|---|---|---|
| 1 | 64-D LORO logistic regression; PCA and PLS at k = 1, 2, 3, 4, 5, 8, 9, 12, 15, 16, 24, 32, 48, 64 | cache | 10 fold processes, 4 BLAS threads each (40 cores). No GPU | 2 and 3 |
| 2 | Within-metro PLS bases (k = 1, 2, 3, 5, 9), principal angles, class-mean directions | cache | 10 processes, 2 threads (20 cores). No GPU | 1 and 3 |
| 3 | Full source→target transfer matrix: 64-D, PCA, PLS at k = 2 and 9. Diagonal is an 80/20 stratified split | cache | 10 processes, 2 threads (20 cores). No GPU | 1 and 2 |
| 4 | Spearman/Pearson, permutation test, bootstrap, geographic distance, AEF mean/MMD/sliced-Wasserstein, metro-ID probe | stages 1–3 | one process, trivial after the matrices exist | — |
| 4b | v4 checkpoint: class directions of every metro inside each fold’s frozen 3D space (inference only) | existing checkpoints | 1 GPU | after 1–3, or alongside 4 |
| 5 | Linear map 64→d, d ∈ {2, 4, 8, 16}; class-only, SupCon, cross-region SupCon, class + direction alignment. Lambda chosen on source metros only | stages 1–4 understood | up to 8 GPUs, one outer metro per GPU | not before stage 4 |

80 hardware threads. Stage 1+2+3 together request 40+20+20 = 80 threads. No nested process pools. Each worker sets `n_jobs=1` on scikit-learn.

## Run directories

```text
week4/geo_subspace_analysis/outputs/<experiment>/<run_id>/
    train.log
    config.json
    summary.json
    metrics.csv
    folds or matrices
    DONE or FAILED
```

A run directory that already has files is refused. Failed runs stay put; a fix gets a new `run_id`.

Stable tables for the report are copied into `week4/geo_subspace_analysis/results/` only after the run’s `DONE` file exists. Figures are redrawn from those tables by `scripts/make_figures.py`.

## What “done” means for each stage

- Stage 1: 10 metros × (1 baseline + 14 PCA + 14 PLS) rows, no NaN F1, predictions saved, leakage assert passed.
- Stage 2: `Q_r` for every metro and k, 10×10 similarity and angle matrices, class-direction cosine matrix.
- Stage 3: transfer matrices including a diagonal that does not score the training rows.
- Stage 4: correlation table with permutation p-values and the pair scatter.
- Stage 5: only if stages 1–4 have been read. Held-out metros do not pick lambda, width, or epoch.
