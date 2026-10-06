# Geographic subspace analysis

Within-US study of whether the residential vs non-residential direction in frozen 64-D AlphaEarth is shared across ten downtown metros, and whether that sharing tracks classifier transfer.

## What is reused

The pixel table is `datasets/week4_pixel_gt_balanced/pixels.parquet`.

- Features: `A00`–`A63`
- Label: `y` (1 residential, 0 non-residential), from pure cells under `resi_def_v1`
- Stratum rule: residential if `frac_res_full > 0` and `frac_nres_full == 0`, and the swap for non-residential. Mixed cells are already excluded
- Balance: within each metro, the majority class was undersampled to the minority, seed 42
- Region column: `region`
- n = 1,751,814, 875,907 per class, 10 metros

The arrays already materialized for the week-4 PLS run are read and not rewritten:

- `outputs/week4_pls_loro/_cache/X.npy` float32 `(1751814, 64)`
- `outputs/week4_pls_loro/_cache/y_bin.npy`
- `outputs/week4_pls_loro/_cache/regions.npy`

Leave-one-metro-out matches `scripts/train_week4_pls_loro.py`: the held-out metro is absent from `StandardScaler`, PCA, PLS, and logistic regression. The decision threshold stays at 0.5. It is not tuned.

The published 10 m reference is LORO macro-F1 0.779 for 64-D logistic regression (`outputs/week4_pls_10m_vs_100m/sweep_summary.csv`). This suite refits that baseline so every later F1 uses the same predictions.

v1–v4 ViT embeddings are 100 m tiles (`outputs/week4_vit_100m_3d*`). They are a separate analysis. Their 3D coordinates are not pooled across folds.

## Layout

| Path | Role |
|---|---|
| `PLAN.md` | Stages, CPU/GPU split, dependencies |
| `scripts/` | Experiment entry points |
| `outputs/<experiment>/<run_id>/` | Immutable run: log, config, metrics, fitted parameters |
| `results/` | Tables copied from completed runs |
| `figures/` | Plots drawn from `results/` |
| `RESULTS.md` | Written after the runs finish |
| `results/leakage_audit.md` | Assertions from the runs |

## How a run is started

Jobs are detached (`setsid` / `nohup`) and write `train.log`, `train.pid`, `metrics.csv`, and `summary.json` under the run directory. CPU experiments do not take a GPU. See `PLAN.md` for the thread split.
