# Week 3 — Pixel k×k neighborhood + Conv encoder

**Status:** trained (k=3, 8× TITAN RTX)  
**Notebook:** [`notebooks/week3/06_pixel_neighborhood_kxk.ipynb`](../../notebooks/week3/06_pixel_neighborhood_kxk.ipynb)  
**Train CLI:** [`scripts/train_week3_pixel_neighborhood_loro.py`](../../scripts/train_week3_pixel_neighborhood_loro.py)  
**Builder:** [`scripts/build_week3_pixel_neighborhoods.py`](../../scripts/build_week3_pixel_neighborhoods.py)  
**Data:** `datasets/week3_pixel_neighborhood_k{k}/`  
**Outputs:** `outputs/week3_pixel_neighborhood_k{k}/`

## Results (k=3 vs notebook 05 single-pixel)

| | OOF macro-F1 | LORO mean |
|---|---|---|
| Nb06 CNN clf | 0.353 | 0.345 |
| Nb06 CNN reg→decode | 0.334 | 0.324 |
| Nb05 single-pixel clf | **0.457** | 0.446 |
| Nb05 single-pixel reg | **0.521** | 0.503 |

Neighborhood CNN underperforms the single-pixel linear heads on this first recipe (unstable folds; empty over-predicted).

## Embeddings / UMAP

OOF pooled 128-D vectors from `NbCNN.encode` (masked GAP), from LORO fold checkpoints:

```bash
.venv/bin/python -u scripts/extract_week3_neighborhood_umap.py --k 3
```

Writes `oof_{clf,reg}_embeddings.npy`, UMAP coords, and
`figures/umap_{clf,reg}_by_{label,region}.png` + PCA overviews.

## Input

Centered odd window `k` (default **3**) on the 10 m AE grid:

- `X_nb`: `(N, 64, k, k)`
- `mask`: `(N, 1, k, k)` validity (OOB / nodata = 0)
- Train pack: `(N, 65, k, k)` = AE + mask

Labels / soft fracs remain **center-pixel** from `week3_pixel_gt_v2` (drop mixed).

## Model

`NbCNN`: Conv(65→128)→BN→ReLU → Conv(128→128)→BN→ReLU → masked GAP → Linear→3.

- Clf: ROS on hard labels + CE  
- Reg: ROS carrying soft `frac_*_full` + softmax MSE  
- Decode: `building_share_max=0.15`

## Rebuild

```bash
.venv/bin/python scripts/build_week3_pixel_neighborhoods.py --k 3 --workers "$(nproc)"
.venv/bin/python scripts/build_week3_pixel_neighborhood_notebook.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u scripts/train_week3_pixel_neighborhood_loro.py --k 3
```
