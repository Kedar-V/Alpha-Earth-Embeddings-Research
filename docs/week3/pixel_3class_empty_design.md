# Week 3 — Pixel 3-class RES / NRES / empty

**Status:** trained (8× TITAN RTX) · decode τ=0.15  
**Notebook:** [`notebooks/week3/05_pixel_res_nres_empty.ipynb`](../../notebooks/week3/05_pixel_res_nres_empty.ipynb)  
**Train CLI:** [`scripts/train_week3_pixel_3class_loro.py`](../../scripts/train_week3_pixel_3class_loro.py)  
**Data:** [`datasets/week3_pixel_gt_v2/`](../../datasets/week3_pixel_gt_v2/)  
**Outputs:** [`outputs/week3_pixel_3class/`](../../outputs/week3_pixel_3class/)

## Labels

| Label | Rule |
|---|---|
| `residential` | Track-A RES overlap only |
| `non_residential` | Track-A NRES overlap only |
| `mixed` | both (dropped from train) |
| `empty` | no footprint overlap |

Empties are **AOI-sampled** (`n_empty ≈ n_residential` per metro).  
`frac_empty_full = clip(1 - frac_res_full - frac_nres_full, 0, 1)`.

## Models

1. **Classifier:** hard labels + SMOTE + CE (LORO).  
2. **Regression:** soft GT `(frac_res_full, frac_nres_full, frac_empty_full)` + RandomOverSampler (carry soft Y) + softmax MSE + decode.

### Decode head

`BUILDING_SHARE_MAX = 0.15` (OOF-tuned; maximizes macro-F1 / empty F1):

| Condition | Output |
|---|---|
| `frac_res + frac_nres < 0.15` | `empty` |
| else, `frac_res ≥ frac_nres` | `residential` |
| else | `non_residential` |

## Results (τ=0.15)

| Head | LORO mean macro-F1 | OOF macro-F1 | Empty recall |
|---|---|---|---|
| Classifier | 0.446 ± 0.054 | 0.457 | 0.621 |
| Regression→decode | **0.503 ± 0.061** | **0.521** | 0.447 |

Regression wins **9/10** metros.

## Rebuild

```bash
.venv/bin/python scripts/build_week3_pixel_3class_notebook.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u scripts/train_week3_pixel_3class_loro.py
```
