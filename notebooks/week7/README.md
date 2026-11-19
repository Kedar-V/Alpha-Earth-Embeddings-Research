# Week 7 — XGBoost vs Section 3 baseline

**Locked baseline:** Week 6 Section 3 fraction MAE in Overture polygon space  
(AE Ridge · AE PLS · GHS-BUILT-S; buildings n=71,497; 100 m tiles n=25,220).

Week 7 replaces the **AE Ridge** soft-fraction head with **XGBoost**, scored on the **exact same** building IDs and tile IDs as that Ridge board.

| Piece | Choice |
|---|---|
| Train | LORO XGB multi-output on week4 balanced pure cells (same as Ridge) |
| Eval buildings | `ridge_section3_building_ids.parquet` (71,497) |
| Eval 100 m | `ridge_section3_tile_ids.parquet` (25,220) |
| Metric | `mae_mean_res_nres` soft GT |

## Run

```bash
bash scripts/submit_week7_xgb_loro.sh          # train OOF (multi-GPU)
bash scripts/submit_week7_section3_xgb_mae.sh  # after train completes
```

## Notebooks

- [`00_briefing.ipynb`](00_briefing.ipynb) — protocol
- [`01_xgb_vs_section3.ipynb`](01_xgb_vs_section3.ipynb) — results
