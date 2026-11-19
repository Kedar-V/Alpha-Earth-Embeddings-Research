# Week 6 — Fair AE vs GHS-BUILT-S on Overture polygons

**This week’s Section 3 board is the locked AE vs GHSL baseline** for later head swaps (Week 7 XGBoost, etc.).

Both predictors are summarized **independently into each Overture footprint**. AE stays in per-metro UTM; GHSL stays in Mollweide. Neither is sampled onto the other's lattice.

| Piece | Choice |
|---|---|
| Reference | Overture Track-A footprints (`resi_def_v1`) |
| Building panel | Soft cell GT inside polygon; upstream buildings 1:1 on `y_res`; score **n=71,497** soft∩AE∩GHSL∩PLS |
| 100 m panel | Soft-argmax 1:1 cells → mean-pool (`min≥10`) → soft-argmax 1:1 tiles · **n=25,220** |
| AE | Mean week4 Ridge LORO OOF on `inside_polygon` cells (+ PLS OOF) |
| GHSL | GHS-BUILT-S soft fractions under polygon mask |
| Metric | `mae_mean_res_nres` on shared coverage |

## Run

```bash
bash scripts/submit_week6_overture_polygon_mae.sh
bash scripts/submit_week6_overture_polygon_mae_100m.sh
```

Canonical artifacts:

- Buildings: `outputs/week6_overture_polygon_mae/20261110_041557/`
- 100 m: `outputs/week6_overture_polygon_mae_100m/20261110_062739/`

## Notebooks

- [`00_briefing.ipynb`](00_briefing.ipynb) — protocol
- [`01_ae_vs_ghsl_overture_polygons.ipynb`](01_ae_vs_ghsl_overture_polygons.ipynb) — results

Week 7 reuses these **exact** building/tile IDs: [`../week7/`](../week7/).
