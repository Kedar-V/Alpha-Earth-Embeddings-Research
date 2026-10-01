#!/usr/bin/env python3
"""Rewrite notebooks/week4/00_briefing.ipynb from week4 artifacts (plain technical prose)."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "week4" / "00_briefing.ipynb"


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    gt = jload(ROOT / "datasets" / "week4_pixel_gt" / "manifest.json")
    bal = jload(ROOT / "datasets" / "week4_pixel_gt_balanced" / "manifest.json")
    train = jload(ROOT / "outputs" / "week4_pixel_balanced_train" / "summary.json")
    mae = pd.read_csv(ROOT / "outputs" / "week4_100m_fraction_eval" / "mae_summary.csv")
    f1_lr = pd.read_csv(ROOT / "outputs" / "week4_100m_fraction_eval" / "f1_lr_balanced_summary.csv")
    best = pd.read_csv(ROOT / "outputs" / "week4_100m_tiles_balanced" / "f1_best_summary.csv")
    bld = pd.read_parquet(
        ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet", columns=["y_res"]
    )

    def pick(df, **kw):
        hit = df
        for k, v in kw.items():
            hit = hit.loc[hit[k] == v]
        return hit.iloc[0]

    ridge10 = pick(mae, model="balanced_train_full", scale="10m")
    ridge100 = pick(mae, model="balanced_train_full", scale="100m")
    ghsl10 = pick(mae, model="ghsl", scale="10m")
    ghsl100 = pick(mae, model="ghsl", scale="100m")
    lr10 = pick(f1_lr, model="balanced_train_full", scale="10m", min_cells=5)
    lr100 = pick(f1_lr, model="balanced_train_full", scale="100m", min_cells=5)
    ghsl_lr10 = pick(f1_lr, model="ghsl", scale="10m", min_cells=5)
    ghsl_lr100 = pick(f1_lr, model="ghsl", scale="100m", min_cells=5)
    tile = best.loc[(best["min_cells"] == 5) & (best["primary"] == True)]
    w4_tile = tile.loc[tile["model"] == "balanced_train_full"].iloc[0]
    ghsl_tile = tile.loc[tile["model"] == "ghsl"].iloc[0]
    full = train.get("full_gt_oof") or {}

    n_bld = len(bld)
    pct_bld = 100 * float(bld["y_res"].mean())
    n_pix = int(gt["n_pixels"])
    fr, fn, fe = float(gt["frac_res"]["mean"]), float(gt["frac_nres"]["mean"]), float(gt["frac_empty"]["mean"])
    n_bal = int(bal["n_pixels"])
    n_per = int(bal["n_residential"])
    n_pure_res = int(bal["n_pure_res_before"])
    n_pure_nres = int(bal["n_pure_nres_before"])
    f1 = float(train["oof_macro_f1"])
    f1_loro = float(train["loro_macro_f1_mean"])
    f1_std = float(train["loro_macro_f1_std"])
    mae_m = float(train["oof_mae_mean"])
    mae_r = float(train["oof_mae_frac_res"])
    mae_n = float(train["oof_mae_frac_nres"])
    mae_e = float(train["oof_mae_frac_empty"])

    cells = []

    def md(s: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

    md(f"""
# Week 4

Track-A Overture footprints in ten US downtown boxes, mapped to AlphaEarth 10 m cells, then pooled to 100 m tiles. Labels: `resi_def_v1`. Evaluation: leave-one-region-out (LORO).

Polygons are unstratified this week. Older Track-A draws capped each Overture class (~80/class) and ended ~27% residential. The unstratified draw keeps the natural mix (~{pct_bld:.0f}% residential among typed buildings, n={n_bld:,}). Training balance comes from undersampling residential *pixels*, not from oversampling rare non-residential polygons.

| Head | Metric | Value |
|---|---|---:|
| Logistic regression | macro-F1 (OOF) | {f1:.3f} |
| Logistic regression | LORO mean ± std | {f1_loro:.3f} ± {f1_std:.3f} |
| Ridge (3 fractions) | mean MAE | {mae_m:.3f} |
| Ridge | MAE res / nres / empty | {mae_r:.3f} / {mae_n:.3f} / {mae_e:.3f} |
""")

    md(f"""
## Data

### Fraction ground truth at 10 m

For each AlphaEarth cell that intersects at least one Track-A footprint:

```text
frac_res   = area(union(RES footprints)  ∩ cell) / area(cell)
frac_nres  = area(union(NRES footprints) ∩ cell) / area(cell)
frac_empty = clip(1 − frac_res − frac_nres, 0, 1)
```

{n_pix:,} cells. Mean coverage: residential {fr:.3f}, non-residential {fn:.3f}, empty {fe:.3f}. Spec: [`docs/week4/pixel_eval_gt_design.md`](../../docs/week4/pixel_eval_gt_design.md).

![Coverage histograms](../../datasets/week4_pixel_gt/figures/frac_histograms.png)

![Mean fractions by metro](../../datasets/week4_pixel_gt/figures/frac_by_region.png)

### Pure-cell balance

A cell is pure residential if `frac_res > 0` and `frac_nres == 0` (swap for non-residential). Before balance: {n_pure_res:,} RES / {n_pure_nres:,} NRES. Per metro, keep all NRES and draw RES down to match (seed 42) → {n_bal:,} cells, {n_per:,} per class.

![Pure counts before balance](../../datasets/week4_pixel_gt_balanced/figures/imbalanced_counts_by_region.png)

![Pure counts after balance](../../datasets/week4_pixel_gt_balanced/figures/balanced_counts_by_region.png)

### Why the old GT looked NRES-heavy

Stratified Track-A oversampled rare classes. Unstratified Overture in the same AOIs is ~{pct_bld:.0f}% residential by building count (~59% by typed area). Global GHSL / UN DESA built-up surface is also ~90% residential. Notebook: [`06_res_nres_eda.ipynb`](06_res_nres_eda.ipynb).

![Building mix: legacy stratified vs unstratified](../../outputs/week4_res_nres_eda/figures/global_vs_stratified_building_mix.png)
""")

    md(f"""
## Models at 10 m

Same LORO folds on the balanced pure cells. Features: AlphaEarth `A00`–`A63`. No SMOTE.

1. Ridge multi-output on `(frac_res, frac_nres, frac_empty)`.
2. Logistic regression on the binary stratum label.

Held-out fold models also score every pure cell in the full GT ({int(full.get('n', 0)):,} cells; F1 {float(full.get('macro_f1', float('nan'))):.3f}) so 100 m tiles have dense coverage.

![Fraction true vs predicted](../../outputs/week4_pixel_balanced_train/figures/oof_true_vs_pred.png)

![Fraction MAE by metro](../../outputs/week4_pixel_balanced_train/figures/loro_mae_by_region.png)

![Binary F1 by metro](../../outputs/week4_pixel_balanced_train/figures/loro_f1_by_region.png)

![Binary confusion](../../outputs/week4_pixel_balanced_train/figures/oof_confusion.png)
""")

    umap_sum = jload(ROOT / "outputs" / "week4_region_umap" / "summary.json")
    umap_cmp_path = ROOT / "outputs" / "week4_region_umap" / "w2_vs_w4_in_umap_f1.csv"
    wrong_png = (
        ROOT / "outputs" / "week4_region_umap" / "figures" / "umap_gt_wrong_side_by_region.png"
    )
    if umap_sum and umap_cmp_path.exists() and wrong_png.exists():
        uc = pd.read_csv(umap_cmp_path)
        w2_mean = float(umap_sum.get("in_umap_lr_f1_w2_cnn_mean", float("nan")))
        w4_gt_mean = float(umap_sum.get("in_umap_lr_f1_gt_mean", float("nan")))
        pct_mean = float(umap_sum.get("pct_wrong_mean", float("nan")))
        worst = umap_sum.get("pct_wrong_max_region", "?")
        worst_pct = float(umap_sum.get("pct_wrong_max", float("nan")))
        best = umap_sum.get("pct_wrong_min_region", "?")
        best_pct = float(umap_sum.get("pct_wrong_min", float("nan")))
        rows = []
        for _, r in uc.iterrows():
            pct = (
                float(r["w4_ae_gt_pct_wrong"])
                if "w4_ae_gt_pct_wrong" in uc.columns
                else float("nan")
            )
            rows.append(
                f"| {r['region']} | {float(r['w2_cnn_in_umap_f1']):.2f} | "
                f"{float(r['w4_ae_gt_in_umap_f1']):.2f} | {pct:.0f}% |"
            )
        table_body = "\n".join(rows)
        md(f"""
## Region UMAP (Week 2 layout)

Same ten metros as the Week 2 CNN panel. Blue/red wash = 2D logistic `P(residential)`; black curve = `LR P=0.5`. Numbers below are *in-UMAP* linear separability, not 64-D LORO F1.

Week 2 = CNN on buildings. Week 4 = AlphaEarth on balanced pure 10 m cells (n={int(umap_sum.get('n_pixels', 0)):,}). Mean in-UMAP F1: CNN {w2_mean:.2f}, AE GT {w4_gt_mean:.2f}. Mean share of AE pixels on the wrong side of the cut: {pct_mean:.0f}% ({worst} {worst_pct:.0f}%, {best} {best_pct:.0f}%).

Grey = correct side; blue/orange = true class of wrong-side pixels. Panel titles print `% wrong` on the full metro; scatter is stratified to ≤{int(umap_sum.get('max_plot_points_per_region', 400))}/metro.

![Wrong-side AE pixels by metro](../../outputs/week4_region_umap/figures/umap_gt_wrong_side_by_region.png)

![Wrong-side rate by metro](../../outputs/week4_region_umap/figures/umap_gt_wrong_side_rate_by_region.png)

Week 2 CNN reference (building patches, same panel order):

![Week 2 CNN UMAP by region + LR boundary](../../outputs/week2_cnn_bbox/umap_cnn_by_region_panels_residential_boundary.png)

| Region | W2 CNN F1 | W4 AE GT F1 | W4 % wrong |
|---|---:|---:|---:|
{table_body}
| mean | {w2_mean:.2f} | {w4_gt_mean:.2f} | {pct_mean:.0f}% |

Dense GT-colored Week 4 panels (overplots; use wrong-side figure for counts): `outputs/week4_region_umap/figures/umap_gt_by_region_panels_residential_boundary.png`. Rebuild: `scripts/plot_week4_region_umap_boundary.py`.
""")

    umap3 = jload(ROOT / "outputs" / "week4_region_umap_3d" / "summary.json")
    cmp3_path = ROOT / "outputs" / "week4_region_umap_3d" / "umap_2d_vs_3d_wrong_side.csv"
    png3 = ROOT / "outputs" / "week4_region_umap_3d" / "figures" / "umap3d_gt_wrong_side_by_region.png"
    if umap3 and png3.exists():
        pct3 = float(umap3.get("pct_wrong_mean", float("nan")))
        f13 = float(umap3.get("in_umap_lr_f1_mean", float("nan")))
        w3 = umap3.get("pct_wrong_max_region", "?")
        w3p = float(umap3.get("pct_wrong_max", float("nan")))
        b3 = umap3.get("pct_wrong_min_region", "?")
        b3p = float(umap3.get("pct_wrong_min", float("nan")))
        table3 = ""
        if cmp3_path.exists():
            c3 = pd.read_csv(cmp3_path)
            lines = []
            for _, r in c3.iterrows():
                lines.append(
                    f"| {r['region']} | {float(r['pct_wrong_2d']):.0f}% | "
                    f"{float(r['pct_wrong_3d']):.0f}% | {float(r['f1_2d']):.2f} | "
                    f"{float(r['f1_3d']):.2f} |"
                )
            table3 = "\n".join(lines)
            mean2 = float(c3["pct_wrong_2d"].mean())
            mean3 = float(c3["pct_wrong_3d"].mean())
            meanf2 = float(c3["f1_2d"].mean())
            meanf3 = float(c3["f1_3d"].mean())
            table3 = f"""
| Region | 2D % wrong | 3D % wrong | 2D F1 | 3D F1 |
|---|---:|---:|---:|---:|
{table3}
| mean | {mean2:.0f}% | {mean3:.0f}% | {meanf2:.2f} | {meanf3:.2f} |
"""
        html_note = ""
        html_rel = (umap3.get("figures") or {}).get("html")
        if html_rel:
            html_note = f" Interactive (rotate): [`{html_rel}`](../../{html_rel})."
        md(f"""
## Region UMAP 3D

Same {int(umap3.get('n_pixels', 0)):,} pixels, UMAP with 3 components, logistic cut is a plane. Grey = correct side; blue/orange = wrong-side by true class. Scatter ≤{int(umap3.get('max_plot_points_per_region', 400))}/metro; `% wrong` from full n. Mean wrong-side {pct3:.0f}% ({w3} {w3p:.0f}%, {b3} {b3p:.0f}%); mean in-UMAP F1 {f13:.2f}.{html_note}

![3D wrong-side AE pixels by metro](../../outputs/week4_region_umap_3d/figures/umap3d_gt_wrong_side_by_region.png)

![3D wrong-side rate by metro](../../outputs/week4_region_umap_3d/figures/umap3d_gt_wrong_side_rate_by_region.png)
{table3}
Rebuild: `scripts/plot_week4_region_umap_3d.py`.
""")

    md(f"""
## 10 m cells vs 100 m tiles

Drop tiles with fewer than five AE 10 m cells (`min ≥ 5`); means and majority votes are unstable on thin tiles. Folders also keep `min ≥ 20`. The three blocks below use different scoring sets; do not mix their numbers.

### Fraction MAE (Ridge)

Full-GT OOF; mean of residential and non-residential MAE. Tile fractions = mean of equal-area 10 m cells.

| | W4 Ridge | GHSL soft share |
|---|---:|---:|
| 10 m | {float(ridge10['mae_mean_res_nres']):.3f} | {float(ghsl10['mae_mean_res_nres']):.3f} |
| 100 m (min ≥ 20) | {float(ridge100['mae_mean_res_nres']):.3f} | {float(ghsl100['mae_mean_res_nres']):.3f} |
| Drop (10 m − 100 m) | {float(ridge100['delta_10m_minus_100m']):.3f} | {float(ghsl100['delta_10m_minus_100m']):.3f} |

![MAE at 10 m vs 100 m](../../outputs/week4_100m_fraction_eval/figures/mae_10m_vs_100m_ladder.png)

![MAE at 100 m by model](../../outputs/week4_100m_fraction_eval/figures/mae_100m_ladder.png)

### Binary F1 on tiles from balanced cells

Map each balanced train cell to its 100 m tile, keep tiles with ≥ 5 cells, predict with `mean(p_res) ≥ 0.50`.

| | F1 @ 10 m | F1 @ 100 m |
|---|---:|---:|
| W4 LR | {float(lr10['macro_f1']):.3f} | {float(lr100['macro_f1']):.3f} |
| GHSL | {float(ghsl_lr10['macro_f1']):.3f} | {float(ghsl_lr100['macro_f1']):.3f} |

![LR F1 at 10 m vs 100 m, min ≥ 5](../../outputs/week4_100m_fraction_eval/figures/f1_lr_balanced_10m_vs_100m_min5.png)

### Binary F1 on a 1:1 tile sample

Label eligible tiles by `argmax(frac_res, frac_nres)`, then undersample per metro to equal RES/NRES ({int(w4_tile['n_tiles']):,} tiles at min ≥ 5). Score with `mean(p_res) ≥ 0.50`.

| | F1 @ τ=0.50 | Best F1 |
|---|---:|---:|
| W4 LR | {float(w4_tile['f1_at_0.5']):.3f} | {float(w4_tile['macro_f1']):.3f} (τ={float(w4_tile['threshold']):.2f}) |
| GHSL | {float(ghsl_tile['f1_at_0.5']):.3f} | {float(ghsl_tile['macro_f1']):.3f} (τ={float(ghsl_tile['threshold']):.2f}) |

At the shared operating point τ=0.50, W4 LR beats GHSL. GHSL's best threshold is far from 0.50.

![F1 on 1:1 tiles, min ≥ 5](../../outputs/week4_100m_tiles_balanced/figures/f1_balanced_tiles_ladder_min5.png)

![Threshold sweep on 1:1 tiles](../../outputs/week4_100m_tiles_balanced/figures/f1_balanced_tiles_threshold_curves_min5.png)

![min ≥ 5 support](../../outputs/week4_100m_tiles_balanced/figures/min5_support_explained.png)
""")

    tile_dir = ROOT / "outputs" / "week4_100m_tile_train"
    tile_sum = jload(tile_dir / "summary.json")
    td5 = tile_sum.get("min5") or jload(tile_dir / "summary_min5.json")
    td20 = tile_sum.get("min20") or jload(tile_dir / "summary_min20.json")
    if td5:
        pool_f1 = td5.get("cell_pool_baseline_f1_at_0.5")
        pool_s = f"{float(pool_f1):.3f}" if pool_f1 is not None else "n/a"
        delta = td5.get("delta_f1_vs_cell_pool")
        delta_s = f"{float(delta):+.3f}" if delta is not None else "n/a"
        extra20 = ""
        if td20:
            d20 = td20.get("delta_f1_vs_cell_pool")
            d20_s = f"{float(d20):+.3f}" if d20 is not None else "n/a"
            extra20 = f"""
### min ≥ 20

| | Tile-direct | Cell-pool @ 0.50 |
|---|---:|---:|
| OOF F1 | {float(td20['oof_macro_f1']):.3f} | {float(td20['cell_pool_baseline_f1_at_0.5']):.3f} |
| Mean MAE | {float(td20['oof_mae_mean']):.3f} | — |
| Δ F1 | {d20_s} | |

![Tile-direct F1, min≥20](../../outputs/week4_100m_tile_train/figures/loro_f1_by_region_min20.png)
"""
        md(f"""
## Tile-direct train (10×10 AE grid)

Train on the tile itself: flatten the 10×10 AlphaEarth grid (64-D per cell) plus occupancy mask → 6500-D. Same 1:1 balanced tiles as above. Heads: Ridge on `(frac_res, frac_nres)`, LR on RES vs NRES. LORO over metros. Notebook: [`07_train_100m_tiles.ipynb`](07_train_100m_tiles.ipynb).

### min ≥ 5

| | Tile-direct | Cell-pool `mean(p_res)≥0.50` |
|---|---:|---:|
| OOF F1 | {float(td5['oof_macro_f1']):.3f} | {pool_s} |
| LORO F1 mean ± std | {float(td5['loro_macro_f1_mean']):.3f} ± {float(td5['loro_macro_f1_std']):.3f} | — |
| Mean MAE (frac_res, frac_nres) | {float(td5['oof_mae_mean']):.3f} | — |
| Δ F1 (direct − pool) | {delta_s} | |

Linear heads on the flattened grid trail cell-pool here.

![Tile-direct F1 by metro](../../outputs/week4_100m_tile_train/figures/loro_f1_by_region_min5.png)

![Tile-direct MAE by metro](../../outputs/week4_100m_tile_train/figures/loro_mae_by_region_min5.png)

![Tile-direct confusion](../../outputs/week4_100m_tile_train/figures/oof_confusion_min5.png)
{extra20}
""")

    md("""
## Paths

| Artifact | Path |
|---|---|
| Fraction GT | `datasets/week4_pixel_gt/` |
| Balanced pure cells | `datasets/week4_pixel_gt_balanced/` |
| Train OOF + checkpoints | `outputs/week4_pixel_balanced_train/` |
| 100 m fraction / LR eval | `outputs/week4_100m_fraction_eval/` |
| 1:1 tile eval | `outputs/week4_100m_tiles_balanced/` |
| Tile-direct train | `outputs/week4_100m_tile_train/` |
| RES/NRES EDA | `outputs/week4_res_nres_eda/` |
| Region UMAP | `outputs/week4_region_umap/` |
| Region UMAP 3D | `outputs/week4_region_umap_3d/` |
""")

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
