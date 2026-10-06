#!/usr/bin/env python3
"""Write notebooks/week4/16_briefing.ipynb from the PLS and 3D ViT artifacts.

The presented note is notebooks/week4/00_briefing.ipynb. This one starts after that.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "week4" / "16_briefing.ipynb"

NAMES = {
    "Atlanta_SE": "Atlanta",
    "Boston_NE": "Boston",
    "Chicago_MW": "Chicago",
    "Denver_W": "Denver",
    "Houston_S": "Houston",
    "LA_W": "Los Angeles",
    "NYC_NE": "New York",
    "Phoenix_SW": "Phoenix",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}

# NYC-fold validation split at best.pt. Mean matches the logged best val gap (3.334).
VAL_NYC = [
    ("Phoenix", 5.44, 0.553, 0.102, 289, 225),
    ("Atlanta", 4.98, 0.458, 0.092, 180, 187),
    ("Denver", 3.74, 0.468, 0.125, 380, 363),
    ("Los Angeles", 3.71, 0.440, 0.118, 531, 537),
    ("Seattle", 3.59, 0.396, 0.110, 121, 104),
    ("Houston", 2.80, 0.324, 0.116, 325, 312),
    ("Chicago", 2.01, 0.264, 0.131, 259, 246),
    ("Boston", 1.90, 0.261, 0.138, 271, 279),
    ("San Francisco", 1.82, 0.173, 0.095, 50, 61),
]


def _denver_mass() -> dict[str, float]:
    import torch

    fold = ROOT / "outputs" / "week4_vit_100m_3d_v4" / "folds" / "Denver_W"
    z = np.load(fold / "oof_z3.npy")
    idx = np.load(fold / "oof_idx.npy")
    y = np.load(ROOT / "outputs" / "week4_vit_100m_3d_v4" / "_cache" / "y.npy")[idx].astype(int)
    blob = torch.load(fold / "best.pt", map_location="cpu", weights_only=False)
    scale = float(np.asarray(blob["model"]["scale"]).reshape(-1)[0])
    w = np.load(fold / "head_w.npy").astype(np.float64).reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)
    if scale < 0:
        w = -w
    if (z[y == 1] @ w).mean() < (z[y == 0] @ w).mean():
        w = -w
    score = z @ w
    left = score <= np.quantile(score, 0.3)
    right = score >= np.quantile(score, 0.7)
    mid = ~left & ~right
    r0, r1 = score[y == 0], score[y == 1]
    return {
        "frac_res_pos": float((r1 > 0).mean()),
        "frac_nres_neg": float((r0 < 0).mean()),
        "left_res": float(y[left].mean()),
        "mid_res": float(y[mid].mean()),
        "right_res": float(y[right].mean()),
        "nres_in_res": float(((r0 >= r1.min()) & (r0 <= r1.max())).mean()),
        "res_in_nres": float(((r1 >= r0.min()) & (r1 <= r0.max())).mean()),
    }


def main() -> None:
    sweep = pd.read_csv(ROOT / "outputs" / "week4_pls_10m_vs_100m" / "sweep_summary.csv")
    v1 = pd.read_csv(ROOT / "outputs" / "week4_vit_100m_3d" / "loro_metrics.csv")
    v2 = pd.read_csv(ROOT / "outputs" / "week4_vit_100m_3d_v2" / "loro_metrics.csv")
    v3 = pd.read_csv(ROOT / "outputs" / "week4_vit_100m_3d_v3" / "loro_metrics.csv")
    v4 = pd.read_csv(ROOT / "outputs" / "week4_vit_100m_3d_v4" / "loro_metrics.csv")
    s1 = json.loads((ROOT / "outputs" / "week4_vit_100m_3d" / "summary.json").read_text())
    s2 = json.loads((ROOT / "outputs" / "week4_vit_100m_3d_v2" / "summary.json").read_text())
    s3 = json.loads((ROOT / "outputs" / "week4_vit_100m_3d_v3" / "summary.json").read_text())
    s4 = json.loads((ROOT / "outputs" / "week4_vit_100m_3d_v4" / "summary.json").read_text())
    vit = json.loads((ROOT / "outputs" / "week4_vit_100m" / "summary.json").read_text())
    l1 = json.loads((ROOT / "outputs" / "week4_l1_pls_loro" / "summary.json").read_text())
    fine = json.loads((ROOT / "outputs" / "week4_pls_loro" / "fine_k_sweep.json").read_text())
    den = _denver_mass()

    def mean_ratio(df: pd.DataFrame) -> float:
        if "test_ratio" in df.columns:
            return float(df["test_ratio"].mean())
        return float((df["test_centroid_sep"] / df["test_within"]).mean())

    g1, g2, g3m, g4 = mean_ratio(v1), mean_ratio(v2), mean_ratio(v3), mean_ratio(v4)

    def best(scale: str, model: str) -> pd.Series:
        hit = sweep[(sweep["scale"] == scale) & (sweep["model"] == model)]
        return hit.loc[hit["loro_macro_f1_mean"].idxmax()]

    rows_f1 = []
    for scale, label in (("10m", "10 m cells"), ("100m", "100 m mean pool")):
        for model, name in (
            ("baseline_lr", "64-D logistic regression"),
            ("pls_da_lr", "PLS-DA + logistic regression"),
            ("pca_lr", "PCA + logistic regression"),
            ("l1_pls", "L1 on 32 PLS scores"),
            ("t1_only", "t1 only"),
        ):
            r = best(scale, model)
            c = "" if pd.isna(r["C"]) else f", C={r['C']:g}"
            k = int(r["n_components"])
            rows_f1.append(
                f"| {label} | {name} | {k}{c} | {r['loro_macro_f1_mean']:.3f} ± {r['loro_macro_f1_std']:.3f} |"
            )

    v3i = v3.set_index("region")
    v1i = v1.set_index("region")
    order = v4.sort_values("test_ratio", ascending=False)
    metro_rows = []
    for _, r in order.iterrows():
        region = r["region"]
        g3 = float(v3i.loc[region, "test_centroid_sep"] / v3i.loc[region, "test_within"])
        metro_rows.append(
            f"| {NAMES[region]} | {r['macro_f1']:.3f} | {float(v3i.loc[region, 'macro_f1']):.3f} | "
            f"{float(v1i.loc[region, 'macro_f1']):.3f} | {r['test_ratio']:.2f} | {g3:.2f} | "
            f"{r['best_metric_value']:.2f} | {int(r['epochs_ran'])} |"
        )
    clear = [NAMES[r] for r in order.loc[order["test_ratio"] >= 3, "region"]]
    clear_s = ", ".join(clear[:-1]) + f", and {clear[-1]}" if len(clear) > 1 else (clear[0] if clear else "none")

    val_rows = [
        f"| {name} | {gap:.2f} | {sep:.3f} | {within:.3f} | {n0 + n1} |"
        for name, gap, sep, within, n0, n1 in VAL_NYC
    ]

    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))

    md(f"""
# Week 4 follow-up

This note starts after [`00_briefing.ipynb`](00_briefing.ipynb). Same ten downtown boxes, same `resi_def_v1` labels, same leave-one-metro-out split. The question here is whether a small embedding of a 100 m tile separates residential from non-residential inside each metro.

Two summaries. Linear models on AlphaEarth already sit near 0.76 LORO macro-F1 at 100 m, and PLS does not beat the 64-D logistic regression. Forcing the ViT down to 3 numbers raised the held-out gap and left F1 about where v1 and v3 were. {clear_s} clear a held-out gap of 3. New York does not.

| Run | LORO macro-F1 | Mean held-out gap |
|---|---:|---:|
| 100 m mean-pool logistic regression | {best('100m','baseline_lr')['loro_macro_f1_mean']:.3f} | — |
| ViT, 128-D class token | {vit['loro_macro_f1_mean']:.3f} | — |
| v1, BCE + SupCon | {s1['loro_macro_f1_mean']:.3f} | {g1:.2f} |
| v2, detached head | {s2['loro_macro_f1_mean']:.3f} | {g2:.2f} |
| v3, same loss, longer training | {s3['loro_macro_f1_mean']:.3f} | {g3m:.2f} |
| v4, within-region pull and gap | {s4['loro_macro_f1_mean']:.3f} | {g4:.2f} |

Gap = distance between the two class centers ÷ mean radius of each class. A gap of 3 was the training target. A gap near 1 is one cloud. Raw sep is not comparable across runs: v4's clouds are much smaller than v1's, and the ratio is the number to compare. Each fold has its own 3D coordinates.
""")

    md(f"""
## Linear baselines

100 m features are the mean of the AlphaEarth cells inside the tile. Tiles need at least 5 occupied cells, then residential and non-residential are balanced (n={vit['n_tiles']:,}). The 10 m numbers use the balanced pure cells (n=1,751,814).

| Scale | Model | Width | LORO macro-F1 |
|---|---|---:|---:|
{chr(10).join(rows_f1)}
| 100 m | flat 10×10 grid, logistic regression | 6500 | {vit['baselines']['flat_grid_lr']:.3f} |
| 100 m | ViT, 128-D class token | 128 | {vit['loro_macro_f1_mean']:.3f} ± {vit['loro_macro_f1_std']:.3f} |

Moving from 10 m cells to a 100 m mean costs 0.018 F1 on the 64-D logistic regression (0.779 → 0.761) and 0.013 on the best PLS-DA head. PLS-DA at 100 m, k=24, is 0.762, level with the full 64-D head. PCA is lower. The first PLS score alone is 0.720.

L1 on the 32 PLS scores drops the mean number of nonzero weights from 32 to about 1.2 at C=10, and that survivor is t1 on every fold where anything survives (t1 survival 1.0, t2 survival 0.2). LORO F1 at that setting is {l1['l1_pls_nnz_le_2']['loro_macro_f1_mean']:.3f}, the same neighborhood as the t1-only head. Stronger L1 zeros every score.

k=9 is the smallest PLS width within 0.02 F1 of the 64-D logistic regression (F1 {fine['min_k_criteria']['within_0.02_F1']['f1']:.3f}, gap {fine['min_k_criteria']['within_0.02_F1']['gap']:.3f}). The ViT reads those 9 scores. Empty cells in the 10×10 grid are masked.

![LORO F1 at 10 m and 100 m](../../outputs/week4_pls_10m_vs_100m/figures/f1_family_10m_vs_100m.png)

![Which PLS scores survive L1](../../outputs/week4_l1_pls_loro/figures/l1_survival_pls32.png)

![ViT against the linear tile heads](../../outputs/week4_vit_100m/figures/vit_vs_baselines.png)

Notebooks: [`08_pls_comparison.ipynb`](08_pls_comparison.ipynb), [`09_l1_pls_sparsity.ipynb`](09_l1_pls_sparsity.ipynb), [`10_pls_10m_vs_100m.ipynb`](10_pls_10m_vs_100m.ipynb).
""")

    md(f"""
## Why the 3D runs changed

The 128-D ViT matches the mean-pool logistic regression and does not show two clusters, because there is no 3D space to look at. The next runs keep the same ViT (d_model 128, 4 layers, 4 heads) and replace the class token with 3 numbers. The aim is two residential / non-residential clusters inside each metro after fine-tuning.

v1 adds supervised contrastive loss on the L2-normalized 3D vector, λ=1, temperature 0.1, and lets the classifier gradient into the encoder. Held-out F1 is {s1['loro_macro_f1_mean']:.3f}. The clouds are horseshoes: the linear head only needs one direction, so the extra two axes collapse into a curve. The tips of the curve are class-pure. The bottom is mixed. SupCon's minimum for two classes is two opposite poles. This run did not reach that minimum.

v2 makes four changes: stop the classifier gradient at the 3D vector, use a cosine head with a learnable scale, set λ=10, and anneal temperature from 0.5 to 0.1. The encoder's only training signal is SupCon. Held-out F1 falls to {s2['loro_macro_f1_mean']:.3f}. SupCon stays near the chance value for a batch of 256 (about log 255 ≈ 5.54). Atlanta is the clear failure: the checkpoint is epoch 1, training stops at epoch 8, and every held-out tile is predicted non-residential (F1 0.333). v1 Atlanta was {float(v1i.loc['Atlanta_SE','macro_f1']):.3f}. The stop rule was validation F1, while training BCE was still falling.

v3 keeps the v2 loss and changes the schedule. Every fold runs at least 35 epochs, then stops on validation loss (BCE + λ τ SupCon), patience 6, cap 60. F1 is logged and does not choose the checkpoint. Held-out F1 returns to {s3['loro_macro_f1_mean']:.3f}, in line with v1. All 10 folds run to the cap. Temperature is still annealing across the 60-epoch budget, so the weighted validation loss keeps falling while validation SupCon moves back toward chance. Longer training did not make two poles.

v4 drops SupCon, the detach, and the temperature. Each batch is one metro, half residential and half non-residential. On the raw 3D vectors:

```text
loss = BCE(scale * (z · unit w), y)
     + mean squared distance to the class mean
     + relu(3 − sep/within)^2
```

Both the classifier and the cluster terms send gradient into z. The stop metric is the mean sep/within on a 10% split of the other metros. Higher is better. The floor is still 35 epochs, patience 6, cap 60. F1 does not choose the checkpoint. Held-out F1 is {s4['loro_macro_f1_mean']:.3f}. Mean held-out gap is {g4:.2f}, compared with {g3m:.2f} in v3.

Notebooks: [`12_vit_100m_3d.ipynb`](12_vit_100m_3d.ipynb) through [`15_vit_100m_3d_v4.ipynb`](15_vit_100m_3d_v4.ipynb).
""")

    md(f"""
## Held-out metros under v4

{clear_s} are the held-out metros with gap ≥ 3. The gap is higher than v3 in every metro. F1 moves by metro and the mean is flat ({s4['loro_macro_f1_mean']:.3f} vs v3 {s3['loro_macro_f1_mean']:.3f}). Chicago drops from {float(v3i.loc['Chicago_MW','macro_f1']):.3f} to {float(v4.set_index('region').loc['Chicago_MW','macro_f1']):.3f}. New York stays one cloud: gap {float(v4.set_index('region').loc['NYC_NE','test_ratio']):.2f}, F1 {float(v4.set_index('region').loc['NYC_NE','macro_f1']):.3f}.

The validation column is the mean gap on a 10% split of the other metros. That is the number training stopped on. New York's fold reached a validation gap of {float(v4.set_index('region').loc['NYC_NE','best_metric_value']):.2f} and a held-out gap of {float(v4.set_index('region').loc['NYC_NE','test_ratio']):.2f}.

| Metro | v4 F1 | v3 F1 | v1 F1 | v4 gap | v3 gap | v4 val gap | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metro_rows)}

![Held-out gap, v4 against v3](../../outputs/week4_vit_100m_3d_v4/figures/v4_vs_v3_gap.png)
""")

    md(f"""
## How to read a cluster figure

Left panel: the three encoder outputs, unrotated. The gray plane is the classifier cut, `z · w = 0`. Right panel: the same tiles on the classifier score and the widest direction perpendicular to it. The dashed line is the same cut. Residential is on the positive side of the score.

Color is the mix of the two class centers. Orange is non-residential, blue is residential, mauve is where both centers still reach. Influence of a center falls with distance, and the radius reaches at least halfway to the other center, so the overlap is a blend instead of one class drawn on top of the other. Marker size falls off with that influence. The white-edged dots are the two centers.

A high gap does not mean an empty band between the classes. Denver's held-out gap is {float(v4.set_index('region').loc['Denver_W','test_ratio']):.2f} and its F1 is {float(v4.set_index('region').loc['Denver_W','macro_f1']):.3f}. {den['frac_nres_neg']:.0%} of non-residential tiles have score below 0, and {den['frac_res_pos']:.0%} of residential tiles have score above 0. Along the score, the leftmost 30% of tiles are {den['left_res']:.0%} residential, the middle 40% are {den['mid_res']:.0%} residential, and the rightmost 30% are {den['right_res']:.0%} residential. Almost every tile of either class still sits inside the other class's score range (over 99.9% of each). The cut sorts the mass. The supports overlap.

The geometry is a thin line along the class axis. A different camera angle does not open a gap.

Atlanta clears the gap. Denver is the high-F1 case above. New York is the held-out cloud.

![Atlanta, held out](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_Atlanta_SE.png)

![Denver, held out](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_Denver_W.png)

![New York, held out](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_NYC_NE.png)
""")

    md(f"""
## Validation while New York is held out

The New York fold's best validation gap is {float(v4.set_index('region').loc['NYC_NE','best_metric_value']):.2f}, measured on 10% of the other metros at the checkpoint (logged best {3.3337091472020046:.3f}). The same checkpoint's held-out New York gap is {float(v4.set_index('region').loc['NYC_NE','test_ratio']):.2f}. The other cities in that fold do separate. They are still thin lines with the classes at opposite ends.

| Metro in the New York fold's validation split | Gap | Sep | Within | n |
|---|---:|---:|---:|---:|
{chr(10).join(val_rows)}

![Phoenix in the New York fold's validation split](../../outputs/week4_vit_100m_3d_v4/figures/val_nycfold_Phoenix_SW.png)

![San Francisco in the New York fold's validation split](../../outputs/week4_vit_100m_3d_v4/figures/val_nycfold_SF_W.png)

The other held-out figures, and the rest of this validation split, are in [`08_pls_comparison.ipynb`](08_pls_comparison.ipynb).
""")

    md("""
## Paths

| Artifact | Path |
|---|---|
| PLS LORO | `outputs/week4_pls_loro/` |
| 10 m vs 100 m | `outputs/week4_pls_10m_vs_100m/` |
| L1 on PLS scores | `outputs/week4_l1_pls_loro/` |
| ViT 128-D | `outputs/week4_vit_100m/` |
| 3D v1–v3 | `outputs/week4_vit_100m_3d/`, `_v2/`, `_v3/` |
| 3D v4 folds, checkpoints, figures | `outputs/week4_vit_100m_3d_v4/` |
| This note | `notebooks/week4/16_briefing.ipynb` |
""")

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    nbf.write(nb, NB)
    print(f"Wrote {NB} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
