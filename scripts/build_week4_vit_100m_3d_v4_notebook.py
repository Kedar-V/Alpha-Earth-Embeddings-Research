#!/usr/bin/env python3
"""Write notebooks/week4/15_vit_100m_3d_v4.ipynb and, once training has finished, its figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs" / "week4_vit_100m_3d_v4"
FIG = OUT / "figures"
NB = ROOT / "notebooks" / "week4" / "15_vit_100m_3d_v4.ipynb"

CODE = '''"""v4 within-region cluster loss.

Matches scripts/train_week4_vit_100m_3d_v4.py.
Each batch is one metro. The gap is sep/within in the raw 3D output.
The run trains at least 35 epochs, then stops if that validation gap
has not improved for 6 epochs. The cap is 60. Higher is better.
"""
import torch
import torch.nn.functional as F

TARGET_RATIO = 3.0


def cluster_terms(z, y, target_ratio=TARGET_RATIO):
    labels = y.view(-1).long()
    centers, pulls, radii = [], [], []
    for k in (0, 1):
        pts = z[labels == k]
        if pts.shape[0] < 2:
            return None
        center = pts.mean(dim=0)
        centers.append(center)
        pulls.append((pts - center).pow(2).sum(dim=-1).mean())
        radii.append((pts - center).norm(dim=-1).mean())
    pull = torch.stack(pulls).mean()
    within = torch.stack(radii).mean()
    sep = (centers[1] - centers[0]).norm()
    ratio = sep / within.clamp_min(1e-6)
    short = F.relu(target_ratio - ratio).pow(2)
    return pull, short, ratio, sep, within


# logits, z = model(x, mask)     # z is not detached
# bce = BCEWithLogits(logits, y) # logits = scale * (z · unit w)
# pull, short, ratio, sep, within = cluster_terms(z, y)
# loss = bce + pull + short
'''


def _results_markdown() -> str:
    summary_path = OUT / "summary.json"
    metrics_path = OUT / "loro_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return """## Results

Training is still running. Per-fold lines are in `outputs/week4_vit_100m_3d_v4/folds/<metro>/train.log`. Epoch rows are appended to `folds/<metro>/metrics.csv` during the run. The driver log is `outputs/week4_vit_100m_3d_v4/train.log`.
"""
    summary = json.loads(summary_path.read_text())
    metrics = pd.read_csv(metrics_path)
    if "ratio" not in metrics.columns and "test_ratio" in metrics.columns:
        metrics["ratio"] = metrics["test_ratio"]
    metrics = metrics.sort_values("macro_f1", ascending=False)
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            f"| {r['region']} | {r['macro_f1']:.3f} | {r['test_centroid_sep']:.2f} | {r['test_within']:.2f} | {r['test_ratio']:.2f} | {r['best_metric_value']:.2f} | {int(r['best_epoch']) + 1} |"
        )
    z_path = OUT / "oof_z3.npy"
    figures = ""
    if z_path.exists() and (OUT / "oof_predictions.parquet").exists() and (OUT / "head_w_by_region.npz").exists():
        import build_week4_vit_100m_3d_notebook as viz

        oof = pd.read_parquet(OUT / "oof_predictions.parquet")
        z = np.load(z_path)
        heads = np.load(OUT / "head_w_by_region.npz")
        viz.FIG = FIG
        plot_metrics = metrics.copy()
        plot_metrics["ratio"] = plot_metrics["test_ratio"]
        viz.write_figures(
            z,
            oof["y_true"].to_numpy().astype(int),
            oof["region"].to_numpy(),
            heads,
            plot_metrics,
        )
        by_gap = plot_metrics.sort_values("ratio", ascending=False)
        gallery = []
        for _, r in by_gap.iterrows():
            region = r["region"]
            gallery.append(
                f"### {region.replace('_', ' ')}\n\n"
                f"![3D {region}](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_{region}.png)\n"
            )
        figures = f"""
![Gap by metro](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_sep_ratio.png)

![2D projection](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_projection.png)

## Encoder output

{chr(10).join(gallery)}
"""
    return f"""## Results

LORO macro-F1 = **{summary['loro_macro_f1_mean']:.3f}** ± {summary['loro_macro_f1_std']:.3f}. Pooled OOF F1 = {summary['oof_macro_f1']:.3f}. v1 was {summary['v1_loro_macro_f1_mean']:.3f}. v3 was {summary['v3_loro_macro_f1_mean']:.3f}.

Mean held-out gap (sep/within) = **{summary['mean_test_ratio']:.2f}**. A gap of 3 is the clear-separation line. Elapsed {summary['elapsed_s']} s.

| Metro | F1 | sep | within | sep/within | best val gap | best epoch |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The 3D axes are the raw encoder outputs. Do not pool them across metros. Each fold has its own axes.
{figures}
"""


def main() -> None:
    results = _results_markdown()
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# 3D ViT v4 — within-region clusters

v3 detached the classifier from the 3D vector and trained the encoder with SupCon. Validation BCE fell. Validation SupCon rose back to chance. The two losses were not training the same weights, so a better classifier did not make two clusters.

This run keeps the same ViT, the same 9 PLS scores, the same 10 metros, and the same leave-one-metro-out split. The 3D output is the thing that has to cluster.

Each training batch is one metro, with an equal number of residential and non-residential tiles. On the raw encoder vectors in that batch:

$$
\\mathcal{L} = \\mathrm{BCE}\\big(s \\cdot (z \\cdot \\hat w),\\ y\\big) + \\overline{\\|z - c_y\\|^2} + \\mathrm{relu}(3 - \\mathrm{sep}/\\mathrm{within})^2
$$

$c_y$ is the mean of that class in the batch. sep is the distance between the two class means. within is the mean distance to the class mean. The gradient of BCE and of both cluster terms reaches $z$. $\\hat w$ is a unit vector in the same 3D space, so residential stays on one side of the gap in every metro.

The number that stops training is the mean validation sep/within, averaged over the training metros in the 10% split. Higher is better. A fold always runs 35 epochs. After that it stops if the gap has not improved by 0.001 for 6 epochs. The cap is 60. F1 is logged and does not choose the checkpoint.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## What each log field means

One file per held-out metro: `outputs/week4_vit_100m_3d_v4/folds/<metro>/train.log`. The same rows are appended, during training, to `folds/<metro>/metrics.csv`.

| Field | Meaning | What to look for |
|---|---|---|
| bce | Classification loss. The head reads raw z | Should fall. Its gradient moves the encoder |
| pull | Mean squared distance to the class center | Should fall if each class tightens |
| short | relu(3 − sep/within)² | Reaches 0 when the gap is at least 3 |
| ratio, sep, within | Gap on the training batches | ratio is sep/within. The target is 3 |
| val_ratio | Mean sep/within on the validation metros | This is the stop metric. Higher is better |
| val_f1, prec, rec | Macro scores on 10% of the training metros | Logged only. Not the held-out city |
| best_ratio, bad | Best val gap and epochs since it improved | A stop is allowed only after epoch 35 |
| ckpt | Which files were written this epoch | `best` is the highest val gap. `last` is every epoch. `epoch_005` every fifth epoch |

Checkpoints for a fold live in `outputs/week4_vit_100m_3d_v4/folds/<metro>/` as `best.pt`, `last.pt`, and `epoch_005.pt` onward. Each file has the weights, optimizer, scheduler, epoch, best gap, early-stop count, seed, and RNG state. `last.pt` is resumable when the config matches. The driver log is `outputs/week4_vit_100m_3d_v4/train.log`.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Code

The training driver is `scripts/train_week4_vit_100m_3d_v4.py`. The cell below is the cluster term. The ViT, the PLS projection, and the fold loop are in the driver.

```bash
PYTHONWARNINGS=ignore setsid nohup .venv/bin/python -u scripts/train_week4_vit_100m_3d_v4.py \\
  > outputs/week4_vit_100m_3d_v4/train.log 2>&1 < /dev/null &
```
"""
        ),
        nbf.v4.new_code_cell(CODE),
        nbf.v4.new_markdown_cell(results),
    ]
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
