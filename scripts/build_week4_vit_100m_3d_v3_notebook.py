#!/usr/bin/env python3
"""Write notebooks/week4/14_vit_100m_3d_v3.ipynb and, once training has finished, its figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs" / "week4_vit_100m_3d_v3"
FIG = OUT / "figures"
NB = ROOT / "notebooks" / "week4" / "14_vit_100m_3d_v3.ipynb"

CODE = '''"""v3 cosine head and the loss that trains it.

Matches scripts/train_week4_vit_100m_3d_v3.py.
The run trains at least 35 epochs. After that it stops if validation loss
(BCE + λ τ SupCon) has not improved for 6 epochs. The cap is 60.
The encoder is updated only by the contrastive term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 3
GRID = 10
TAU_START = 0.5
TAU_END = 0.1
SUPCON_LAMBDA = 10.0


def tau_at(epoch, epochs, tau_start=TAU_START, tau_end=TAU_END):
    if epochs <= 1:
        return tau_end
    t = epoch / (epochs - 1)
    return tau_start + t * (tau_end - tau_start)


def supervised_contrastive(z, y, tau):
    u = F.normalize(z, dim=1)
    logits = u @ u.T / tau
    b = z.shape[0]
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    labels = y.view(-1).long()
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    mean_log_prob = (log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    return -mean_log_prob[valid].mean()


class AEPatchViT3DCosine(nn.Module):
    """Same transformer as v1. The head is a cosine classifier and does not train the encoder."""

    def __init__(self, patch_dim, d_model=128, n_heads=4, n_layers=4, dropout=0.1, emb_dim=EMB_DIM):
        super().__init__()
        n_patches = GRID * GRID
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.to_3d = nn.Linear(d_model, emb_dim)
        self.head = nn.Linear(emb_dim, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def encode(self, x, mask):
        b = x.shape[0]
        tokens = self.patch_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        pad = torch.zeros(b, 1, device=x.device, dtype=torch.bool)
        key_pad = torch.cat([pad, mask < 0.5], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        return self.to_3d(self.norm(h[:, 0]))

    def cosine_logits(self, u):
        w = F.normalize(self.head.weight, dim=-1)
        return (self.scale * (u @ w.T)).squeeze(-1)

    def forward(self, x, mask):
        z = self.encode(x, mask)
        u = F.normalize(z, dim=-1)
        return self.cosine_logits(u.detach()), z


# One training step, tau = tau_at(epoch, epochs):
#   logits, z = model(x, mask)          # logits do not carry grad into z
#   bce = BCEWithLogits(logits, y)
#   sup = supervised_contrastive(z, y, tau)
#   loss = bce + SUPCON_LAMBDA * tau * sup
'''


def _results_markdown() -> str:
    summary_path = OUT / "summary.json"
    metrics_path = OUT / "loro_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return """## Results

Training is still running. Per-fold lines are in `outputs/week4_vit_100m_3d_v3/logs/<metro>.log`. The driver log is `outputs/week4_vit_100m_3d_v3/train.log`.
"""
    summary = json.loads(summary_path.read_text())
    metrics = pd.read_csv(metrics_path)
    metrics["ratio"] = metrics["test_centroid_sep"] / metrics["test_within"]
    metrics["unit_ratio"] = metrics["test_unit_sep"] / metrics["test_unit_within"]
    metrics = metrics.sort_values("macro_f1", ascending=False)
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            f"| {r['region']} | {r['macro_f1']:.3f} | {r['test_centroid_sep']:.2f} | {r['test_within']:.2f} | {r['ratio']:.2f} | {r['test_unit_sep']:.2f} | {r['test_unit_within']:.2f} | {r['unit_ratio']:.2f} |"
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
                f"![3D {region}](../../outputs/week4_vit_100m_3d_v3/figures/vit3d_{region}.png)\n"
            )
        figures = f"""
![Gap by metro](../../outputs/week4_vit_100m_3d_v3/figures/vit3d_sep_ratio.png)

![2D projection](../../outputs/week4_vit_100m_3d_v3/figures/vit3d_projection.png)

## Encoder output

{chr(10).join(gallery)}
"""
    return f"""## Results

LORO macro-F1 = **{summary['loro_macro_f1_mean']:.3f}** ± {summary['loro_macro_f1_std']:.3f}. Pooled OOF F1 = {summary['oof_macro_f1']:.3f}. v1 on the same tiles was 0.732.

Mean raw gap (sep/within) = {summary['mean_test_centroid_sep'] / summary['mean_test_within']:.2f}. Mean unit-sphere sep = {summary['mean_test_unit_sep']:.2f} (2.00 is the two-pole target). Elapsed {summary['elapsed_s']} s.

| Metro | F1 | sep | within | sep/within | unit sep | unit within | unit sep/within |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

`unit sep` is the distance between class centers after each vector is scaled to length 1. The contrastive term sees that sphere. A value near 2 with a small unit within means the two classes collapsed to opposite poles.
{figures}
"""


def main() -> None:
    results = _results_markdown()
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# 3D ViT v3 — at least 35 epochs, then stop on validation loss

v1 (`notebooks/week4/12_vit_100m_3d.ipynb`) put a supervised contrastive term on the 3D encoder output and still drew one curve per metro. The classifier gradient reached that vector, and the contrastive term stayed near chance (about 5.3, against 5.54).

This run keeps the same ViT, the same 9 PLS scores, the same 10 metros, and the same leave-one-metro-out split. Four changes:

| # | v1 | v2 |
|---|---|---|
| 1 | BCE backprops into the 3D vector | BCE stops at the 3D vector. Only SupCon trains the encoder |
| 2 | Linear head on the raw 3D vector, with a bias | Cosine head. The vector and the weight are L2-normalized. No bias. Learnable logit scale, init 10 |
| 3 | loss = BCE + 1 · SupCon | loss = BCE + λ · τ · SupCon, λ = 10 |
| 4 | τ fixed at 0.1 | τ anneals linearly from 0.5 to 0.1 over the 30-epoch budget |

$$
\\mathcal{L} = \\mathrm{BCE}\\big(s \\cdot \\cos(\\hat z_{\\mathrm{stop}}, \\hat w),\\ y\\big) + \\lambda \\, \\tau \\, \\mathcal{L}_{\\mathrm{SupCon}}(z, y)
$$

$\\hat z_{\\mathrm{stop}}$ is the unit encoder output with the gradient stopped. $\\hat w$ is the unit head weight. $s$ starts at 10. For two classes, SupCon's minimum is two opposite poles on the sphere (Graf et al., 2021). Chance at batch 256 is about 5.54. Two poles sit near 4.84.

v2 stopped on validation F1. Atlanta's probabilities never crossed 0.5, so F1 sat at 0.33 while BCE was still falling, and the fold halted at epoch 8. This run always completes **35 epochs**. After that, a fold stops if validation loss has not improved for 6 epochs. Validation loss is BCE plus λ τ SupCon, the same objective as training. The cap is 60. F1 is still logged. It does not decide the checkpoint.

Khosla et al. (2020) train the encoder with SupCon and fit the linear head with its gradient blocked. They also multiply the contrastive loss by τ, because the gradient scales as 1/τ. λ = 10 is the low end of the range that matters when the classifier gradient is still attached. Here the gradient is stopped, and λ · τ still sets the step size on the encoder.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## What each log field means

One file per held-out metro: `outputs/week4_vit_100m_3d_v3/logs/<metro>.log`. The first lines restate the four changes. Then one line per epoch.

| Field | Meaning | What to look for |
|---|---|---|
| tau | Temperature this epoch | Falls from 0.50 toward 0.10. Early stop can cut the anneal short |
| bce | Classification loss on the cosine head | Should fall. It does not move the encoder |
| supcon | Supervised contrastive loss | Leave 5.54. Toward 4.84 means the poles are forming |
| wsup | λ · τ · supcon, the term added to the loss | The encoder step scales with this |
| scale | Learnable logit scale s | Starts at 10. Grows if the head wants sharper probabilities |
| val_bce, val_supcon, val_loss | Validation BCE, SupCon, and their sum (BCE + λ τ SupCon) | Early stop watches val_loss after epoch 35. Patience 6. Cap 60 |
| val_f1 | Macro-F1 on 10% of the training metros | Logged only. This is not the held-out city, and it does not stop the fold |
| sep, within | Class-center distance and mean radius in the raw 3D output | v1's metric. A curve can have a large sep and a large within |
| unit_sep, unit_within | Same two numbers after each vector is scaled to length 1 | The geometry SupCon sees. unit_sep near 2 and unit_within near 0 is two poles |
| bad | Epochs since val_loss improved | Counted from the start. A stop is allowed only once the epoch number is at least 35 |

The driver log is `outputs/week4_vit_100m_3d_v3/train.log`. The process id is `outputs/week4_vit_100m_3d_v3/train.pid`. Checkpoints are `outputs/week4_vit_100m_3d_v3/checkpoints/fold_<metro>.pt` (weights, PLS rotation, score mean and std, the epoch history). Held-out 3D vectors are `oof_z3.npy`. Do not pool those coordinates across metros. Each fold has its own axes.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Code

The training driver is `scripts/train_week4_vit_100m_3d_v3.py`. The cell below is the head and the loss. The data loader, the PLS projection, and the fold loop are the same as v1.

```bash
PYTHONWARNINGS=ignore setsid nohup .venv/bin/python -u scripts/train_week4_vit_100m_3d_v3.py \\
  > outputs/week4_vit_100m_3d_v3/train.log 2>&1 < /dev/null &
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
