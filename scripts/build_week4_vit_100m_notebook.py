#!/usr/bin/env python3
"""Build notebooks/week4/11_vit_100m.ipynb from ViT LORO outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_vit_100m"
NB = ROOT / "notebooks" / "week4" / "11_vit_100m.ipynb"


def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def main() -> None:
    if not (OUT / "summary.json").exists():
        raise SystemExit(f"Missing {OUT / 'summary.json'}; run scripts/train_week4_vit_100m_loro.py first")

    s = json.loads((OUT / "summary.json").read_text())
    metrics = pd.read_csv(OUT / "loro_metrics.csv")
    arch = s.get("arch", {})
    train = s.get("train", {})
    base = s.get("baselines", {}) or {}

    metro_rows = []
    for _, r in metrics.sort_values("region").iterrows():
        metro_rows.append(
            f"| {r['region']} | {fmt(r['macro_f1'])} | {int(r['best_epoch'])} | {int(r['n_test']):,} |"
        )

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# ViT on 100 m tiles (AE cells = patches)

Each 100 m tile is a **10×10** grid of AlphaEarth 10 m cells. Each cell is projected
onto the **minimum PLS-DA scores** that stay within 0.02 LORO F1 of the 64-D
logistic (k from the 10 m fine sweep; held-out metro's own rotation). Empty
slots are masked in attention.

| | |
|---|---|
| Data | balanced min{s.get('min_cells')} tiles · n={s.get('n_tiles'):,} (RES/NRES {s.get('n_res'):,}/{s.get('n_nres'):,}) |
| Arch | d_model={arch.get('d_model')}, layers={arch.get('n_layers')}, heads={arch.get('n_heads')}, dropout={arch.get('dropout')} |
| Train | AdamW lr={train.get('lr')}, batch={train.get('batch_size')}, ≤{train.get('epochs_max')} epochs, early stop patience={train.get('patience')} |
| Protocol | LORO (10 metros), {s.get('n_gpus')} GPUs |

**Result.** LORO mean macro-F1 = **{fmt(s.get('loro_macro_f1_mean'))}** ± {fmt(s.get('loro_macro_f1_std'))}.  
Pooled OOF F1 = {fmt(s.get('oof_macro_f1'))}.

```bash
.venv/bin/python -u scripts/train_week4_vit_100m_loro.py
.venv/bin/python -u scripts/build_week4_vit_100m_notebook.py
```
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## How patches are built

1. Reuse the week-4 10×10 grid cache (`X_grid_min5_*.npy`): 100×64 AE bands + occupancy.
2. Per held-out metro, load that fold's PLS-DA rotation and keep the first k scores
   (k = smallest width within 0.02 macro-F1 of full 64-D LR).
3. Occupied cells → patch tokens via a linear `k → d_model` projection.
4. Learnable positional embeddings on the 10×10 lattice + a CLS token.
5. `TransformerEncoder` with `src_key_padding_mask` so empty cells never attend.
6. Binary head on the CLS representation → $P(\\mathrm{{RES}})$.

Per-fold epoch logs: `outputs/week4_vit_100m/logs/<metro>.log`.
"""
        )
    )

    delta_mp = s.get("delta_vs_mean_pool_lr")
    delta_flat = s.get("delta_vs_flat_grid_lr")
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Vs other 100 m heads

![vs baselines](../../outputs/week4_vit_100m/figures/vit_vs_baselines.png)

| Model | LORO macro-F1 |
|---|---:|
| ViT (this notebook) | {fmt(s.get('loro_macro_f1_mean'))} |
| Mean-pool 64-D LR | {fmt(base.get('mean_pool_lr'))} |
| PLS-DA on mean-pool | {fmt(base.get('pls_mean_pool'))} |
| Flat 6500-D LR (grid) | {fmt(base.get('flat_grid_lr'))} |

Δ vs mean-pool LR: {fmt(delta_mp, 3)}. Δ vs flat grid LR: {fmt(delta_flat, 3)}.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Per-metro

![by metro](../../outputs/week4_vit_100m/figures/vit_f1_by_metro.png)

| Metro | macro-F1 | best epoch | n test |
|---|---:|---:|---:|
"""
            + "\n".join(metro_rows)
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## OOF confusion

![confusion](../../outputs/week4_vit_100m/figures/vit_oof_confusion.png)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Artifacts

| Path | Contents |
|---|---|
| `outputs/week4_vit_100m/checkpoints/fold_*.pt` | Per-fold weights + train μ/σ + history |
| `outputs/week4_vit_100m/oof_predictions.parquet` | OOF labels / probs |
| `outputs/week4_vit_100m/oof_cls_embeddings.npy` | OOF CLS vectors ({arch.get('d_model')}-D) |
| `outputs/week4_vit_100m/loro_metrics.csv` | Per-metro F1 |
| `outputs/week4_vit_100m/summary.json` | Manifest |

Elapsed ≈ {s.get('elapsed_s')} s.
"""
        )
    )

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
