#!/usr/bin/env python3
"""Build notebooks/week4/07_train_100m_tiles.ipynb from tile-direct train outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_100m_tile_train"
NB = ROOT / "notebooks" / "week4" / "07_train_100m_tiles.ipynb"


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    summary = jload(OUT / "summary.json")
    m5 = summary.get("min5") or jload(OUT / "summary_min5.json")
    m20 = summary.get("min20") or jload(OUT / "summary_min20.json")
    if not m5:
        raise SystemExit(f"Missing min5 summary under {OUT}")

    cells: list = []

    def md(s: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

    def code(s: str) -> None:
        cells.append(nbf.v4.new_code_cell(s.strip() + "\n"))

    pool = m5.get("cell_pool_baseline_f1_at_0.5")
    pool_s = f"{float(pool):.3f}" if pool is not None else "n/a"
    delta = m5.get("delta_f1_vs_cell_pool")
    delta_s = f"{float(delta):+.3f}" if delta is not None else "n/a"

    md(f"""
# Week 4: tile-direct 100 m train (10×10 AE grid)

Train on the **1:1 balanced 100 m tiles** (`tiles_balanced_min5/min20`), not on pooled cell scores.

**Input** per tile: up to 10×10 AlphaEarth 10 m cells (64-D each), missing slots zero-filled, plus a 10×10 occupancy mask → **6500-D**.

**Heads** (shared LORO folds):
1. Ridge multi-output on `(frac_res, frac_nres)`
2. Logistic regression on `y ∈ {{NRES, RES}}`

Compare to the cell-pool baseline that scores the same tiles with `mean(p_res) ≥ 0.50` from the 10 m LR (`outputs/week4_100m_tiles_balanced/`).

```bash
.venv/bin/python -u scripts/train_week4_100m_tiles_loro.py --min-cells 5
.venv/bin/python -u scripts/train_week4_100m_tiles_loro.py --min-cells 20
```

### min ≥ 5 (primary)

| Metric | Value |
|---|---:|
| Tiles | {int(m5['n_tiles']):,} |
| OOF macro-F1 (LR) | {float(m5['oof_macro_f1']):.3f} |
| LORO F1 mean ± std | {float(m5['loro_macro_f1_mean']):.3f} ± {float(m5['loro_macro_f1_std']):.3f} |
| OOF mean MAE (Ridge) | {float(m5['oof_mae_mean']):.3f} |
| Cell-pool F1 @ 0.50 | {pool_s} |
| Δ F1 (tile-direct − pool) | {delta_s} |
| Backend | {m5.get('backend', '—')} · {m5.get('n_gpus', 0)} GPUs / {m5.get('n_cpus', 0)} CPUs |
| Runtime | {m5.get('elapsed_s', '—')} s |
""")

    md("""
![LORO F1 by metro](../../outputs/week4_100m_tile_train/figures/loro_f1_by_region_min5.png)

![LORO MAE by metro](../../outputs/week4_100m_tile_train/figures/loro_mae_by_region_min5.png)

![OOF true vs pred](../../outputs/week4_100m_tile_train/figures/oof_true_vs_pred_min5.png)

![OOF confusion](../../outputs/week4_100m_tile_train/figures/oof_confusion_min5.png)
""")

    if m20:
        pool20 = m20.get("cell_pool_baseline_f1_at_0.5")
        pool20_s = f"{float(pool20):.3f}" if pool20 is not None else "n/a"
        d20 = m20.get("delta_f1_vs_cell_pool")
        d20_s = f"{float(d20):+.3f}" if d20 is not None else "n/a"
        md(f"""
### min ≥ 20

| Metric | Value |
|---|---:|
| Tiles | {int(m20['n_tiles']):,} |
| OOF macro-F1 (LR) | {float(m20['oof_macro_f1']):.3f} |
| LORO F1 mean ± std | {float(m20['loro_macro_f1_mean']):.3f} ± {float(m20['loro_macro_f1_std']):.3f} |
| OOF mean MAE (Ridge) | {float(m20['oof_mae_mean']):.3f} |
| Cell-pool F1 @ 0.50 | {pool20_s} |
| Δ F1 (tile-direct − pool) | {d20_s} |

![LORO F1 by metro, min≥20](../../outputs/week4_100m_tile_train/figures/loro_f1_by_region_min20.png)

![LORO MAE by metro, min≥20](../../outputs/week4_100m_tile_train/figures/loro_mae_by_region_min20.png)

![OOF true vs pred, min≥20](../../outputs/week4_100m_tile_train/figures/oof_true_vs_pred_min20.png)

![OOF confusion, min≥20](../../outputs/week4_100m_tile_train/figures/oof_confusion_min20.png)
""")

    code("""
from pathlib import Path
import json
import pandas as pd

ROOT = Path("../..").resolve()
OUT = ROOT / "outputs" / "week4_100m_tile_train"
summary = json.loads((OUT / "summary.json").read_text())
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "paths"} for k, v in summary.items()}, indent=2))
for tag in (5, 20):
    p = OUT / f"loro_metrics_min{tag}.csv"
    if p.exists():
        print(p.name)
        display(pd.read_csv(p))
""")

    md("""
## Paths

| Artifact | Path |
|---|---|
| OOF preds | `outputs/week4_100m_tile_train/oof_predictions_min{5,20}.parquet` |
| Per-fold metrics | `outputs/week4_100m_tile_train/loro_metrics_min{5,20}.csv` |
| Checkpoints | `outputs/week4_100m_tile_train/checkpoints/` |
| Figures | `outputs/week4_100m_tile_train/figures/` |
| Train script | `scripts/train_week4_100m_tiles_loro.py` |
""")

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
