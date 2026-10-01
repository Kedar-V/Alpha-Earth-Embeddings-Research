#!/usr/bin/env python3
"""Generate notebooks/week3/03_pixel_fraction_regression.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "week3" / "03_pixel_fraction_regression.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells = [
        md(
            """
# Week 3 — Pixel fraction regression + majority head

**Goal.** Predict continuous `frac_res` / `frac_nres` from **AlphaEarth embeddings only**,
then apply a **majority** post-process for RES / NRES / mixed.

**Design.** [`docs/week3/pixel_fraction_regression_design.md`](../../docs/week3/pixel_fraction_regression_design.md)

**Model.** `StandardScaler` + multi-output `Ridge` on `A00`–`A63`, LORO.
**Compute.** CPU-only (`ProcessPoolExecutor` over folds).

**Data.** All Stage-1 pixels including mixed (`datasets/week3_pixel_gt/pixels.parquet`).
"""
        ),
        md(
            """
## Results (from last run)

CPU-only Ridge LORO. Saved under `outputs/week3_fraction_regression/`.

| Metric | Value |
|---|---:|
| OOF MAE `frac_res` | **0.087** |
| OOF MAE `frac_nres` | **0.323** |
| Majority pure-binary LORO F1 | **0.561** |
| Stage-2 pixel LR F1 | 0.563 |

Re-run:

```bash
.venv/bin/python scripts/train_week3_fraction_regression_loro.py
```

![True vs pred](../../outputs/week3_fraction_regression/figures/oof_true_vs_pred_scatter.png)

![MAE by region](../../outputs/week3_fraction_regression/figures/loro_mae_by_region.png)

![Majority confusion](../../outputs/week3_fraction_regression/figures/oof_majority_confusion.png)

![vs Stage-2](../../outputs/week3_fraction_regression/figures/majority_vs_stage2_f1.png)
"""
        ),
        md("## Setup"),
        code(
            r"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent
if not (ROOT / "requirements.txt").exists() and (ROOT.parent.parent / "requirements.txt").exists():
    ROOT = ROOT.parent.parent

OUT = ROOT / "outputs" / "week3_fraction_regression"
print("ROOT:", ROOT)
print("OUT exists:", OUT.exists())
"""
        ),
        md("## Load metrics"),
        code(
            r"""
summary = json.loads((OUT / "summary.json").read_text())
reg = pd.read_csv(OUT / "loro_regression_metrics.csv")
maj = pd.read_csv(OUT / "loro_majority_metrics.csv")

print(f"GPUs used: {summary['n_gpus']} · {summary.get('gpu_names')}")
print(
    f"OOF MAE frac_res={summary['oof_mae_res']:.3f}  "
    f"frac_nres={summary['oof_mae_nres']:.3f}"
)
print(
    f"LORO pure-binary F1 (majority)={summary['loro_f1_macro_pure_binary_mean']:.3f}  "
    f"Stage2 pixel LR={summary.get('stage2_pixel_lr_loro_macro_f1_mean')}"
)
display(reg)
display(maj.sort_values("f1_macro_pure_binary", ascending=False))
"""
        ),
        md("## Figures"),
        code(
            r"""
for name in [
    "oof_true_vs_pred_scatter.png",
    "loro_mae_by_region.png",
    "oof_majority_confusion.png",
    "majority_vs_stage2_f1.png",
]:
    path = OUT / "figures" / name
    print(path.relative_to(ROOT))
    display(Image(filename=str(path)))
"""
        ),
        md(
            """
## How to read this

| Piece | Meaning |
|---|---|
| Features | AE `A00`–`A63` only |
| Targets | `frac_res_full`, `frac_nres_full` |
| Head | majority on predicted fractions |
| vs Stage 2 | pure-binary macro-F1 of majority vs pixel LogReg |

Retrain: `scripts/train_week3_fraction_regression_loro.py`
"""
        ),
    ]
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "AlphaEarth (.venv)",
                "language": "python",
                "name": "alphaearth",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
