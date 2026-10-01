#!/usr/bin/env python3
"""Generate notebooks/week3/02_pixel_ae_loro_classifier.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "week3" / "02_pixel_ae_loro_classifier.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# Week 3 — Pixel AE LORO residential classifier

**Goal.** Train a **binary residential vs non-residential** model at the **10 m AlphaEarth cell**,
using Stage 1 pixel GT. Mixed cells are excluded.

**Design.** [`docs/week3/pixel_ae_classifier_design.md`](../../docs/week3/pixel_ae_classifier_design.md)

**Data.** `datasets/week3_pixel_gt/pixels.parquet` — pure RES / NRES hard labels + `A00`–`A63`.
Coverage fractions (`frac_res` / `frac_nres`) are QA only, not features.

**Model.** `StandardScaler` + balanced `LogisticRegression` on the 64-D embedding (AlphaEarth-only).

**Protocol.** Leave-one-region-out (LORO) macro-F1 over 10 US metros.

On the pure subset, Kyle’s \(f_{res}\) is 0 or 1 and matches this hard label.
"""
        ),
        md(
            """
## Results (from last run — no retrain needed)

Saved under `outputs/week3_pixel_ae/`. Mean LORO macro-F1: **0.563 ± 0.082** (OOF 0.558). Mixed dropped: **137**.

| | |
|---|---:|
| Pure pixels | 283,119 |
| RES / NRES | 29,798 / 253,321 |
| Best region (macro-F1) | Phoenix_SW 0.714 |
| Worst region | Houston_S 0.423 |

**vs polygon (building) baselines** (same 10 metros, AE-only; week2 binary LORO):

| Unit / model | Mean LORO macro-F1 |
|---|---:|
| Building centroid LR | **0.696** |
| Building bbox CNN | **0.746** |
| Pixel AE LR (this notebook) | **0.563** |

Units differ (building vs 10 m cell) — compare directionally.

Re-run:

```bash
.venv/bin/python scripts/train_week3_pixel_ae_loro.py
.venv/bin/python scripts/compare_week3_polygon_vs_pixel.py
```

![LORO macro-F1 by region](../../outputs/week3_pixel_ae/figures/loro_macro_f1_by_region.png)

![OOF confusion](../../outputs/week3_pixel_ae/figures/oof_confusion.png)

![Polygon vs pixel by region](../../outputs/week3_pixel_ae/figures/polygon_vs_pixel_f1_by_region.png)

![Polygon vs pixel means](../../outputs/week3_pixel_ae/figures/polygon_vs_pixel_mean_f1.png)

![Polygon vs pixel scatter](../../outputs/week3_pixel_ae/figures/polygon_vs_pixel_scatter.png)

![Polygon − pixel Δ](../../outputs/week3_pixel_ae/figures/polygon_vs_pixel_delta.png)
"""
        ),
        md("## Setup"),
        code(
            r"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent
if not (ROOT / "requirements.txt").exists() and (ROOT.parent.parent / "requirements.txt").exists():
    ROOT = ROOT.parent.parent

PIXELS = ROOT / "datasets" / "week3_pixel_gt" / "pixels.parquet"
OUT = ROOT / "outputs" / "week3_pixel_ae"
BUILDING_BIN = ROOT / "outputs" / "week2_cnn_bbox" / "binary_loro_comparison.csv"
TRIAL_LORO = ROOT / "outputs" / "trial_residential" / "loro_metrics.csv"
BANDS = [f"A{i:02d}" for i in range(64)]
print("ROOT:", ROOT)
print("OUT exists:", OUT.exists())
"""
        ),
        md(
            """
## 1. Load pure pixels

Drop `mixed`. Map `residential` → `y=1`, `non_residential` → `y=0`.
"""
        ),
        code(
            r"""
raw = pd.read_parquet(PIXELS)
n_mixed = int((raw["label"] == "mixed").sum())
pure = raw.loc[raw["label"] != "mixed"].copy()
pure["y"] = (pure["label"] == "residential").astype(int)

print(f"raw={len(raw)}  mixed_dropped={n_mixed}  pure={len(pure)}")
print(pure["label"].value_counts())
print("by region:")
display(
    pure.groupby("region")["y"]
    .agg(n="size", res="sum")
    .assign(nres=lambda d: d["n"] - d["res"])
)
"""
        ),
        md(
            """
## 2. Load LORO results

Training lives in `scripts/train_week3_pixel_ae_loro.py` (threaded LORO over regions).
This section loads saved metrics / OOF preds.
"""
        ),
        code(
            r"""
summary = json.loads((OUT / "summary.json").read_text())
metrics = pd.read_csv(OUT / "loro_metrics.csv")
oof = pd.read_parquet(OUT / "oof_predictions.parquet")

print(
    f"LORO mean macro-F1 = {summary['loro_macro_f1_mean']:.3f} "
    f"± {summary['loro_macro_f1_std']:.3f}"
)
print(f"OOF macro-F1 = {summary['oof_macro_f1']:.3f}  acc = {summary['oof_accuracy']:.3f}")
print(f"mixed excluded = {summary['n_mixed_excluded']}")
display(metrics.sort_values("f1_macro", ascending=False))
"""
        ),
        md("## 3. Pixel-only figures"),
        code(
            r"""
from IPython.display import Image, display as idisplay

for name in ["loro_macro_f1_by_region.png", "oof_confusion.png"]:
    path = OUT / "figures" / name
    print(path.relative_to(ROOT))
    idisplay(Image(filename=str(path)))
"""
        ),
        md(
            """
## 4. Polygon (building) vs pixel comparison

Building baselines from week2 binary LORO on the same ~11k Overture buildings
(`outputs/week2_cnn_bbox/binary_loro_comparison.csv`):

- **Building centroid LR** — 64-D AE at footprint centroid (closest like-for-like head)
- **Building bbox CNN** — week2 patch CNN (stronger polygon model)
- **Pixel AE LR** — this notebook (10 m cell)

Also shown on the mean plot: notebook-02 trial building LR (~0.77) on a **balanced**
400-building/region subsample — easier setting, not the same sample as week2/week3.

Regenerate plots: `scripts/compare_week3_polygon_vs_pixel.py`
"""
        ),
        code(
            r"""
cmp_path = OUT / "polygon_vs_pixel_loro_comparison.csv"
if not cmp_path.exists():
    %run -i ../../scripts/compare_week3_polygon_vs_pixel.py

cmp = pd.read_csv(cmp_path)
display(cmp.sort_values("delta_centroid_minus_pixel", ascending=False))

pv = summary.get("polygon_vs_pixel", {})
if pv:
    print(
        f"means · building_centroid={pv['building_centroid_lr']:.3f}  "
        f"building_cnn={pv['building_bbox_cnn']:.3f}  "
        f"pixel={pv['pixel_ae_lr']:.3f}"
    )

for name in [
    "polygon_vs_pixel_f1_by_region.png",
    "polygon_vs_pixel_mean_f1.png",
    "polygon_vs_pixel_scatter.png",
    "polygon_vs_pixel_delta.png",
]:
    path = OUT / "figures" / name
    print(path.relative_to(ROOT))
    idisplay(Image(filename=str(path)))
"""
        ),
        md(
            """
## 5. Optional: retrain

Uncomment to re-run LORO + comparison from this notebook.
"""
        ),
        code(
            r"""
# %run ../../scripts/train_week3_pixel_ae_loro.py
# %run ../../scripts/compare_week3_polygon_vs_pixel.py
print("Training script:", ROOT / "scripts" / "train_week3_pixel_ae_loro.py")
print("Compare script:", ROOT / "scripts" / "compare_week3_polygon_vs_pixel.py")
print("Design doc:", ROOT / "docs" / "week3" / "pixel_ae_classifier_design.md")
"""
        ),
        md(
            """
## How to read this

| Piece | Meaning |
|---|---|
| Unit | 10 m AlphaEarth cell (not building) |
| Features | AE `A00`–`A63` only |
| Labels | Stage 1 pure RES/NRES; mixed dropped |
| Fractions | QA coverage; not in the model |
| Protocol | LORO macro-F1 (fills missing regions) |
| vs polygon | Building centroid LR / bbox CNN from week2 |

**Claim language.** AlphaEarth-only on high-precision Overture GT. Pixel F1 is lower than
building-level on the same metros — expected when the unit shrinks from footprint to 10 m cell
and class imbalance is harsher at pixel scale. Do not pool the two F1s as one scoreboard.
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
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
