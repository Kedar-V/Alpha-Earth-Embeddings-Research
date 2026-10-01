#!/usr/bin/env python3
"""Generate notebooks/week3/01_pixel_res_nres_dataset.ipynb.

Week 3 Stage 1 — pixel-level RES / NRES / mixed GT at 10 m AlphaEarth cells.
Heavy lift lives in scripts/build_week3_pixel_gt.py; this notebook documents
the label rule, runs (or reloads) the build, and embeds QA plots.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "week3" / "01_pixel_res_nres_dataset.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# Week 3 — Pixel RES / NRES / mixed GT dataset

**Stage 1 only:** build training labels at the **10 m AlphaEarth cell**. No model,
no dense AOI inference, no GHSL wiring.

**Label rule (presence-based mixed — not Track B τ = 0.7 \(f_{\\mathrm{res}}\))**

For each unique AE pixel that overlaps ≥1 Track-A GT Overture building
([`resi_def_v1`](../../docs/residential_definition_v1.md) Building GT, `y_res ∈ {0,1}`):

| Condition | `label` |
|---|---|
| Overlaps RES only | `residential` |
| Overlaps NRES only | `non_residential` |
| Overlaps both | `mixed` |

Coverage columns (pixel area fractions, clamped to \\([0,1]\\), **written at 2 decimals**):

- `frac_res` — fraction of the **pixel** covered by residential GT polygons
- `frac_nres` — fraction of the **pixel** covered by non-residential GT polygons

Hard `label` uses **pre-round** overlap (`frac > 0`), so tiny intersections are not
zeroed into the wrong class. Geometry uses **union-per-class then ∩ pixel** so
overlapping same-class footprints do not inflate fractions above 1.

Optional later: \(f_{\\mathrm{res}} \\approx \\mathrm{frac\\_res}/(\\mathrm{frac\\_res}+\\mathrm{frac\\_nres})\) when both > 0.

**Inputs (reuse — no EE / COG re-download)**

- `datasets/building_bbox_grid/buildings.parquet`
- `datasets/building_bbox_grid/building_cells.parquet` (AABB cells; do **not** trust
  `inside_polygon` for fractions — recompute intersection)

**Outputs** → `datasets/week3_pixel_gt/`
"""
        ),
        md("## 0. Setup"),
        code(
            r"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", lambda x: f"{x:0.4f}")

ROOT = Path(".").resolve()
for cand in [ROOT, *ROOT.parents]:
    if (cand / "docs" / "residential_definition_v1.md").exists():
        ROOT = cand
        break
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "datasets" / "week3_pixel_gt"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BANDS = [f"A{i:02d}" for i in range(64)]
LABEL_DEFINITION = "resi_def_v1_pixel_overlap_v1"

print("ROOT", ROOT)
print("OUT ", OUT)
print("CPUs", os.cpu_count())
"""
        ),
        md(
            """
## 1. Build (or reload) pixel GT

Calls `scripts/build_week3_pixel_gt.py` logic: parallel by region
(`ProcessPoolExecutor`, `os.cpu_count()` workers), STRtree over all Track-A
buildings in the metro, union-per-class ∩ pixel.

Set `REBUILD = True` to recompute; `False` reloads existing parquet.
"""
        ),
        code(
            r"""
REBUILD = not (OUT / "pixels.parquet").exists()
# REBUILD = True  # uncomment to force

if REBUILD:
    import build_week3_pixel_gt as pix

    df, manifest = pix.build(workers=os.cpu_count())
    front = [
        "pixel_id", "pixel_wkt", "pixel_lon", "pixel_lat", "region", "crs",
        "pixel_size_m", "pixel_area_m2", "area_res_m2", "area_nres_m2",
        "frac_res", "frac_nres", "frac_res_full", "frac_nres_full", "label",
        "n_res_buildings", "n_nres_buildings", "building_ids", "label_definition",
    ]
    df = df[front + BANDS]
    # Round already applied inside build(); write artifacts.
    df.to_parquet(OUT / "pixels.parquet", index=False)
    fig_paths = pix.write_qa_plots(df, FIG)
    manifest["figures"] = [str(p.relative_to(ROOT)) for p in fig_paths]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT / 'pixels.parquet'} ({len(df):,} rows)")
else:
    df = pd.read_parquet(OUT / "pixels.parquet")
    manifest = json.loads((OUT / "manifest.json").read_text())
    print(f"Loaded {OUT / 'pixels.parquet'} ({len(df):,} rows)")

display(Markdown(f"**label_definition:** `{manifest.get('label_definition')}`"))
display(pd.Series(manifest["label_counts"], name="n").to_frame())
print(f"mixed_rate = {manifest['mixed_rate']:.4f}  |  elapsed_s = {manifest.get('elapsed_s')}")
df.head(3)
"""
        ),
        md("## 2. Schema check"),
        code(
            r"""
assert set(df["label"].unique()) <= {"residential", "non_residential", "mixed"}
assert df["label_definition"].eq(LABEL_DEFINITION).all()
# Dropped AABB-only empties at build time; rounded fracs can still be 0.00 with label from full precision.
n_both_zero_full = ((df["frac_res_full"] == 0) & (df["frac_nres_full"] == 0)).sum()
assert n_both_zero_full == 0, n_both_zero_full
assert df["pixel_id"].is_unique
assert all(b in df.columns for b in BANDS)

print("rows", len(df))
print("regions", sorted(df["region"].unique().tolist()))
print(df.dtypes.head(20))
print("---")
print(df[["frac_res", "frac_nres", "frac_res_full", "frac_nres_full"]].describe())
"""
        ),
        md(
            """
## 3. QA

Plots are written under `datasets/week3_pixel_gt/figures/` by the builder so they
survive without re-running geometry.
"""
        ),
        code(
            r"""
for name in [
    "label_counts.png",
    "frac_histograms.png",
    "frac_scatter.png",
    "label_by_region.png",
]:
    path = FIG / name
    if not path.exists():
        import build_week3_pixel_gt as pix
        pix.write_qa_plots(df, FIG)
    display(Markdown(f"### `{name}`"))
    display(Image(filename=str(path)))

by_region = (
    df.groupby(["region", "label"]).size().unstack(fill_value=0)
    .reindex(columns=["residential", "non_residential", "mixed"], fill_value=0)
)
by_region["total"] = by_region.sum(axis=1)
by_region["mixed_rate"] = by_region["mixed"] / by_region["total"]
display(by_region.sort_values("total", ascending=False))

# Sanity: buildings per pixel
print("n_res_buildings:", df["n_res_buildings"].describe().to_dict())
print("n_nres_buildings:", df["n_nres_buildings"].describe().to_dict())
print(
    "pixels with ≥2 buildings:",
    int(((df["n_res_buildings"] + df["n_nres_buildings"]) >= 2).sum()),
)
"""
        ),
        md("## 4. Manifest snapshot"),
        code(
            r"""
display(Markdown("```json\n" + json.dumps(manifest, indent=2)[:2500] + "\n...\n```"))
print("paths:")
for k, v in manifest.get("paths", {}).items():
    print(f"  {k}: {v}")
"""
        ),
        md(
            """
## 5. Why are labels so imbalanced?

Not a bug in the pixel builder — two stacked effects from Track A:

1. **Building sampling is already NRES-heavy** (~2.7∶1 NRES∶RES buildings).
2. **NRES footprints are much larger**, so they dominate 10 m cells
   (median ~1.5k m² NRES vs ~280 m² RES → ~93% of footprint area / inside cells).

The pixel table is roughly **area-weighted over the Track A mix**, not
building-count-weighted. Single-family homes barely appear (median 1 inside
cell). Mixed is rare because RES and NRES rarely share the same 10 m cell.

For a balanced trainer later: subsample NRES pixels, class-weight, or upsample
RES — do not change the GT rule.
"""
        ),
        code(
            r"""
b = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet")
b = b.loc[b["y_res"].isin([0, 1])].copy()
b["class_res"] = np.where(b["y_res"] == 1, "RES", "NRES")

print("=== Building counts (Track A) ===")
bc = b["class_res"].value_counts()
display(bc.to_frame("n"))
print(f"ratio NRES/RES buildings: {bc['NRES'] / bc['RES']:.2f}")

print("\n=== Building footprint area (area_m2_approx) ===")
area = b.groupby("class_res")["area_m2_approx"].agg(["count", "median", "mean", "sum"])
display(area)
print(f"area share NRES: {area.loc['NRES', 'sum'] / area['sum'].sum():.3f}")

print("\n=== Cells per building ===")
for col in ["n_cells", "n_cells_inside_polygon"]:
    g = b.groupby("class_res")[col].agg(["median", "mean", "sum"])
    print(col)
    display(g)
    print(f"  share NRES: {g.loc['NRES', 'sum'] / g['sum'].sum():.3f}")

print("\n=== By study label (median footprint + inside cells) ===")
by_label = (
    b.groupby("label")
    .agg(
        n=("id", "size"),
        y_res=("y_res", "first"),
        med_area=("area_m2_approx", "median"),
        mean_area=("area_m2_approx", "mean"),
        med_cells_in=("n_cells_inside_polygon", "median"),
        sum_cells_in=("n_cells_inside_polygon", "sum"),
        sum_area=("area_m2_approx", "sum"),
    )
    .sort_values("sum_cells_in", ascending=False)
)
display(by_label)

print("\n=== Pixel labels ===")
pc = df["label"].value_counts()
display(pc.to_frame("n"))
print(
    f"ratio NRES/RES pixels: "
    f"{pc['non_residential'] / pc['residential']:.2f}"
)
print(
    f"nres share of covered area: "
    f"{df['area_nres_m2'].sum() / (df['area_res_m2'].sum() + df['area_nres_m2'].sum()):.3f}"
)

# Figure: building count vs pixel count vs footprint area
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
colors = {"RES": "#2ca02c", "NRES": "#d62728"}

axes[0].bar(["RES", "NRES"], [bc["RES"], bc["NRES"]], color=[colors["RES"], colors["NRES"]])
axes[0].set_title("Buildings")
axes[0].set_ylabel("count")

axes[1].bar(
    ["RES", "NRES"],
    [area.loc["RES", "sum"] / 1e6, area.loc["NRES", "sum"] / 1e6],
    color=[colors["RES"], colors["NRES"]],
)
axes[1].set_title("Footprint area")
axes[1].set_ylabel("10⁶ m²")

axes[2].bar(
    ["residential", "non_residential"],
    [pc["residential"], pc["non_residential"]],
    color=[colors["RES"], colors["NRES"]],
)
axes[2].set_title("Pixel labels")
axes[2].set_ylabel("pixels")
for ax in axes:
    ax.tick_params(axis="x", rotation=15)
fig.suptitle("Imbalance amplifies from buildings → area → pixels", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "imbalance_stack.png", dpi=120, bbox_inches="tight")
plt.show()
display(Markdown(f"Saved `{FIG / 'imbalance_stack.png'}`"))
"""
        ),
        md(
            """
## Notes / later stages (out of scope here)

- Trainer / LORO on these pixel rows
- Dense AOI inference beyond GT-covered pixels
- GHSL `pixel_eval` wiring
- Track B τ = 0.7 hard labels from \(f_{\\mathrm{res}}\) — different rule; do not mix metrics
"""
        ),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }})
    nbf.write(nb, OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
