#!/usr/bin/env python3
"""Map figures for the research deck (slides 3-4), drawn from repo datasets.

Run with the project venv:  .venv/bin/python presentation/make_figures.py
Outputs PNGs to presentation/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from shapely import wkt as swkt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Deck palette
NAVY, SLATE, TEAL, AMBER, GREY = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C", "#B8BEC6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": NAVY,
                     "axes.edgecolor": GREY, "axes.labelcolor": SLATE})

# Cambridge, MA (Boston_NE metro): dense mix of apartments, retail, offices, university
W = (-71.1000, 42.3600, -71.0950, 42.3640)
REGION = "Boston_NE"


def polys(wkts):
    out = []
    for w in wkts:
        g = swkt.loads(w)
        geoms = getattr(g, "geoms", [g])
        for gg in geoms:
            out.append(np.asarray(gg.exterior.coords))
    return out


def frame(ax):
    ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3])
    ax.set_aspect(1 / np.cos(np.radians((W[1] + W[3]) / 2)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("white")


def scalebar(ax, metres=100):
    lat = (W[1] + W[3]) / 2
    dlon = metres / (111_320 * np.cos(np.radians(lat)))
    x0, y0 = W[0] + 0.00015, W[1] + 0.00012
    ax.plot([x0, x0 + dlon], [y0, y0], color=NAVY, lw=2.5, solid_capstyle="butt")
    ax.text(x0 + dlon / 2, y0 + 0.00006, f"{metres} m", ha="center", va="bottom", fontsize=10, color=NAVY)


def in_window(df, lon, lat, pad=0.0004):
    return df[df[lon].between(W[0] - pad, W[2] + pad) & df[lat].between(W[1] - pad, W[3] + pad)]


# ---------- data ----------
b = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet",
                    columns=["id", "region", "lon", "lat", "wkt", "label", "y_res"])
b = in_window(b[b.region == REGION], "lon", "lat", pad=0.002)
p = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet",
                    columns=["pixel_wkt", "pixel_lon", "pixel_lat", "region", "frac_res_full", "frac_nres_full"])
p = in_window(p[p.region == REGION], "pixel_lon", "pixel_lat")
share = p.frac_res_full / (p.frac_res_full + p.frac_nres_full)
p = p.assign(share=share)
p = p[p.share.notna()]
pure_res = (p.frac_res_full > 0) & (p.frac_nres_full == 0)
pure_nres = (p.frac_nres_full > 0) & (p.frac_res_full == 0)
p = p.assign(stratum=np.where(pure_res, "res", np.where(pure_nres, "nres", "mixed")))

cmap = LinearSegmentedColormap.from_list("nres_res", [AMBER, "#EFE7DA", TEAL])

# ---------- Figure A: ground truth (footprints -> cell share) ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=200)
ax = axes[0]
frame(ax)
res_b, nres_b = b[b.y_res == 1], b[b.y_res == 0]
ax.add_collection(PolyCollection(polys(res_b.wkt), facecolor=TEAL, edgecolor="white", lw=0.6))
ax.add_collection(PolyCollection(polys(nres_b.wkt), facecolor=AMBER, edgecolor="white", lw=0.6))
ax.set_title("1  Tagged Overture buildings", loc="left", fontsize=14, fontweight="bold", color=NAVY)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=AMBER)],
          labels=["Residential", "Non-residential"], loc="upper right", frameon=True, fontsize=10)
scalebar(ax)

ax = axes[1]
frame(ax)
pc = PolyCollection(polys(p.pixel_wkt), array=p.share.values, cmap=cmap, clim=(0, 1), edgecolor="white", lw=0.15)
ax.add_collection(pc)
ax.add_collection(PolyCollection(polys(b.wkt), facecolor="none", edgecolor=NAVY, lw=0.5, alpha=0.6))
ax.set_title("2  Residential share of each 10 m cell", loc="left", fontsize=14, fontweight="bold", color=NAVY)
cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
cb.ax.set_yticklabels(["0 non-res", "0.5 mixed", "1 res"])
cb.outline.set_visible(False)
scalebar(ax)
fig.tight_layout()
fig.savefig(OUT / "gt_footprints_to_cells.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------- Figure B: balancing on the map (Phoenix: residential-heavy metro) ----------
mixed_note = f"mixed cells in Cambridge window: {int((p.stratum == 'mixed').sum())} of {len(p)}"
W = (-112.144, 33.5125, -112.140, 33.5160)
REGION = "Phoenix_SW"
full = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet",
                       columns=["pixel_wkt", "pixel_lon", "pixel_lat", "region", "frac_res_full", "frac_nres_full"])
counts_before = full.assign(
    res=(full.frac_res_full > 0) & (full.frac_nres_full == 0),
    nres=(full.frac_nres_full > 0) & (full.frac_res_full == 0),
).groupby("region")[["res", "nres"]].sum()
pw = in_window(full[full.region == REGION], "pixel_lon", "pixel_lat")
pw_res = pw[(pw.frac_res_full > 0) & (pw.frac_nres_full == 0)]
pw_nres = pw[(pw.frac_nres_full > 0) & (pw.frac_res_full == 0)]
bal = pd.read_parquet(ROOT / "datasets/week4_pixel_gt_balanced/pixels.parquet",
                      columns=["pixel_wkt", "pixel_lon", "pixel_lat", "region", "y"])
counts_after = bal.groupby(["region", "y"]).size().unstack()
bw = in_window(bal[bal.region == REGION], "pixel_lon", "pixel_lat")
bph = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet", columns=["region", "lon", "lat", "wkt"])
bph = in_window(bph[bph.region == REGION], "lon", "lat", pad=0.002)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), dpi=200)
for ax, title, r, n in [
    (axes[0], "Before: every pure cell", pw_res, pw_nres),
    (axes[1], "After: 1:1 per metro", bw[bw.y == 1], bw[bw.y == 0]),
]:
    frame(ax)
    ax.add_collection(PolyCollection(polys(bph.wkt), facecolor="none", edgecolor=GREY, lw=0.5))
    ax.add_collection(PolyCollection(polys(r.pixel_wkt), facecolor=TEAL, edgecolor="white", lw=0.15))
    ax.add_collection(PolyCollection(polys(n.pixel_wkt), facecolor=AMBER, edgecolor="white", lw=0.15))
    ax.set_title(f"{title}\n{len(r):,} residential · {len(n):,} non-residential", loc="left",
                 fontsize=13, fontweight="bold", color=NAVY)
    scalebar(ax)
fig.tight_layout()
fig.savefig(OUT / "balancing_before_after_map.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

tbl = pd.DataFrame({
    "res_before": counts_before.res.astype(int), "nres_before": counts_before.nres.astype(int),
    "per_class_after": counts_after[1].astype(int),
})
tbl.to_csv(OUT / "balancing_counts_by_metro.csv")
print(tbl)
print(mixed_note)
print("wrote", sorted(x.name for x in OUT.glob("*.png")))
