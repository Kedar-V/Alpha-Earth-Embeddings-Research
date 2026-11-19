#!/usr/bin/env python3
"""Sentinel-2 figure for the GHSL-heuristics slide.

Left: Sentinel-2 true colour (summer 2018 mosaic from the week-9 run) with tagged Overture buildings.
Right: each building filled with the week-9 fuzzy NRES score, recomputed with the week-9 script's own
feature function on the same mosaic, so the values match what the model was given.

Run:  .venv/bin/python presentation/make_figure_s2.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely import wkt as swkt
from shapely.ops import transform as stransform

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RUN = ROOT / "outputs/week9_ae_s2_nres_loro/20261113_055624/artifacts"
NAVY, SLATE, TEAL, AMBER, GREY = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C", "#B8BEC6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": NAVY})

spec = importlib.util.spec_from_file_location("w9", ROOT / "scripts/run_week9_ae_s2_nres_loro.py")
w9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w9)

REGION = "Denver_W"
W = (-105.048, 39.770, -105.042, 39.775)

ds = rasterio.open(RUN / "s2_mosaics" / f"{REGION}_bgrn.tif")
to_utm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
xs, ys = to_utm.transform([W[0], W[2]], [W[1], W[3]])
ext = (min(xs), max(xs), min(ys), max(ys))
chip = ds.read(window=from_bounds(ext[0], ext[2], ext[1], ext[3], transform=ds.transform), boundless=True, fill_value=0)
rgb = np.stack([chip[2], chip[1], chip[0]], -1)
valid = rgb[rgb > 0]
lo, hi = np.percentile(valid, [2, 98])
rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8

b = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet",
                    columns=["id", "region", "lon", "lat", "wkt", "y_res", "area_m2_approx"])
b = b[(b.region == REGION) & b.lon.between(W[0] - 0.0005, W[2] + 0.0005) & b.lat.between(W[1] - 0.0005, W[3] + 0.0005)]
proj = lambda g: stransform(lambda x, y, z=None: to_utm.transform(x, y), g)

geoms, scores = [], []
for row in b.itertuples(index=False):
    g = proj(swkt.loads(row.wkt))
    area = float(row.area_m2_approx) if np.isfinite(row.area_m2_approx) and row.area_m2_approx > 0 else float(g.area)
    compact = 4 * np.pi * g.area / (g.length ** 2) if g.length > 0 else np.nan
    minx, miny, maxx, maxy = g.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    if win.width < 8 or win.height < 8:  # same padding rule as week 9
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        pad = max(ds.res[0] * 8, (maxx - minx) / 2, (maxy - miny) / 2)
        win = from_bounds(cx - pad, cy - pad, cx + pad, cy + pad, transform=ds.transform)
    c = ds.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    f = w9._features_from_chip(c[0], c[1], c[2], c[3], area_m2=area, compactness=compact) if (c[2] > 0).sum() >= 4 else None
    geoms.append(g)
    scores.append(f["s2_nres_score"] if f else np.nan)
b = b.assign(score=scores)


def rings(gs):
    out = []
    for g in gs:
        for gg in getattr(g, "geoms", [g]):
            out.append(np.asarray(gg.exterior.coords))
    return out


fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=200)
ax = axes[0]
ax.imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
ax.add_collection(PolyCollection(rings([g for g, y in zip(geoms, b.y_res) if y == 1]), facecolor="none", edgecolor=TEAL, lw=1.3))
ax.add_collection(PolyCollection(rings([g for g, y in zip(geoms, b.y_res) if y == 0]), facecolor="none", edgecolor=AMBER, lw=1.8))
ax.set_title("Sentinel-2, summer 2018 (10 m pixels)", loc="left", fontsize=13, fontweight="bold")
ax.legend(handles=[plt.Line2D([], [], color=TEAL, lw=2.5), plt.Line2D([], [], color=AMBER, lw=2.5)],
          labels=["Residential", "Non-residential"], loc="lower right", fontsize=10, framealpha=0.92)

ax = axes[1]
ax.set_facecolor("#F4F4F4")
cmap = LinearSegmentedColormap.from_list("score", ["#E6F2F4", TEAL, NAVY])
ok = np.isfinite(b.score.to_numpy())
pc = PolyCollection(rings([g for g, k in zip(geoms, ok) if k]), array=b.score.to_numpy()[ok], cmap=cmap,
                    clim=(0.10, 0.45), edgecolor="white", lw=0.4)
ax.add_collection(pc)
ax.set_title("GHSL-style non-residential score (0–1)", loc="left", fontsize=13, fontweight="bold")
cb = fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02, ticks=[0.10, 0.275, 0.45])
cb.ax.set_yticklabels(["0.10 house-like", "0.28", "0.45+ shop or warehouse"]); cb.outline.set_visible(False)

for i, ax in enumerate(axes):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    col = "white" if i == 0 else NAVY
    ax.plot([ext[0] + 25, ext[0] + 125], [ext[2] + 22] * 2, color=col, lw=3, solid_capstyle="butt")
    ax.text(ext[0] + 75, ext[2] + 32, "100 m", ha="center", fontsize=10, fontweight="bold", color=col)
fig.tight_layout()
fig.savefig(OUT / "s2_heuristics_denver.png", facecolor="white", bbox_inches="tight")
print("buildings", len(b), "scored", int(ok.sum()), "median score by y_res:", b.groupby("y_res").score.median().round(3).to_dict())
