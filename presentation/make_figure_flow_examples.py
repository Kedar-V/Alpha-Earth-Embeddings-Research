#!/usr/bin/env python3
"""Real-data companion to the GHSL-style feature flow diagram: one house and one shop in Denver.

For each building, the five diagram stages:
  1 Sentinel-2 summer-2018 mosaic (week-9 run) with Overture footprints on top
  2 the chip the week-9 code reads (bounding box, padded to >= 8 x 8 pixels), pixels shown at 10 m
  3 the measured values (from the texture-fixed week-9 feature table)
  4 the six cue scores, recomputed with the week-9 script's own ramps and calibrated texture-ratio range
  5 the model's held-out prediction (week-9 texture-fixed run, one metro held out) next to the truth
Run:  .venv/bin/python presentation/make_figure_flow_examples.py
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
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely import wkt as swkt
from shapely.ops import transform as stransform

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures" / "flow_examples_denver.png"
MOSAIC = ROOT / "outputs/week9_ae_s2_nres_loro/20261113_055624/artifacts/s2_mosaics/Denver_W_bgrn.tif"
FIXED = ROOT / "outputs/week9_ae_s2_nres_loro/20261117_044358/artifacts/buildings_oof_ae_s2.parquet"
NAVY, SLATE, TEAL, AMBER, BLUE, LIGHT = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C", "#1F6FA8", "#E4E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY, "axes.edgecolor": "#B8BEC6"})

spec = importlib.util.spec_from_file_location("w9", ROOT / "scripts/run_week9_ae_s2_nres_loro.py")
w9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w9)

EXAMPLES = [("f61b998e-b1fb-4dc1-b480-09ed165ad128", "House"),
            ("23edc293-d510-4953-b48a-ca6ebfafec43", "Shop")]

bld = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet",
                      columns=["id", "region", "lon", "lat", "wkt", "y_res", "class"])
bld = bld[bld.region == "Denver_W"]
feat = pd.read_parquet(FIXED).set_index("id")
ds = rasterio.open(MOSAIC)
to_utm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
proj = lambda g: stransform(lambda x, y, z=None: to_utm.transform(x, y), g)


def rgb(chip):
    a = np.stack([chip[2], chip[1], chip[0]], -1).astype(float)
    v = a[a > 0]
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
    return np.clip((a - lo) / (hi - lo), 0, 1) ** 0.8


def rings(gs):
    out = []
    for g in gs:
        for gg in getattr(g, "geoms", [g]):
            out.append(np.asarray(gg.exterior.coords))
    return out


def clamp(x):
    return float(np.clip(x, 0, 1)) if np.isfinite(x) else 0.0


fig = plt.figure(figsize=(11.9, 4.15), dpi=220)
col_x = [0.0, 0.2, 0.39, 0.6, 0.83]       # left edges for the five stages
col_w = [0.18, 0.17, 0.19, 0.2, 0.16]
row_y = [0.47, 0.035]; row_h = 0.355
heads = ["1  Inputs", "2  Read a chip", "3  Measure", "4  Score like GHSL", "5  Model"]
for x, h in zip(col_x, heads):
    fig.text(x + 0.005, 0.965, h, fontsize=11.5, fontweight="bold", color=NAVY, va="top")

for r, (bid, name) in enumerate(EXAMPLES):
    b = bld[bld.id == bid].iloc[0]; f = feat.loc[bid]
    g = proj(swkt.loads(b.wkt))
    y0 = row_y[r]
    fig.text(0.003, y0 + row_h + 0.015, f"{name} · {f.fp_area_m2:,.0f} m² footprint", fontsize=10.5,
             color=TEAL if b.y_res == 1 else AMBER, fontweight="bold", va="bottom")

    # 1 inputs: S2 + footprints, ~300 m box
    cx, cy = g.centroid.x, g.centroid.y; R = 150
    chip = ds.read(window=from_bounds(cx - R, cy - R, cx + R, cy + R, transform=ds.transform), boundless=True, fill_value=0)
    ax = fig.add_axes([col_x[0], y0, col_w[0] * 0.95, row_h])
    ax.imshow(rgb(chip), extent=(cx - R, cx + R, cy - R, cy + R), interpolation="nearest")
    near = bld[(bld.lon - b.lon).abs().lt(0.003) & (bld.lat - b.lat).abs().lt(0.003)]
    ng = [proj(swkt.loads(w)) for w in near.wkt]
    ax.add_collection(PolyCollection(rings([q for q, y in zip(ng, near.y_res) if y == 1]), facecolor="none", edgecolor=TEAL, lw=0.8))
    ax.add_collection(PolyCollection(rings([q for q, y in zip(ng, near.y_res) if y == 0]), facecolor="none", edgecolor=AMBER, lw=0.8))
    ax.add_collection(PolyCollection(rings([g]), facecolor="none", edgecolor="white", lw=2.2))
    ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R); ax.set_xticks([]); ax.set_yticks([])

    # 2 chip: same window logic as week 9
    minx, miny, maxx, maxy = g.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    if win.width < 8 or win.height < 8:
        pad = max(ds.res[0] * 8, (maxx - minx) / 2, (maxy - miny) / 2)
        minx, miny, maxx, maxy = cx - pad, cy - pad, cx + pad, cy + pad
        win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    c2 = ds.read(window=win, boundless=True, fill_value=0)
    ax = fig.add_axes([col_x[1], y0, col_w[1] * 0.95, row_h])
    ax.imshow(rgb(c2), extent=(minx, maxx, miny, maxy), interpolation="nearest")
    ax.add_collection(PolyCollection(rings([g]), facecolor="none", edgecolor="white", lw=2.2))
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.03, 0.03, f"{c2.shape[2]} × {c2.shape[1]} px", transform=ax.transAxes, fontsize=9, color="white",
            fontweight="bold", va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc=NAVY, ec="none"))

    # 3 measured values
    lines = [("Brightness", f"{f.s2_brightness:,.0f}"), ("Blue share", f"{f.s2_blue_dom:.2f}"), ("NDVI", f"{f.s2_ndvi:.2f}"),
             ("Fine texture", f"{f.s2_contrast_small:,.0f}"), ("Texture ratio", f"{f.s2_contrast_ratio:.2f}"),
             ("Diameter", f"{f.fp_equiv_diam_m:.0f} m"), ("Compactness", f"{f.fp_compactness:.2f}")]
    for k, (lab, val) in enumerate(lines):
        yy = y0 + row_h - 0.02 - k * row_h / 7.2
        fig.text(col_x[2] + 0.005, yy, lab, fontsize=9.5, color=SLATE, va="top")
        fig.text(col_x[2] + col_w[2] - 0.01, yy, val, fontsize=9.5, color=NAVY, fontweight="bold", va="top", ha="right")

    # 4 cue bars with the week-9 ramps
    lo, hi = w9.CONTRAST_RATIO_LO, w9.CONTRAST_RATIO_HI
    cues = [("Bright", clamp((f.s2_brightness - 800) / 2500)), ("Bluish", clamp((f.s2_blue_dom - 0.30) / 0.25)),
            ("Smooth", clamp(1 - f.s2_contrast_small / 400)), ("Coarse", clamp((f.s2_contrast_ratio - lo) / (hi - lo))),
            ("Large", clamp((f.fp_equiv_diam_m - 20) / 60)), ("Compact", clamp((f.fp_compactness - 0.4) / 0.5))]
    score = float(np.mean([v for _, v in cues]))
    ax = fig.add_axes([col_x[3] + 0.055, y0, col_w[3] - 0.065, row_h])
    names = [c[0] for c in cues] + ["Score"]; vals = [c[1] for c in cues] + [score]
    ax.barh(range(7), vals, color=[BLUE] * 6 + [NAVY], height=0.7)
    for k, v in enumerate(vals):
        ax.text(v + 0.03, k, f"{v:.2f}", va="center", fontsize=8.5, color=NAVY)
    ax.set_yticks(range(7)); ax.set_yticklabels(names, fontsize=9); ax.set_ylim(6.6, -0.6)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, 1]); ax.tick_params(labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    assert abs(score - f.s2_nres_score) < 0.01, (score, f.s2_nres_score)   # matches the stored week-9 score

    # 5 model output vs truth
    ax = fig.add_axes([col_x[4] + 0.04, y0 + 0.05, col_w[4] - 0.04, row_h - 0.1])
    ax.barh([0, 1], [f.s2_frac_res, f.s2_frac_nres], color=[TEAL, AMBER], height=0.55)
    ax.scatter([f.soft_gt_frac_res, f.soft_gt_frac_nres], [0, 1], marker="|", s=320, linewidths=2.5, color=NAVY, zorder=3)
    for k, v in enumerate([f.s2_frac_res, f.s2_frac_nres]):
        if v > 0.45:   # long bar: label inside so the truth tick stays readable
            ax.text(0.03, k, f"{v:.0%}", va="center", ha="left", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(v + 0.03, k, f"{v:.0%}", va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Res.", "Non-res."], fontsize=9); ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, 1.15); ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "100%"], fontsize=8); ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.text(0.83, 0.005, "bar = predicted share · | = truth", fontsize=8.5, color=SLATE, va="bottom")
fig.text(0.0, 0.005, "white = this building · teal / amber = other residential / non-residential footprints", fontsize=8.5,
         color=SLATE, va="bottom")
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
for bid, name in EXAMPLES:
    f = feat.loc[bid]
    print(name, "score", round(f.s2_nres_score, 3), "pred res/nres", round(f.s2_frac_res, 2), round(f.s2_frac_nres, 2),
          "truth", round(f.soft_gt_frac_res, 2), round(f.soft_gt_frac_nres, 2))
