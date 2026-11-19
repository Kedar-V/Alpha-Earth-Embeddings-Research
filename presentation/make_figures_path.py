#!/usr/bin/env python3
"""Research-path figures for the deck (slides 7-11), drawn from repo data.

Run:  .venv/bin/python presentation/make_figures_path.py
Writes PNGs to presentation/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from shapely import wkt as swkt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
NAVY, SLATE, TEAL, AMBER, GREY = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C", "#B8BEC6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": NAVY,
                     "axes.edgecolor": GREY, "axes.labelcolor": SLATE, "xtick.color": SLATE, "ytick.color": SLATE})
A = [f"A{i:02d}" for i in range(64)]

W = (-71.1000, 42.3600, -71.0950, 42.3640)  # Cambridge, MA (Boston_NE)
REGION = "Boston_NE"
COSLAT = np.cos(np.radians((W[1] + W[3]) / 2))


def polys(wkts):
    out = []
    for w in wkts:
        g = swkt.loads(w)
        for gg in getattr(g, "geoms", [g]):
            out.append(np.asarray(gg.exterior.coords))
    return out


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------- load window data ----------
pad = 0.0025
bld = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet",
                      columns=["id", "region", "lon", "lat", "wkt", "y_res", "n_rows", "n_cols", "area_m2_approx"])
bld = bld[(bld.region == REGION) & bld.lon.between(W[0] - pad, W[2] + pad) & bld.lat.between(W[1] - pad, W[3] + pad)]
cells = ds.dataset(ROOT / "datasets/building_bbox_grid/building_cells.parquet").to_table(
    columns=["id", "i", "j", "pixel_lon", "pixel_lat", "pixel_wkt", "inside_polygon"] + A,
    filter=ds.field("id").isin(bld.id.tolist())).to_pandas()
uniq = cells.drop_duplicates(["pixel_lon", "pixel_lat"]).copy()

# PCA on window cells -> RGB
X = uniq[A].to_numpy(np.float64)
mu = X.mean(0)
U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
pcs = Vt[:3]


def to_rgb(arr):
    z = (arr - mu) @ pcs.T
    lo, hi = np.percentile((X - mu) @ pcs.T, [2, 98], axis=0)
    return np.clip((z - lo) / (hi - lo), 0, 1)


uniq["rgb"] = list(to_rgb(X))

# ---------- (a) AlphaEarth as colour ----------
fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=200)
inwin = uniq[uniq.pixel_lon.between(W[0], W[2]) & uniq.pixel_lat.between(W[1], W[3])]
ax.add_collection(PolyCollection(polys(inwin.pixel_wkt), facecolors=np.stack(inwin.rgb), edgecolor="none"))
ax.add_collection(PolyCollection(polys(bld.wkt), facecolor="none", edgecolor="white", lw=0.7))
ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3]); ax.set_aspect(1 / COSLAT); clean(ax)
lat = (W[1] + W[3]) / 2
dlon = 100 / (111_320 * COSLAT)
ax.plot([W[0] + 0.00015, W[0] + 0.00015 + dlon], [W[1] + 0.00012] * 2, color="white", lw=3, solid_capstyle="butt")
ax.text(W[0] + 0.00015 + dlon / 2, W[1] + 0.00018, "100 m", color="white", ha="center", fontsize=10, fontweight="bold")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "ae_rgb_cambridge.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------- (c) one building as model input: CNN vs ViT ----------
cand = bld[(bld.n_rows.between(11, 16)) & (bld.n_cols.between(11, 16)) & (bld.y_res == 0)
           & bld.lon.between(W[0], W[2]) & bld.lat.between(W[1], W[3])]
bid = cand.sort_values("area_m2_approx", ascending=False).id.iloc[0]
bc = cells[cells.id == bid]
nr, nc = int(bc.i.max()) + 1, int(bc.j.max()) + 1
P = 16
grid = np.ones((P, P, 3))
mask = np.zeros((P, P), bool)
oi, oj = (P - nr) // 2, (P - nc) // 2
rgb = to_rgb(bc[A].to_numpy(np.float64))
for (i, j, ins), c in zip(bc[["i", "j", "inside_polygon"]].itertuples(index=False), rgb):
    grid[oi + i, oj + j] = c
    mask[oi + i, oj + j] = bool(ins)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
for k, ax in enumerate(axes):
    ax.imshow(grid, interpolation="nearest")
    # footprint mask outline
    for i in range(P):
        for j in range(P):
            if mask[i, j]:
                for di, dj, xs, ys in [(-1, 0, [j - .5, j + .5], [i - .5, i - .5]), (1, 0, [j - .5, j + .5], [i + .5, i + .5]),
                                       (0, -1, [j - .5, j - .5], [i - .5, i + .5]), (0, 1, [j + .5, j + .5], [i - .5, i + .5])]:
                    ii, jj = i + di, j + dj
                    if not (0 <= ii < P and 0 <= jj < P and mask[ii, jj]):
                        ax.plot(xs, ys, color="white", lw=2.2)
    clean(ax)
    if k == 0:
        ax.add_patch(Rectangle((2.5, 2.5), 3, 3, fill=False, edgecolor=AMBER, lw=3))
        ax.annotate("", xy=(9.2, 4), xytext=(6.0, 4),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.5))
        ax.set_title("CNN: a small filter slides over the grid", loc="left", fontsize=13, fontweight="bold")
    else:
        for t in range(0, P + 1, 4):
            ax.axhline(t - .5, color=NAVY, lw=2); ax.axvline(t - .5, color=NAVY, lw=2)
        for n in range(16):
            r, c = divmod(n, 4)
            ax.text(c * 4 + 1.5, r * 4 + 1.5, str(n + 1), ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25", fc=NAVY, ec="none"))
        ax.set_title("ViT: 16 patches, each compared with all others", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlim(-.5, P - .5); ax.set_ylim(P - .5, -.5)
fig.tight_layout()
fig.savefig(OUT / "building_input_cnn_vit.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------- (d) neighbourhood rings ----------
mid = bld[bld.lon.between(W[0] + 0.0018, W[2] - 0.0018) & bld.lat.between(W[1] + 0.0013, W[3] - 0.0013)]
centre = mid.sort_values("area_m2_approx", ascending=False).iloc[0]
R = 200
dlat = R / 111_320
dlon_r = R / (111_320 * COSLAT)
V = (centre.lon - dlon_r * 1.12, centre.lat - dlat * 1.12, centre.lon + dlon_r * 1.12, centre.lat + dlat * 1.12)
near = uniq[uniq.pixel_lon.between(V[0], V[2]) & uniq.pixel_lat.between(V[1], V[3])]
fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=200)
ax.add_collection(PolyCollection(polys(near.pixel_wkt), facecolors=np.stack(near.rgb), edgecolor="none", alpha=0.85))
ax.add_collection(PolyCollection(polys([centre.wkt]), facecolor="none", edgecolor="white", lw=2.5))
for r, col in [(50, "white"), (100, AMBER), (200, NAVY)]:
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((centre.lon, centre.lat), 2 * r / (111_320 * COSLAT), 2 * r / 111_320,
                         fill=False, edgecolor=col, lw=3))
    ax.text(centre.lon + r / (111_320 * COSLAT) * 0.72, centre.lat + r / 111_320 * 0.72, f"{r} m",
            color="white", fontsize=11, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=col if col != "white" else SLATE, ec="none"))
ax.set_xlim(V[0], V[2]); ax.set_ylim(V[1], V[3]); ax.set_aspect(1 / COSLAT); clean(ax)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "neighbourhood_rings.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------- (b) metro-to-metro transfer: 64-D vs PLS-2 ----------
m = pd.read_csv(ROOT / "week4/geo_subspace_analysis/results/transfer/metrics.csv")
order = ["Atlanta_SE", "Boston_NE", "Chicago_MW", "Denver_W", "Houston_S", "LA_W", "NYC_NE", "Phoenix_SW", "SF_W", "Seattle_NW"]
short = [o.split("_")[0] for o in order]
cmap = LinearSegmentedColormap.from_list("deck", ["#FFFFFF", "#CFE5E8", TEAL, NAVY])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=200, gridspec_kw={"wspace": 0.12})
for n_ax, (ax, (model, k, label)) in enumerate(zip(axes, [("lr64", 64, "Full 64-D"), ("pls", 2, "PLS, 2 dimensions")])):
    sub = m[(m.model == model) & (m.k == k)]
    M = sub.pivot(index="source", columns="target", values="macro_f1").loc[order, order].to_numpy()
    off = M[~np.eye(10, dtype=bool)].mean()
    im = ax.imshow(M, cmap=cmap, vmin=0.3, vmax=0.95)
    ax.set_xticks(range(10)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(10)); ax.set_yticklabels(short, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{label} · other metros {off:.2f}", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tested on", fontsize=11)
    if n_ax == 0:
        ax.set_ylabel("Trained on", fontsize=11)
    else:
        ax.set_yticklabels([])
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Macro-F1", color=SLATE); cb.outline.set_visible(False)
fig.savefig(OUT / "transfer_64d_vs_pls2.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

print("building used for CNN/ViT figure:", bid, "grid", nr, "x", nc)
print("ring centre:", centre.id)
print("wrote", sorted(x.name for x in OUT.glob("*.png")))
