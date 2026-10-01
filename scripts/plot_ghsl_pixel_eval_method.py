#!/usr/bin/env python3
"""Visualize how GHSL pixel-level evaluation works.

Writes under outputs/validation/figures/:
  ghsl_pixel_eval_schematic.png
  ghsl_pixel_eval_real_chip.png
  ghsl_pixel_eval_agree_disagree.png
  ghsl_pixel_eval_center_vs_area.png
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from rasterio.transform import array_bounds, xy as transform_xy
from shapely.geometry import box, Point

from validation.io import load_config, load_gt_buildings
from validation.ghsl import open_ghsl, sample_window, map_ghsl_value
from validation.rasterize import rasterize_buildings_on_window
from validation.spatial import to_ghsl_crs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation" / "figures"


def _draw_flow(path: Path) -> None:
    steps = [
        ("1. GT footprints", "Overture Track-A\nRES / NRES polygons\n→ Mollweide (GHSL CRS)", "#e8f5e9", "#2e7d32"),
        ("2. Exact GHSL grid", "Same 10 m origin / size\nas GHS-BUILT-C FUN\n(no resampling of GHSL)", "#e3f2fd", "#1565c0"),
        ("3. Paint GT on grid", "Burn footprints with\nall_touched=False\n(center-in-polygon)", "#fff8e1", "#f9a825"),
        ("4. Keep eval cells", "Cell enters set if center\nis inside a footprint\n(config says ≥50%; code is binary)", "#fce4ec", "#c2185b"),
        ("5. Score", "Read GHSL code at cell\n1=RES, 2=NRES\nvs GT majority label → F1", "#f3e5f5", "#7b1fa2"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("GHSL pixel evaluation pipeline", fontsize=13, fontweight="bold", pad=8)
    w, h = 2.35, 2.4
    y = 0.7
    for i, (title, body, fc, ec) in enumerate(steps):
        x = 0.35 + i * 2.75
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=fc, edgecolor=ec, linewidth=1.5,
            )
        )
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=8.2, linespacing=1.25)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w + 0.32, y + h / 2),
                xytext=(x + w + 0.05, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4),
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def _pick_chip(gt: gpd.GeoDataFrame, aoi: str = "Boston_NE", pad: float = 90.0) -> gpd.GeoDataFrame:
    """Tight local window around a dense building neighborhood."""
    sub = gt[gt["aoi_id"] == aoi].copy()
    sub = sub[sub["label"].isin(["RES", "NRES"])]
    if sub.empty:
        raise ValueError(f"no buildings in {aoi}")
    seed = sub[sub["label"] == "NRES"]
    if seed.empty:
        seed = sub
    cents = seed.geometry.centroid
    sindex = sub.sindex
    best_i, best_n = None, -1
    for i, g in cents.items():
        hits = list(sindex.query(g.buffer(150.0), predicate="intersects"))
        if len(hits) > best_n:
            best_n, best_i = len(hits), i
    center = cents.loc[best_i]
    view = box(center.x - pad, center.y - pad, center.x + pad, center.y + pad)
    chip = sub[sub.intersects(view)].copy().reset_index(drop=True)
    if chip.empty:
        raise ValueError("empty chip after window")
    return chip


def _true_area_fraction(
    geoms, transform, shape, focus: np.ndarray | None = None
) -> np.ndarray:
    """Footprint coverage fraction per cell (only focus cells + 1-cell ring)."""
    from shapely.ops import unary_union

    h, w = shape
    frac = np.zeros((h, w), dtype=np.float64)
    union = unary_union([g for g in geoms if g is not None and not g.is_empty])
    if union.is_empty:
        return frac
    if focus is None:
        rows, cols = np.where(np.ones(shape, dtype=bool))
    else:
        try:
            from scipy import ndimage

            dil = ndimage.binary_dilation(focus.astype(bool), iterations=1)
        except Exception:
            dil = focus.astype(bool)
        rows, cols = np.where(dil)
    for r, c in zip(rows.tolist(), cols.tolist()):
        minx, maxy = transform * (c, r)
        maxx, miny = transform * (c + 1, r + 1)
        pix = box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))
        inter = pix.intersection(union)
        if not inter.is_empty:
            a = float(pix.area)
            if a > 0:
                frac[r, c] = float(inter.area) / a
    return frac


def plot_real_chip(cfg, gt_moll, path_chip: Path, path_agree: Path, path_center: Path) -> None:
    chip = _pick_chip(gt_moll, "Boston_NE", pad=90.0)
    # Fixed square window so GHSL cells are visible
    cx = float(chip.geometry.centroid.x.mean())
    cy = float(chip.geometry.centroid.y.mean())
    half = 100.0
    minx, miny, maxx, maxy = cx - half, cy - half, cx + half, cy + half
    view = box(minx, miny, maxx, maxy)
    chip = chip[chip.intersects(view)].copy()
    mosaic = open_ghsl(cfg)
    try:
        data, window, transform = sample_window(mosaic, minx, miny, maxx, maxy, pad_px=2)
        maj, frac_proxy, best_idx = rasterize_buildings_on_window(chip, "label", transform, data.shape, cfg)

        # GHSL labels mapped
        ghsl_lab = np.zeros(data.shape, dtype=np.int16)  # 0 empty/other, 1 RES, 2 NRES
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                lab = map_ghsl_value(int(data[r, c]), cfg)
                if lab == "RES":
                    ghsl_lab[r, c] = 1
                elif lab == "NRES":
                    ghsl_lab[r, c] = 2

        eval_mask = (maj == 1) | (maj == 2)
        agree = eval_mask & (maj == ghsl_lab)
        disagree = eval_mask & (ghsl_lab > 0) & (maj != ghsl_lab)
        ghsl_uncov = eval_mask & (ghsl_lab == 0)

        # --- panel: footprints + eval cells on GHSL ---
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
        extent = array_bounds(data.shape[0], data.shape[1], transform)

        # left: GHSL raw codes
        ax = axes[0]
        cmap_g = ListedColormap(["#f5f5f5", "#2ca02c", "#d62728", "#bbbbbb"])
        # map data: 0 nonbuilt, 1 res, 2 nres, else nodata-ish
        show = np.full(data.shape, 3, dtype=np.int16)
        show[data == 0] = 0
        show[data == 1] = 1
        show[data == 2] = 2
        ax.imshow(show, extent=_extent_xy(extent), origin="upper", cmap=cmap_g, vmin=0, vmax=3, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="black", linewidth=0.9)
        ax.set_title("GHSL Built-C FUN\n+ GT footprints", fontsize=10)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        # mid: GT paint (eval cells)
        ax = axes[1]
        show_gt = np.zeros(data.shape, dtype=np.int16)
        show_gt[maj == 1] = 1
        show_gt[maj == 2] = 2
        show_gt[maj == -1] = 3
        cmap_gt = ListedColormap(["#ffffff", "#2ca02c", "#d62728", "#9467bd"])
        ax.imshow(show_gt, extent=_extent_xy(extent), origin="upper", cmap=cmap_gt, vmin=0, vmax=3, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="black", linewidth=0.9)
        # mark centers of eval cells
        rs, cs = np.where(eval_mask)
        if len(rs):
            xs, ys = transform_xy(transform, rs, cs)
            ax.scatter(xs, ys, s=8, c="k", marker=".", zorder=5)
        ax.set_title("GT paint on GHSL grid\n(eval cells; dots = centers)", fontsize=10)

        # right: which cells are scored
        ax = axes[2]
        score = np.zeros(data.shape, dtype=np.int16)
        score[eval_mask] = 1
        ax.imshow(score, extent=_extent_xy(extent), origin="upper",
                  cmap=ListedColormap(["#f0f0f0", "#fdae61"]), vmin=0, vmax=1, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="black", linewidth=0.9)
        ax.set_title(f"Eval set (n={int(eval_mask.sum())})\ncenter inside footprint", fontsize=10)

        for ax in axes:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        handles = [
            mpatches.Patch(facecolor="#2ca02c", label="RES"),
            mpatches.Patch(facecolor="#d62728", label="NRES"),
            mpatches.Patch(facecolor="#fdae61", label="in pixel eval"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"GHSL pixel eval chip · {chip['aoi_id'].iloc[0]} (Mollweide meters)", fontsize=12, y=1.02)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(path_chip, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path_chip)

        # --- agree / disagree ---
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
        panels = [
            (axes[0], maj, "GT label on eval cells", ListedColormap(["#ffffff", "#2ca02c", "#d62728", "#9467bd"]), 3),
            (axes[1], ghsl_lab, "GHSL label (same cells)", ListedColormap(["#f5f5f5", "#2ca02c", "#d62728"]), 2),
            (None, None, None, None, None),
        ]
        ax = axes[0]
        show = maj.copy()
        show[~eval_mask] = 0
        ax.imshow(show, extent=_extent_xy(extent), origin="upper",
                  cmap=ListedColormap(["#ffffff", "#2ca02c", "#d62728", "#9467bd"]), vmin=0, vmax=3, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="k", linewidth=0.8)
        ax.set_title("GT on eval cells")

        ax = axes[1]
        show = ghsl_lab.copy()
        show[~eval_mask] = 0
        ax.imshow(show, extent=_extent_xy(extent), origin="upper",
                  cmap=ListedColormap(["#ffffff", "#2ca02c", "#d62728"]), vmin=0, vmax=2, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="k", linewidth=0.8)
        ax.set_title("GHSL on same cells")

        ax = axes[2]
        status = np.zeros(data.shape, dtype=np.int16)
        status[agree] = 1
        status[disagree] = 2
        status[ghsl_uncov] = 3
        ax.imshow(status, extent=_extent_xy(extent), origin="upper",
                  cmap=ListedColormap(["#ffffff", "#1b9e77", "#d95f02", "#7570b3"]),
                  vmin=0, vmax=3, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="k", linewidth=0.8)
        n_a, n_d, n_u = int(agree.sum()), int(disagree.sum()), int(ghsl_uncov.sum())
        ax.set_title(f"Agree {n_a} · Disagree {n_d}\nGHSL uncovered {n_u}")

        for ax in axes:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        handles = [
            mpatches.Patch(facecolor="#1b9e77", label="agree"),
            mpatches.Patch(facecolor="#d95f02", label="disagree"),
            mpatches.Patch(facecolor="#7570b3", label="GHSL nonbuilt/nodata"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("Functional comparison on the shared eval cells", fontsize=12, y=1.02)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(path_agree, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path_agree)

        # --- center proxy vs true area fraction ---
        true_frac = _true_area_fraction(
            list(chip.geometry), transform, data.shape, focus=eval_mask
        )
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))

        ax = axes[0]
        im = ax.imshow(true_frac, extent=_extent_xy(extent), origin="upper", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="white", linewidth=0.9)
        ax.set_title("True footprint area / cell area")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1]
        ax.imshow(frac_proxy, extent=_extent_xy(extent), origin="upper", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="white", linewidth=0.9)
        ax.set_title("Code proxy (0/1 center burn)")

        ax = axes[2]
        # cells in eval with true frac < 0.5
        edge = eval_mask & (true_frac < 0.5)
        show = np.zeros(data.shape, dtype=np.int16)
        show[eval_mask & (true_frac >= 0.5)] = 1
        show[edge] = 2
        ax.imshow(show, extent=_extent_xy(extent), origin="upper",
                  cmap=ListedColormap(["#f0f0f0", "#4c78a8", "#e45756"]), vmin=0, vmax=2, interpolation="nearest")
        chip.boundary.plot(ax=ax, color="k", linewidth=0.8)
        ax.set_title(
            f"Eval cells by true frac\n≥0.5: {int((eval_mask & (true_frac >= 0.5)).sum())}  "
            f"<0.5: {int(edge.sum())}"
        )

        for ax in axes:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        handles = [
            mpatches.Patch(facecolor="#4c78a8", label="eval & true frac ≥ 0.5"),
            mpatches.Patch(facecolor="#e45756", label="eval but true frac < 0.5"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(
            "Config says ≥50% overlap; implementation is center-in-polygon (binary)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        fig.savefig(path_center, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path_center)
    finally:
        mosaic.close()


def _extent_xy(bounds):
    # array_bounds returns (west, south, east, north)
    west, south, east, north = bounds
    return (west, east, south, north)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "validation" / "config.yaml")
    _draw_flow(OUT / "ghsl_pixel_eval_schematic.png")

    gt = load_gt_buildings(cfg["paths"]["gt_buildings"], cfg)
    gt_m = to_ghsl_crs(gt)
    plot_real_chip(
        cfg,
        gt_m,
        OUT / "ghsl_pixel_eval_real_chip.png",
        OUT / "ghsl_pixel_eval_agree_disagree.png",
        OUT / "ghsl_pixel_eval_center_vs_area.png",
    )


if __name__ == "__main__":
    main()
