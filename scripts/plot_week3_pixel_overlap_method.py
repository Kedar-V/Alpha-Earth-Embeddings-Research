#!/usr/bin/env python3
"""Visual explainers for how Week 3 pixels meet Overture footprints.

Writes under outputs/week3_pixel_3class/figures/:
  pixel_overlap_schematic.png   — frac_res / frac_nres / empty cartoon
  pixel_sampling_two_paths.png  — building-grid vs AOI empty sampling
  pixel_overlap_real_examples.png — real Overture overlays (partial + empty)
"""
from __future__ import annotations

import ast
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon as MplPolygon
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
PIX = ROOT / "datasets" / "week3_pixel_gt_v2" / "pixels.parquet"
BLD = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT = ROOT / "outputs" / "week3_pixel_3class" / "figures"


def parse_ids(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i) for i in x if str(i) not in ("", "nan", "None")]
    s = str(x).strip()
    if s in ("", "[]", "None", "nan"):
        return []
    if s.startswith("["):
        try:
            return [str(i) for i in ast.literal_eval(s)]
        except Exception:
            return []
    if ";" in s:
        return [p for p in s.split(";") if p]
    return [s]


def plot_schematic(path: Path) -> None:
    """Cartoon 10 m cell with RES / NRES footprint clips and empty remainder."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # Shared pixel square in data coords 0..1
    def draw_pixel(ax, title, res_poly, nres_poly, label, fr, fn, fe):
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="#222", linewidth=2.0, zorder=3)
        )
        ax.text(0.5, 1.12, title, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(0.5, -0.08, "10 m AlphaEarth cell", ha="center", va="top", fontsize=8, color="#555")

        if res_poly is not None:
            ax.add_patch(
                MplPolygon(res_poly, closed=True, facecolor="#2ca02c", alpha=0.55, edgecolor="#1a6b1a", lw=1.2)
            )
        if nres_poly is not None:
            ax.add_patch(
                MplPolygon(nres_poly, closed=True, facecolor="#d62728", alpha=0.55, edgecolor="#8a1a1a", lw=1.2)
            )
        # empty hatch over remainder hint
        if fe > 0.4:
            ax.add_patch(
                Rectangle((0, 0), 1, 1, fill=False, hatch="..", edgecolor="#999", linewidth=0, zorder=1)
            )

        ax.text(
            0.5,
            0.5,
            f"frac_res={fr:.2f}\nfrac_nres={fn:.2f}\nfrac_empty={fe:.2f}\n\nlabel = {label}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#ccc", alpha=0.92),
            zorder=5,
        )

    # Panel A: RES-only partial (edge clip)
    draw_pixel(
        axes[0],
        "A. Building edge clip",
        res_poly=[(0.55, 0.0), (1.0, 0.0), (1.0, 0.45), (0.7, 0.45)],
        nres_poly=None,
        label="residential",
        fr=0.12,
        fn=0.00,
        fe=0.88,
    )

    # Panel B: NRES dominant
    draw_pixel(
        axes[1],
        "B. Mostly covered NRES",
        res_poly=None,
        nres_poly=[(0.05, 0.1), (0.95, 0.1), (0.95, 0.9), (0.05, 0.9)],
        label="non_residential",
        fr=0.00,
        fn=0.72,
        fe=0.28,
    )

    # Panel C: true empty
    draw_pixel(
        axes[2],
        "C. True empty (no footprint)",
        res_poly=None,
        nres_poly=None,
        label="empty",
        fr=0.00,
        fn=0.00,
        fe=1.00,
    )

    handles = [
        mpatches.Patch(facecolor="#2ca02c", alpha=0.55, edgecolor="#1a6b1a", label="RES footprint ∩ cell"),
        mpatches.Patch(facecolor="#d62728", alpha=0.55, edgecolor="#8a1a1a", label="NRES footprint ∩ cell"),
        mpatches.Patch(facecolor="white", hatch="..", edgecolor="#999", label="empty remainder"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Coverage fractions = intersection area / 10 m cell area  (presence label if any overlap)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sampling_paths(path: Path) -> None:
    """Two-path cartoon: bbox-grid near buildings vs AOI empty reject sample."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    rng = np.random.default_rng(7)

    # Left: building-overlapped
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Path 1 — building-overlapped pixels", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888")

    # AOI box
    ax.add_patch(Rectangle((0.3, 0.3), 9.4, 9.4, fill=False, ls="--", color="#888", lw=1))
    ax.text(0.5, 9.5, "metro AOI", fontsize=8, color="#666")

    # Fake buildings
    bld_boxes = [
        (2.0, 6.5, 1.6, 1.2, "#2ca02c"),
        (5.5, 5.8, 1.4, 1.8, "#d62728"),
        (3.2, 2.5, 2.0, 1.0, "#2ca02c"),
        (7.0, 2.0, 1.5, 1.4, "#d62728"),
    ]
    cell = 0.45  # visual cell size; tiles with no gap (real AE cells are contiguous)
    for x, y, w, h, c in bld_boxes:
        pad = 0.15
        gx0, gy0 = x - pad, y - pad
        ncols = int(np.ceil((w + 2 * pad) / cell))
        nrows = int(np.ceil((h + 2 * pad) / cell))
        for i in range(ncols):
            for j in range(nrows):
                cx = gx0 + i * cell
                cy = gy0 + j * cell
                hit = not (cx + cell <= x or cx >= x + w or cy + cell <= y or cy >= y + h)
                face = "#fdae61" if hit else "#f0f0f0"
                ax.add_patch(
                    Rectangle((cx, cy), cell, cell, facecolor=face, edgecolor="#555", lw=0.4, alpha=0.95)
                )
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, alpha=0.45, edgecolor="#222", lw=1.2))

    ax.text(
        5,
        0.55,
        "Keep unique 10 m cells that intersect\n≥1 Track-A footprint → RES / NRES / mixed\n"
        "(cells tile with no gap; orange = kept)",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e6", edgecolor="#e0c070"),
    )

    # Right: AOI empty sample
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Path 2 — AOI-sampled empties", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888")
    ax.add_patch(Rectangle((0.3, 0.3), 9.4, 9.4, fill=False, ls="--", color="#888", lw=1))
    ax.text(0.5, 9.5, "metro AOI", fontsize=8, color="#666")

    for x, y, w, h, c in bld_boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=c, alpha=0.35, edgecolor="#333", lw=1))

    # scatter candidate points; reject near buildings
    xs = rng.uniform(0.8, 9.2, size=80)
    ys = rng.uniform(0.8, 9.2, size=80)
    kept_x, kept_y, rej_x, rej_y = [], [], [], []
    for x, y in zip(xs, ys):
        hit = False
        for bx, by, bw, bh, _ in bld_boxes:
            if bx - 0.15 <= x <= bx + bw + 0.15 and by - 0.15 <= y <= by + bh + 0.15:
                hit = True
                break
        if hit:
            rej_x.append(x)
            rej_y.append(y)
        else:
            kept_x.append(x)
            kept_y.append(y)

    ax.scatter(rej_x, rej_y, s=18, c="#bbbbbb", marker="x", linewidths=0.8, label="reject (hits footprint)", zorder=3)
    ax.scatter(kept_x, kept_y, s=28, c="#4c78a8", marker="s", edgecolors="#1f3a5f", linewidths=0.5, label="keep as empty", zorder=4)

    ax.text(
        5,
        0.55,
        "Uniform sample → snap to 10 m grid →\nreject if any footprint overlap\nCap: n_empty ≈ n_residential per metro",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f1fa", edgecolor="#7aa0c4"),
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    fig.suptitle("Two constructions in the same pixel table", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pick_empty_with_neighbors(sub: pd.DataFrame, bg: gpd.GeoDataFrame, window_m: float = 120.0):
    """Prefer an empty cell that has Track-A buildings nearby (context only)."""
    empties = sub[sub["label"] == "empty"]
    if empties.empty:
        return None
    best_row, best_n = None, -1
    sample = empties.sample(min(400, len(empties)), random_state=3)
    for _, erow in sample.iterrows():
        pix = shapely_wkt.loads(erow.pixel_wkt)
        lat = float(erow.pixel_lat)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
        dlon = window_m / m_per_deg_lon
        dlat = window_m / m_per_deg_lat
        minx, miny, maxx, maxy = pix.bounds
        view = box(minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)
        n = int(bg.intersects(view).sum())
        if n > best_n:
            best_n, best_row = n, erow
            if n >= 8:
                break
    return best_row


def plot_real_examples(path: Path) -> None:
    """Real Overture overlays: partial RES, covered NRES, AOI empty with context."""
    df = pd.read_parquet(PIX)
    bld = pd.read_parquet(BLD)
    bld["geometry"] = bld["wkt"].map(shapely_wkt.loads)
    bg = gpd.GeoDataFrame(bld, geometry="geometry", crs="EPSG:4326")
    id_to_row = bg.set_index(bg["id"].astype(str))

    sub = df[df["region"] == "Boston_NE"].copy()
    partial = sub[
        (sub["label"] == "residential")
        & (sub["frac_res_full"] > 0.05)
        & (sub["frac_res_full"] < 0.35)
        & (sub["frac_nres_full"] == 0)
    ].sort_values("frac_res_full")
    fullish = sub[
        (sub["label"] == "non_residential")
        & (sub["frac_nres_full"] > 0.55)
        & (sub["frac_res_full"] == 0)
    ].sort_values("frac_nres_full", ascending=False)
    empty_row = _pick_empty_with_neighbors(sub, bg)

    if partial.empty or fullish.empty or empty_row is None:
        partial = df[
            (df["label"] == "residential")
            & (df["frac_res_full"] > 0.05)
            & (df["frac_res_full"] < 0.35)
        ]
        fullish = df[(df["label"] == "non_residential") & (df["frac_nres_full"] > 0.55)]
        empty_row = _pick_empty_with_neighbors(df, bg) or df[df["label"] == "empty"].iloc[0]

    rows = [
        ("Partial RES overlap\n(still labeled residential)", partial.iloc[len(partial) // 3], 40.0),
        ("Mostly covered NRES", fullish.iloc[0], 40.0),
        (
            "AOI-sampled empty\n(no footprint on cell; buildings may sit nearby)",
            empty_row,
            90.0,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, (title, row, pad_m) in zip(axes, rows):
        pix = shapely_wkt.loads(row.pixel_wkt)
        lat = float(row.pixel_lat)
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
        dlon = pad_m / m_per_deg_lon
        dlat = pad_m / m_per_deg_lat
        minx, miny, maxx, maxy = pix.bounds
        view = box(minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)

        nearby = bg[bg.intersects(view)]
        for _, b in nearby.iterrows():
            color = "#2ca02c" if int(b["y_res"]) == 1 else "#d62728"
            gpd.GeoSeries([b.geometry], crs="EPSG:4326").plot(
                ax=ax, facecolor=color, edgecolor="#222", linewidth=0.8, alpha=0.55
            )

        for bid in parse_ids(row.get("building_ids")):
            if bid in id_to_row.index:
                g = id_to_row.loc[bid].geometry
                if isinstance(g, pd.Series):
                    g = g.iloc[0]
                gpd.GeoSeries([g], crs="EPSG:4326").plot(
                    ax=ax, facecolor="none", edgecolor="#000", linewidth=1.6, alpha=1.0
                )

        gpd.GeoSeries([pix], crs="EPSG:4326").plot(
            ax=ax, facecolor="#fdae61", edgecolor="#000", linewidth=2.0, alpha=0.35
        )

        ax.set_xlim(minx - dlon, maxx + dlon)
        ax.set_ylim(miny - dlat, maxy + dlat)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.text(
            0.02,
            0.02,
            f"frac_res={row.frac_res_full:.2f}  frac_nres={row.frac_nres_full:.2f}\n"
            f"frac_empty={row.frac_empty_full:.2f}  label={row.label}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.92),
        )

    handles = [
        mpatches.Patch(facecolor="#2ca02c", alpha=0.55, label="RES footprint"),
        mpatches.Patch(facecolor="#d62728", alpha=0.55, label="NRES footprint"),
        mpatches.Patch(facecolor="#fdae61", alpha=0.45, edgecolor="#000", label="10 m AE pixel"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Real Boston examples (Overture Track-A × AlphaEarth cell)", fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = OUT / "pixel_overlap_schematic.png"
    p2 = OUT / "pixel_sampling_two_paths.png"
    p3 = OUT / "pixel_overlap_real_examples.png"
    print("writing", p1, flush=True)
    plot_schematic(p1)
    print("writing", p2, flush=True)
    plot_sampling_paths(p2)
    print("writing", p3, flush=True)
    plot_real_examples(p3)
    print("done", flush=True)


if __name__ == "__main__":
    main()
