#!/usr/bin/env python3
"""Build 10 m AlphaEarth pixel fraction GT from Track-A Overture polygons.

For each unique AE cell that intersects ≥1 Track-A footprint:

  frac_res   = area(union(RES)  ∩ cell) / area(cell)
  frac_nres  = area(union(NRES) ∩ cell) / area(cell)
  frac_empty = clip(1 - frac_res - frac_nres, 0, 1)

No hard labels / τ thresholds — fractions only.

Writes datasets/week4_pixel_gt/{pixels.parquet, manifest.json, figures/}.

Usage (from repo root):
  .venv/bin/python scripts/build_week4_pixel_gt.py
  .venv/bin/python scripts/build_week4_pixel_gt.py --regions NYC_NE --limit-pixels 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
SRC_BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
SRC_CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
OUT_DIR = ROOT / "datasets" / "week4_pixel_gt"
FIG_DIR = OUT_DIR / "figures"

BANDS = [f"A{i:02d}" for i in range(64)]
LABEL_DEFINITION = "resi_def_v1_pixel_frac_v1"
LABEL_DOC = "docs/residential_definition_v1.md"
DESIGN_DOC = "docs/week4/pixel_eval_gt_design.md"


def pixel_id_from_wkt(pixel_wkt: str) -> str:
    return hashlib.sha1(pixel_wkt.encode("utf-8")).hexdigest()[:16]


def _process_region(payload: dict) -> pd.DataFrame:
    """Worker: unique pixels for one metro → coverage fractions."""
    region = payload["region"]
    buildings = pd.DataFrame(payload["buildings"])
    cells = pd.DataFrame(payload["cells"])
    crs = payload["crs"]

    if buildings.empty or cells.empty:
        return pd.DataFrame()

    b_geoms = []
    for w in buildings["wkt"].tolist():
        g = shapely_wkt.loads(w)
        if not g.is_valid:
            g = g.buffer(0)
        b_geoms.append(g)
    bg = gpd.GeoDataFrame(
        {
            "id": buildings["id"].astype(str).tolist(),
            "y_res": buildings["y_res"].astype(int).tolist(),
        },
        geometry=b_geoms,
        crs="EPSG:4326",
    ).to_crs(crs)

    pix = (
        cells.drop_duplicates("pixel_wkt", keep="first")
        .loc[:, ["pixel_wkt", "pixel_lon", "pixel_lat", "pixel_size_m", *BANDS]]
        .reset_index(drop=True)
    )
    p_geoms = [shapely_wkt.loads(w) for w in pix["pixel_wkt"].tolist()]
    pg = gpd.GeoDataFrame(pix, geometry=p_geoms, crs="EPSG:4326").to_crs(crs)

    sindex = bg.sindex
    rows: list[dict] = []

    for _, prow in pg.iterrows():
        pixel = prow.geometry
        if pixel is None or pixel.is_empty:
            continue
        pixel_area = float(pixel.area)
        if pixel_area <= 0:
            continue

        cand_idx = list(sindex.query(pixel, predicate="intersects"))
        if not cand_idx:
            continue

        hits = bg.iloc[cand_idx]
        res_parts = []
        nres_parts = []
        res_ids: list[str] = []
        nres_ids: list[str] = []
        for _, h in hits.iterrows():
            if not pixel.intersects(h.geometry):
                continue
            if int(h["y_res"]) == 1:
                res_parts.append(h.geometry)
                res_ids.append(str(h["id"]))
            else:
                nres_parts.append(h.geometry)
                nres_ids.append(str(h["id"]))

        if not res_parts and not nres_parts:
            continue

        area_res = (
            float(pixel.intersection(unary_union(res_parts)).area) if res_parts else 0.0
        )
        area_nres = (
            float(pixel.intersection(unary_union(nres_parts)).area) if nres_parts else 0.0
        )
        frac_res = float(np.clip(area_res / pixel_area, 0.0, 1.0))
        frac_nres = float(np.clip(area_nres / pixel_area, 0.0, 1.0))
        frac_empty = float(np.clip(1.0 - frac_res - frac_nres, 0.0, 1.0))

        row = {
            "pixel_id": pixel_id_from_wkt(str(prow["pixel_wkt"])),
            "pixel_wkt": str(prow["pixel_wkt"]),
            "pixel_lon": float(prow["pixel_lon"]),
            "pixel_lat": float(prow["pixel_lat"]),
            "region": region,
            "crs": crs,
            "pixel_size_m": float(prow["pixel_size_m"]),
            "pixel_area_m2": pixel_area,
            "area_res_m2": area_res,
            "area_nres_m2": area_nres,
            "frac_res": frac_res,
            "frac_nres": frac_nres,
            "frac_empty": frac_empty,
            "n_res_buildings": len(res_ids),
            "n_nres_buildings": len(nres_ids),
            "n_buildings": len(set(res_ids + nres_ids)),
            "building_ids": ";".join(sorted(set(res_ids + nres_ids))),
            "label_definition": LABEL_DEFINITION,
        }
        for b in BANDS:
            row[b] = float(prow[b])
        rows.append(row)

    return pd.DataFrame(rows)


def _region_payload(
    region: str,
    buildings: pd.DataFrame,
    cells: pd.DataFrame,
    limit_pixels: int | None,
) -> dict:
    b = buildings.loc[buildings["region"].astype(str) == region].copy()
    b = b.loc[b["y_res"].isin([0, 1])].copy()
    b["id"] = b["id"].astype(str)
    crs = str(b["crs"].mode().iloc[0]) if len(b) else "EPSG:4326"

    c = cells.loc[cells["id"].isin(set(b["id"]))].copy()
    c = c.drop_duplicates("pixel_wkt", keep="first")
    if limit_pixels is not None and len(c):
        c = c.iloc[:limit_pixels].copy()

    return {
        "region": region,
        "crs": crs,
        "buildings": b[["id", "wkt", "y_res"]].to_dict(orient="list"),
        "cells": c[
            ["pixel_wkt", "pixel_lon", "pixel_lat", "pixel_size_m", *BANDS]
        ].to_dict(orient="list"),
    }


def build(
    regions: list[str] | None = None,
    workers: int | None = None,
    limit_pixels: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    buildings = pd.read_parquet(SRC_BUILDINGS)
    buildings["id"] = buildings["id"].astype(str)
    buildings = buildings.loc[buildings["y_res"].isin([0, 1])].copy()

    cell_cols = ["id", "pixel_wkt", "pixel_lon", "pixel_lat", "pixel_size_m", *BANDS]
    cells = pd.read_parquet(SRC_CELLS, columns=cell_cols)
    cells["id"] = cells["id"].astype(str)

    all_regions = sorted(buildings["region"].astype(str).unique().tolist())
    if regions:
        all_regions = [r for r in all_regions if r in set(regions)]
    if not all_regions:
        raise ValueError("No regions to process")

    n_workers = workers or min(len(all_regions), os.cpu_count() or 1)
    print(
        f"Buildings (y_res∈{{0,1}}): {len(buildings):,} | "
        f"cell rows: {len(cells):,} | regions: {all_regions} | workers={n_workers}",
        flush=True,
    )

    payloads = [
        _region_payload(r, buildings, cells, limit_pixels) for r in all_regions
    ]
    del buildings, cells

    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_region, p): p["region"] for p in payloads}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="regions"):
            region = futs[fut]
            df = fut.result()
            print(f"  {region}: {len(df):,} pixels", flush=True)
            if len(df):
                parts.append(df)

    if not parts:
        raise RuntimeError("No pixels produced")

    out = pd.concat(parts, ignore_index=True)

    before = len(out)
    out = out.drop_duplicates("pixel_id", keep="first").reset_index(drop=True)
    if len(out) < before:
        print(f"Dropped {before - len(out)} cross-region duplicate pixel_ids", flush=True)

    out["frac_res_full"] = out["frac_res"].astype(float)
    out["frac_nres_full"] = out["frac_nres"].astype(float)
    out["frac_empty_full"] = out["frac_empty"].astype(float)
    out["frac_res"] = out["frac_res_full"].round(2)
    out["frac_nres"] = out["frac_nres_full"].round(2)
    out["frac_empty"] = out["frac_empty_full"].round(2)

    elapsed = time.time() - t0
    manifest = {
        "name": "week4_pixel_gt_v1",
        "purpose": (
            "Pixel-level coverage fractions (frac_res, frac_nres, frac_empty) "
            "at 10 m AlphaEarth cells from Overture Track-A footprints. "
            "No hard labels."
        ),
        "label_definition": LABEL_DEFINITION,
        "label_definition_doc": LABEL_DOC,
        "design_doc": DESIGN_DOC,
        "fraction_rule": {
            "frac_res": "area(union(RES footprints) ∩ cell) / cell.area",
            "frac_nres": "area(union(NRES footprints) ∩ cell) / cell.area",
            "frac_empty": "clip(1 - frac_res - frac_nres, 0, 1)",
            "note": "Rounded frac_* at 2 decimals; *_full keeps full precision.",
        },
        "y_res_rule": "resi_def_v1 Building GT: 1 iff label in {single_family, multi_family}",
        "geometry_method": (
            "per unique pixel: unary_union(RES footprints) ∩ pixel / pixel.area "
            "(and same for NRES); STRtree over all region GT buildings"
        ),
        "sources": {
            "buildings": str(SRC_BUILDINGS.relative_to(ROOT)),
            "cells": str(SRC_CELLS.relative_to(ROOT)),
        },
        "n_pixels": int(len(out)),
        "n_regions": int(out["region"].nunique()),
        "regions": sorted(out["region"].astype(str).unique().tolist()),
        "frac_res": {
            "mean": float(out["frac_res"].mean()),
            "median": float(out["frac_res"].median()),
        },
        "frac_nres": {
            "mean": float(out["frac_nres"].mean()),
            "median": float(out["frac_nres"].median()),
        },
        "frac_empty": {
            "mean": float(out["frac_empty"].mean()),
            "median": float(out["frac_empty"].median()),
        },
        "frac_means_by_region": (
            out.groupby("region")[["frac_res", "frac_nres", "frac_empty"]]
            .mean()
            .round(4)
            .to_dict(orient="index")
        ),
        "embedding": {
            "bands": BANDS,
            "dim": 64,
            "year": 2024,
            "product": "AlphaEarth Foundations Satellite Embedding V1",
        },
        "workers": n_workers,
        "limit_pixels": limit_pixels,
        "elapsed_s": round(elapsed, 1),
        "paths": {
            "pixels": str((OUT_DIR / "pixels.parquet").relative_to(ROOT)),
            "manifest": str((OUT_DIR / "manifest.json").relative_to(ROOT)),
            "figures": str(FIG_DIR.relative_to(ROOT)),
        },
    }
    return out, manifest


def write_qa_plots(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # Colorblind-safe: blue / orange / grey
    C_RES, C_NRES, C_EMPTY = "#0072B2", "#E69F00", "#7F7F7F"

    pure = ((df["frac_res"] >= 0.99) | (df["frac_nres"] >= 0.99)).mean()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    specs = [
        ("frac_res", C_RES, "Residential"),
        ("frac_nres", C_NRES, "Non-residential"),
        ("frac_empty", C_EMPTY, "Empty (uncovered)"),
    ]
    for ax, (col, color, title) in zip(axes, specs):
        ax.hist(df[col], bins=40, color=color, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Cell coverage (0–1)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Pixels")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(
        0.01, 0.98,
        f"Most cells are nearly pure or empty — only {100*(1-pure):.0f}% are mixed",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        f"Week 4 Overture→10 m AE coverage fractions (n={len(df):,}).",
        fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "frac_histograms.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    sc = ax.scatter(
        df["frac_res"], df["frac_nres"], c=df["frac_empty"],
        s=4, alpha=0.35, cmap="cividis_r", rasterized=True, vmin=0, vmax=1,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Empty fraction")
    ax.set_xlabel("Residential coverage")
    ax.set_ylabel("Non-residential coverage")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.text(
        0.01, 0.98,
        "Coverage lives on the RES–NRES–empty simplex (few mixed cells)",
        fontsize=11, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "frac_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    means = (
        df.groupby("region")[["frac_res", "frac_nres", "frac_empty"]]
        .mean()
        .sort_values("frac_empty", ascending=True)
    )
    means.index = [str(i).replace("_", " ") for i in means.index]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    means.plot(
        kind="barh", stacked=True, ax=ax,
        color=[C_RES, C_NRES, C_EMPTY], width=0.8, legend=False,
    )
    ax.set_xlabel("Mean coverage share")
    ax.set_xlim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Direct labels in title colors
    ax.text(0.02, 1.02, "residential", color=C_RES, fontsize=9, transform=ax.transAxes)
    ax.text(0.22, 1.02, "non-residential", color=C_NRES, fontsize=9, transform=ax.transAxes)
    ax.text(0.48, 1.02, "empty", color=C_EMPTY, fontsize=9, transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(
        0.01, 0.98,
        "Empty share dominates every metro — buildings are a minority of AE cells",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "frac_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    counts = df["region"].value_counts()
    counts.index = [str(i).replace("_", " ") for i in counts.index]
    counts = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(counts.index.astype(str), counts.values, color="#0072B2", height=0.75)
    for rect, v in zip(bars, counts.values):
        ax.text(v + max(counts.values) * 0.01, rect.get_y() + rect.get_height() / 2,
                f"{v:,}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Pixels")
    ax.set_xlim(0, max(counts.values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"Sample sizes range {counts.min():,}–{counts.max():,} pixels per metro",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "pixel_counts_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="*", default=None, help="Subset of metros")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--limit-pixels",
        type=int,
        default=None,
        help="Per-region unique-pixel cap (smoke test)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, manifest = build(
        regions=args.regions,
        workers=args.workers,
        limit_pixels=args.limit_pixels,
    )

    front = [
        "pixel_id",
        "pixel_wkt",
        "pixel_lon",
        "pixel_lat",
        "region",
        "crs",
        "pixel_size_m",
        "pixel_area_m2",
        "area_res_m2",
        "area_nres_m2",
        "frac_res",
        "frac_nres",
        "frac_empty",
        "frac_res_full",
        "frac_nres_full",
        "frac_empty_full",
        "n_res_buildings",
        "n_nres_buildings",
        "n_buildings",
        "building_ids",
        "label_definition",
    ]
    cols = front + BANDS
    df = df[cols]

    out_path = OUT_DIR / "pixels.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(df):,} rows)", flush=True)

    fig_paths = write_qa_plots(df, FIG_DIR)
    manifest["figures"] = [str(p.relative_to(ROOT)) for p in fig_paths]

    man_path = OUT_DIR / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {man_path}", flush=True)
    print(
        f"frac means: res={manifest['frac_res']['mean']:.3f} "
        f"nres={manifest['frac_nres']['mean']:.3f} "
        f"empty={manifest['frac_empty']['mean']:.3f} | "
        f"n={manifest['n_pixels']:,} | {manifest['elapsed_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
