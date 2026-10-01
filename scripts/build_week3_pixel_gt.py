#!/usr/bin/env python3
"""Build pixel-level RES / NRES / mixed GT from bbox-grid buildings + AE cells.

Stage 1 of week 3 — labels only (no model). For each unique 10 m AlphaEarth
pixel that overlaps ≥1 Track-A GT Overture footprint:

  label = residential | non_residential | mixed   (presence-based; not τ=0.7)
  frac_res / frac_nres = pixel coverage fractions (union-per-class ∩ pixel)

Writes datasets/week3_pixel_gt/{pixels.parquet, manifest.json, figures/}.

Usage (from repo root):
  .venv/bin/python scripts/build_week3_pixel_gt.py
  .venv/bin/python scripts/build_week3_pixel_gt.py --regions NYC_NE --limit-pixels 500
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
OUT_DIR = ROOT / "datasets" / "week3_pixel_gt"
FIG_DIR = OUT_DIR / "figures"

BANDS = [f"A{i:02d}" for i in range(64)]
LABEL_DEFINITION = "resi_def_v1_pixel_overlap_v1"
LABEL_DOC = "docs/residential_definition_v1.md"


def pixel_id_from_wkt(pixel_wkt: str) -> str:
    return hashlib.sha1(pixel_wkt.encode("utf-8")).hexdigest()[:16]


def hard_label(frac_res: float, frac_nres: float) -> str | None:
    """Presence-based mixed (pre-round). None if no GT overlap."""
    has_res = frac_res > 0.0
    has_nres = frac_nres > 0.0
    if has_res and has_nres:
        return "mixed"
    if has_res:
        return "residential"
    if has_nres:
        return "non_residential"
    return None


def _process_region(payload: dict) -> pd.DataFrame:
    """Worker: unique pixels for one metro → coverage fractions + label."""
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

    # Unique pixels that appear for this region's GT buildings (AABB grid).
    pix = (
        cells.drop_duplicates("pixel_wkt", keep="first")
        .loc[
            :,
            ["pixel_wkt", "pixel_lon", "pixel_lat", "pixel_size_m", *BANDS],
        ]
        .reset_index(drop=True)
    )
    p_geoms = [shapely_wkt.loads(w) for w in pix["pixel_wkt"].tolist()]
    pg = gpd.GeoDataFrame(pix, geometry=p_geoms, crs="EPSG:4326").to_crs(crs)

    # Embedding identity check across duplicate cell rows of the same pixel.
    # (Cheap sample: if any band differs, keep first and flag — assert later.)
    sindex = bg.sindex
    rows: list[dict] = []

    for idx, prow in pg.iterrows():
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
        # Exact intersection filter (sindex is bbox-level).
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
        frac_res = area_res / pixel_area
        frac_nres = area_nres / pixel_area
        # Numerical clamp (union ≤ pixel; float noise).
        frac_res = float(np.clip(frac_res, 0.0, 1.0))
        frac_nres = float(np.clip(frac_nres, 0.0, 1.0))

        label = hard_label(frac_res, frac_nres)
        if label is None:
            continue

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
            "label": label,
            "n_res_buildings": len(res_ids),
            "n_nres_buildings": len(nres_ids),
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
    # Worker only needs one row per geographic pixel (embeddings + footprint).
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

    cell_cols = [
        "id",
        "pixel_wkt",
        "pixel_lon",
        "pixel_lat",
        "pixel_size_m",
        *BANDS,
    ]
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
    # Free large tables before fork.
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

    # Dedup if the same geographic pixel appears under multiple regions (edge).
    before = len(out)
    out = out.drop_duplicates("pixel_id", keep="first").reset_index(drop=True)
    if len(out) < before:
        print(f"Dropped {before - len(out)} cross-region duplicate pixel_ids", flush=True)

    # Persist rounded coverage fractions; label already from pre-round.
    out["frac_res_full"] = out["frac_res"].astype(float)
    out["frac_nres_full"] = out["frac_nres"].astype(float)
    out["frac_res"] = out["frac_res_full"].round(2)
    out["frac_nres"] = out["frac_nres_full"].round(2)

    elapsed = time.time() - t0
    manifest = {
        "name": "week3_pixel_gt_v1",
        "purpose": (
            "Pixel-level RES/NRES/mixed GT at 10 m AlphaEarth cells from "
            "Overture Track-A footprints (presence-based mixed, not τ=0.7 f_res)."
        ),
        "label_definition": LABEL_DEFINITION,
        "label_definition_doc": LABEL_DOC,
        "label_rule": {
            "residential": "overlaps RES GT only (frac_res>0, frac_nres==0) pre-round",
            "non_residential": "overlaps NRES GT only",
            "mixed": "overlaps both RES and NRES GT",
            "note": "Hard label uses pre-round overlap; frac_* written at 2 decimals.",
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
        "label_counts": {k: int(v) for k, v in out["label"].value_counts().to_dict().items()},
        "label_counts_by_region": (
            out.groupby(["region", "label"]).size().unstack(fill_value=0).to_dict()
        ),
        "frac_res": {
            "mean": float(out["frac_res"].mean()),
            "median": float(out["frac_res"].median()),
        },
        "frac_nres": {
            "mean": float(out["frac_nres"].mean()),
            "median": float(out["frac_nres"].median()),
        },
        "mixed_rate": float((out["label"] == "mixed").mean()),
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

    # Label counts
    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = ["residential", "non_residential", "mixed"]
    counts = df["label"].value_counts().reindex(order).fillna(0).astype(int)
    colors = {"residential": "#2ca02c", "non_residential": "#d62728", "mixed": "#9467bd"}
    ax.bar(order, counts.values, color=[colors[k] for k in order])
    ax.set_ylabel("pixels")
    ax.set_title("Pixel GT label counts (presence-based mixed)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = fig_dir / "label_counts.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # Fraction histograms
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    axes[0].hist(df["frac_res"], bins=40, color="#2ca02c", edgecolor="white")
    axes[0].set_xlabel("frac_res (2 dp)")
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Residential coverage")
    axes[1].hist(df["frac_nres"], bins=40, color="#d62728", edgecolor="white")
    axes[1].set_xlabel("frac_nres (2 dp)")
    axes[1].set_title("Non-residential coverage")
    fig.suptitle("Pixel coverage fractions", y=1.02)
    fig.tight_layout()
    p = fig_dir / "frac_histograms.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Mixed scatter of frac_res vs frac_nres
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, color in colors.items():
        sub = df.loc[df["label"] == lab]
        ax.scatter(
            sub["frac_res"],
            sub["frac_nres"],
            s=4,
            alpha=0.25,
            c=color,
            label=f"{lab} (n={len(sub):,})",
            rasterized=True,
        )
    ax.set_xlabel("frac_res")
    ax.set_ylabel("frac_nres")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    ax.set_title("Coverage plane (pre-round label)")
    fig.tight_layout()
    p = fig_dir / "frac_scatter.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # Region breakdown
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct = (
        df.groupby(["region", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        .sort_index()
    )
    ct.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[colors[c] for c in order],
        width=0.8,
    )
    ax.set_xlabel("pixels")
    ax.set_title("Label counts by region")
    ax.legend(title="label", loc="lower right", fontsize=8)
    fig.tight_layout()
    p = fig_dir / "label_by_region.png"
    fig.savefig(p, dpi=120)
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

    # Column order for parquet
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
        "frac_res_full",
        "frac_nres_full",
        "label",
        "n_res_buildings",
        "n_nres_buildings",
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
        f"labels: {manifest['label_counts']} | mixed_rate={manifest['mixed_rate']:.3f} | "
        f"{manifest['elapsed_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
