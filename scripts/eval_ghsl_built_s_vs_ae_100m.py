#!/usr/bin/env python3
"""AE vs GHS-BUILT-S fraction MAE at 100 m tiles + map visuals.

Uses existing:
  - Week-4 100 m tile IDs (datasets/week4_100m_tiles/cells_with_tile.parquet)
  - AE W4 Ridge OOF (10 m) aggregated by mean within tile
  - GHS-BUILT-S samples at AE centers (from 10 m BUILT-S run), mean within tile

Also estimates AE-100 m tile vs Mollweide-100 m cell IoU on a sample to show
the lattice shift is much milder at 100 m.

Run dir: outputs/ghsl_built_s_vs_ae_100m/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.patches import Patch, FancyBboxPatch
from pyproj import Transformer
from shapely.geometry import box, Point, Polygon
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
AE_OOF = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
BUILT_S_10M = ROOT / "outputs" / "ghsl_built_s_vs_ae" / "20261103_010752" / "checkpoints" / "built_s_at_ae_centers.parquet"
FUN_CACHE = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
MIN_CELLS = 20
TILE_M = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def fun_onehot(labels: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fun_frac_res": (labels == "RES").astype(float),
            "fun_frac_nres": (labels == "NRES").astype(float),
            "fun_frac_empty": (~labels.isin(["RES", "NRES"])).astype(float),
        }
    )


def mae_pack(yt_r, yt_n, yt_e, yp_r, yp_n, yp_e) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yp_e)
    if m.sum() == 0:
        return {"n": 0}
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_r[m], yp_r[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_n[m], yp_n[m])),
        "mae_frac_empty": float(mean_absolute_error(yt_e[m], yp_e[m])),
        "mae_mean_res_nres": float(
            0.5
            * (
                mean_absolute_error(yt_r[m], yp_r[m])
                + mean_absolute_error(yt_n[m], yp_n[m])
            )
        ),
        "mae_mean_3": float(
            np.mean(
                [
                    mean_absolute_error(yt_r[m], yp_r[m]),
                    mean_absolute_error(yt_n[m], yp_n[m]),
                    mean_absolute_error(yt_e[m], yp_e[m]),
                ]
            )
        ),
        "r2_frac_res": float(r2_score(yt_r[m], yp_r[m])),
        "r2_frac_nres": float(r2_score(yt_n[m], yp_n[m])),
        "r2_frac_empty": float(r2_score(yt_e[m], yp_e[m])),
        "r2_built": float(r2_score(yt_r[m] + yt_n[m], yp_r[m] + yp_n[m])),
    }


def aggregate_tiles(run_dir: Path) -> pd.DataFrame:
    write_progress(run_dir, stage="load", last_progress_at=utc_now())
    cells = pd.read_parquet(
        CELLS,
        columns=["pixel_id", "region", "tile_id", "tile_i", "tile_j", "utm_x", "utm_y"],
    )
    log(run_dir, f"cells_with_tile n={len(cells):,}")

    ae = pd.read_parquet(
        AE_OOF,
        columns=[
            "pixel_id",
            "region",
            "pixel_lon",
            "pixel_lat",
            "frac_res",
            "frac_nres",
            "frac_empty",
            "pred_frac_res",
            "pred_frac_nres",
            "pred_frac_empty",
        ],
    )
    # CRS from full GT (mode per region)
    crs_map = (
        pd.read_parquet(ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet", columns=["region", "crs"])
        .drop_duplicates("region")
        .set_index("region")["crs"]
        .to_dict()
    )
    bs = pd.read_parquet(BUILT_S_10M)
    if "pred_frac_res" not in bs.columns:
        bu = bs["built_s_raw"].astype(float)
        nr = bs["built_s_nres_raw"].astype(float)
        valid = (bu != 255) & (nr != 255)
        bu_f = np.clip(bu / 100.0, 0, 1)
        is_nres = nr == 1
        bs = bs.copy()
        bs["pred_frac_res"] = np.where(valid, np.where(is_nres, 0.0, bu_f), np.nan)
        bs["pred_frac_nres"] = np.where(valid, np.where(is_nres, bu_f, 0.0), np.nan)
        bs["pred_frac_empty"] = np.where(valid, 1.0 - bu_f, np.nan)

    fun = pd.read_parquet(FUN_CACHE, columns=["pixel_id", "ghsl_label"]) if FUN_CACHE.exists() else None

    write_progress(run_dir, stage="join", last_progress_at=utc_now())
    df = cells.merge(ae, on=["pixel_id", "region"], how="inner")
    df["crs"] = df["region"].map(crs_map)
    df = df.merge(
        bs[["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]].rename(
            columns={
                "pred_frac_res": "bs_frac_res",
                "pred_frac_nres": "bs_frac_nres",
                "pred_frac_empty": "bs_frac_empty",
            }
        ),
        on="pixel_id",
        how="inner",
    )
    if fun is not None:
        f = fun.drop_duplicates("pixel_id")
        df = df.merge(f, on="pixel_id", how="left")
        oh = fun_onehot(df["ghsl_label"].fillna("NODATA"))
        df = pd.concat([df.reset_index(drop=True), oh.reset_index(drop=True)], axis=1)

    df = df.loc[(df["frac_res"] + df["frac_nres"]) > 0].copy()
    log(run_dir, f"joined building-linked cells n={len(df):,}")

    write_progress(run_dir, stage="aggregate", last_progress_at=utc_now())
    agg = {
        "n_cells": ("pixel_id", "size"),
        "frac_res": ("frac_res", "mean"),
        "frac_nres": ("frac_nres", "mean"),
        "frac_empty": ("frac_empty", "mean"),
        "ae_frac_res": ("pred_frac_res", "mean"),
        "ae_frac_nres": ("pred_frac_nres", "mean"),
        "ae_frac_empty": ("pred_frac_empty", "mean"),
        "bs_frac_res": ("bs_frac_res", "mean"),
        "bs_frac_nres": ("bs_frac_nres", "mean"),
        "bs_frac_empty": ("bs_frac_empty", "mean"),
        "utm_x_c": ("utm_x", "mean"),
        "utm_y_c": ("utm_y", "mean"),
        "pixel_lon_c": ("pixel_lon", "mean"),
        "pixel_lat_c": ("pixel_lat", "mean"),
    }
    if "fun_frac_res" in df.columns:
        agg.update(
            {
                "fun_frac_res": ("fun_frac_res", "mean"),
                "fun_frac_nres": ("fun_frac_nres", "mean"),
                "fun_frac_empty": ("fun_frac_empty", "mean"),
            }
        )
    tiles = (
        df.groupby(["tile_id", "region", "tile_i", "tile_j", "crs"], sort=False)
        .agg(**agg)
        .reset_index()
    )
    tiles = tiles.loc[tiles["n_cells"] >= MIN_CELLS].copy()
    log(run_dir, f"tiles with n_cells>={MIN_CELLS}: {len(tiles):,}")
    tiles.to_parquet(run_dir / "artifacts" / "tiles_100m_joined.parquet", index=False)
    return tiles


def score(tiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yt_r = tiles["frac_res"].to_numpy()
    yt_n = tiles["frac_nres"].to_numpy()
    yt_e = tiles["frac_empty"].to_numpy()
    rows = []
    models = [
        ("ae_w4_ridge", "ae_frac_res", "ae_frac_nres", "ae_frac_empty"),
        ("ghsl_built_s", "bs_frac_res", "bs_frac_nres", "bs_frac_empty"),
    ]
    if "fun_frac_res" in tiles.columns:
        models.append(("ghsl_fun_onehot", "fun_frac_res", "fun_frac_nres", "fun_frac_empty"))
    for name, a, b, c in models:
        met = mae_pack(yt_r, yt_n, yt_e, tiles[a].to_numpy(), tiles[b].to_numpy(), tiles[c].to_numpy())
        met.update({"model": name, "eval": "100m_min20", "scale": "100m"})
        rows.append(met)

    per = []
    for reg, sub in tiles.groupby("region"):
        for name, a, b, c in models[:2]:
            met = mae_pack(
                sub["frac_res"].to_numpy(),
                sub["frac_nres"].to_numpy(),
                sub["frac_empty"].to_numpy(),
                sub[a].to_numpy(),
                sub[b].to_numpy(),
                sub[c].to_numpy(),
            )
            met.update({"model": name, "eval": "100m_min20", "region": reg})
            per.append(met)
    return pd.DataFrame(rows), pd.DataFrame(per)


def iou_shift_same_crs(side_m: float, dist_m: float = 4.07) -> float:
    """IoU of two axis-aligned squares of side side_m with center offset dist_m on a diagonal."""
    dx = dy = dist_m / math.sqrt(2.0)
    if dx >= side_m or dy >= side_m:
        return 0.0
    overlap = (side_m - dx) * (side_m - dy)
    union = 2 * side_m * side_m - overlap
    return float(overlap / union)


def iou_100m_sample(tiles: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Two IoUs for each sampled AE 100 m tile:

    1) origin_shift_iou: same-CRS 100 m square vs square shifted by the empirical
       ~4.07 m AE↔GHSL center offset (isolates lattice origin error).
    2) cross_crs_iou: AE UTM square projected to Mollweide vs Mollweide-aligned
       100 m cell at the centroid (includes projection skew; usually worse).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tiles), size=min(n, len(tiles)), replace=False)
    sample = tiles.iloc[idx].copy()
    shift_m = 4.07
    origin_iou = iou_shift_same_crs(TILE_M, shift_m)
    origin_iou_10 = iou_shift_same_crs(10.0, shift_m)
    rows = []
    for _, r in sample.iterrows():
        crs = str(r.crs)
        x0 = float(r.tile_i) * TILE_M
        y0 = float(r.tile_j) * TILE_M
        ae = box(x0, y0, x0 + TILE_M, y0 + TILE_M)
        # same-CRS shifted partner (diagonal 4.07 m)
        dx = dy = shift_m / math.sqrt(2.0)
        gh_shift = box(x0 + dx, y0 + dy, x0 + dx + TILE_M, y0 + dy + TILE_M)
        inter_s = ae.intersection(gh_shift).area
        union_s = ae.union(gh_shift).area
        iou_shift = inter_s / union_s if union_s > 0 else 0.0

        to_moll = Transformer.from_crs(crs, "ESRI:54009", always_xy=True)
        xs = [x0, x0 + TILE_M, x0 + TILE_M, x0, x0]
        ys = [y0, y0, y0 + TILE_M, y0 + TILE_M, y0]
        mx, my = to_moll.transform(xs, ys)
        ae_m = Polygon(zip(mx, my))
        cx, cy = ae_m.centroid.x, ae_m.centroid.y
        gi = math.floor(cx / TILE_M) * TILE_M
        gj = math.floor(cy / TILE_M) * TILE_M
        gh = box(gi, gj, gi + TILE_M, gj + TILE_M)
        inter = ae_m.intersection(gh).area
        union = ae_m.union(gh).area
        iou_crs = inter / union if union > 0 else 0.0
        dist = Point(gh.centroid.x, gh.centroid.y).distance(ae_m.centroid)
        rows.append(
            {
                "region": r.region,
                "iou_100m_origin_shift": iou_shift,
                "iou_100m_cross_crs": iou_crs,
                "iou_100m": iou_shift,  # primary: origin-shift story
                "dist_m_100m": dist,
                "dist_m_origin_shift": shift_m,
                "n_cells": r.n_cells,
                "ref_iou_10m_origin_shift": origin_iou_10,
                "ref_iou_100m_origin_shift_closed_form": origin_iou,
            }
        )
    return pd.DataFrame(rows)


def make_metric_figs(run_dir: Path, summary: pd.DataFrame, per: pd.DataFrame) -> list[Path]:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    paths = []
    s = summary.sort_values("mae_mean_res_nres")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    colors = ["#2ca02c" if m.startswith("ae") else ("#ff7f0e" if "built_s" in m else "#7f7f7f") for m in s["model"]]
    ax.barh(s["model"], s["mae_mean_res_nres"], color=colors)
    for y, v in enumerate(s["mae_mean_res_nres"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center")
    ax.set_xlabel("MAE mean(res, nres) ↓")
    ax.set_title(f"100 m fraction MAE · min {MIN_CELLS} cells/tile")
    fig.tight_layout()
    p = fig_dir / "mae_ladder_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    piv = per.pivot_table(index="region", columns="model", values="mae_mean_res_nres")
    piv = piv.sort_values("ae_w4_ridge")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(piv))
    h = 0.35
    ax.barh(y - h / 2, piv["ae_w4_ridge"], height=h, color="#2ca02c", label="AE W4 Ridge")
    ax.barh(y + h / 2, piv["ghsl_built_s"], height=h, color="#ff7f0e", label="GHS-BUILT-S")
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("MAE mean(res, nres)")
    ax.set_title("Per-metro 100 m MAE")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "mae_by_metro_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # component
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(s))
    w = 0.25
    ax.bar(x - w, s["mae_frac_res"], width=w, label="res", color="#4C78A8")
    ax.bar(x, s["mae_frac_nres"], width=w, label="nres", color="#F58518")
    ax.bar(x + w, s["mae_frac_empty"], width=w, label="empty", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(s["model"], rotation=15)
    ax.set_ylabel("MAE")
    ax.set_title("100 m component MAE")
    ax.legend()
    fig.tight_layout()
    p = fig_dir / "mae_components_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def make_map_visuals(run_dir: Path, tiles: pd.DataFrame, iou_df: pd.DataFrame) -> list[Path]:
    """Step-by-step 100 m process on a real Houston neighborhood."""
    import io
    import urllib.request

    fig_dir = run_dir / "figures"
    paths = []

    # pick Houston tile with many cells near prior industrial area if possible
    h = tiles[tiles.region == "Houston_S"].copy()
    if h.empty:
        h = tiles
    focus = h.sort_values("n_cells", ascending=False).iloc[0]
    # local window of neighboring tiles
    ti0, tj0 = int(focus.tile_i), int(focus.tile_j)
    near = tiles[
        (tiles.region == focus.region)
        & (tiles.tile_i.between(ti0 - 2, ti0 + 2))
        & (tiles.tile_j.between(tj0 - 2, tj0 + 2))
    ].copy()
    crs = str(focus.crs)

    # load 10 m cells for this tile for inset detail
    cells = pd.read_parquet(
        CELLS,
        columns=["pixel_id", "region", "tile_id", "utm_x", "utm_y"],
    )
    cells = cells[cells.tile_id == focus.tile_id].copy()
    g10 = gpd.GeoDataFrame(
        cells,
        geometry=[box(x - 5, y - 5, x + 5, y + 5) for x, y in zip(cells.utm_x, cells.utm_y)],
        crs=crs,
    )
    # 100 m tiles
    g100 = gpd.GeoDataFrame(
        near,
        geometry=[
            box(float(i) * TILE_M, float(j) * TILE_M, float(i) * TILE_M + TILE_M, float(j) * TILE_M + TILE_M)
            for i, j in zip(near.tile_i, near.tile_j)
        ],
        crs=crs,
    )
    focus_poly = box(ti0 * TILE_M, tj0 * TILE_M, ti0 * TILE_M + TILE_M, tj0 * TILE_M + TILE_M)

    # Mollweide 100 m cell at focus center
    to_moll = Transformer.from_crs(crs, "ESRI:54009", always_xy=True)
    to_utm = Transformer.from_crs("ESRI:54009", crs, always_xy=True)
    cx = ti0 * TILE_M + 50
    cy = tj0 * TILE_M + 50
    mx, my = to_moll.transform(cx, cy)
    gi = math.floor(mx / TILE_M) * TILE_M
    gj = math.floor(my / TILE_M) * TILE_M
    # project GHSL 100 m corners to UTM for drawing
    gxs = [gi, gi + TILE_M, gi + TILE_M, gi, gi]
    gys = [gj, gj, gj + TILE_M, gj + TILE_M, gj]
    ux, uy = to_utm.transform(gxs, gys)
    gh_poly = Polygon(zip(ux, uy))
    ae_m = gpd.GeoSeries([focus_poly], crs=crs).to_crs("ESRI:54009").iloc[0]
    gh_m = box(gi, gj, gi + TILE_M, gj + TILE_M)
    inter = ae_m.intersection(gh_m).area
    union = ae_m.union(gh_m).area
    iou_focus = inter / union if union > 0 else 0.0
    dist_focus = Point(gh_m.centroid.x, gh_m.centroid.y).distance(ae_m.centroid)

    # basemap
    def fetch_osm(bounds_4326, zoom=17):
        from PIL import Image

        minlon, minlat, maxlon, maxlat = bounds_4326

        def deg2num(lat, lon, z):
            lat_r = math.radians(lat)
            n = 2.0**z
            return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(lat_r)) / math.pi) / 2 * n)

        def lonlat_to_merc(lon, lat):
            x = lon * 20037508.34 / 180
            y = math.log(math.tan((90 + lat) * math.pi / 360)) * 20037508.34 / math.pi
            return x, y

        def tile_y_to_lat(y, z):
            n = 2.0**z
            return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))

        x0, ys = deg2num(minlat, minlon, zoom)
        x1, yn = deg2num(maxlat, maxlon, zoom)
        xs = range(min(x0, x1), max(x0, x1) + 1)
        ys = range(min(yn, ys), max(yn, ys) + 1)
        tiles_img = []
        for y in ys:
            row = []
            for x in xs:
                url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                req = urllib.request.Request(url, headers={"User-Agent": "AlphaEarthResearch/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    row.append(Image.open(io.BytesIO(resp.read())).convert("RGB"))
            tiles_img.append(row)
        tw, th = tiles_img[0][0].size
        canvas = Image.new("RGB", (tw * len(list(xs)), th * len(list(ys))))
        for yi, row in enumerate(tiles_img):
            for xi, im in enumerate(row):
                canvas.paste(im, (xi * tw, yi * th))
        lon_left = min(xs) / (2**zoom) * 360 - 180
        lon_right = (max(xs) + 1) / (2**zoom) * 360 - 180
        lat_top = tile_y_to_lat(min(ys), zoom)
        lat_bot = tile_y_to_lat(max(ys) + 1, zoom)
        west, _ = lonlat_to_merc(lon_left, 0)
        east, _ = lonlat_to_merc(lon_right, 0)
        _, north = lonlat_to_merc(0, lat_top)
        _, south = lonlat_to_merc(0, lat_bot)
        return np.asarray(canvas), (west, east, south, north)

    g100_4326 = g100.to_crs("EPSG:4326")
    b = g100_4326.total_bounds
    pad = 0.0008
    try:
        basemap = fetch_osm((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), zoom=16)
    except Exception as e:
        log(run_dir, f"basemap failed: {e}")
        basemap = None

    g100_m = g100.to_crs("EPSG:3857")
    g10_m = g10.to_crs("EPSG:3857")
    focus_m = gpd.GeoSeries([focus_poly], crs=crs).to_crs("EPSG:3857").iloc[0]
    gh_m3857 = gpd.GeoSeries([gh_poly], crs=crs).to_crs("EPSG:3857").iloc[0]

    fig = plt.figure(figsize=(15.5, 10.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.22)

    def paint_base(ax, title):
        if basemap is not None:
            img, extent = basemap
            ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", alpha=0.88)
            minx, miny, maxx, maxy = g100_m.total_bounds
            ax.set_xlim(minx - 40, maxx + 40)
            ax.set_ylim(miny - 40, maxy + 40)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            minx, miny, maxx, maxy = g100.total_bounds
            ax.set_xlim(minx - 40, maxx + 40)
            ax.set_ylim(miny - 40, maxy + 40)
            ax.set_aspect("equal")
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")

    # 1: 10 m AE cells inside 100 m tile
    ax = fig.add_subplot(gs[0, 0])
    paint_base(ax, "1 · Group AE 10 m cells into a 100 m tile")
    geoms = g10_m.geometry if basemap is not None else g10.geometry
    for geom in geoms:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, facecolor="#4C78A8", edgecolor="#1f4e79", alpha=0.35, linewidth=0.4)
    fp = focus_m if basemap is not None else focus_poly
    xs, ys = fp.exterior.xy
    ax.plot(xs, ys, color="#0b2540", lw=2.5)
    ax.text(
        0.02,
        0.02,
        f"focus tile n_cells={int(focus.n_cells)}\nGT mean frac on tile",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.92),
    )

    # 2: AE 100 m lattice
    ax = fig.add_subplot(gs[0, 1])
    paint_base(ax, "2 · AE 100 m tile is the compare unit")
    geoms = g100_m.geometry if basemap is not None else g100.geometry
    for geom in geoms:
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color="#1f4e79", lw=1.2)
    xs, ys = fp.exterior.xy
    ax.plot(xs, ys, color="#0b2540", lw=2.8)
    ax.text(
        0.02,
        0.02,
        f"GT res={focus.frac_res:.2f} nres={focus.frac_nres:.2f}\nempty={focus.frac_empty:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.92),
    )

    # 3: GHSL 100 m cell vs AE 100 m tile (overlap)
    ax = fig.add_subplot(gs[0, 2])
    paint_base(ax, "3 · Mollweide 100 m GHSL cell vs AE tile")
    xs, ys = fp.exterior.xy
    ax.fill(xs, ys, facecolor="#4C78A8", edgecolor="#0b2540", alpha=0.35, linewidth=2.0)
    gp = gh_m3857 if basemap is not None else gh_poly
    xs, ys = gp.exterior.xy
    ax.fill(xs, ys, facecolor="#F58518", edgecolor="#6b2d00", alpha=0.35, linewidth=2.0)
    # centers
    if basemap is not None:
        ae_c = gpd.GeoSeries([focus_poly.centroid], crs=crs).to_crs("EPSG:3857").iloc[0]
        gh_c = gpd.GeoSeries([gh_poly.centroid], crs=crs).to_crs("EPSG:3857").iloc[0]
    else:
        ae_c, gh_c = focus_poly.centroid, gh_poly.centroid
    ax.plot(ae_c.x, ae_c.y, "o", color="#0b2540", ms=7)
    ax.plot(gh_c.x, gh_c.y, "o", color="#6b2d00", ms=7)
    ax.annotate("", xy=(gh_c.x, gh_c.y), xytext=(ae_c.x, ae_c.y), arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    origin_iou_100 = iou_shift_same_crs(100.0, 4.07)
    origin_iou_10 = iou_shift_same_crs(10.0, 4.07)
    ax.text(
        0.02,
        0.02,
        f"origin-shift IoU @100 m≈{origin_iou_100:.2f}\n(same 4.1 m offset @10 m≈{origin_iou_10:.2f})\ncross-CRS tile IoU≈{iou_focus:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#c45c00", alpha=0.95),
    )
    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", edgecolor="#0b2540", alpha=0.5, label="AE 100 m"),
            Patch(facecolor="#F58518", edgecolor="#6b2d00", alpha=0.5, label="GHSL 100 m"),
        ],
        loc="upper right",
        fontsize=8,
    )

    # 4: aggregate BUILT-S 10 m samples inside tile
    ax = fig.add_subplot(gs[1, 0])
    paint_base(ax, "4 · Mean GHS-BUILT-S soft fracs over AE centers in tile")
    xs, ys = fp.exterior.xy
    ax.plot(xs, ys, color="#0b2540", lw=2.5)
    # plot 10 m centers
    if basemap is not None:
        pts = gpd.GeoSeries(gpd.points_from_xy(cells.utm_x, cells.utm_y), crs=crs).to_crs("EPSG:3857")
        ax.scatter([p.x for p in pts], [p.y for p in pts], s=8, c="#c45c00", alpha=0.8)
    else:
        ax.scatter(cells.utm_x, cells.utm_y, s=8, c="#c45c00", alpha=0.8)
    ax.text(
        0.02,
        0.02,
        f"tile BUILT-S\nres={focus.bs_frac_res:.2f} nres={focus.bs_frac_nres:.2f}\nempty={focus.bs_frac_empty:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#c45c00", alpha=0.95),
    )

    # 5: AE ridge tile mean
    ax = fig.add_subplot(gs[1, 1])
    paint_base(ax, "5 · Mean AE Ridge preds over same 10 m cells")
    # color nearby tiles by AE pred res
    geoms = list(g100_m.geometry) if basemap is not None else list(g100.geometry)
    for geom, (_, row) in zip(geoms, near.iterrows()):
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, facecolor=(0.2 + 0.5 * row.ae_frac_res, 0.4, 0.55), edgecolor="#1f4e79", alpha=0.45, linewidth=1.0)
    xs, ys = fp.exterior.xy
    ax.plot(xs, ys, color="#0b2540", lw=2.8)
    ax.text(
        0.02,
        0.02,
        f"tile AE\nres={focus.ae_frac_res:.2f} nres={focus.ae_frac_nres:.2f}\nempty={focus.ae_frac_empty:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.95),
    )

    # 6: scoreboard + IoU distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("6 · Score MAE on shared 100 m tiles", fontsize=10.5, fontweight="bold", loc="left")
    # will fill numbers after summary known — placeholder filled by caller via text file; compute from focus + iou_df
    ax.text(0.05, 0.88, "Focus tile |error| vs GT (3-way mean)", fontsize=10, fontweight="bold", transform=ax.transAxes)
    gtv = np.array([focus.frac_res, focus.frac_nres, focus.frac_empty])
    aev = np.array([focus.ae_frac_res, focus.ae_frac_nres, focus.ae_frac_empty])
    bsv = np.array([focus.bs_frac_res, focus.bs_frac_nres, focus.bs_frac_empty])
    ax.text(0.05, 0.78, f"AE  mean |err| = {np.mean(np.abs(aev - gtv)):.3f}", color="#2e7d32", fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.70, f"BUILT-S mean |err| = {np.mean(np.abs(bsv - gtv)):.3f}", color="#c45c00", fontsize=11, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((0.04, 0.28), 0.92, 0.34, boxstyle="round,pad=0.02", transform=ax.transAxes, facecolor="#f5f5f5", edgecolor="#999"))
    ax.text(0.08, 0.52, "Lattice shift at 100 m (sample)", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.08,
        0.40,
        f"origin-shift IoU @100 m = {iou_df.iou_100m_origin_shift.median():.2f}\n"
        f"origin-shift IoU @10 m  = {iou_df.ref_iou_10m_origin_shift.median():.2f}\n"
        f"cross-CRS IoU @100 m    = {iou_df.iou_100m_cross_crs.median():.2f}",
        fontsize=9,
        transform=ax.transAxes,
        family="monospace",
    )
    ax.text(0.05, 0.16, "Full MAE numbers written into summary.json", fontsize=9, style="italic", transform=ax.transAxes)

    fig.suptitle(
        f"How AE vs GHS-BUILT-S works at 100 m · {focus.region}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    out = fig_dir / "comparison_steps_100m_on_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(out)

    # IoU histogram 10m vs 100m comparison figure
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    # closed-form / empirical origin-shift IoUs
    vals = [
        ("10 m origin shift\n(4.1 m offset)", float(iou_df.ref_iou_10m_origin_shift.median())),
        ("100 m origin shift\n(same 4.1 m)", float(iou_df.iou_100m_origin_shift.median())),
        ("100 m cross-CRS\n(UTM vs Mollweide)", float(iou_df.iou_100m_cross_crs.median())),
    ]
    colors = ["#4C78A8", "#F58518", "#7f7f7f"]
    ax.bar([v[0] for v in vals], [v[1] for v in vals], color=colors)
    for i, (_, v) in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU")
    ax.set_title("Same 4.1 m lattice offset: bad at 10 m, mild at 100 m")
    fig.tight_layout()
    p = fig_dir / "iou_10m_vs_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    return paths, focus, iou_focus, dist_focus


def update_notebook(run_dir: Path, summary: pd.DataFrame) -> Path:
    nb_dir = ROOT / "notebooks" / "ghsl_built_s_vs_ae"
    nb_dir.mkdir(parents=True, exist_ok=True)
    # copy figures
    fig_out = nb_dir / "figures"
    fig_out.mkdir(exist_ok=True)
    for p in (run_dir / "figures").glob("*.png"):
        (fig_out / p.name).write_bytes(p.read_bytes())

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GHS-BUILT-S vs AE fraction MAE\n",
                "\n",
                "10 m center-sample compare, plus **100 m tile** aggregation where lattice shift is milder.\n",
                "\n",
                f"100 m run: `{run_dir}`\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import pandas as pd\n",
                "from IPython.display import Image, display, Markdown\n",
                "\n",
                f"RUN10 = Path({str(ROOT / 'outputs/ghsl_built_s_vs_ae/20261103_010752')!r})\n",
                f"RUN100 = Path({str(run_dir)!r})\n",
                "s10 = pd.read_csv(RUN10 / 'mae_summary.csv')\n",
                "s100 = pd.read_csv(RUN100 / 'mae_summary.csv')\n",
                "display(Markdown('## 10 m (shared 59k)'))\n",
                "display(s10[s10.eval=='shared_59k'][['model','n','mae_mean_res_nres','mae_frac_res','mae_frac_nres','mae_frac_empty']])\n",
                "display(Markdown('## 100 m tiles (min 20 cells)'))\n",
                "display(s100[['model','n','mae_mean_res_nres','mae_frac_res','mae_frac_nres','mae_frac_empty','r2_built']])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Markdown('## 100 m process maps'))\n",
                "for p in sorted((RUN100 / 'figures').glob('*.png')):\n",
                "    display(Markdown(f'### {p.name}'))\n",
                "    display(Image(filename=str(p)))\n",
                "meta = json.loads((RUN100 / 'summary.json').read_text())\n",
                "print(meta.get('verdict',''))\n",
            ],
        },
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = nb_dir / "01_built_s_vs_ae_mae.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    # also dedicated 100m notebook
    out2 = nb_dir / "02_built_s_vs_ae_100m.ipynb"
    out2.write_text(json.dumps(nb, indent=1))
    return out2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "ghsl_built_s_vs_ae_100m" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"cmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n")
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=time.time(),
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count()},
    )
    log(run_dir, f"Run dir {run_dir}")
    if not BUILT_S_10M.exists():
        raise FileNotFoundError(f"missing 10 m BUILT-S cache: {BUILT_S_10M}")

    tiles = aggregate_tiles(run_dir)
    write_progress(run_dir, stage="score", last_progress_at=utc_now())
    summary, per = score(tiles)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)
    per.to_csv(run_dir / "mae_per_region.csv", index=False)
    log(run_dir, "Scores: " + ", ".join(f"{r.model}={r.mae_mean_res_nres:.4f}" for r in summary.itertuples()))

    write_progress(run_dir, stage="iou_sample", last_progress_at=utc_now())
    iou_df = iou_100m_sample(tiles, n=2000)
    iou_df.to_csv(run_dir / "iou_100m_sample.csv", index=False)
    log(
        run_dir,
        f"origin-shift IoU 10 m={iou_df.ref_iou_10m_origin_shift.median():.3f}  "
        f"100 m={iou_df.iou_100m_origin_shift.median():.3f}  "
        f"cross-CRS 100 m={iou_df.iou_100m_cross_crs.median():.3f}",
    )

    write_progress(run_dir, stage="figures", last_progress_at=utc_now())
    figs = make_metric_figs(run_dir, summary, per)
    map_figs, focus, iou_focus, dist_focus = make_map_visuals(run_dir, tiles, iou_df)
    figs.extend(map_figs)

    # annotate ladder figure path etc into summary panel text already done
    shared = summary.set_index("model")
    ae_mae = float(shared.loc["ae_w4_ridge", "mae_mean_res_nres"])
    bs_mae = float(shared.loc["ghsl_built_s", "mae_mean_res_nres"])
    if ae_mae < bs_mae:
        verdict = (
            f"At 100 m (min {MIN_CELLS} cells), AE W4 Ridge MAE={ae_mae:.3f} beats GHS-BUILT-S MAE={bs_mae:.3f} "
            f"(Δ={bs_mae - ae_mae:.3f}). Origin-shift IoU for a 4.1 m offset rises from "
            f"{iou_df.ref_iou_10m_origin_shift.median():.2f} at 10 m to "
            f"{iou_df.iou_100m_origin_shift.median():.2f} at 100 m."
        )
    else:
        verdict = (
            f"At 100 m (min {MIN_CELLS} cells), GHS-BUILT-S MAE={bs_mae:.3f} beats AE W4 Ridge MAE={ae_mae:.3f} "
            f"(Δ={ae_mae - bs_mae:.3f}). Origin-shift IoU for a 4.1 m offset rises from "
            f"{iou_df.ref_iou_10m_origin_shift.median():.2f} at 10 m to "
            f"{iou_df.iou_100m_origin_shift.median():.2f} at 100 m."
        )

    # patch step-6 board with global MAE by rewriting a small overlay fig? already have ladders.
    nb = update_notebook(run_dir, summary)
    payload = {
        "status": "completed",
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "method": {
            "scale": "100m",
            "tile_id": "per-metro UTM floor(easting/100), floor(northing/100)",
            "min_cells": MIN_CELLS,
            "gt_ae_agg": "mean of 10 m fractions / Ridge preds inside tile",
            "built_s_agg": "mean of 10 m BUILT-S soft fractions at AE centers inside tile",
            "iou_100m": "AE UTM 100 m square vs Mollweide-aligned 100 m cell at tile center",
        },
        "mae_summary": summary.to_dict(orient="records"),
        "iou_100m": {
            "n": int(len(iou_df)),
            "origin_shift_iou_10m": float(iou_df.ref_iou_10m_origin_shift.median()),
            "origin_shift_iou_100m": float(iou_df.iou_100m_origin_shift.median()),
            "cross_crs_iou_100m_median": float(iou_df.iou_100m_cross_crs.median()),
            "focus_tile_cross_crs_iou": float(iou_focus),
            "focus_tile_cross_crs_dist_m": float(dist_focus),
        },
        "verdict": verdict,
        "figures": [str(p) for p in figs],
        "notebook": str(nb),
        "n_tiles": int(len(tiles)),
    }
    atomic_json(run_dir / "summary.json", payload)
    write_progress(run_dir, status="completed", stage="done", last_progress_at=utc_now(), verdict=verdict)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, verdict)
    log(run_dir, f"DONE · notebook={nb}")


if __name__ == "__main__":
    main()
