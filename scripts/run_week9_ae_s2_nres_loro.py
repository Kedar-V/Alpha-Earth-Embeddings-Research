#!/usr/bin/env python3
"""Week 9: AE embeddings + Sentinel-2 NRES-rule features → XGB LORO.

Apples-to-apples with Section 3/4 (docs/eval_protocol_apples_to_apples.md):
  TRAIN: soft_gt_covered Track-A buildings (n≈86k)
  EVAL:  locked cohort IDs only (buildings 71,497 · tiles 25,220)
  METRIC: 0.5*(MAE_res + MAE_nres) on soft fractions
  SPLIT:  leave-one-region-out

S2 features approximate Pesaresi et al. 2024 NRES heuristics from Element84
sentinel-2-l2a COGs (2018 summer, low cloud): brightness, blue dominance,
NDVI, multi-scale luminance contrast (PANTEX-like), footprint size/compactness,
and a fuzzy nres_score. Concatenated to mean AE 64-d of inside_polygon cells.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
W6_BUILD = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)
W6_100M = ROOT / "outputs" / "week6_overture_polygon_mae_100m" / "20261110_062739"
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
COHORT_TILE = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_tile_ids.parquet"
XGB_BUILD = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "buildings_section3_with_xgb.parquet"
)
XGB_TILE = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "tiles_section3_with_xgb.parquet"
)

BANDS = [f"A{i:02d}" for i in range(64)]
S2_FEATS = [
    "s2_blue",
    "s2_green",
    "s2_red",
    "s2_nir",
    "s2_brightness",
    "s2_blue_dom",
    "s2_ndvi",
    "s2_contrast_small",
    "s2_contrast_large",
    "s2_contrast_ratio",
    "fp_area_m2",
    "fp_equiv_diam_m",
    "fp_compactness",
    "s2_nres_score",
    "s2_n_pix",
    "s2_ok",
]
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
S2_DATETIME = "2018-06-01T00:00:00Z/2018-09-30T23:59:59Z"
SEED = 42

METRO_META = [
    ("Boston_NE", "Boston NE", "#ECE8F4"),
    ("NYC_NE", "NYC NE", "#ECE8F4"),
    ("Atlanta_SE", "Atlanta SE", "#E2EFE6"),
    ("Houston_S", "Houston S", "#F3EDD5"),
    ("Chicago_MW", "Chicago MW", "#F3E3E7"),
    ("Phoenix_SW", "Phoenix SW", "#EFE4D9"),
    ("Denver_W", "Denver W", "#E1ECEE"),
    ("LA_W", "LA W", "#E1ECEE"),
    ("SF_W", "SF W", "#E1ECEE"),
    ("Seattle_NW", "Seattle NW", "#E9EDDC"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    if "_t0" not in prev and "_t0" not in kwargs:
        kwargs["_t0"] = time.time()
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0.0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if int(m.sum()) == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    mae_r = float(mean_absolute_error(y_res[m], p_res[m]))
    mae_n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": mae_r, "nres": mae_n, "mean": 0.5 * (mae_r + mae_n)}


def feat_colnames() -> list[str]:
    return BANDS + S2_FEATS


def rasterio_env():
    import rasterio

    return rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        AWS_NO_SIGN_REQUEST="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    )


def stac_search(bbox: list[float], limit: int = 40) -> list[dict]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": S2_DATETIME,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": 20}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    r = requests.post(STAC_URL, json=body, timeout=120)
    r.raise_for_status()
    return list(r.json().get("features", []))


def _asset_href(feat: dict, keys: tuple[str, ...]) -> str | None:
    assets = feat.get("assets", {})
    for k in keys:
        if k in assets and "href" in assets[k]:
            return assets[k]["href"]
    return None


def select_scenes(bbox: list[float], max_scenes: int = 8) -> list[dict]:
    feats = stac_search(bbox, limit=50)
    picked: list[dict] = []
    seen_tiles: set[str] = set()
    for f in feats:
        props = f.get("properties", {})
        tile = props.get("grid:code") or props.get("mgrs:utm_zone")
        key = f"{tile}|{f.get('id')}"
        hrefs = {
            "blue": _asset_href(f, ("blue", "B02")),
            "green": _asset_href(f, ("green", "B03")),
            "red": _asset_href(f, ("red", "B04")),
            "nir": _asset_href(f, ("nir", "B08", "nir08")),
        }
        if not all(hrefs.values()):
            continue
        # prefer diverse MGRS coverage then stop
        mgrs = str(tile)
        if mgrs in seen_tiles and len(picked) >= 3:
            continue
        seen_tiles.add(mgrs)
        picked.append({"id": f["id"], "cloud": props.get("eo:cloud_cover"), "hrefs": hrefs, "bbox": f.get("bbox")})
        if len(picked) >= max_scenes:
            break
    return picked


def _contrast(arr: np.ndarray) -> float:
    """Nan-aware std over finite values; needs ≥4 finite samples."""
    a = np.asarray(arr, dtype=np.float32)
    if a.size < 4 or not np.isfinite(a).any():
        return float("nan")
    a = a[np.isfinite(a)]
    if a.size < 4:
        return float("nan")
    return float(np.std(a))


def _nanmean_valid(a: np.ndarray) -> float:
    m = np.isfinite(a)
    return float(np.mean(a[m])) if m.any() else float("nan")


def _block_mean_std(lum: np.ndarray, min_valid_per_block: int = 2, min_blocks: int = 4) -> float:
    """Std of 2×2 block means over a 2-D luminance chip (NaN = nodata).

    Crops to even H/W, reshapes to (h/2, 2, w/2, 2), nanmean over the 2×2,
    keeps blocks with ≥ min_valid_per_block finite pixels, then nanstd over
    blocks (requires ≥ min_blocks valid blocks).
    """
    a = np.asarray(lum, dtype=np.float32)
    if a.ndim != 2:
        return float("nan")
    h, w = a.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 < 2 or w2 < 2:
        return float("nan")
    crop = a[:h2, :w2]
    blocks = crop.reshape(h2 // 2, 2, w2 // 2, 2)
    valid_count = np.sum(np.isfinite(blocks), axis=(1, 3))
    with np.errstate(all="ignore"):
        block_means = np.nanmean(blocks, axis=(1, 3))
    block_means = np.where(valid_count >= min_valid_per_block, block_means, np.nan)
    finite = block_means[np.isfinite(block_means)]
    if finite.size < min_blocks:
        return float("nan")
    return float(np.std(finite))


# Contrast-ratio ramp endpoints for the fuzzy NRES score.
# Chosen on soft-covered train buildings excluding locked 71,497 eval IDs
# (run 20261117_044358 / contrast_ratio_calibration.json):
#   p10=0.7650, p90=0.8948, flip=False, use_ratio=True (AUC_nres=0.603).
# (ratio-1)/2 was wrong once large = true 2×2 block-mean std (ratio usually ≤1).
CONTRAST_RATIO_LO = 0.7649924497421877
CONTRAST_RATIO_HI = 0.8948352902414409
CONTRAST_RATIO_FLIP = False  # if True, lower ratio → NRES
USE_CONTRAST_RATIO_IN_NRES = True


def _fuzzy_nres(
    brightness: float,
    blue_dom: float,
    contrast_ratio: float,
    contrast_small: float,
    equiv_diam: float,
    compactness: float,
    ratio_lo: float | None = None,
    ratio_hi: float | None = None,
    ratio_flip: bool | None = None,
    use_ratio: bool | None = None,
) -> float:
    """Rough fuzzy score mirroring GHSL NRES heuristics (0–1).

    Contrast-ratio term uses calibrated endpoints (p10/p90 on train-only
    buildings). If USE_CONTRAST_RATIO_IN_NRES is False (AUC < 0.55 on the
    calibration set), that term is dropped from the mean.
    """

    def clamp01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))

    lo = CONTRAST_RATIO_LO if ratio_lo is None else float(ratio_lo)
    hi = CONTRAST_RATIO_HI if ratio_hi is None else float(ratio_hi)
    flip = CONTRAST_RATIO_FLIP if ratio_flip is None else bool(ratio_flip)
    use = USE_CONTRAST_RATIO_IN_NRES if use_ratio is None else bool(use_ratio)

    # reflectance often ~0–4000 for L2A scaled; normalize loosely
    b_hi = clamp01((brightness - 800.0) / 2500.0)
    blue = clamp01((blue_dom - 0.30) / 0.25)
    if hi > lo and np.isfinite(contrast_ratio):
        large_raw = (float(contrast_ratio) - lo) / (hi - lo)
        large = clamp01(1.0 - large_raw if flip else large_raw)
    else:
        large = 0.0
    smooth = clamp01(1.0 - (contrast_small / 400.0))
    size = clamp01((equiv_diam - 20.0) / 60.0)
    compact = clamp01((compactness - 0.4) / 0.5)
    terms = [b_hi, blue, smooth, size, compact]
    if use:
        terms.insert(2, large)
    return float(np.mean(terms))


def _features_from_chip(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    area_m2: float,
    compactness: float,
) -> dict[str, float]:
    """S2 + footprint features from a BGRN chip.

    Nodata (B=G=R=0) is set to NaN so the chip stays 2-D. Band means / brightness /
    blue share / NDVI use nan-aware means over valid pixels (numerically equal to
    the old boolean-index path). Texture keeps spatial structure: s2_contrast_small
    = nanstd(lum); s2_contrast_large = nanstd of 2×2 block means.
    """
    b = blue.astype(np.float32).copy()
    g = green.astype(np.float32).copy()
    r = red.astype(np.float32).copy()
    n = nir.astype(np.float32).copy()
    # nodata zeros common on COG edges — keep 2-D layout
    nodata = (b <= 0) & (g <= 0) & (r <= 0)
    valid = ~nodata
    n_pix = int(valid.sum())
    for arr in (b, g, r, n):
        arr[nodata] = np.nan

    # Equality with the old 1-D boolean-index means (same valid pixels).
    if n_pix > 0:
        old_mb = float(np.mean(b[valid]))
        old_mg = float(np.mean(g[valid]))
        old_mr = float(np.mean(r[valid]))
        old_mn = float(np.mean(n[valid]))
    else:
        old_mb = old_mg = old_mr = old_mn = float("nan")

    mb, mg, mr, mn = _nanmean_valid(b), _nanmean_valid(g), _nanmean_valid(r), _nanmean_valid(n)
    if n_pix > 0:
        assert abs(mb - old_mb) < 1e-5 and abs(mg - old_mg) < 1e-5
        assert abs(mr - old_mr) < 1e-5 and abs(mn - old_mn) < 1e-5

    brightness = float(np.nanmean([mb, mg, mr])) if np.isfinite(mb) else float("nan")
    blue_dom = mb / (mb + mg + mr + 1e-6) if np.isfinite(mb) else float("nan")
    ndvi = (mn - mr) / (mn + mr + 1e-6) if np.isfinite(mn) and np.isfinite(mr) else float("nan")

    lum = 0.3 * r + 0.59 * g + 0.11 * b  # 2-D, NaN at nodata
    c_small = _contrast(lum)
    c_large = _block_mean_std(lum)
    c_ratio = (c_large / (c_small + 1e-6)) if np.isfinite(c_small) and np.isfinite(c_large) else float("nan")
    equiv = float(2.0 * np.sqrt(max(area_m2, 0.0) / np.pi)) if np.isfinite(area_m2) else float("nan")
    nres = _fuzzy_nres(brightness, blue_dom, c_ratio, c_small, equiv, compactness)
    return {
        "s2_blue": mb,
        "s2_green": mg,
        "s2_red": mr,
        "s2_nir": mn,
        "s2_brightness": float(brightness) if np.isfinite(brightness) else float("nan"),
        "s2_blue_dom": float(blue_dom) if np.isfinite(blue_dom) else float("nan"),
        "s2_ndvi": float(ndvi) if np.isfinite(ndvi) else float("nan"),
        "s2_contrast_small": c_small,
        "s2_contrast_large": c_large,
        "s2_contrast_ratio": float(c_ratio) if np.isfinite(c_ratio) else float("nan"),
        "fp_area_m2": float(area_m2) if np.isfinite(area_m2) else float("nan"),
        "fp_equiv_diam_m": equiv,
        "fp_compactness": float(compactness) if np.isfinite(compactness) else float("nan"),
        "s2_nres_score": nres,
        "s2_n_pix": float(n_pix),
        "s2_ok": 1.0 if np.isfinite(mb) and n_pix >= 4 else 0.0,
    }


def _geom_metrics(wkt_str: str) -> tuple[float, float]:
    try:
        g = shapely_wkt.loads(wkt_str)
        if g.is_empty:
            return float("nan"), float("nan")
        # WGS84 area is deg^2 — use geodesic approx via local UTM if possible
        area = float(g.area)
        perim = float(g.length)
        # better: use shapely in projected if we have bounds — caller may pass area_m2
        compactness = (4.0 * np.pi * area / (perim * perim + 1e-12)) if perim > 0 else float("nan")
        return area, float(compactness)
    except Exception:
        return float("nan"), float("nan")


def _empty_s2_row(bid: str, region: str, area_m2: float, compact: float) -> dict[str, float | str]:
    equiv = float(2.0 * np.sqrt(max(area_m2, 0.0) / np.pi)) if np.isfinite(area_m2) else float("nan")
    feat: dict[str, float | str] = {k: float("nan") for k in S2_FEATS}
    feat.update(
        {
            "fp_area_m2": area_m2 if np.isfinite(area_m2) else float("nan"),
            "fp_equiv_diam_m": equiv,
            "fp_compactness": compact if np.isfinite(compact) else float("nan"),
            "s2_nres_score": 0.0,
            "s2_n_pix": 0.0,
            "s2_ok": 0.0,
            "id": bid,
            "region": region,
        }
    )
    return feat


def _build_metro_mosaic(
    scenes: list[dict],
    bbox_wgs84: list[float],
    mosaic_path: Path,
) -> tuple[str, int]:
    """Download/merge S2 RGB+NIR for metro bbox once → local 4-band GeoTIFF.

    Returns (crs_str, n_scenes_used).
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    band_names = ("blue", "green", "red", "nir")
    opened_by_band: dict[str, list] = {b: [] for b in band_names}
    crs = None
    with rasterio_env():
        try:
            for sc in scenes:
                try:
                    red_ds = rasterio.open(sc["hrefs"]["red"])
                except Exception:
                    continue
                crs = red_ds.crs
                red_ds.close()
                for b in band_names:
                    try:
                        opened_by_band[b].append(rasterio.open(sc["hrefs"][b]))
                    except Exception:
                        pass
            if not opened_by_band["red"]:
                raise RuntimeError("no readable S2 scenes")

            # project WGS84 bbox into scene CRS
            west, south, east, north = transform_bounds("EPSG:4326", crs, *bbox_wgs84, densify_pts=21)
            bounds = (west, south, east, north)

            arrays = []
            transform = None
            for b in band_names:
                srcs = opened_by_band[b]
                if not srcs:
                    raise RuntimeError(f"missing band {b}")
                mosaic, transform = merge(srcs, bounds=bounds, nodata=0, dtype="float32")
                arrays.append(mosaic[0])

            data = np.stack(arrays, axis=0).astype(np.float32)
            profile = {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "count": 4,
                "dtype": "float32",
                "crs": crs,
                "transform": transform,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "nodata": 0,
            }
            tmp = mosaic_path.with_suffix(".tmp.tif")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data)
                dst.update_tags(bands="blue,green,red,nir")
            tmp.replace(mosaic_path)
            return str(crs), len(opened_by_band["red"])
        finally:
            for srcs in opened_by_band.values():
                for ds in srcs:
                    try:
                        ds.close()
                    except Exception:
                        pass


def process_metro_s2_path(
    region: str,
    buildings_path: Path,
    out_path: Path,
    mosaic_dir: str | None = None,
    allow_download: bool = True,
) -> dict[str, Any]:
    """Fast path: one local S2 mosaic per metro, then sample all buildings from disk.

    If mosaic_dir is set, read `{region}_bgrn.tif` from there (no STAC download).
    Otherwise write/read under out_path's sibling s2_mosaics/ and optionally download.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.ops import transform as shp_transform

    t0 = time.time()
    df = pd.read_parquet(buildings_path)
    bbox = [
        float(df["bbox_west"].min()) - 0.005,
        float(df["bbox_south"].min()) - 0.005,
        float(df["bbox_east"].max()) + 0.005,
        float(df["bbox_north"].max()) + 0.005,
    ]
    if mosaic_dir:
        mosaic_path = Path(mosaic_dir) / f"{region}_bgrn.tif"
    else:
        mosaic_path = out_path.parent.parent / "s2_mosaics" / f"{region}_bgrn.tif"
    n_scenes = 0
    if mosaic_path.exists():
        n_scenes = -1  # reused cache
    elif allow_download:
        scenes = select_scenes(bbox, max_scenes=12)
        if scenes:
            try:
                mosaic_path.parent.mkdir(parents=True, exist_ok=True)
                _, n_scenes = _build_metro_mosaic(scenes, bbox, mosaic_path)
            except Exception as e:
                # fall through to empty features but record error beside output
                err_path = out_path.with_suffix(".mosaic_error.txt")
                err_path.write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                n_scenes = 0
    else:
        n_scenes = 0

    rows: list[dict[str, Any]] = []
    n_ok = 0
    transformers: dict[str, Transformer] = {}

    def utm_metrics(geom, lon: float, lat: float, area_m2: float) -> tuple[float, float]:
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        key = str(epsg)
        if key not in transformers:
            transformers[key] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        tr = transformers[key]
        g2 = shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)
        if not np.isfinite(area_m2) or area_m2 <= 0:
            area_m2 = float(g2.area)
        perim = float(g2.length)
        compact = float(4.0 * np.pi * float(g2.area) / (perim * perim + 1e-12)) if perim > 0 else float("nan")
        return area_m2, compact

    if not mosaic_path.exists():
        for _, row in df.iterrows():
            geom = shapely_wkt.loads(str(row["wkt"]))
            area_m2, compact = utm_metrics(geom, float(row["lon"]), float(row["lat"]), float(row.get("area_m2_approx", np.nan)))
            rows.append(_empty_s2_row(str(row["id"]), region, area_m2, compact))
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        return {
            "region": region,
            "n": len(rows),
            "n_ok": 0,
            "n_scenes": n_scenes,
            "seconds": round(time.time() - t0, 1),
            "status": "no_mosaic",
            "out": str(out_path),
        }

    with rasterio_env():
        with rasterio.open(mosaic_path) as ds:
            trf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            for _, row in df.iterrows():
                bid = str(row["id"])
                geom = shapely_wkt.loads(str(row["wkt"]))
                area_m2, compact = utm_metrics(
                    geom, float(row["lon"]), float(row["lat"]), float(row.get("area_m2_approx", np.nan))
                )
                minx, miny, maxx, maxy = geom.bounds
                xs, ys = trf.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
                rb = (min(xs), min(ys), max(xs), max(ys))
                try:
                    win = from_bounds(*rb, transform=ds.transform)
                    if win.width < 8 or win.height < 8:
                        cx = (rb[0] + rb[2]) / 2
                        cy = (rb[1] + rb[3]) / 2
                        pad = max(ds.res[0] * 8, (rb[2] - rb[0]) / 2, (rb[3] - rb[1]) / 2)
                        win = from_bounds(cx - pad, cy - pad, cx + pad, cy + pad, transform=ds.transform)
                    chip = ds.read(window=win, boundless=True, fill_value=0).astype(np.float32)
                    blue, green, red, nir = chip[0], chip[1], chip[2], chip[3]
                    if (red > 0).sum() >= 4:
                        feat = _features_from_chip(blue, green, red, nir, area_m2=area_m2, compactness=compact)
                        n_ok += int(feat["s2_ok"] > 0.5)
                    else:
                        feat = _empty_s2_row(bid, region, area_m2, compact)
                        feat.pop("id", None)
                        feat.pop("region", None)
                except Exception:
                    feat = _empty_s2_row(bid, region, area_m2, compact)
                    feat.pop("id", None)
                    feat.pop("region", None)
                feat["id"] = bid
                feat["region"] = region
                rows.append(feat)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return {
        "region": region,
        "n": len(out),
        "n_ok": int(n_ok),
        "n_scenes": int(n_scenes),
        "seconds": round(time.time() - t0, 1),
        "status": "ok",
        "mosaic": str(mosaic_path),
        "out": str(out_path),
    }


def build_ae_means(run_dir: Path, regions: list[str]) -> Path:
    """Mean AE embedding of inside_polygon cells per soft-covered building."""
    out = run_dir / "artifacts" / "ae_means.parquet"
    if out.exists():
        return out
    write_progress(run_dir, status="running", stage="ae_means", completed=0, total=len(regions), failed=0)
    parts = []
    for i, region in enumerate(regions):
        t0 = time.time()
        cells = pd.read_parquet(
            CELLS,
            columns=["id", "inside_polygon"] + BANDS,
            filters=[("inside_polygon", "==", True)],
        )
        # filter region via buildings join
        bids = pd.read_parquet(
            run_dir / "artifacts" / "metro_buildings" / f"{region}.parquet",
            columns=["id"],
        )
        bids["id"] = bids["id"].astype(str)
        cells["id"] = cells["id"].astype(str)
        # cells id is building id in this dataset
        sub = cells[cells["id"].isin(set(bids["id"]))].copy()
        means = sub.groupby("id", sort=False)[BANDS].mean().reset_index()
        means["region"] = region
        parts.append(means)
        log(run_dir, f"ae_means {region}: n={len(means):,} in {time.time()-t0:.1f}s")
        write_progress(run_dir, status="running", stage="ae_means", completed=i + 1, total=len(regions), failed=0)
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    return out


def prepare_metro_building_tables(run_dir: Path) -> list[str]:
    w6 = pd.read_parquet(W6_BUILD)
    w6 = w6[w6["soft_gt_covered"] == True].copy()  # noqa: E712
    w6["id"] = w6["id"].astype(str)
    meta = pd.read_parquet(
        BUILDINGS,
        columns=[
            "id",
            "region",
            "lon",
            "lat",
            "wkt",
            "area_m2_approx",
            "bbox_west",
            "bbox_south",
            "bbox_east",
            "bbox_north",
            "y_res",
        ],
    )
    meta["id"] = meta["id"].astype(str)
    df = w6.merge(meta, on=["id", "region"], how="left", suffixes=("", "_m"))
    if "y_res" not in df.columns and "y_res_m" in df.columns:
        df["y_res"] = df["y_res_m"]
    out_dir = run_dir / "artifacts" / "metro_buildings"
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = []
    for region, g in df.groupby("region", sort=True):
        path = out_dir / f"{region}.parquet"
        g.to_parquet(path, index=False)
        regions.append(str(region))
        log(run_dir, f"prepared {region}: n={len(g):,}")
    return regions


def run_s2_extraction(
    run_dir: Path,
    regions: list[str],
    workers: int,
    mosaic_dir: Path | None = None,
    allow_download: bool = True,
) -> pd.DataFrame:
    feat_dir = run_dir / "artifacts" / "s2_feats"
    feat_dir.mkdir(parents=True, exist_ok=True)
    write_progress(
        run_dir,
        status="running",
        stage="s2_extract",
        completed=0,
        total=len(regions),
        failed=0,
        n_cpus=os.cpu_count(),
        workers=workers,
        mosaic_dir=str(mosaic_dir) if mosaic_dir else None,
    )
    pending = []
    for region in regions:
        outp = feat_dir / f"{region}.parquet"
        if outp.exists():
            log(run_dir, f"s2 skip existing {region}")
            continue
        pending.append(region)

    failed = 0
    done = len(regions) - len(pending)
    write_progress(run_dir, completed=done, total=len(regions), failed=failed)

    if pending:
        # Parallelize across metros (local mosaics are disk-bound; use all cores when cached).
        max_w = max(1, min(workers, len(pending), os.cpu_count() or 4))
        log(
            run_dir,
            f"S2 extract metros={pending} workers={max_w} "
            f"mosaic_dir={mosaic_dir} allow_download={allow_download}",
        )
        mosaic_dir_s = str(mosaic_dir) if mosaic_dir is not None else None
        with ProcessPoolExecutor(max_workers=max_w) as pool:
            futs = {
                pool.submit(
                    process_metro_s2_path,
                    region,
                    run_dir / "artifacts" / "metro_buildings" / f"{region}.parquet",
                    feat_dir / f"{region}.parquet",
                    mosaic_dir_s,
                    allow_download,
                ): region
                for region in pending
            }
            for fut in as_completed(futs):
                region = futs[fut]
                try:
                    info = fut.result()
                    log(
                        run_dir,
                        f"s2 {region}: n={info['n']:,} ok={info['n_ok']:,} "
                        f"scenes={info['n_scenes']} in {info['seconds']}s",
                    )
                    atomic_json(run_dir / "checkpoints" / f"s2_{region}.json", info)
                except Exception as e:
                    failed += 1
                    err = {"region": region, "error": str(e), "tb": traceback.format_exc()}
                    atomic_json(run_dir / "failures" / f"s2_{region}.json", err)
                    log(run_dir, f"s2 FAIL {region}: {e}")
                done += 1
                write_progress(
                    run_dir,
                    status="running",
                    stage="s2_extract",
                    completed=done,
                    total=len(regions),
                    failed=failed,
                )

    parts = [pd.read_parquet(feat_dir / f"{r}.parquet") for r in regions if (feat_dir / f"{r}.parquet").exists()]
    if not parts:
        raise RuntimeError("no S2 feature tables written")
    return pd.concat(parts, ignore_index=True)


def calibrate_contrast_ratio_ramp(
    run_dir: Path,
    s2: pd.DataFrame,
    train_ids: set[str],
    locked_eval_ids: set[str],
) -> dict[str, Any]:
    """Fit contrast-ratio fuzzy endpoints on train-only buildings (exclude locked eval).

    Endpoints = p10/p90 of pooled ratio on calibration rows. Orientation: the
    direction that better separates NRES (higher fuzzy 'large' → NRES). If
    |AUC - 0.5| implies AUC < 0.55 toward NRES, drop the term from the score.
    """
    global CONTRAST_RATIO_LO, CONTRAST_RATIO_HI, CONTRAST_RATIO_FLIP, USE_CONTRAST_RATIO_IN_NRES
    from sklearn.metrics import roc_auc_score

    cal_ids = train_ids - locked_eval_ids
    s2 = s2.copy()
    s2["id"] = s2["id"].astype(str)
    # y_res from metro building tables
    y_parts = []
    metro_dir = run_dir / "artifacts" / "metro_buildings"
    for p in sorted(metro_dir.glob("*.parquet")):
        y_parts.append(pd.read_parquet(p, columns=["id", "y_res"]))
    ymap = pd.concat(y_parts, ignore_index=True)
    ymap["id"] = ymap["id"].astype(str)
    m = s2.merge(ymap, on="id", how="inner")
    m = m[m["id"].isin(cal_ids)].copy()
    m = m.dropna(subset=["s2_contrast_ratio", "y_res"])
    ratio = m["s2_contrast_ratio"].to_numpy(dtype=float)
    y_res = m["y_res"].to_numpy(dtype=int)
    y_nres = 1 - y_res

    lo = float(np.nanpercentile(ratio, 10))
    hi = float(np.nanpercentile(ratio, 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.70, 1.00

    # AUC of raw ratio for NRES (positive = higher ratio → NRES)
    try:
        auc_raw = float(roc_auc_score(y_nres, ratio))
    except Exception:
        auc_raw = 0.5
    flip = bool(auc_raw < 0.5)  # if higher ratio → RES, flip so large term → NRES
    # effective separation after orientation
    auc_nres = auc_raw if not flip else (1.0 - auc_raw)
    use_ratio = bool(auc_nres >= 0.55)

    CONTRAST_RATIO_LO = lo
    CONTRAST_RATIO_HI = hi
    CONTRAST_RATIO_FLIP = flip
    USE_CONTRAST_RATIO_IN_NRES = use_ratio

    # Endpoints chosen on train-only (excl. locked 71,497 eval IDs), 2026-11-17 texture fix:
    # p10/p90 of s2_contrast_ratio; flip if raw AUC favors RES; drop term if AUC_nres < 0.55.
    meta = {
        "contrast_ratio_lo": lo,
        "contrast_ratio_hi": hi,
        "contrast_ratio_flip": flip,
        "use_contrast_ratio_in_nres": use_ratio,
        "auc_ratio_toward_nres": auc_nres,
        "auc_ratio_raw_toward_nres": auc_raw,
        "n_calibrate": int(len(m)),
        "n_calibrate_res": int((y_res == 1).sum()),
        "n_calibrate_nres": int((y_res == 0).sum()),
        "median_ratio_res": float(np.nanmedian(ratio[y_res == 1])) if (y_res == 1).any() else float("nan"),
        "median_ratio_nres": float(np.nanmedian(ratio[y_res == 0])) if (y_res == 0).any() else float("nan"),
        "note": (
            f"Endpoints p10={lo:.4f} p90={hi:.4f} on train-excl-locked; "
            f"flip={flip}; use_ratio={use_ratio} (AUC_nres={auc_nres:.3f})"
        ),
    }
    atomic_json(run_dir / "artifacts" / "contrast_ratio_calibration.json", meta)
    log(run_dir, f"contrast_ratio calib: {meta['note']}")
    return meta


def recompute_nres_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute s2_nres_score with calibrated contrast-ratio ramp."""
    out = df.copy()
    scores = []
    for row in out.itertuples(index=False):
        scores.append(
            _fuzzy_nres(
                float(getattr(row, "s2_brightness", np.nan)),
                float(getattr(row, "s2_blue_dom", np.nan)),
                float(getattr(row, "s2_contrast_ratio", np.nan)),
                float(getattr(row, "s2_contrast_small", np.nan)),
                float(getattr(row, "fp_equiv_diam_m", np.nan)),
                float(getattr(row, "fp_compactness", np.nan)),
            )
        )
    out["s2_nres_score"] = np.asarray(scores, dtype=np.float32)
    return out


def build_training_table(run_dir: Path, s2: pd.DataFrame, ae_path: Path) -> pd.DataFrame:
    w6 = pd.read_parquet(W6_BUILD)
    w6 = w6[w6["soft_gt_covered"] == True].copy()  # noqa: E712
    w6["id"] = w6["id"].astype(str)
    s2 = s2.copy()
    s2["id"] = s2["id"].astype(str)
    ae = pd.read_parquet(ae_path)
    ae["id"] = ae["id"].astype(str)
    meta = pd.read_parquet(BUILDINGS, columns=["id", "region", "y_res"])
    meta["id"] = meta["id"].astype(str)
    df = w6.merge(ae, on=["id", "region"], how="inner")
    df = df.merge(s2, on=["id", "region"], how="left", suffixes=("", "_s2"))
    if "y_res" not in df.columns:
        df = df.merge(meta, on=["id", "region"], how="left")
    # fill missing s2
    for c in S2_FEATS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].astype(np.float32)
    for c in BANDS:
        df[c] = df[c].astype(np.float32)
    df.to_parquet(run_dir / "artifacts" / "train_buildings_ae_s2.parquet", index=False)
    log(run_dir, f"training table n={len(df):,} s2_ok={int((df['s2_ok']>0.5).sum()):,}")
    return df


def train_loro_xgb(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    from xgboost import XGBRegressor

    cols = feat_colnames()
    regions = sorted(df["region"].unique())
    oof = np.full((len(df), 2), np.nan, dtype=np.float32)
    fold_metrics = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_progress(run_dir, status="running", stage="train_loro", completed=0, total=len(regions), failed=0)

    for i, hold in enumerate(regions):
        t0 = time.time()
        tr = df["region"] != hold
        te = ~tr
        Xtr = np.nan_to_num(df.loc[tr, cols].to_numpy(np.float32), nan=0.0)
        Ytr = df.loc[tr, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
        Xte = np.nan_to_num(df.loc[te, cols].to_numpy(np.float32), nan=0.0)
        Yte = df.loc[te, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        base = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=max(1, (os.cpu_count() or 8) // max(1, min(2, len(regions)))),
            random_state=SEED,
            tree_method="hist",
        )
        model = MultiOutputRegressor(base)
        model.fit(Xtr_s, Ytr)
        pred = np.clip(model.predict(Xte_s), 0, 1).astype(np.float32)
        oof[te.to_numpy()] = pred
        mae_r = float(mean_absolute_error(Yte[:, 0], pred[:, 0]))
        mae_n = float(mean_absolute_error(Yte[:, 1], pred[:, 1]))
        fold_metrics.append(
            {
                "holdout": hold,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "mae_res": mae_r,
                "mae_nres": mae_n,
                "mae_mean": 0.5 * (mae_r + mae_n),
                "seconds": round(time.time() - t0, 1),
            }
        )
        log(
            run_dir,
            f"LORO hold={hold}: mean={0.5*(mae_r+mae_n):.4f} res={mae_r:.4f} "
            f"nres={mae_n:.4f} n_te={int(te.sum()):,}",
        )
        write_progress(run_dir, status="running", stage="train_loro", completed=i + 1, total=len(regions), failed=0)

    out = df[
        [
            "id",
            "region",
            "soft_gt_frac_res",
            "soft_gt_frac_nres",
            "ae_frac_res",
            "ae_frac_nres",
            "ghsl_frac_res",
            "ghsl_frac_nres",
            "pls_frac_res",
            "pls_frac_nres",
        ]
        + [c for c in ["y_res"] if c in df.columns]
        + S2_FEATS
    ].copy()
    out["s2_frac_res"] = oof[:, 0]
    out["s2_frac_nres"] = oof[:, 1]
    out.to_parquet(run_dir / "artifacts" / "buildings_oof_ae_s2.parquet", index=False)
    pd.DataFrame(fold_metrics).to_csv(run_dir / "artifacts" / "fold_metrics.csv", index=False)
    atomic_json(run_dir / "artifacts" / "fold_metrics.json", fold_metrics)
    return out


def decorate_metro(rows: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in rows}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        out.append(r)
    return out


def score_buildings(run_dir: Path, oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    oof = oof.copy()
    oof["id"] = oof["id"].astype(str)
    m = cohort.merge(oof, on=["id", "region"], how="left", suffixes=("", "_oof"))
    assert len(m) == 71497, f"building cohort size {len(m)} != 71497"

    if XGB_BUILD.exists():
        xb = pd.read_parquet(XGB_BUILD, columns=["id", "xgb_frac_res", "xgb_frac_nres"])
        xb["id"] = xb["id"].astype(str)
        m = m.merge(xb, on="id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ae_frac_res", "ae_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in m.columns:
            soft_pooled[name] = {
                "n": 0,
                "res": float("nan"),
                "nres": float("nan"),
                "mean": float("nan"),
                "model": name,
                "gt": "soft",
            }
            continue
        soft_pooled[name] = {
            **mae_pair(
                m["soft_gt_frac_res"].to_numpy(float),
                m["soft_gt_frac_nres"].to_numpy(float),
                m[pr].to_numpy(float),
                m[pn].to_numpy(float),
            ),
            "model": name,
            "gt": "soft",
        }

    metro = []
    for region, sub in m.groupby("region", sort=False):
        row: dict[str, Any] = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            if pr not in sub.columns:
                continue
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    metro_soft = decorate_metro(metro)
    cov = float(np.mean(np.isfinite(m["s2_frac_res"].to_numpy(float))))
    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_ae_s2.parquet", index=False)
    for name in ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb", "ae_s2"]:
        r = soft_pooled[name]
        log(run_dir, f"building {name:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    return {
        "n_shared": len(m),
        "coverage": cov,
        "soft_pooled": soft_pooled,
        "metro_soft": metro_soft,
        "s2_beats_xgb": sum(
            1
            for r in metro_soft
            if "ae_s2" in r
            and "ae_xgb" in r
            and np.isfinite(r["ae_s2"]["mean"])
            and np.isfinite(r["ae_xgb"]["mean"])
            and r["ae_s2"]["mean"] + 1e-12 < r["ae_xgb"]["mean"]
        ),
    }


def score_tiles_from_buildings(run_dir: Path, building_oof: pd.DataFrame) -> dict[str, Any]:
    """Tile MAE: join locked tiles; AE+S2 tile pred = mean of building OOF in tile if available,
    else leave NaN (coverage reported). Prefer cell-based tile training in a follow-up.
    """
    cohort = pd.read_parquet(COHORT_TILE)
    base = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    base_cols = [c for c in base.columns if c != "region" or "region" not in cohort.columns]
    t = cohort.merge(base[base_cols], on="tile_id", how="left")
    assert len(t) == 25220, f"tile cohort size {len(t)} != 25220"

    # Map buildings → tiles via cell membership is expensive; use building lon/lat → tile_id if present
    # Fallback: no s2 tile preds (coverage 0) but still report baselines for board integrity.
    if "s2_frac_res" not in t.columns:
        t["s2_frac_res"] = np.nan
        t["s2_frac_nres"] = np.nan

    if XGB_TILE.exists():
        xb = pd.read_parquet(XGB_TILE)
        cols = ["tile_id"] + [c for c in xb.columns if c.startswith("xgb_")]
        t = t.merge(xb[cols], on="tile_id", how="left")

    # Build tile preds from mean of soft buildings whose centroid falls in tile
    meta = pd.read_parquet(BUILDINGS, columns=["id", "region", "lon", "lat", "crs", "utm_x", "utm_y"] if False else ["id", "region", "lon", "lat"])
    # get utm from cells? use lonlat + approximate: parse tile_id region|i|j
    bo = building_oof[["id", "region", "s2_frac_res", "s2_frac_nres"]].copy()
    bo["id"] = bo["id"].astype(str)
    meta["id"] = meta["id"].astype(str)
    bm = bo.merge(meta, on=["id", "region"], how="left")
    # assign tile via nearest: need utm — read from metro building tables if present
    # Use week6 tile assignment if buildings have tile_id somewhere; else skip
    # Practical: load cells for inside buildings is heavy — mark tile S2 as optional.
    # Attempt: for each region, assign using building cells mean utm
    try:
        cell_xy = pd.read_parquet(
            CELLS,
            columns=["id", "utm_x", "utm_y", "inside_polygon"],
            filters=[("inside_polygon", "==", True)],
        )
        cell_xy["id"] = cell_xy["id"].astype(str)
        centers = cell_xy.groupby("id", sort=False).agg(utm_x=("utm_x", "mean"), utm_y=("utm_y", "mean")).reset_index()
        bm = bm.merge(centers, on="id", how="left")
        bm = bm[np.isfinite(bm["utm_x"]) & np.isfinite(bm["utm_y"])].copy()
        bm["tile_id"] = (
            bm["region"]
            + "|"
            + (bm["utm_x"] // 100).astype(int).astype(str)
            + "|"
            + (bm["utm_y"] // 100).astype(int).astype(str)
        )
        agg = (
            bm.groupby("tile_id", sort=False)
            .agg(s2_frac_res=("s2_frac_res", "mean"), s2_frac_nres=("s2_frac_nres", "mean"))
            .reset_index()
        )
        t = t.drop(columns=["s2_frac_res", "s2_frac_nres"], errors="ignore").merge(agg, on="tile_id", how="left")
    except Exception as e:
        log(run_dir, f"tile s2 aggregation skipped: {e}")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in t.columns:
            soft_pooled[name] = {
                "n": 0,
                "res": float("nan"),
                "nres": float("nan"),
                "mean": float("nan"),
                "model": name,
                "gt": "soft",
            }
            continue
        soft_pooled[name] = {
            **mae_pair(
                t["soft_gt_frac_res"].to_numpy(float),
                t["soft_gt_frac_nres"].to_numpy(float),
                t[pr].to_numpy(float),
                t[pn].to_numpy(float),
            ),
            "model": name,
            "gt": "soft",
        }
    metro = []
    for region, sub in t.groupby("region", sort=False):
        row: dict[str, Any] = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            if pr not in sub.columns:
                continue
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    t.to_parquet(run_dir / "artifacts" / "tiles_section3_with_ae_s2.parquet", index=False)
    for name in ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb", "ae_s2"]:
        r = soft_pooled[name]
        log(run_dir, f"tile {name:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    return {
        "n_shared": len(t),
        "coverage": float(np.mean(np.isfinite(t["s2_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "metro_soft": decorate_metro(metro),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 10))
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week9_ae_s2_nres_loro")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument(
        "--mosaic-dir",
        type=Path,
        default=None,
        help="Reuse cached metro mosaics ({region}_bgrn.tif). Skips STAC download when files exist.",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Fail metros missing from --mosaic-dir instead of downloading new scenes.",
    )
    ap.add_argument(
        "--copy-ae-means-from",
        type=Path,
        default=None,
        help="Optional path to an existing ae_means.parquet to copy (identical AE features).",
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.resume_dir is not None:
        run_dir = args.resume_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    latest = args.out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    (run_dir / "launch.txt").write_text(
        f"started={utc_now()}\npid={os.getpid()}\nworkers={args.workers}\n"
        f"mosaic_dir={args.mosaic_dir}\nno_download={args.no_download}\n"
        f"texture_fix=2d_nan_blockmean\n"
        f"protocol=docs/eval_protocol_apples_to_apples.md\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpus={os.cpu_count()}\npython={os.sys.executable}\ncwd={os.getcwd()}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="init",
        completed=0,
        total=10,
        failed=0,
        _t0=time.time(),
        output=str(run_dir),
        n_cpus=os.cpu_count(),
        workers=args.workers,
        mosaic_dir=str(args.mosaic_dir) if args.mosaic_dir else None,
    )
    log(run_dir, f"Week9 AE+S2 NRES LORO start → {run_dir}")
    if args.mosaic_dir:
        log(run_dir, f"reusing mosaics from {args.mosaic_dir}")

    regions = prepare_metro_building_tables(run_dir)
    write_progress(run_dir, stage="prepared_metros", completed=1, total=10)

    s2 = run_s2_extraction(
        run_dir,
        regions,
        workers=args.workers,
        mosaic_dir=args.mosaic_dir,
        allow_download=not args.no_download,
    )
    write_progress(run_dir, stage="s2_done", completed=3, total=10)

    # Calibrate contrast-ratio ramp on train buildings excluding locked eval IDs.
    locked_ids = set(pd.read_parquet(COHORT_BUILD, columns=["id"])["id"].astype(str))
    train_ids = set(s2["id"].astype(str))
    calib = calibrate_contrast_ratio_ramp(run_dir, s2, train_ids, locked_ids)
    s2 = recompute_nres_scores(s2)
    # rewrite per-metro feature tables with calibrated nres_score
    feat_dir = run_dir / "artifacts" / "s2_feats"
    for region, g in s2.groupby("region", sort=True):
        g.to_parquet(feat_dir / f"{region}.parquet", index=False)
    s2.to_parquet(run_dir / "artifacts" / "s2_feats_all.parquet", index=False)
    write_progress(run_dir, stage="contrast_calib", completed=4, total=10, **{k: calib[k] for k in ("contrast_ratio_lo", "contrast_ratio_hi", "use_contrast_ratio_in_nres") if k in calib})

    # AE means: filter cells by building ids only (avoid loading all regions repeatedly)
    ae_path = run_dir / "artifacts" / "ae_means.parquet"
    if not ae_path.exists() and args.copy_ae_means_from and Path(args.copy_ae_means_from).exists():
        import shutil

        shutil.copy2(args.copy_ae_means_from, ae_path)
        log(run_dir, f"copied ae_means from {args.copy_ae_means_from}")
    if not ae_path.exists():
        write_progress(run_dir, stage="ae_means", completed=4, total=10)
        # efficient: one pass over cells filtered by all soft ids
        all_ids = set()
        for r in regions:
            ids = pd.read_parquet(run_dir / "artifacts" / "metro_buildings" / f"{r}.parquet", columns=["id"])
            all_ids.update(ids["id"].astype(str).tolist())
        t0 = time.time()
        cells = pd.read_parquet(CELLS, columns=["id", "inside_polygon"] + BANDS)
        cells = cells[cells["inside_polygon"] == True].copy()  # noqa: E712
        cells["id"] = cells["id"].astype(str)
        cells = cells[cells["id"].isin(all_ids)]
        means = cells.groupby("id", sort=False)[BANDS].mean().reset_index()
        # attach region from metro tables
        reg_map = []
        for r in regions:
            g = pd.read_parquet(run_dir / "artifacts" / "metro_buildings" / f"{r}.parquet", columns=["id", "region"])
            reg_map.append(g)
        reg_map = pd.concat(reg_map, ignore_index=True)
        reg_map["id"] = reg_map["id"].astype(str)
        means = means.merge(reg_map, on="id", how="left")
        means.to_parquet(ae_path, index=False)
        log(run_dir, f"ae_means n={len(means):,} in {time.time()-t0:.1f}s")
    write_progress(run_dir, stage="ae_means_done", completed=5, total=10)

    df = build_training_table(run_dir, s2, ae_path)
    write_progress(run_dir, stage="train_table", completed=6, total=10, train_n=len(df))

    oof = train_loro_xgb(df, run_dir)
    write_progress(run_dir, stage="score", completed=8, total=10)

    b_metrics = score_buildings(run_dir, oof)
    t_metrics = score_tiles_from_buildings(run_dir, oof)

    metrics = {
        "protocol": "docs/eval_protocol_apples_to_apples.md",
        "train_n_buildings": int(len(df)),
        "eval_n_buildings": 71497,
        "eval_n_tiles": 25220,
        "features": feat_colnames(),
        "s2_datetime": S2_DATETIME,
        "texture_fix": "2d_nan_blockmean",
        "contrast_ratio_calibration": calib,
        "mosaic_dir": str(args.mosaic_dir) if args.mosaic_dir else None,
        "buildings": b_metrics,
        "tiles": t_metrics,
        "checklist": {
            "eval_ids_locked": True,
            "soft_gt_from_week6": True,
            "spatial_split": "LORO",
            "metric": "0.5*(MAE_res+MAE_nres)",
            "baselines_joined": True,
        },
    }
    atomic_json(run_dir / "artifacts" / "section4_ae_s2_metrics.json", metrics)
    # also write under the name rebuild_section6 expects
    atomic_json(run_dir / "artifacts" / "section6_ae_s2_metrics.json", metrics)
    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "started_at": json.loads((run_dir / "progress.json").read_text()).get("started_at"),
            "finished_at": utc_now(),
            "output": str(run_dir),
            "building_mean_ae_s2": b_metrics["soft_pooled"]["ae_s2"]["mean"],
            "building_mean_ae_xgb": b_metrics["soft_pooled"].get("ae_xgb", {}).get("mean"),
            "building_coverage": b_metrics["coverage"],
            "tile_mean_ae_s2": t_metrics["soft_pooled"]["ae_s2"]["mean"],
            "texture_fix": "2d_nan_blockmean",
            "contrast_ratio_calibration": calib,
            "metrics": str(run_dir / "artifacts" / "section4_ae_s2_metrics.json"),
        },
    )
    write_progress(run_dir, status="completed", stage="done", completed=10, total=10, failed=0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, "DONE")


if __name__ == "__main__":
    main()
