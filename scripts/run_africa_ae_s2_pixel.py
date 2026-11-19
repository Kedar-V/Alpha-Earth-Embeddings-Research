#!/usr/bin/env python3
"""Africa-5 AE vs AE+S2 vs GHSL fraction MAE (pixel-level).

Uses the locked Africa GHSL/AE run's balanced pure cells:
  outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/balanced_pure.parquet

Per city:
  1. Element84 STAC S2 L2A mosaic (BGRN) over the pixel bbox
  2. Sample Week-9-style S2 feats at each AE 10 m center (5×5 chip)
  3. Within-city 5-fold Ridge OOF for AE-64 and AE+S2 (1 city/country)
  4. Score MAE mean(res, nres) vs soft fractions; compare to GHSL BUILT-S

Note: city-LORO is impossible with 1 city/country — same KFold fallback as
the Africa GHSL AE heads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"A{i:02d}" for i in range(64)]
AE_COLS = [f"a{i:02d}" for i in range(64)]

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

# Pixel footprint proxies (10 m AE cell)
FP_AREA_M2 = 100.0
FP_COMPACT = 1.0

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
DEFAULT_BALANCED = (
    ROOT
    / "outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/balanced_pure.parquet"
)
DEFAULT_SCORED = (
    ROOT
    / "outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/scored.parquet"
)

# Prefer dry-season windows; tropical cities get a wider / cloudier search.
CITY_S2_DATETIME = {
    "EGY_CAIRO": "2018-05-01T00:00:00Z/2018-09-30T23:59:59Z",
    "GHA_ACCRA": "2017-12-01T00:00:00Z/2018-02-28T23:59:59Z",
    "KEN_NAIROBI": "2018-06-01T00:00:00Z/2018-09-30T23:59:59Z",
    "NGA_LAGOS": "2017-12-01T00:00:00Z/2018-02-28T23:59:59Z",
    "ZAF_JOHANNESBURG": "2018-06-01T00:00:00Z/2018-08-31T23:59:59Z",
}
CITY_CLOUD_LT = {
    "EGY_CAIRO": 30,
    "GHA_ACCRA": 40,
    "KEN_NAIROBI": 30,
    "NGA_LAGOS": 45,
    "ZAF_JOHANNESBURG": 25,
}


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


def mae_rn(yt_r, yt_n, yp_r, yp_n) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yt_r) & np.isfinite(yt_n)
    if int(m.sum()) == 0:
        return {
            "n": 0,
            "mae_mean_res_nres": float("nan"),
            "mae_frac_res": float("nan"),
            "mae_frac_nres": float("nan"),
        }
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_r[m], yp_r[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_n[m], yp_n[m])),
        "mae_mean_res_nres": float(
            0.5
            * (
                mean_absolute_error(yt_r[m], yp_r[m])
                + mean_absolute_error(yt_n[m], yp_n[m])
            )
        ),
    }


def rasterio_env():
    import rasterio

    return rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        AWS_NO_SIGN_REQUEST="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
    )


def _asset_href(feat: dict, keys: tuple[str, ...]) -> str | None:
    assets = feat.get("assets", {})
    for k in keys:
        if k in assets and "href" in assets[k]:
            return assets[k]["href"]
    return None


def stac_search(bbox: list[float], datetime_range: str, cloud_lt: float, limit: int = 50) -> list[dict]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": cloud_lt}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    r = requests.post(STAC_URL, json=body, timeout=120)
    r.raise_for_status()
    return list(r.json().get("features", []))


def select_scenes(bbox: list[float], datetime_range: str, cloud_lt: float, max_scenes: int = 4) -> list[dict]:
    feats = stac_search(bbox, datetime_range, cloud_lt, limit=40)
    if len(feats) < 2 and cloud_lt < 70:
        feats = stac_search(bbox, datetime_range, min(80.0, cloud_lt + 35), limit=40)
    picked: list[dict] = []
    seen_tiles: set[str] = set()
    for f in feats:
        props = f.get("properties", {})
        tile = props.get("grid:code") or props.get("mgrs:utm_zone") or f.get("id")
        hrefs = {
            "blue": _asset_href(f, ("blue", "B02")),
            "green": _asset_href(f, ("green", "B03")),
            "red": _asset_href(f, ("red", "B04")),
            "nir": _asset_href(f, ("nir", "B08", "nir08")),
        }
        if not all(hrefs.values()):
            continue
        mgrs = str(tile)
        # Prefer diverse MGRS tiles early, then allow duplicates for denser fill.
        if mgrs in seen_tiles and len(seen_tiles) < 3 and len(picked) >= 2:
            continue
        seen_tiles.add(mgrs)
        picked.append(
            {
                "id": f["id"],
                "cloud": props.get("eo:cloud_cover"),
                "hrefs": hrefs,
            }
        )
        if len(picked) >= max_scenes:
            break
    return picked


def _contrast(arr: np.ndarray) -> float:
    a = arr.astype(np.float32)
    if a.size < 4 or not np.isfinite(a).any():
        return float("nan")
    a = a[np.isfinite(a)]
    if a.size < 4:
        return float("nan")
    return float(np.std(a))


def _fuzzy_nres(
    brightness: float,
    blue_dom: float,
    contrast_ratio: float,
    contrast_small: float,
    equiv_diam: float,
    compactness: float,
) -> float:
    def clamp01(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))

    b_hi = clamp01((brightness - 800.0) / 2500.0)
    blue = clamp01((blue_dom - 0.30) / 0.25)
    large = clamp01((contrast_ratio - 1.0) / 2.0)
    smooth = clamp01(1.0 - (contrast_small / 400.0))
    size = clamp01((equiv_diam - 20.0) / 60.0)
    compact = clamp01((compactness - 0.4) / 0.5)
    return float(np.mean([b_hi, blue, large, smooth, size, compact]))


def _features_from_chip(blue: np.ndarray, green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> dict[str, float]:
    def mean_finite(a: np.ndarray) -> float:
        m = np.isfinite(a)
        return float(np.mean(a[m])) if m.any() else float("nan")

    b = blue.astype(np.float32)
    g = green.astype(np.float32)
    r = red.astype(np.float32)
    n = nir.astype(np.float32)
    mask = (b > 0) | (g > 0) | (r > 0)
    if mask.any():
        b, g, r, n = b[mask], g[mask], r[mask], n[mask]
    mb, mg, mr, mn = mean_finite(b), mean_finite(g), mean_finite(r), mean_finite(n)
    brightness = np.nanmean([mb, mg, mr])
    blue_dom = mb / (mb + mg + mr + 1e-6) if np.isfinite(mb) else float("nan")
    ndvi = (mn - mr) / (mn + mr + 1e-6) if np.isfinite(mn) and np.isfinite(mr) else float("nan")
    lum = (0.3 * r + 0.59 * g + 0.11 * b) if r.size else np.array([], dtype=np.float32)
    c_small = _contrast(lum)
    if lum.size >= 16:
        side = int(np.floor(np.sqrt(lum.size)))
        lum2 = lum[: side * side].reshape(side, side)
        c_large = _contrast(lum2[::2, ::2])
    else:
        c_large = c_small
    c_ratio = (c_large / (c_small + 1e-6)) if np.isfinite(c_small) else float("nan")
    equiv = float(2.0 * np.sqrt(FP_AREA_M2 / np.pi))
    nres = _fuzzy_nres(brightness, blue_dom, c_ratio, c_small, equiv, FP_COMPACT)
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
        "fp_area_m2": FP_AREA_M2,
        "fp_equiv_diam_m": equiv,
        "fp_compactness": FP_COMPACT,
        "s2_nres_score": nres,
        "s2_n_pix": float(int(mask.sum()) if mask.size else 0),
        "s2_ok": 1.0 if np.isfinite(mb) and int(mask.sum()) >= 4 else 0.0,
    }


def _empty_s2_feats() -> dict[str, float]:
    equiv = float(2.0 * np.sqrt(FP_AREA_M2 / np.pi))
    feat = {k: float("nan") for k in S2_FEATS}
    feat.update(
        {
            "fp_area_m2": FP_AREA_M2,
            "fp_equiv_diam_m": equiv,
            "fp_compactness": FP_COMPACT,
            "s2_nres_score": 0.0,
            "s2_n_pix": 0.0,
            "s2_ok": 0.0,
        }
    )
    return feat


def _build_city_mosaic(scenes: list[dict], bbox_wgs84: list[float], mosaic_path: Path) -> tuple[str, int]:
    """Window-read S2 COGs into a local BGRN mosaic for the city bbox.

    Avoids rasterio.merge over many HTTP COGs (hangs on large Africa AOIs).
    Composites up to a few single-scene window reads with nanmax fill.
    """
    import rasterio
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.warp import transform_bounds, reproject, Resampling

    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    band_names = ("blue", "green", "red", "nir")
    # Target grid in WGS84 lon/lat at ~10 m (~1e-4 deg) — simple & CRS-stable.
    west, south, east, north = bbox_wgs84
    # ~10 m in deg at equator; fine enough for chip stats
    res = 0.0001
    width = max(32, int(np.ceil((east - west) / res)))
    height = max(32, int(np.ceil((north - south) / res)))
    # Cap to keep memory sane (~8k px)
    if width > 8000 or height > 8000:
        scale = max(width / 8000, height / 8000)
        width = int(width / scale)
        height = int(height / scale)
        res = (east - west) / width
    dst_transform = transform_from_bounds(west, south, east, north, width, height)
    acc = np.zeros((4, height, width), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    n_used = 0
    last_err: Exception | None = None

    with rasterio_env():
        for sc in scenes:
            if n_used >= 3:
                break
            try:
                band_arrs = []
                for b in band_names:
                    with rasterio.open(sc["hrefs"][b]) as src:
                        dst = np.zeros((height, width), dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=dst,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs="EPSG:4326",
                            resampling=Resampling.bilinear,
                            src_nodata=0,
                            dst_nodata=0,
                        )
                        band_arrs.append(dst)
                stack = np.stack(band_arrs, axis=0)
                mask = stack[0] > 0
                if float(mask.mean()) < 0.02:
                    continue
                # fill only still-empty pixels
                fill = mask & ~covered
                if not fill.any():
                    continue
                for i in range(4):
                    acc[i][fill] = stack[i][fill]
                covered |= fill
                n_used += 1
                if float(covered.mean()) >= 0.85:
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

    if n_used == 0 or float(covered.mean()) < 0.02:
        raise RuntimeError(f"no usable S2 window composite ({last_err!r})")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }
    tmp = mosaic_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(acc)
        dst.update_tags(bands="blue,green,red,nir", n_scenes=str(n_used), covered=f"{float(covered.mean()):.3f}")
    tmp.replace(mosaic_path)
    return "EPSG:4326", n_used


def _sample_pixels_from_mosaic(
    mosaic_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    chip: int = 5,
) -> list[dict[str, float]]:
    import rasterio
    from pyproj import Transformer

    half = chip // 2
    out: list[dict[str, float]] = []
    with rasterio_env():
        with rasterio.open(mosaic_path) as ds:
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = tr.transform(lons, lats)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            # Affine inverse → col/row (vectorized)
            inv = ~ds.transform
            cols, rows = inv * (xs, ys)
            cols = np.asarray(cols, dtype=np.float64)
            rows = np.asarray(rows, dtype=np.float64)
            data = ds.read()  # (4, H, W)
            H, W = int(data.shape[1]), int(data.shape[2])
            for r_f, c_f in zip(rows, cols):
                if not (np.isfinite(r_f) and np.isfinite(c_f)):
                    out.append(_empty_s2_feats())
                    continue
                r, c = int(r_f), int(c_f)
                if r < 0 or c < 0 or r >= H or c >= W:
                    out.append(_empty_s2_feats())
                    continue
                r0, r1 = max(0, r - half), min(H, r + half + 1)
                c0, c1 = max(0, c - half), min(W, c + half + 1)
                chip_arr = data[:, r0:r1, c0:c1]
                out.append(
                    _features_from_chip(chip_arr[0], chip_arr[1], chip_arr[2], chip_arr[3])
                )
    return out


def process_city_s2(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: mosaic + sample one city → parquet of S2 feats keyed by pixel_id."""
    city_id = payload["city_id"]
    pixels_path = Path(payload["pixels_path"])
    out_path = Path(payload["out_path"])
    mosaic_path = Path(payload["mosaic_path"])
    run_dir = Path(payload["run_dir"])
    t0 = time.time()
    if out_path.exists():
        df = pd.read_parquet(out_path)
        return {
            "city_id": city_id,
            "status": "cached",
            "n": len(df),
            "s2_ok": float(df["s2_ok"].mean()) if len(df) else 0.0,
            "seconds": round(time.time() - t0, 1),
            "out": str(out_path),
        }

    df = pd.read_parquet(pixels_path)
    pad = 0.005
    bbox = [
        float(df["pixel_lon"].min()) - pad,
        float(df["pixel_lat"].min()) - pad,
        float(df["pixel_lon"].max()) + pad,
        float(df["pixel_lat"].max()) + pad,
    ]
    dt = CITY_S2_DATETIME.get(city_id, "2018-06-01T00:00:00Z/2018-09-30T23:59:59Z")
    cloud = float(CITY_CLOUD_LT.get(city_id, 30))
    max_scenes = int(payload.get("max_scenes", 4))
    scenes = select_scenes(bbox, dt, cloud, max_scenes=max_scenes)
    if len(scenes) < 2:
        # broader seasonal / cloud fallback for sparse tropics
        scenes = select_scenes(bbox, "2018-01-01T00:00:00Z/2018-12-31T23:59:59Z", 70, max_scenes=max_scenes)
    if not scenes:
        rows = []
        for pid in df["pixel_id"].to_numpy():
            feat = _empty_s2_feats()
            feat["pixel_id"] = pid
            feat["city_id"] = city_id
            rows.append(feat)
        out = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        return {
            "city_id": city_id,
            "status": "no_scenes",
            "n": len(out),
            "s2_ok": 0.0,
            "seconds": round(time.time() - t0, 1),
            "out": str(out_path),
            "bbox": bbox,
        }

    if not mosaic_path.exists():
        try:
            crs, n_sc = _build_city_mosaic(scenes, bbox, mosaic_path)
        except Exception as exc:
            err = mosaic_path.with_suffix(".error.txt")
            err.write_text(repr(exc) + "\n")
            rows = []
            for pid in df["pixel_id"].to_numpy():
                feat = _empty_s2_feats()
                feat["pixel_id"] = pid
                feat["city_id"] = city_id
                rows.append(feat)
            out = pd.DataFrame(rows)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(out_path, index=False)
            return {
                "city_id": city_id,
                "status": "mosaic_fail",
                "error": repr(exc),
                "n": len(out),
                "s2_ok": 0.0,
                "seconds": round(time.time() - t0, 1),
                "out": str(out_path),
            }
    else:
        crs, n_sc = "cached", 0

    feats = _sample_pixels_from_mosaic(
        mosaic_path,
        df["pixel_lon"].to_numpy(np.float64),
        df["pixel_lat"].to_numpy(np.float64),
        chip=5,
    )
    out = pd.DataFrame(feats)
    out["pixel_id"] = df["pixel_id"].to_numpy()
    out["city_id"] = city_id
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return {
        "city_id": city_id,
        "status": "ok",
        "n": len(out),
        "s2_ok": float(out["s2_ok"].mean()),
        "n_scenes": n_sc,
        "crs": crs,
        "bbox": bbox,
        "datetime": dt,
        "cloud_lt": cloud,
        "seconds": round(time.time() - t0, 1),
        "out": str(out_path),
        "mosaic": str(mosaic_path),
    }


def ensure_ae_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a, A in zip(AE_COLS, BANDS):
        if A not in out.columns and a in out.columns:
            out[A] = out[a].astype(np.float32)
    return out


def kfold_ridge_oof(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_splits: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    X = df[feat_cols].to_numpy(np.float64)
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(np.float64)
    # impute non-finite feats with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~np.isfinite(col)] = med
        X[:, j] = col
    oof = np.full_like(Y, np.nan)
    n = len(df)
    splits = n_splits if n >= 500 else max(2, min(5, n // 50 or 2))
    if n < 2 * splits:
        splits = 2 if n >= 4 else 0
    if splits < 2:
        return oof
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    for tr, te in kf.split(X):
        if len(tr) < 50:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        est = Ridge(alpha=alpha)
        est.fit(Xtr, Y[tr])
        pred = np.clip(est.predict(Xte), 0.0, 1.0)
        s = pred.sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        oof[te] = pred / s
    return oof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--balanced", type=Path, default=DEFAULT_BALANCED)
    ap.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    ap.add_argument("--workers", type=int, default=max(1, min(5, (os.cpu_count() or 4))))
    ap.add_argument("--cities", nargs="*", default=None, help="optional city_id filter")
    ap.add_argument("--max-scenes", type=int, default=4, help="S2 scenes per city mosaic (keep small)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "africa_ae_s2_pixel" / ts)
    for sub in ("checkpoints", "artifacts", "figures", "failures", "s2_mosaics", "s2_feats", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    (run_dir / "launch.txt").write_text(
        f"cmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n"
        f"balanced={args.balanced}\nscored={args.scored}\nworkers={args.workers}\n"
    )

    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="load",
        started_at=utc_now(),
        _t0=t0,
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count(), "gpus": 0, "workers": args.workers},
    )
    log(run_dir, f"Run dir {run_dir}")
    if not args.balanced.exists():
        raise SystemExit(f"missing balanced parquet: {args.balanced}")

    bal = ensure_ae_bands(pd.read_parquet(args.balanced))
    if args.cities:
        bal = bal[bal["city_id"].isin(args.cities)].copy()
    cities = sorted(bal["city_id"].unique())
    log(run_dir, f"Balanced pure n={len(bal):,} cities={cities}")
    write_progress(run_dir, stage="s2_extract", total=len(cities), completed=0, cities=cities)

    # Per-city pixel shards for workers
    payloads = []
    for cid in cities:
        shard = run_dir / "checkpoints" / f"pixels_{cid}.parquet"
        if not shard.exists():
            bal.loc[bal["city_id"] == cid, ["pixel_id", "pixel_lon", "pixel_lat"]].to_parquet(
                shard, index=False
            )
        payloads.append(
            {
                "city_id": cid,
                "pixels_path": str(shard),
                "out_path": str(run_dir / "s2_feats" / f"{cid}.parquet"),
                "mosaic_path": str(run_dir / "s2_mosaics" / f"{cid}_bgrn.tif"),
                "run_dir": str(run_dir),
                "max_scenes": int(args.max_scenes),
            }
        )

    results = []
    # Process cities in parallel (I/O + mosaic heavy)
    n_workers = max(1, min(int(args.workers), len(payloads)))
    log(run_dir, f"S2 extract cities={len(payloads)} workers={n_workers}")
    if n_workers == 1:
        for i, p in enumerate(payloads, 1):
            r = process_city_s2(p)
            results.append(r)
            log(
                run_dir,
                f"S2 {r['city_id']}: status={r['status']} n={r['n']:,} "
                f"s2_ok={r.get('s2_ok', 0):.3f} t={r['seconds']}s",
            )
            write_progress(run_dir, completed=i, current_city=r["city_id"])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(process_city_s2, p): p["city_id"] for p in payloads}
            done_n = 0
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    (run_dir / "failures" / f"{cid}.txt").write_text(repr(exc) + "\n")
                    r = {"city_id": cid, "status": "failed", "error": repr(exc), "n": 0, "s2_ok": 0.0, "seconds": 0}
                results.append(r)
                done_n += 1
                log(
                    run_dir,
                    f"S2 {r['city_id']}: status={r.get('status')} n={r.get('n', 0):,} "
                    f"s2_ok={float(r.get('s2_ok', 0)):.3f} t={r.get('seconds', 0)}s",
                )
                write_progress(run_dir, completed=done_n, current_city=cid)

    atomic_json(run_dir / "artifacts" / "s2_extract_summary.json", {"results": results})
    s2_parts = []
    for cid in cities:
        p = run_dir / "s2_feats" / f"{cid}.parquet"
        if p.exists():
            s2_parts.append(pd.read_parquet(p))
    if not s2_parts:
        raise SystemExit("no S2 features produced")
    s2 = pd.concat(s2_parts, ignore_index=True)
    log(run_dir, f"S2 feats n={len(s2):,} s2_ok_rate={float(s2['s2_ok'].mean()):.3f}")

    write_progress(run_dir, stage="train_oof", last_progress_at=utc_now())
    merged = bal.merge(s2, on=["pixel_id", "city_id"], how="left", suffixes=("", "_s2"))
    for c in S2_FEATS:
        if c not in merged.columns:
            merged[c] = np.nan

    oof_parts = []
    for cid, g in merged.groupby("city_id", sort=False):
        g = g.reset_index(drop=True)
        log(run_dir, f"KFold Ridge AE / AE+S2 {cid} n={len(g):,}")
        ae_oof = kfold_ridge_oof(g, BANDS)
        aes2_oof = kfold_ridge_oof(g, BANDS + S2_FEATS)
        part = g[
            [
                "pixel_id",
                "city_id",
                "country",
                "frac_res_full",
                "frac_nres_full",
                "frac_empty_full",
                "pixel_lon",
                "pixel_lat",
            ]
            + S2_FEATS
        ].copy()
        part["ae_frac_res"] = ae_oof[:, 0]
        part["ae_frac_nres"] = ae_oof[:, 1]
        part["ae_frac_empty"] = ae_oof[:, 2]
        part["aes2_frac_res"] = aes2_oof[:, 0]
        part["aes2_frac_nres"] = aes2_oof[:, 1]
        part["aes2_frac_empty"] = aes2_oof[:, 2]
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)

    # Attach GHSL from prior scored run if available
    if args.scored.exists():
        sc = pd.read_parquet(
            args.scored,
            columns=[
                "pixel_id",
                "city_id",
                "bs_frac_res",
                "bs_frac_nres",
                "ridge_frac_res",
                "ridge_frac_nres",
            ],
        )
        oof = oof.merge(sc, on=["pixel_id", "city_id"], how="left")
        log(run_dir, f"Joined GHSL/prior AE from {args.scored}")
    else:
        oof["bs_frac_res"] = np.nan
        oof["bs_frac_nres"] = np.nan
        oof["ridge_frac_res"] = np.nan
        oof["ridge_frac_nres"] = np.nan

    oof.to_parquet(run_dir / "artifacts" / "oof_ae_s2.parquet", index=False)

    write_progress(run_dir, stage="score", last_progress_at=utc_now())
    rows = []
    scopes = [("all", oof)]
    scopes += [(f"country:{c}", g) for c, g in oof.groupby("country")]
    scopes += [(f"city:{c}", g) for c, g in oof.groupby("city_id")]
    models = [
        ("ae_ridge_kfold", "ae_frac_res", "ae_frac_nres"),
        ("ae_s2_ridge_kfold", "aes2_frac_res", "aes2_frac_nres"),
        ("ae_ridge_prior", "ridge_frac_res", "ridge_frac_nres"),
        ("ghsl_built_s", "bs_frac_res", "bs_frac_nres"),
    ]
    for scope, sub in scopes:
        yt_r, yt_n = sub["frac_res_full"].to_numpy(), sub["frac_nres_full"].to_numpy()
        for name, a, b in models:
            if a not in sub.columns:
                continue
            met = mae_rn(yt_r, yt_n, sub[a].to_numpy(), sub[b].to_numpy())
            met.update({"model": name, "scope": scope})
            rows.append(met)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)

    country_tbl = (
        summary[summary.scope.str.startswith("country:")]
        .assign(country=lambda d: d.scope.str.replace("country:", "", regex=False))
        .pivot_table(index="country", columns="model", values="mae_mean_res_nres", aggfunc="first")
    )
    country_tbl.to_csv(run_dir / "mae_by_country.csv")
    log(run_dir, "MAE by country:\n" + country_tbl.to_string())

    write_progress(run_dir, stage="figures", last_progress_at=utc_now())
    fig, ax = plt.subplots(figsize=(9, 4.0))
    countries_sorted = sorted(country_tbl.index)
    x = np.arange(len(countries_sorted))
    plot_models = [
        ("ghsl_built_s", "#ff7f0e"),
        ("ae_ridge_kfold", "#2ca02c"),
        ("ae_s2_ridge_kfold", "#9467bd"),
    ]
    w = 0.25
    for i, (model, color) in enumerate(plot_models):
        vals = []
        for c in countries_sorted:
            if model in country_tbl.columns and c in country_tbl.index:
                vals.append(float(country_tbl.loc[c, model]))
            else:
                vals.append(np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=model, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(countries_sorted)
    ax.set_ylabel("MAE mean(res, nres) ↓")
    ax.set_title("Africa-5 · AE vs AE+S2 vs GHSL (within-city KFold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mae_by_country.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = []
    for c in countries_sorted:
        def _v(m: str) -> float:
            if m not in country_tbl.columns:
                return float("nan")
            return float(country_tbl.loc[c, m])

        lines.append(
            f"{c}: GHSL={_v('ghsl_built_s'):.3f} AE={_v('ae_ridge_kfold'):.3f} "
            f"AE+S2={_v('ae_s2_ridge_kfold'):.3f}"
        )
    verdict = " | ".join(lines)
    log(run_dir, verdict)

    payload = {
        "status": "completed",
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "balanced": str(args.balanced),
        "scored": str(args.scored),
        "cities": cities,
        "n": int(len(oof)),
        "s2_ok_rate": float(s2["s2_ok"].mean()),
        "protocol": "within-city 5-fold Ridge; AE-64 vs AE+S2 Week9-style feats; GHSL from prior Africa run",
        "verdict": verdict,
        "mae_by_country": country_tbl.reset_index().to_dict(orient="records"),
        "s2_extract": results,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    atomic_json(run_dir / "summary.json", payload)
    write_progress(run_dir, status="completed", stage="done", percent_complete=100.0, verdict=verdict)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, f"DONE · {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
