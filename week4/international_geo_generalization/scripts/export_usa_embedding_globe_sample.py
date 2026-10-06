#!/usr/bin/env python3
"""Sample AlphaEarth cells (USA / India / Japan) for the globe + PLS viewer.

Observability (submit-job):
  viz/usa_embedding_globe/runs/<stamp>/
    job.log, job.pid, progress.json, summary.json, DONE, launch.txt

Durable outputs (shared / resume-safe):
  viz/usa_embedding_globe/data/points.json
  viz/usa_embedding_globe/data/chips/{cell_id}_{patch|context|context500}.jpg
  viz/usa_embedding_globe/data/intl_folds/fold_{CITY}.joblib
  viz/usa_embedding_globe/data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `from intl_globe_folds import ...` when launched as a script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz" / "usa_embedding_globe"
DATA = OUT / "data"
CHIPS = DATA / "chips"

# XYZ tiles parallelize better than the export API and avoid tiny-bbox 500s.
ESRI_TILE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

A_COLS = [f"a{i:02d}" for i in range(64)]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "aef-international-viz/1.0"})


def connect():
    import sys

    sys.path.insert(0, str(ROOT))
    from lib.db import connect as _connect, load_dotenv

    load_dotenv()
    return _connect()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def update_progress(run_dir: Path, **kwargs) -> None:
    path = run_dir / "progress.json"
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text())
        except Exception:
            cur = {}
    cur.update(kwargs)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    cur["updated_at"] = now
    if kwargs.get("status") == "running" or "completed" in kwargs:
        cur["last_progress_at"] = now
    atomic_json(path, cur)


def sample_cells(
    n_per_city: int,
    seed: int,
    countries: list[str] | None = None,
    cities: list[str] | None = None,
    exclude_cell_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Sample pure res / non-res cells with 1:1 class balance per city.

    Takes ``n_per_city // 2`` cells from each ``binary_label`` ∈ {0,1} per city
    (random md5 order). Total per city is even; odd ``n_per_city`` floors.

    Optional ``cities`` limits to those city_ids. ``exclude_cell_ids`` skips
    already-drawn cells (for weak-city top-ups).
    """
    countries = [c.upper() for c in (countries or ["USA"])]
    prefixes = [f"{c}_" for c in countries]
    n_per_class = max(1, int(n_per_city) // 2)
    emb = ", ".join(f"c.{a}" for a in A_COLS)
    prefix_sql = " OR ".join(["starts_with(c.city_id, %s)"] * len(prefixes))
    city_filter_sql = ""
    exclude_sql = ""
    params: list = [str(seed), *prefixes]
    if cities:
        city_filter_sql = " AND c.city_id = ANY(%s)"
        params.append(list(cities))
    if exclude_cell_ids:
        exclude_sql = " AND NOT (c.cell_id = ANY(%s))"
        params.append(list(int(x) for x in exclude_cell_ids))
    params.append(int(n_per_class))
    sql = f"""
    WITH ranked AS (
      SELECT
        c.cell_id,
        c.city_id,
        ci.city_name,
        ST_X(ST_Centroid(c.geom)) AS lon,
        ST_Y(ST_Centroid(c.geom)) AS lat,
        ST_XMin(c.geom) AS west,
        ST_YMin(c.geom) AS south,
        ST_XMax(c.geom) AS east,
        ST_YMax(c.geom) AS north,
        cl.binary_label,
        cl.frac_res,
        cl.frac_nres,
        cl.frac_empty,
        cl.purity_status,
        m.frac_res_full,
        m.frac_nres_full,
        m.frac_empty_full,
        m.area_res_m2,
        m.area_nres_m2,
        {emb},
        row_number() OVER (
          PARTITION BY c.city_id, cl.binary_label
          ORDER BY md5(c.cell_id::text || %s::text)
        ) AS rn
      FROM alphaearth_cells c
      JOIN cities ci ON ci.city_id = c.city_id
      JOIN cell_labels cl
        ON cl.cell_id = c.cell_id AND cl.label_version = 'resi_def_v1_pixel_frac_v1'
      LEFT JOIN cell_label_metrics m
        ON m.cell_id = c.cell_id AND m.label_version = cl.label_version
      WHERE ({prefix_sql})
        AND c.a00 IS NOT NULL
        AND cl.binary_label IN (0, 1)
        {city_filter_sql}
        {exclude_sql}
    )
    SELECT * FROM ranked WHERE rn <= %s
    ORDER BY city_id, binary_label, rn
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for a in A_COLS:
        df[a] = pd.to_numeric(df[a], errors="coerce")
    df["binary_label"] = pd.to_numeric(df["binary_label"], errors="coerce").astype(int)
    df["country"] = df["city_id"].astype(str).str.split("_").str[0]
    df = df.dropna(subset=A_COLS).reset_index(drop=True)

    keep = []
    for city_id, g in df.groupby("city_id", sort=True):
        n0 = int((g["binary_label"] == 0).sum())
        n1 = int((g["binary_label"] == 1).sum())
        if n0 < n_per_class or n1 < n_per_class:
            print(
                f"  WARN {city_id}: unbalanced pool res={n1} nres={n0} "
                f"(need {n_per_class}/class) — taking min",
                flush=True,
            )
        k = min(n0, n1, n_per_class)
        if k < 1:
            print(f"  DROP {city_id}: no both-class sample", flush=True)
            continue
        g0 = g[g["binary_label"] == 0].head(k)
        g1 = g[g["binary_label"] == 1].head(k)
        keep.append(pd.concat([g0, g1], ignore_index=True))
    if not keep:
        return pd.DataFrame()
    out = pd.concat(keep, ignore_index=True)
    out.attrs["n_per_class"] = n_per_class
    out.attrs["balanced"] = True
    return out


def flag_weak_cities(
    city_ids: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    *,
    recall_floor: float = 0.15,
    mono_floor: float = 0.05,
    min_hi: int = 3,
) -> list[dict]:
    """Cities where LORO collapses to one class / lacks hi-conf correct endpoints."""
    out = []
    for city in sorted(set(map(str, city_ids))):
        m = np.asarray([str(c) == city for c in city_ids])
        yt, pt, pr = y[m], pred[m], proba[m]
        n = int(m.sum())
        if n == 0:
            continue
        gt0 = int((yt == 0).sum())
        gt1 = int((yt == 1).sum())
        pr0 = int((pt == 0).sum())
        pr1 = int((pt == 1).sum())
        corr0 = int(((pt == yt) & (yt == 0)).sum())
        corr1 = int(((pt == yt) & (yt == 1)).sum())
        r0 = corr0 / gt0 if gt0 else 1.0
        r1 = corr1 / gt1 if gt1 else 1.0
        hi0 = int(((pt == yt) & (yt == 0) & np.isfinite(pr) & (pr < 0.1)).sum())
        hi1 = int(((pt == yt) & (yt == 1) & np.isfinite(pr) & (pr > 0.9)).sum())
        mono = min(pr0, pr1) / n < mono_floor
        bad_recall = r0 < recall_floor or r1 < recall_floor
        thin_hi = hi0 < min_hi or hi1 < min_hi
        if mono or bad_recall or thin_hi:
            reasons = []
            if mono:
                reasons.append("pred_mono")
            if bad_recall:
                reasons.append(f"recall_res={r1:.2f}/nres={r0:.2f}")
            if thin_hi:
                reasons.append(f"hi_conf_res={hi1}/nres={hi0}")
            out.append(
                {
                    "city_id": city,
                    "n": n,
                    "recall_res": round(r1, 3),
                    "recall_nres": round(r0, 3),
                    "pred_res": pr1,
                    "pred_nres": pr0,
                    "hi_res": hi1,
                    "hi_nres": hi0,
                    "reasons": reasons,
                }
            )
    return out


CITY_TO_WEEK4_REGION = {
    "USA_ATLANTA": "Atlanta_SE",
    "USA_BOSTON": "Boston_NE",
    "USA_CHICAGO": "Chicago_MW",
    "USA_DENVER": "Denver_W",
    "USA_HOUSTON": "Houston_S",
    "USA_LOS_ANGELES": "LA_W",
    "USA_NEW_YORK": "NYC_NE",
    "USA_PHOENIX": "Phoenix_SW",
    "USA_SAN_FRANCISCO": "SF_W",
    "USA_SEATTLE": "Seattle_NW",
}

WEEK4_CKPT_DIR = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "week4_pls_loro"
    / "checkpoints"
)


def week4_baseline_lr_predictions(
    X: np.ndarray,
    y: np.ndarray,
    city_ids: np.ndarray,
    ckpt_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply week4 leave-one-metro-out 64-D logistic (baseline_lr) fold checkpoints.

    For each viz pixel, uses the fold trained with that metro held out — same
    protocol as ``scripts/train_week4_pls_loro.py`` / week4 geo subspace.
    """
    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    ckpt_dir = Path(ckpt_dir or WEEK4_CKPT_DIR)
    n = len(y)
    pred = np.full(n, -1, dtype=np.int64)
    proba = np.full(n, np.nan, dtype=np.float64)
    fold_rows = []
    missing = []

    by_region: dict[str, list[int]] = {}
    for i, cid in enumerate(city_ids):
        region = CITY_TO_WEEK4_REGION.get(str(cid))
        if region is None:
            missing.append(str(cid))
            continue
        by_region.setdefault(region, []).append(i)

    if missing:
        uniq = sorted(set(missing))
        raise RuntimeError(f"no week4 region map for city_ids: {uniq}")

    for region, idxs in sorted(by_region.items()):
        path = ckpt_dir / f"fold_{region}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"missing week4 fold checkpoint: {path}")
        fold = joblib.load(path)
        mu = np.asarray(fold["mu"], dtype=np.float64).reshape(1, -1)
        sd = np.asarray(fold["sd"], dtype=np.float64).reshape(1, -1)
        w = np.asarray(fold["baseline_lr_w"], dtype=np.float64).reshape(-1)
        b = float(fold["baseline_lr_b"])
        if w.shape[0] != X.shape[1]:
            raise RuntimeError(f"fold {region}: weight dim {w.shape[0]} != X dim {X.shape[1]}")

        idx = np.asarray(idxs, dtype=np.int64)
        Xs = (X[idx] - mu) / np.clip(sd, 1e-6, None)
        logits = Xs @ w + b
        # numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        yhat = (p >= 0.5).astype(np.int64)
        pred[idx] = yhat
        proba[idx] = p
        fold_rows.append(
            {
                "held_out_region": region,
                "n_test": int(len(idx)),
                "accuracy": float(accuracy_score(y[idx], yhat)),
                "macro_f1": float(f1_score(y[idx], yhat, average="macro", zero_division=0)),
                "checkpoint": str(path),
            }
        )

    ok = pred >= 0
    if not ok.all():
        raise RuntimeError(f"week4 model left {int((~ok).sum())} pixels unscored")

    summary = {
        "method": "week4_pls_loro_baseline_lr_loro",
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "threshold": 0.5,
        "checkpoint_dir": str(ckpt_dir),
        "n_labeled": int(n),
        "n_scored": int(ok.sum()),
        "accuracy": float(accuracy_score(y[ok], pred[ok])),
        "macro_f1": float(f1_score(y[ok], pred[ok], average="macro", zero_division=0)),
        "folds": fold_rows,
    }
    return pred, proba, summary


def loro_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    city_ids: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Leave-one-city-out logistic on standardized 64-D embeddings.

    Returns pred (0/1), proba_res in [0,1], and summary metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    n = len(y)
    pred = np.full(n, -1, dtype=np.int64)
    proba = np.full(n, np.nan, dtype=np.float64)
    cities = sorted(set(map(str, city_ids)))
    fold_rows = []
    for city in cities:
        te = np.asarray([str(c) == city for c in city_ids])
        tr = ~te
        if te.sum() == 0 or tr.sum() < 2 or len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(
            solver="newton-cholesky",
            C=1.0,
            max_iter=200,
            random_state=seed,
        )
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(np.int64)
        pred[te] = yhat
        proba[te] = p
        fold_rows.append(
            {
                "held_out_city": city,
                "n_test": int(te.sum()),
                "accuracy": float(accuracy_score(y[te], yhat)),
                "macro_f1": float(f1_score(y[te], yhat, average="macro", zero_division=0)),
            }
        )
    ok = pred >= 0
    summary = {
        "method": "leave_one_city_out_logistic_64d",
        "threshold": 0.5,
        "n_labeled": int(n),
        "n_scored": int(ok.sum()),
        "accuracy": float(accuracy_score(y[ok], pred[ok])) if ok.any() else None,
        "macro_f1": float(f1_score(y[ok], pred[ok], average="macro", zero_division=0))
        if ok.any()
        else None,
        "folds": fold_rows,
    }
    return pred, proba, summary


def _expand_bbox(west, south, east, north, pad_m: float):
    mid_lat = 0.5 * (south + north)
    dlat = pad_m / 111_320.0
    dlon = pad_m / (111_320.0 * max(0.2, math.cos(math.radians(mid_lat))))
    return west - dlon, south - dlat, east + dlon, north + dlat


def _square_bbox_m(west, south, east, north, side_m: float):
    """Centered square footprint of side_m meters around the cell."""
    mid_lat = 0.5 * (south + north)
    mid_lon = 0.5 * (west + east)
    half = side_m / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * max(0.2, math.cos(math.radians(mid_lat))))
    return mid_lon - dlon, mid_lat - dlat, mid_lon + dlon, mid_lat + dlat


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    def lat(y_):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y_ / n))))
    north = lat(y)
    south = lat(y + 1)
    return west, south, east, north


def _fetch_tile(z: int, x: int, y: int) -> Image.Image:
    url = ESRI_TILE.format(z=z, x=x, y=y)
    last = None
    for attempt in range(5):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.4 * (2**attempt) + 0.05 * (attempt + 1))
                last = RuntimeError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last = e
            time.sleep(0.3 * (2**attempt))
    raise RuntimeError(f"tile {z}/{x}/{y}: {last}")


def fetch_chip_from_tiles(
    west: float,
    south: float,
    east: float,
    north: float,
    size: int,
    pad_m: float = 0.0,
    side_m: float | None = None,
) -> Image.Image:
    """Mosaic Esri XYZ tiles into a chip.

    pad_m: expand cell bbox by this many meters on each side.
    side_m: if set, replace bbox with a centered square of this side length (meters).
    """
    if side_m is not None:
        west, south, east, north = _square_bbox_m(west, south, east, north, side_m)
    elif pad_m:
        west, south, east, north = _expand_bbox(west, south, east, north, pad_m)
    if east <= west:
        east = west + 1e-5
    if north <= south:
        north = south + 1e-5

    # Choose zoom so ~size pixels cover the bbox width.
    mid_lat = 0.5 * (south + north)
    width_m = (east - west) * 111_320.0 * max(0.2, math.cos(math.radians(mid_lat)))
    mpp_target = max(width_m / max(size, 1), 0.5)
    # WebMercator m/px ≈ 156543.03 * cos(lat) / 2^z
    z = int(round(math.log2(156543.03 * math.cos(math.radians(mid_lat)) / mpp_target)))
    z = max(14, min(19, z))

    x0, y1 = _lonlat_to_tile(west, south, z)  # south -> larger y
    x1, y0 = _lonlat_to_tile(east, north, z)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # Cap mosaic size
    if (x1 - x0 + 1) * (y1 - y0 + 1) > 64:
        z = max(14, z - 1)
        x0, y1 = _lonlat_to_tile(west, south, z)
        x1, y0 = _lonlat_to_tile(east, north, z)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

    tile_w = tile_h = 256
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * tile_w, (y1 - y0 + 1) * tile_h))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = _fetch_tile(z, x, y)
            mosaic.paste(tile, ((x - x0) * tile_w, (y - y0) * tile_h))

    # Crop to exact bbox in mosaic pixel space
    mosa_w, mosa_s = _tile_bounds(x0, y1, z)[0], _tile_bounds(x0, y1, z)[1]
    mosa_e, mosa_n = _tile_bounds(x1, y0, z)[2], _tile_bounds(x1, y0, z)[3]

    def lon_to_px(lon):
        return (lon - mosa_w) / (mosa_e - mosa_w) * mosaic.width

    def lat_to_py(lat):
        return (mosa_n - lat) / (mosa_n - mosa_s) * mosaic.height

    left = max(0, int(lon_to_px(west)))
    right = min(mosaic.width, int(math.ceil(lon_to_px(east))))
    top = max(0, int(lat_to_py(north)))
    bottom = min(mosaic.height, int(math.ceil(lat_to_py(south))))
    if right <= left:
        right = min(mosaic.width, left + 1)
    if bottom <= top:
        bottom = min(mosaic.height, top + 1)
    crop = mosaic.crop((left, top, right, bottom))
    return crop.resize((size, size), Image.Resampling.BILINEAR)


def chip_job(row: dict, force: bool = False) -> dict:
    cell_id = int(row["cell_id"])
    patch_path = CHIPS / f"{cell_id}_patch.jpg"
    ctx_path = CHIPS / f"{cell_id}_context.jpg"
    ctx500_path = CHIPS / f"{cell_id}_context500.jpg"
    out = {
        "cell_id": cell_id,
        "ok": True,
        "error": None,
        "skipped": 0,
        "wrote": 0,
    }
    try:
        # Exact 10 m AlphaEarth cell (no padding)
        if (not force) and patch_path.exists() and patch_path.stat().st_size > 1000:
            out["skipped"] += 1
        else:
            img = fetch_chip_from_tiles(
                float(row["west"]),
                float(row["south"]),
                float(row["east"]),
                float(row["north"]),
                size=256,
                pad_m=0.0,
            )
            tmp = patch_path.with_suffix(".tmp.jpg")
            img.save(tmp, quality=90, optimize=True)
            tmp.replace(patch_path)
            out["wrote"] += 1
        # 50 m square context centered on the cell
        if (not force) and ctx_path.exists() and ctx_path.stat().st_size > 1000:
            out["skipped"] += 1
        else:
            img = fetch_chip_from_tiles(
                float(row["west"]),
                float(row["south"]),
                float(row["east"]),
                float(row["north"]),
                size=384,
                side_m=50.0,
            )
            tmp = ctx_path.with_suffix(".tmp.jpg")
            img.save(tmp, quality=88, optimize=True)
            tmp.replace(ctx_path)
            out["wrote"] += 1
        # 500 m square context
        if (not force) and ctx500_path.exists() and ctx500_path.stat().st_size > 1000:
            out["skipped"] += 1
        else:
            img = fetch_chip_from_tiles(
                float(row["west"]),
                float(row["south"]),
                float(row["east"]),
                float(row["north"]),
                size=512,
                side_m=500.0,
            )
            tmp = ctx500_path.with_suffix(".tmp.jpg")
            img.save(tmp, quality=85, optimize=True)
            tmp.replace(ctx500_path)
            out["wrote"] += 1
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
    return out


def week4_pls_xyz(
    X: np.ndarray,
    city_ids: np.ndarray,
    n_components: int = 3,
    ckpt_dir: Path | None = None,
) -> np.ndarray:
    """Project cells into week4 PLS-DA scores (first n_components) using LORO folds."""
    import joblib

    ckpt_dir = Path(ckpt_dir or WEEK4_CKPT_DIR)
    n = X.shape[0]
    xyz = np.full((n, n_components), np.nan, dtype=np.float64)
    by_region: dict[str, list[int]] = {}
    for i, cid in enumerate(city_ids):
        region = CITY_TO_WEEK4_REGION.get(str(cid))
        if region is None:
            raise RuntimeError(f"no week4 region for city_id={cid}")
        by_region.setdefault(region, []).append(i)

    for region, idxs in sorted(by_region.items()):
        path = ckpt_dir / f"fold_{region}.joblib"
        fold = joblib.load(path)
        mu = np.asarray(fold["mu"], dtype=np.float64).reshape(1, -1)
        sd = np.asarray(fold["sd"], dtype=np.float64).reshape(1, -1)
        R = np.asarray(fold["pls_da"]["x_rotations"], dtype=np.float64)[:, :n_components]
        idx = np.asarray(idxs, dtype=np.int64)
        Xs = (X[idx] - mu) / np.clip(sd, 1e-6, None)
        xyz[idx] = Xs @ R
    if not np.isfinite(xyz).all():
        raise RuntimeError("PLS projection produced NaNs")
    return xyz


def embed_umap_3d(X: np.ndarray, seed: int, n_jobs: int) -> np.ndarray:
    try:
        import umap

        # Omit random_state so UMAP/numba can use all CPU threads (n=1000 is small).
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.15,
            metric="cosine",
            n_jobs=n_jobs,
            low_memory=False,
        )
        return reducer.fit_transform(X)
    except Exception as e:
        print(f"UMAP failed ({e}); falling back to PCA-3", flush=True)
        return PCA(n_components=3, random_state=seed).fit_transform(X)


def _binary_gt_fracs(row, gt: int) -> dict:
    """GT building-class shares for pure exclusive cells: frac_res + frac_nres = 1.

    Empty pixel area is excluded — week4 binary analysis is res vs non-res only.
    """
    r_full = row.get("frac_res_full")
    n_full = row.get("frac_nres_full")
    r = 0.0 if (r_full is None or (isinstance(r_full, float) and pd.isna(r_full))) else float(r_full)
    n = 0.0 if (n_full is None or (isinstance(n_full, float) and pd.isna(n_full))) else float(n_full)
    s = r + n
    if s > 0:
        r, n = r / s, n / s
    elif int(gt) == 1:
        r, n = 1.0, 0.0
    else:
        r, n = 0.0, 1.0
    return {
        "frac_res": float(r),
        "frac_nres": float(n),
        "frac_res_full": None if pd.isna(row.get("frac_res_full")) else float(row["frac_res_full"]),
        "frac_nres_full": None if pd.isna(row.get("frac_nres_full")) else float(row["frac_nres_full"]),
        "area_res_m2": None if pd.isna(row.get("area_res_m2")) else float(row["area_res_m2"]),
        "area_nres_m2": None if pd.isna(row.get("area_nres_m2")) else float(row["area_nres_m2"]),
    }


def city_color(city_id: str) -> str:
    h = hashlib.md5(city_id.encode()).hexdigest()
    r = 80 + int(h[0:2], 16) % 140
    g = 80 + int(h[2:4], 16) % 140
    b = 80 + int(h[4:6], 16) % 140
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> int:
    n_cpus = os.cpu_count() or 8
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-city", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--countries",
        nargs="+",
        default=["USA", "IND", "JPN"],
        help="Country prefixes to sample (USA IND JPN)",
    )
    ap.add_argument(
        "--n-per-city-weak",
        type=int,
        default=500,
        help="Target 1:1 sample size for cities with collapsed LORO (top-up after first score)",
    )
    # I/O-bound: default 2× CPUs (skill guidance). Rate-limit backoff handles Esri.
    ap.add_argument("--workers", type=int, default=max(32, 2 * n_cpus))
    ap.add_argument("--fold-workers", type=int, default=0, help="Parallel intl fold fits")
    ap.add_argument("--skip-imagery", action="store_true")
    ap.add_argument(
        "--force-imagery",
        action="store_true",
        help="Re-download chips even if files already exist",
    )
    ap.add_argument("--force-folds", action="store_true", help="Refit IND/JPN LORO folds")
    ap.add_argument("--run-dir", type=str, default=os.environ.get("RUN_DIR", ""))
    ap.add_argument(
        "--intl-fold-dir",
        type=str,
        default=str(DATA / "intl_folds"),
    )
    ap.add_argument(
        "--extract-dir",
        type=str,
        default="",
        help="Balanced extract for intl LORO fold fits (default: intl_globe_folds DEFAULT_EXTRACT)",
    )
    ap.add_argument(
        "--out-points",
        type=str,
        default="",
        help="Write points JSON here (default: viz/.../data/points.json)",
    )
    args = ap.parse_args()
    countries = [c.upper() for c in args.countries]
    if args.n_per_city_weak < args.n_per_city:
        args.n_per_city_weak = args.n_per_city

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = OUT / "runs" / f"export_{stamp}"
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CHIPS.mkdir(parents=True, exist_ok=True)
    intl_fold_dir = Path(args.intl_fold_dir)
    intl_fold_dir.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    n_gpus = 0
    try:
        import subprocess

        n_gpus = len(
            subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
        )
    except Exception:
        n_gpus = 0

    n_city_guess = {"USA": 10, "IND": 10, "JPN": 10}
    expected = args.n_per_city * sum(n_city_guess.get(c, 5) for c in countries)
    update_progress(
        run_dir,
        status="running",
        stage="sample",
        started_at=started,
        completed=0,
        failed=0,
        total=expected,
        percent_complete=0.0,
        countries=countries,
        resource_allocation={
            "cpus": n_cpus,
            "gpus": n_gpus,
            "gpu_used": False,
            "gpu_note": "Idle by design: DB sample + fold CPU + HTTP tile I/O",
            "workers": args.workers,
            "fold_workers": args.fold_workers or min(10, n_cpus),
        },
        output=str(DATA),
        job_run_dir=str(run_dir),
    )
    print(
        f"resources cpus={n_cpus} gpus={n_gpus} (unused) imagery_workers={args.workers} "
        f"countries={countries}",
        flush=True,
    )

    n_per_class = max(1, args.n_per_city // 2)
    print(
        f"sampling {args.n_per_city}/city (1:1 → {n_per_class} res + {n_per_class} nres) "
        f"for {countries} ...",
        flush=True,
    )
    df = sample_cells(args.n_per_city, args.seed, countries=countries)
    print(
        f"sampled {len(df)} cells across {df['city_id'].nunique()} cities "
        f"({df.groupby('country').size().to_dict() if len(df) else {}})",
        flush=True,
    )
    if df.empty:
        update_progress(run_dir, status="failed", stage="sample")
        raise SystemExit("no cells found")
    bal = (
        df.groupby("city_id")["binary_label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "nres", 1: "res"})
    )
    bad = bal[(bal.get("res", 0) != bal.get("nres", 0)) | (bal.min(axis=1) < n_per_class)]
    if len(bad):
        print(f"  balance check WARN:\n{bad}", flush=True)
    else:
        print(
            f"  balance ok: every city has {n_per_class} res + {n_per_class} nres",
            flush=True,
        )

    update_progress(
        run_dir,
        stage="score",
        total=len(df),
        completed=0,
        cities=int(df["city_id"].nunique()),
    )

    X_raw = df[A_COLS].to_numpy(dtype=np.float64)
    y = df["binary_label"].to_numpy(dtype=np.int64)
    city_ids = df["city_id"].astype(str).to_numpy()
    country_arr = df["country"].astype(str).to_numpy()

    pred = np.full(len(df), -1, dtype=np.int64)
    proba = np.full(len(df), np.nan, dtype=np.float64)
    xyz_raw = np.full((len(df), 3), np.nan, dtype=np.float64)
    pred_model = np.array([""] * len(df), dtype=object)
    summaries = {}

    usa_mask = country_arr == "USA"
    intl_mask = np.isin(country_arr, ["IND", "JPN"])

    if usa_mask.any():
        print(
            f"scoring USA ({int(usa_mask.sum())}) with week4 baseline_lr LORO …",
            flush=True,
        )
        p_u, pr_u, sum_u = week4_baseline_lr_predictions(
            X_raw[usa_mask], y[usa_mask], city_ids[usa_mask]
        )
        pred[usa_mask] = p_u
        proba[usa_mask] = pr_u
        pred_model[usa_mask] = "week4_baseline_lr_loro"
        xyz_raw[usa_mask] = week4_pls_xyz(
            X_raw[usa_mask], city_ids[usa_mask], n_components=3
        )
        summaries["USA"] = sum_u
        atomic_json(run_dir / "artifacts" / "week4_baseline_lr_metrics.json", sum_u)
        print(
            f"  USA acc={sum_u.get('accuracy')} macro_f1={sum_u.get('macro_f1')}",
            flush=True,
        )

    if intl_mask.any():
        from intl_globe_folds import ALL_INTL, ensure_intl_folds, score_with_fold_dir

        intl_cities = sorted({str(c) for c in city_ids[intl_mask]})
        update_progress(run_dir, stage="fit_intl_folds", total=len(intl_cities))
        print(f"ensuring intl LORO folds for {intl_cities} …", flush=True)

        def _fold_progress(done, total, res):
            update_progress(
                run_dir,
                stage="fit_intl_folds",
                completed=done,
                total=total,
                percent_complete=round(100.0 * done / max(total, 1), 2),
                last_fold=res.get("held_out"),
                last_fold_status=res.get("status"),
            )

        fold_kwargs = dict(
            cities=[c for c in ALL_INTL if c in intl_cities],
            out_dir=intl_fold_dir,
            workers=args.fold_workers or min(len(intl_cities), n_cpus),
            force=args.force_folds,
            progress_cb=_fold_progress,
        )
        if args.extract_dir:
            fold_kwargs["extract_dir"] = Path(args.extract_dir)
        fold_summary = ensure_intl_folds(**fold_kwargs)
        atomic_json(run_dir / "artifacts" / "intl_folds_manifest.json", fold_summary)
        if fold_summary.get("n_failed", 0):
            raise SystemExit(f"intl fold fit failed: {fold_summary}")

        update_progress(run_dir, stage="score_intl")
        print(f"scoring IND/JPN ({int(intl_mask.sum())}) with intl LORO …", flush=True)
        p_i, pr_i, xyz_i, sum_i = score_with_fold_dir(
            X_raw[intl_mask],
            y[intl_mask],
            city_ids[intl_mask],
            intl_fold_dir,
        )
        pred[intl_mask] = p_i
        proba[intl_mask] = pr_i
        xyz_raw[intl_mask] = xyz_i
        pred_model[intl_mask] = "intl_loro_lr64"
        summaries["INTL"] = sum_i
        atomic_json(run_dir / "artifacts" / "intl_loro_metrics.json", sum_i)
        print(
            f"  INTL acc={sum_i.get('accuracy')} macro_f1={sum_i.get('macro_f1')}",
            flush=True,
        )

    if not np.isfinite(xyz_raw).all() or (pred < 0).any():
        raise RuntimeError("scoring left unscored rows or NaN PLS coords")

    from sklearn.metrics import accuracy_score, f1_score

    weak = flag_weak_cities(city_ids, y, pred, proba)
    atomic_json(run_dir / "artifacts" / "weak_cities_pass1.json", {"cities": weak})
    weak_topup: list[dict] = []
    if weak and args.n_per_city_weak > args.n_per_city:
        weak_ids = [w["city_id"] for w in weak]
        print(
            f"weak cities ({len(weak_ids)}): {weak_ids} → top-up to "
            f"{args.n_per_city_weak}/city (1:1)",
            flush=True,
        )
        for w in weak:
            print(
                f"  {w['city_id']}: recall res/nres={w['recall_res']}/{w['recall_nres']} "
                f"hi={w['hi_res']}/{w['hi_nres']} reasons={','.join(w['reasons'])}",
                flush=True,
            )
        update_progress(
            run_dir,
            stage="topup_weak",
            weak_cities=weak_ids,
            n_per_city_weak=args.n_per_city_weak,
        )
        have = {
            str(cid): int((city_ids == cid).sum())
            for cid in weak_ids
        }
        # Sample enough NEW balanced cells so each weak city reaches n_per_city_weak.
        # sample_cells(n) draws n/2 per class excluding existing ids.
        extra_n = args.n_per_city_weak  # draw up to this many new; then trim per city
        df_extra = sample_cells(
            extra_n,
            args.seed + 17,
            countries=countries,
            cities=weak_ids,
            exclude_cell_ids=set(int(x) for x in df["cell_id"].tolist()),
        )
        if df_extra.empty:
            print("  top-up: no extra cells available", flush=True)
        else:
            # Keep only what each city still needs to hit the target (balanced).
            pieces = []
            for cid in weak_ids:
                g = df_extra[df_extra["city_id"] == cid]
                if g.empty:
                    continue
                need = max(0, args.n_per_city_weak - have.get(cid, 0))
                need_class = need // 2
                if need_class < 1:
                    continue
                g0 = g[g["binary_label"] == 0].head(need_class)
                g1 = g[g["binary_label"] == 1].head(need_class)
                k = min(len(g0), len(g1))
                if k < 1:
                    continue
                piece = pd.concat([g0.head(k), g1.head(k)], ignore_index=True)
                pieces.append(piece)
                weak_topup.append(
                    {
                        "city_id": cid,
                        "had": have.get(cid, 0),
                        "added": int(len(piece)),
                        "target": args.n_per_city_weak,
                    }
                )
                print(
                    f"  top-up {cid}: +{len(piece)} (was {have.get(cid, 0)} → "
                    f"{have.get(cid, 0) + len(piece)})",
                    flush=True,
                )
            if pieces:
                df_extra = pd.concat(pieces, ignore_index=True)
                X_e = df_extra[A_COLS].to_numpy(dtype=np.float64)
                y_e = df_extra["binary_label"].to_numpy(dtype=np.int64)
                city_e = df_extra["city_id"].astype(str).to_numpy()
                country_e = df_extra["country"].astype(str).to_numpy()
                pred_e = np.full(len(df_extra), -1, dtype=np.int64)
                proba_e = np.full(len(df_extra), np.nan, dtype=np.float64)
                xyz_e = np.full((len(df_extra), 3), np.nan, dtype=np.float64)
                model_e = np.array([""] * len(df_extra), dtype=object)

                usa_e = country_e == "USA"
                intl_e = np.isin(country_e, ["IND", "JPN"])
                if usa_e.any():
                    p_u, pr_u, _ = week4_baseline_lr_predictions(
                        X_e[usa_e], y_e[usa_e], city_e[usa_e]
                    )
                    pred_e[usa_e] = p_u
                    proba_e[usa_e] = pr_u
                    model_e[usa_e] = "week4_baseline_lr_loro"
                    xyz_e[usa_e] = week4_pls_xyz(
                        X_e[usa_e], city_e[usa_e], n_components=3
                    )
                if intl_e.any():
                    from intl_globe_folds import score_with_fold_dir

                    p_i, pr_i, xyz_i, sum_i = score_with_fold_dir(
                        X_e[intl_e], y_e[intl_e], city_e[intl_e], intl_fold_dir
                    )
                    pred_e[intl_e] = p_i
                    proba_e[intl_e] = pr_i
                    xyz_e[intl_e] = xyz_i
                    model_e[intl_e] = "intl_loro_lr64"
                    summaries["INTL_topup"] = {
                        "accuracy": sum_i.get("accuracy"),
                        "macro_f1": sum_i.get("macro_f1"),
                        "n_scored": sum_i.get("n_scored"),
                    }

                if (pred_e < 0).any() or not np.isfinite(xyz_e).all():
                    raise RuntimeError("weak-city top-up left unscored rows")

                df = pd.concat([df, df_extra], ignore_index=True)
                X_raw = np.vstack([X_raw, X_e])
                y = np.concatenate([y, y_e])
                city_ids = np.concatenate([city_ids, city_e])
                country_arr = np.concatenate([country_arr, country_e])
                pred = np.concatenate([pred, pred_e])
                proba = np.concatenate([proba, proba_e])
                xyz_raw = np.vstack([xyz_raw, xyz_e])
                pred_model = np.concatenate([pred_model, model_e])
                # refresh masks used later
                usa_mask = country_arr == "USA"
                intl_mask = np.isin(country_arr, ["IND", "JPN"])

                weak2 = flag_weak_cities(city_ids, y, pred, proba)
                atomic_json(
                    run_dir / "artifacts" / "weak_cities_after_topup.json",
                    {"cities": weak2, "topup": weak_topup},
                )
                print(
                    f"  after top-up still-weak: {[w['city_id'] for w in weak2]} "
                    f"(n={len(df)})",
                    flush=True,
                )

    pred_summary = {
        "method": "+".join(
            m
            for m in [
                summaries.get("USA", {}).get("method"),
                summaries.get("INTL", {}).get("method"),
            ]
            if m
        ),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n_scored": int(len(y)),
        "by_country": summaries,
        "intl_fold_dir": str(intl_fold_dir) if intl_mask.any() else None,
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "weak_cities_pass1": weak,
        "weak_city_topup": weak_topup,
        "n_per_city_weak": args.n_per_city_weak,
    }
    atomic_json(run_dir / "artifacts" / "combined_pred_metrics.json", pred_summary)
    print(
        f"  combined acc={pred_summary['accuracy']:.3f} "
        f"macro_f1={pred_summary['macro_f1']:.3f} n={pred_summary['n_scored']}",
        flush=True,
    )

    update_progress(run_dir, stage="pls3_normalize", completed=0)
    xyz = xyz_raw - xyz_raw.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(xyz, axis=1)) or 1.0)
    xyz = xyz / scale
    # keep a 1-D scalar for optional coloring (PLS-1 normalized)
    pls1 = xyz_raw[:, 0]
    pca1_n = (pls1 - pls1.min()) / max(1e-9, float(pls1.max() - pls1.min()))

    ok = fail = skipped = wrote = 0
    if not args.skip_imagery:
        update_progress(run_dir, stage="imagery", completed=0, failed=0, total=len(df))
        print(f"fetching satellite chips workers={args.workers} (resume-safe) ...", flush=True)
        rows = df.to_dict(orient="records")
        t_img = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(chip_job, r, args.force_imagery) for r in rows]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res["ok"]:
                    ok += 1
                    skipped += int(res.get("skipped") or 0)
                    wrote += int(res.get("wrote") or 0)
                else:
                    fail += 1
                    (run_dir / "failures" / f"{res['cell_id']}.txt").write_text(
                        str(res.get("error") or "unknown") + "\n"
                    )
                    if fail <= 20:
                        print(f"  chip fail cell={res['cell_id']}: {res['error']}", flush=True)
                if i % 25 == 0 or i == len(futs):
                    elapsed = max(1e-3, time.time() - t_img)
                    rate = i / elapsed
                    eta = (len(futs) - i) / max(rate, 1e-6)
                    update_progress(
                        run_dir,
                        stage="imagery",
                        completed=i,
                        failed=fail,
                        total=len(futs),
                        success=ok,
                        skipped_chip_files=skipped,
                        wrote_chip_files=wrote,
                        rate_per_second=round(rate, 2),
                        percent_complete=round(100.0 * i / len(futs), 2),
                        eta_seconds=round(eta, 1),
                    )
                    print(
                        f"  chips {i}/{len(futs)} ok={ok} fail={fail} "
                        f"wrote={wrote} skip={skipped} rate={rate:.2f}/s eta={eta:.0f}s",
                        flush=True,
                    )
    else:
        update_progress(run_dir, stage="imagery_skipped")

    update_progress(run_dir, stage="write_points")
    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        cid = int(row["cell_id"])
        gt = int(row["binary_label"])
        pr = int(pred[i]) if pred[i] >= 0 else None
        pr_p = float(proba[i]) if np.isfinite(proba[i]) else None
        country = str(row["country"])
        points.append(
            {
                "id": cid,
                "country": country,
                "city_id": row["city_id"],
                "city_name": row["city_name"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "west": float(row["west"]),
                "south": float(row["south"]),
                "east": float(row["east"]),
                "north": float(row["north"]),
                "ux": float(xyz[i, 0]),
                "uy": float(xyz[i, 1]),
                "uz": float(xyz[i, 2]),
                "pls1": float(xyz_raw[i, 0]),
                "pls2": float(xyz_raw[i, 1]),
                "pls3": float(xyz_raw[i, 2]),
                "pca1": float(pca1_n[i]),
                "label": str(float(gt)),  # "1.0" / "0.0" (GT)
                "gt": gt,
                "pred": pr,
                "proba_res": pr_p,
                "pred_frac_res": pr_p,
                "pred_frac_nres": (None if pr_p is None else float(1.0 - pr_p)),
                "correct": (None if pr is None else bool(pr == gt)),
                **_binary_gt_fracs(row, gt),
                "purity": None if pd.isna(row.get("purity_status")) else str(row["purity_status"]),
                "pred_model": str(pred_model[i]),
                "week4_region": CITY_TO_WEEK4_REGION.get(str(row["city_id"])),
                "fold_id": (
                    CITY_TO_WEEK4_REGION.get(str(row["city_id"]))
                    if country == "USA"
                    else str(row["city_id"])
                ),
                "color": city_color(str(row["city_id"])),
                "patch": f"chips/{cid}_patch.jpg" if (CHIPS / f"{cid}_patch.jpg").exists() else None,
                "context": f"chips/{cid}_context.jpg" if (CHIPS / f"{cid}_context.jpg").exists() else None,
                "context500": f"chips/{cid}_context500.jpg" if (CHIPS / f"{cid}_context500.jpg").exists() else None,
            }
        )

    cities = sorted({p["city_id"] for p in points})
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_points": len(points),
        "n_per_city": int(2 * n_per_class),
        "n_per_class": int(n_per_class),
        "n_per_city_weak": int(args.n_per_city_weak),
        "sample_balance": "1:1_gt_binary_label_per_city",
        "weak_city_topup": weak_topup,
        "seed": args.seed,
        "countries": countries,
        "embedding": "pls_da_loro_dims_1_2_3 (USA=week4 folds; IND/JPN=intl extract LORO)",
        "imagery": "Esri World Imagery XYZ · patch=exact 10m · context=50m · context=500m",
        "prediction": {
            "method": pred_summary["method"],
            "threshold": pred_summary["threshold"],
            "accuracy": pred_summary.get("accuracy"),
            "macro_f1": pred_summary.get("macro_f1"),
            "n_scored": pred_summary.get("n_scored"),
            "by_country": {
                k: {
                    "accuracy": v.get("accuracy"),
                    "macro_f1": v.get("macro_f1"),
                    "n_scored": v.get("n_scored"),
                    "method": v.get("method"),
                }
                for k, v in summaries.items()
            },
            "intl_fold_dir": str(intl_fold_dir) if intl_mask.any() else None,
            "checkpoint_dir": str(WEEK4_CKPT_DIR) if usa_mask.any() else None,
            "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        },
        "cities": [
            {
                "city_id": c,
                "city_name": next(p["city_name"] for p in points if p["city_id"] == c),
                "country": next(p["country"] for p in points if p["city_id"] == c),
                "color": city_color(c),
                "n": sum(1 for p in points if p["city_id"] == c),
            }
            for c in cities
        ],
        "points": points,
    }
    out_points = Path(args.out_points) if args.out_points else (DATA / "points.json")
    out_points.parent.mkdir(parents=True, exist_ok=True)
    tmp_points = out_points.with_suffix(".tmp.json")
    tmp_points.write_text(json.dumps(payload))
    tmp_points.replace(out_points)

    runtime = round(time.time() - t0, 1)
    summary = {
        "status": "completed" if fail == 0 else "completed_with_failures",
        "n_points": len(points),
        "cities": len(cities),
        "countries": countries,
        "n_res": int((y == 1).sum()),
        "n_nres": int((y == 0).sum()),
        "loro_accuracy": pred_summary.get("accuracy"),
        "loro_macro_f1": pred_summary.get("macro_f1"),
        "prediction_method": pred_summary.get("method"),
        "by_country": pred_summary.get("by_country"),
        "intl_fold_dir": str(intl_fold_dir) if intl_mask.any() else None,
        "chips_ok_cells": ok,
        "chips_failed_cells": fail,
        "chip_files_wrote": wrote,
        "chip_files_skipped": skipped,
        "runtime_seconds": runtime,
        "workers": args.workers,
        "cpus": n_cpus,
        "gpus": n_gpus,
        "gpu_used": False,
        "points_json": str(out_points),
        "viewer": str(OUT / "index.html"),
    }
    atomic_json(DATA / "manifest.json", summary)
    atomic_json(run_dir / "summary.json", summary)
    update_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        completed=len(df),
        failed=fail,
        percent_complete=100.0,
        runtime_seconds=runtime,
    )
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
