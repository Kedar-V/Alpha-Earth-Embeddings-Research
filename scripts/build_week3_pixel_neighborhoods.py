#!/usr/bin/env python3
"""Fetch centered k×k AlphaEarth neighborhoods for week3 pixel GT v2.

Writes datasets/week3_pixel_neighborhood_k{k}/:
  X_nb.npy   float32 (N, 64, k, k)
  mask.npy   uint8   (N, 1, k, k)
  meta.parquet  row-aligned labels / fracs / geometry
  manifest.json

Usage:
  .venv/bin/python scripts/build_week3_pixel_neighborhoods.py --k 3
  .venv/bin/python scripts/build_week3_pixel_neighborhoods.py --k 3 --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week3_pixel_gt_v2" / "pixels.parquet"
CACHE = ROOT / "datasets" / "week3_pixel_gt_v2" / "_cache"
INDEX_URL = (
    "https://storage.googleapis.com/alphaearth_foundations/"
    "satellite_embedding/v1/annual/aef_index.parquet"
)
INDEX_FULL = CACHE / "aef_index.parquet"
INDEX_US = CACHE / "aef_index_us_2024.parquet"
EMBED_YEAR = 2024
BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "268435456",
}


def gs_to_https(path: str) -> str:
    return path.replace("gs://", "https://storage.googleapis.com/")


def dequantize(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32)
    return ((v / 127.5) ** 2) * np.sign(v)


def ensure_index() -> pd.DataFrame:
    import urllib.request

    CACHE.mkdir(parents=True, exist_ok=True)
    if not INDEX_FULL.exists():
        print(f"Downloading AEF index -> {INDEX_FULL} ...", flush=True)
        urllib.request.urlretrieve(INDEX_URL, INDEX_FULL)
    if not INDEX_US.exists():
        print("Filtering 2024 US tiles ...", flush=True)
        t = pd.read_parquet(INDEX_FULL)
        t = t.loc[t["year"] == EMBED_YEAR].copy()
        t = t.loc[
            (t["wgs84_west"] <= -66.0)
            & (t["wgs84_east"] >= -125.0)
            & (t["wgs84_south"] <= 50.0)
            & (t["wgs84_north"] >= 24.0)
        ].copy()
        t.to_parquet(INDEX_US, index=False)
    return pd.read_parquet(INDEX_US)


def out_dir_for(k: int) -> Path:
    return ROOT / "datasets" / f"week3_pixel_neighborhood_k{k}"


def _assign_rows_to_tiles(meta: pd.DataFrame, index: pd.DataFrame) -> list[dict]:
    """Fast numpy bbox assignment (tiles × N vectorized checks)."""
    lon = meta["pixel_lon"].to_numpy(dtype=np.float64)
    lat = meta["pixel_lat"].to_numpy(dtype=np.float64)
    crs = meta["crs"].astype(str).to_numpy()
    pids = meta["pixel_id"].astype(str).to_numpy()
    n = len(meta)
    # best tile index per point (-1 = none); preference score
    best_ti = np.full(n, -1, dtype=np.int32)
    best_score = np.full(n, -1, dtype=np.int8)

    paths = index["path"].astype(str).to_numpy()
    tile_crs = index["crs"].astype(str).to_numpy()
    west = index["wgs84_west"].to_numpy(dtype=np.float64)
    east = index["wgs84_east"].to_numpy(dtype=np.float64)
    south = index["wgs84_south"].to_numpy(dtype=np.float64)
    north = index["wgs84_north"].to_numpy(dtype=np.float64)

    # Restrict to tiles overlapping the data bbox (huge speedup vs full US index)
    overlap = (
        (east >= lon.min())
        & (west <= lon.max())
        & (north >= lat.min())
        & (south <= lat.max())
    )
    ti_list = np.where(overlap)[0]
    print(f"candidate tiles overlapping data bbox: {len(ti_list)}/{len(index)}", flush=True)

    for ti in ti_list:
        inside = (
            (lon >= west[ti])
            & (lon <= east[ti])
            & (lat >= south[ti])
            & (lat <= north[ti])
        )
        if not inside.any():
            continue
        score = (crs == tile_crs[ti]).astype(np.int8)
        # update where inside and score better (or first hit)
        take = inside & (score > best_score)
        # also take first tile if none yet
        take |= inside & (best_ti < 0)
        best_ti[take] = ti
        best_score[take] = np.maximum(best_score[take], score[take])

    buckets: dict[str, dict] = {}
    assigned = 0
    for i in range(n):
        ti = int(best_ti[i])
        if ti < 0:
            continue
        assigned += 1
        path = paths[ti]
        if path not in buckets:
            buckets[path] = {"path": path, "tile_crs": tile_crs[ti], "rows": []}
        buckets[path]["rows"].append(
            {
                "i": i,
                "pixel_id": pids[i],
                "pixel_lon": float(lon[i]),
                "pixel_lat": float(lat[i]),
                "crs": crs[i],
            }
        )
    print(f"assigned {assigned}/{n} pixels to {len(buckets)} tiles", flush=True)
    return list(buckets.values())


def _fetch_tile_windows(payload: dict) -> dict:
    """Read k×k windows for all centers on one COG tile."""
    path = payload["path"]
    k = int(payload["k"])
    h = k // 2
    rows = payload["rows"]
    n = len(rows)
    X = np.zeros((n, 64, k, k), dtype=np.float32)
    M = np.zeros((n, 1, k, k), dtype=np.uint8)
    ok = np.zeros(n, dtype=bool)
    ids = [r["pixel_id"] for r in rows]
    idxs = [r["i"] for r in rows]

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(gs_to_https(path)) as src:
            # group by row crs for transformers
            by_crs: dict[str, list[int]] = {}
            for j, r in enumerate(rows):
                by_crs.setdefault(r["crs"], []).append(j)

            for crs, js in by_crs.items():
                tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                for j in js:
                    r = rows[j]
                    ux, uy = tfm.transform(r["pixel_lon"], r["pixel_lat"])
                    try:
                        rr, cc = src.index(ux, uy)
                    except Exception:
                        continue
                    if not (0 <= rr < src.height and 0 <= cc < src.width):
                        continue
                    # center must be valid
                    center = src.read(window=((rr, rr + 1), (cc, cc + 1)))[:, 0, 0]
                    if (center == -128).any():
                        continue

                    r0, r1 = rr - h, rr + h + 1
                    c0, c1 = cc - h, cc + h + 1
                    # clip to tile; pad offsets
                    rs, re = max(0, r0), min(src.height, r1)
                    cs, ce = max(0, c0), min(src.width, c1)
                    if re <= rs or ce <= cs:
                        continue
                    raw = src.read(window=((rs, re), (cs, ce)))  # (64, h', w')
                    # place into full k×k with pad
                    pr0, pc0 = rs - r0, cs - c0  # offset into k×k
                    ph, pw = raw.shape[1], raw.shape[2]
                    patch = np.zeros((64, k, k), dtype=np.int16)
                    mask = np.zeros((k, k), dtype=np.uint8)
                    patch[:, pr0 : pr0 + ph, pc0 : pc0 + pw] = raw
                    mask[pr0 : pr0 + ph, pc0 : pc0 + pw] = 1
                    # invalidate nodata cells
                    nodata = (raw == -128).any(axis=0)
                    # nodata in the placed region
                    sub = mask[pr0 : pr0 + ph, pc0 : pc0 + pw]
                    sub[nodata] = 0
                    mask[pr0 : pr0 + ph, pc0 : pc0 + pw] = sub
                    if mask[h, h] == 0:
                        continue
                    emb = dequantize(patch)
                    emb[:, mask == 0] = 0.0
                    X[j] = emb
                    M[j, 0] = mask
                    ok[j] = True

    return {
        "idxs": idxs,
        "ids": ids,
        "X": X,
        "M": M,
        "ok": ok,
        "path": path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3, help="odd window size (default 3)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--regions", nargs="*", default=None)
    args = ap.parse_args()

    k = int(args.k)
    if k < 1 or k % 2 == 0:
        raise SystemExit("--k must be a positive odd integer")

    out = out_dir_for(k)
    out.mkdir(parents=True, exist_ok=True)
    x_path = out / "X_nb.npy"
    m_path = out / "mask.npy"
    meta_path = out / "meta.parquet"
    man_path = out / "manifest.json"

    if x_path.exists() and meta_path.exists() and not args.rebuild:
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        if man.get("k") == k:
            print(f"Exists: {out} (pass --rebuild to overwrite)", flush=True)
            return

    t0 = time.time()
    print(f"Loading {PIXELS}", flush=True)
    df = pd.read_parquet(PIXELS)
    if args.regions:
        df = df.loc[df["region"].isin(args.regions)].copy()
    df = df.reset_index(drop=True)
    print(f"n_pixels={len(df)}  k={k}", flush=True)

    index = ensure_index()
    index_slim = index[
        ["path", "crs", "wgs84_west", "wgs84_east", "wgs84_south", "wgs84_north"]
    ].copy()

    jobs = _assign_rows_to_tiles(df, index_slim)
    for j in jobs:
        j["k"] = k
    print(f"tiles={len(jobs)}", flush=True)

    X_all = np.zeros((len(df), 64, k, k), dtype=np.float32)
    M_all = np.zeros((len(df), 1, k, k), dtype=np.uint8)
    ok_all = np.zeros(len(df), dtype=bool)

    n_workers = args.workers or min(len(jobs), os.cpu_count() or 1)
    print(f"Fetching windows · workers={n_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_fetch_tile_windows, job): job["path"] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="tiles"):
            r = fut.result()
            for local_i, (gi, good) in enumerate(zip(r["idxs"], r["ok"])):
                if not good:
                    continue
                X_all[gi] = r["X"][local_i]
                M_all[gi] = r["M"][local_i]
                ok_all[gi] = True

    n_ok = int(ok_all.sum())
    print(f"valid neighborhoods: {n_ok}/{len(df)} ({100 * n_ok / len(df):.1f}%)", flush=True)
    meta = df.loc[ok_all].reset_index(drop=True)
    X_ok = X_all[ok_all]
    M_ok = M_all[ok_all]

    np.save(x_path, X_ok)
    np.save(m_path, M_ok)
    meta.to_parquet(meta_path, index=False)
    man = {
        "name": f"week3_pixel_neighborhood_k{k}",
        "k": k,
        "n": int(len(meta)),
        "n_source": int(len(df)),
        "n_dropped_center_nodata_or_miss": int(len(df) - len(meta)),
        "shape_X": list(X_ok.shape),
        "shape_mask": list(M_ok.shape),
        "bands": BANDS,
        "layout": "channels_first (64, k, k) + mask (1, k, k); train concat → (65, k, k)",
        "source_pixels": str(PIXELS.relative_to(ROOT)),
        "seed": SEED,
        "elapsed_s": round(time.time() - t0, 1),
    }
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
