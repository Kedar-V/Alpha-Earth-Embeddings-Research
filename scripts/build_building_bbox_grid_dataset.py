#!/usr/bin/env python3
"""Build bbox-grid AlphaEarth embedding dataset for CNN training.

For each labeled Overture building:
  1. Take the axis-aligned rectangle enclosing the footprint
  2. Collect every underlying 10 m AlphaEarth cell (no averaging)
  3. Store the ragged cell stack + building metadata/polygon

Reads AlphaEarth 2024 COGs from the public GCS bucket
(gs://alphaearth_foundations) — no Earth Engine auth required.

Writes under datasets/building_bbox_grid/:
  buildings.parquet       1 row / building (meta + polygon + grid shape)
  building_cells.parquet  1 row / 10 m cell (id, i, j, lon/lat, utm, pixel_wkt, A00..A63)
  manifest.json
  load_example.py         tiny loader: id -> (H, W, 64) float32 array

Usage (from repo root):
  python scripts/build_building_bbox_grid_dataset.py
  python scripts/build_building_bbox_grid_dataset.py --limit 50   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.prepared import prep
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
# Prefer unstratified labels_only (natural RES mix); fall back to centroid dataset.
SRC_LABELS = ROOT / "outputs" / "building_type_dataset" / "labels_only.parquet"
SRC_DATASET = ROOT / "outputs" / "building_type_dataset" / "dataset.parquet"
OUT_DIR = ROOT / "datasets" / "building_bbox_grid"
CACHE_DIR = OUT_DIR / "_cache"
TILE_CACHE = CACHE_DIR / "tiles"
INDEX_FULL = CACHE_DIR / "aef_index.parquet"
INDEX_US = CACHE_DIR / "aef_index_2024_us.parquet"

EMBED_YEAR = 2024
BANDS = [f"A{i:02d}" for i in range(64)]
INDEX_URL = (
    "https://storage.googleapis.com/alphaearth_foundations/"
    "satellite_embedding/v1/annual/aef_index.parquet"
)
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "268435456",
}

META_COLS = [
    "id",
    "region",
    "lon",
    "lat",
    "wkt",
    "subtype",
    "class",
    "label",
    "brief_category",
    "y_res",
    "height",
    "num_floors",
    "area_m2_approx",
    "source_primary",
    "label_source",
    "split_seed",
]


def gs_to_https(path: str) -> str:
    return path.replace("gs://", "https://storage.googleapis.com/")


def dequantize(values: np.ndarray) -> np.ndarray:
    """int8 COG values -> float embeddings in [-1, 1] (unit-ish vectors)."""
    v = values.astype(np.float32)
    return ((v / 127.5) ** 2) * np.sign(v)


def ensure_index() -> pd.DataFrame:
    import urllib.request

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


def window_rc(src: rasterio.DatasetReader, west: float, south: float, east: float, north: float):
    rows, cols = [], []
    for x, y in ((west, south), (west, north), (east, south), (east, north)):
        r, c = src.index(x, y)
        rows.append(r)
        cols.append(c)
    r0, r1 = min(rows), max(rows) + 1
    c0, c1 = min(cols), max(cols) + 1
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(src.height, r1)
    c1 = min(src.width, c1)
    if r1 <= r0 or c1 <= c0:
        return None
    return (r0, r1), (c0, c1)


def load_buildings(limit: int | None) -> pd.DataFrame:
    src = SRC_LABELS if SRC_LABELS.exists() else SRC_DATASET
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {SRC_LABELS} (or {SRC_DATASET}). "
            "Run: python scripts/build_building_type_dataset.py --rebuild-labels --labels-only"
        )
    ds = pd.read_parquet(src)
    ds["id"] = ds["id"].astype(str)
    cols = [c for c in META_COLS if c in ds.columns]
    out = ds[cols].copy()
    if "y_res" in out.columns:
        pct = 100.0 * float(out["y_res"].mean())
        print(
            f"Source buildings: {len(out):,} from {src.relative_to(ROOT)} "
            f"(pct_res={pct:.1f}%, unstratified expected ~90%)",
            flush=True,
        )
    else:
        print(f"Source buildings: {len(out):,} from {src.relative_to(ROOT)}", flush=True)
    if limit is not None:
        out = out.iloc[:limit].copy()
        print(f"  limit → {len(out):,}", flush=True)
    return out


def assign_tiles(buildings: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach intersecting AE tile paths per building (centroid + pad; fast vectorized probe)."""
    # ~250 m pad so edge footprints still hit the right COG(s).
    pad = 0.0025
    west = index["wgs84_west"].to_numpy()
    east = index["wgs84_east"].to_numpy()
    south = index["wgs84_south"].to_numpy()
    north = index["wgs84_north"].to_numpy()
    paths = index["path"].to_numpy()
    crss = index["crs"].to_numpy()

    records = []
    for _, b in tqdm(buildings.iterrows(), total=len(buildings), desc="assign tiles"):
        lon = float(b["lon"])
        lat = float(b["lat"])
        # Prefer footprint AABB when WKT present (correct near tile seams).
        try:
            g = shapely_wkt.loads(b["wkt"])
            if not g.is_valid:
                g = g.buffer(0)
            minx, miny, maxx, maxy = g.bounds
        except Exception:
            minx = lon - pad
            miny = lat - pad
            maxx = lon + pad
            maxy = lat + pad
        mask = (west <= maxx) & (east >= minx) & (south <= maxy) & (north >= miny)
        if not mask.any():
            records.append({**b.to_dict(), "tile_paths": [], "crs": None, "bbox": (minx, miny, maxx, maxy)})
            continue
        hit_paths = paths[mask].tolist()
        records.append(
            {
                **b.to_dict(),
                "tile_paths": hit_paths,
                "crs": str(crss[mask][0]),
                "bbox": (minx, miny, maxx, maxy),
            }
        )
    return pd.DataFrame(records)


def extract_from_tile(
    src: rasterio.DatasetReader,
    crs: str,
    buildings_on_tile: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Read one covering window and slice each building bbox from it."""
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Union of building bboxes in WGS84 -> UTM covering window
    xs, ys = [], []
    parsed = []
    for _, b in buildings_on_tile.iterrows():
        g = shapely_wkt.loads(b["wkt"])
        if not g.is_valid:
            g = g.buffer(0)
        minx, miny, maxx, maxy = g.bounds
        for lon, lat in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
            x, y = tfm.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        parsed.append((b, g, (minx, miny, maxx, maxy)))

    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    # pad by half a pixel so edge centers are inside
    pad = 5.0
    win = window_rc(src, west - pad, south - pad, east + pad, north + pad)
    if win is None:
        return [], []
    (r0, r1), (c0, c1) = win
    raw = src.read(window=((r0, r1), (c0, c1)))  # (64, H, W) int8
    # pixel center coordinates in UTM for the window
    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    # affine: x = a*col + c; y = e*row + f  (and b=d=0)
    a = src.transform.a
    c_off = src.transform.c
    e = src.transform.e
    f_off = src.transform.f
    # center of pixel (row, col)
    xs_c = c_off + a * (cols + 0.5)
    ys_c = f_off + e * (rows + 0.5)
    # mesh for quick lookup: shape (H, W)
    XX, YY = np.meshgrid(xs_c, ys_c)

    cell_rows: list[dict] = []
    building_rows: list[dict] = []

    for b, geom, (minx, miny, maxx, maxy) in parsed:
        x0, y0 = tfm.transform(minx, miny)
        x1, y1 = tfm.transform(maxx, maxy)
        bw, be = min(x0, x1), max(x0, x1)
        bs, bn = min(y0, y1), max(y0, y1)

        # pixels whose centers fall inside the enclosing rectangle (in UTM)
        # also include a 1-pixel dilation so thin footprints still get cells
        mask = (XX >= bw - 0.5 * abs(a)) & (XX <= be + 0.5 * abs(a)) & (
            YY >= bs - 0.5 * abs(e)
        ) & (YY <= bn + 0.5 * abs(e))
        if not mask.any():
            # fallback: nearest pixel to centroid
            ux, uy = tfm.transform(float(b["lon"]), float(b["lat"]))
            rr, cc = src.index(ux, uy)
            rr -= r0
            cc -= c0
            if 0 <= rr < raw.shape[1] and 0 <= cc < raw.shape[2]:
                mask = np.zeros_like(XX, dtype=bool)
                mask[rr, cc] = True
            else:
                continue

        prep_g = prep(geom)
        rr_idx, cc_idx = np.where(mask)
        # Raster row/col -> local CNN grid. i=0 at north, j=0 at west.
        # e>0 means row increases with northing, so north = max row.
        r_min, r_max = int(rr_idx.min()), int(rr_idx.max())
        c_min, c_max = int(cc_idx.min()), int(cc_idx.max())
        n_rows = r_max - r_min + 1
        n_cols = c_max - c_min + 1

        n_cells = 0
        n_inside = 0
        seen: set[tuple[int, int]] = set()
        for r_i, c_i in zip(rr_idx.tolist(), cc_idx.tolist()):
            if e >= 0:
                i = r_max - int(r_i)
            else:
                i = int(r_i) - r_min
            j = int(c_i) - c_min
            if (i, j) in seen:
                continue
            pix = raw[:, r_i, c_i]
            if (pix == -128).any():
                continue
            emb = dequantize(pix)
            ux = float(XX[r_i, c_i])
            uy = float(YY[r_i, c_i])
            lon, lat = inv.transform(ux, uy)
            # Exact COG pixel footprint: UTM pixel bounds → WGS84 ring (no approx).
            hx, hy = abs(a) / 2.0, abs(e) / 2.0
            ring = [
                inv.transform(ux - hx, uy - hy),
                inv.transform(ux + hx, uy - hy),
                inv.transform(ux + hx, uy + hy),
                inv.transform(ux - hx, uy + hy),
                inv.transform(ux - hx, uy - hy),
            ]
            pixel_wkt = "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"
            inside = bool(prep_g.contains(Point(lon, lat))) or bool(
                prep_g.intersects(Point(lon, lat).buffer(1e-7))
            )
            seen.add((i, j))
            row = {
                "id": str(b["id"]),
                "i": int(i),
                "j": int(j),
                "pixel_lon": float(lon),
                "pixel_lat": float(lat),
                "utm_x": ux,
                "utm_y": uy,
                "crs": crs,
                "pixel_wkt": pixel_wkt,
                "pixel_size_m": float(abs(a)),
                "inside_polygon": bool(inside),
                "embed_year": EMBED_YEAR,
                "sample_method": "bbox_grid",
            }
            for k, band in enumerate(BANDS):
                row[band] = float(emb[k])
            cell_rows.append(row)
            n_cells += 1
            n_inside += int(inside)

        if n_cells == 0:
            continue

        building_rows.append(
            {
                "id": str(b["id"]),
                "region": b["region"],
                "lon": float(b["lon"]),
                "lat": float(b["lat"]),
                "wkt": b["wkt"],
                "subtype": b.get("subtype"),
                "class": b.get("class"),
                "label": b.get("label"),
                "brief_category": b.get("brief_category"),
                "y_res": b.get("y_res"),
                "height": b.get("height"),
                "num_floors": b.get("num_floors"),
                "area_m2_approx": b.get("area_m2_approx"),
                "source_primary": b.get("source_primary"),
                "label_source": b.get("label_source"),
                "split_seed": b.get("split_seed"),
                "bbox_west": float(minx),
                "bbox_south": float(miny),
                "bbox_east": float(maxx),
                "bbox_north": float(maxy),
                "n_rows": int(n_rows),
                "n_cols": int(n_cols),
                "n_cells": int(n_cells),
                "n_cells_inside_polygon": int(n_inside),
                "embed_year": EMBED_YEAR,
                "sample_method": "bbox_grid",
                "crs": crs,
            }
        )

    return building_rows, cell_rows


def tile_cache_key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _rebuild_one_building(g: pd.DataFrame) -> pd.DataFrame:
    """Recompute i,j from UTM (i=0 north, j=0 west) for one building's cells.

    Key by rounded meter coordinates — NOT utm/10 with np.round, which uses
    banker's rounding and collapses adjacent 10 m centers into a false 20 m lattice.
    """
    g = g.copy()
    g["_n"] = np.rint(g["utm_y"].to_numpy()).astype(np.int64)
    g["_e"] = np.rint(g["utm_x"].to_numpy()).astype(np.int64)
    g = g.drop_duplicates(["_n", "_e"], keep="first")
    uniq_n = sorted(g["_n"].unique().tolist(), reverse=True)
    uniq_e = sorted(g["_e"].unique().tolist())
    n_to_i = {v: i for i, v in enumerate(uniq_n)}
    e_to_j = {v: j for j, v in enumerate(uniq_e)}
    g["i"] = g["_n"].map(n_to_i).astype(int)
    g["j"] = g["_e"].map(e_to_j).astype(int)
    return g.drop(columns=["_n", "_e"]).sort_values(["i", "j"]).reset_index(drop=True)


def _rebuild_chunk_from_path(chunk_path: str) -> pd.DataFrame:
    """Worker: rebuild i,j for all buildings in one parquet chunk."""
    chunk = pd.read_parquet(chunk_path)
    if chunk.empty:
        return chunk
    parts = [_rebuild_one_building(g) for _, g in chunk.groupby("id", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else chunk.iloc[0:0].copy()


def merge_from_tile_parquets(
    building_paths: list[Path],
    cell_paths: list[Path],
    n_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-tile caches, dedup seams, rebuild i,j in parallel, attach building stats."""
    if not cell_paths:
        return pd.DataFrame(), pd.DataFrame()

    print(f"Loading {len(building_paths)} tile building tables ...", flush=True)
    bdf = pd.concat(
        [pd.read_parquet(p) for p in tqdm(building_paths, desc="load buildings")],
        ignore_index=True,
    )
    bdf = bdf.drop_duplicates("id", keep="first")
    print(f"  buildings={len(bdf):,}", flush=True)

    print(f"Loading {len(cell_paths)} tile cell tables ...", flush=True)
    cells = pd.concat(
        [pd.read_parquet(p) for p in tqdm(cell_paths, desc="load cells")],
        ignore_index=True,
    )
    print(f"  raw cells={len(cells):,}", flush=True)

    print("Dedup across tile seams ...", flush=True)
    t_dedup = time.time()
    before = len(cells)
    cells = cells.copy()
    cells["id"] = cells["id"].astype(str)
    cells["_lon_r"] = cells["pixel_lon"].round(6)
    cells["_lat_r"] = cells["pixel_lat"].round(6)
    cells = cells.drop_duplicates(["id", "_lon_r", "_lat_r"], keep="first")
    cells = cells.drop(columns=["_lon_r", "_lat_r"])
    print(
        f"  after dedup={len(cells):,} (dropped {before - len(cells):,}) "
        f"in {time.time() - t_dedup:.1f}s",
        flush=True,
    )

    # Partition by building id → temp parquet chunks → process pool (avoids giant pickles).
    n_workers = max(1, n_workers)
    ids = np.sort(cells["id"].unique())
    id_splits = [s for s in np.array_split(ids, n_workers) if len(s)]
    chunk_dir = CACHE_DIR / "_merge_chunks"
    if chunk_dir.exists():
        for old in chunk_dir.glob("*.parquet"):
            old.unlink()
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(id_splits)} merge chunks for {n_workers} workers ...", flush=True)
    chunk_paths: list[str] = []
    id_to_split = {}
    for i, split in enumerate(id_splits):
        for bid in split:
            id_to_split[bid] = i
    cells["_split"] = cells["id"].map(id_to_split)
    for i in tqdm(range(len(id_splits)), desc="write chunks"):
        p = chunk_dir / f"chunk_{i:03d}.parquet"
        cells.loc[cells["_split"] == i].drop(columns=["_split"]).to_parquet(p, index=False)
        chunk_paths.append(str(p))
    cells = cells.drop(columns=["_split"])

    print(f"Rebuild i,j · {len(chunk_paths)} chunks · {n_workers} workers ...", flush=True)
    t_rebuild = time.time()
    rebuilt_parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_rebuild_chunk_from_path, p): p for p in chunk_paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="rebuild i,j"):
            rebuilt_parts.append(fut.result())
    cells = pd.concat(rebuilt_parts, ignore_index=True)
    print(f"  rebuilt cells={len(cells):,} in {time.time() - t_rebuild:.1f}s", flush=True)

    print("Attach per-building grid stats ...", flush=True)
    stats = (
        cells.groupby("id")
        .agg(
            n_cells=("i", "size"),
            n_cells_inside_polygon=("inside_polygon", "sum"),
            n_rows=("i", "max"),
            n_cols=("j", "max"),
        )
        .reset_index()
    )
    stats["n_rows"] = stats["n_rows"] + 1
    stats["n_cols"] = stats["n_cols"] + 1
    bdf = bdf.drop(columns=["n_cells", "n_cells_inside_polygon", "n_rows", "n_cols"], errors="ignore")
    bdf["id"] = bdf["id"].astype(str)
    bdf = bdf.merge(stats, on="id", how="inner")

    for p in chunk_paths:
        Path(p).unlink(missing_ok=True)
    return bdf, cells


def write_loader(path: Path) -> None:
    path.write_text(
        '''"""Load a building bbox-grid as an (H, W, 64) float32 array for CNNs.

Example:
  from pathlib import Path
  import importlib.util
  spec = importlib.util.spec_from_file_location("load", Path("datasets/building_bbox_grid/load_example.py"))
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  grid, meta, inside = mod.load_grid("<building-id>")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BANDS = [f"A{i:02d}" for i in range(64)]
_buildings = None
_cells = None


def _tables():
    global _buildings, _cells
    if _buildings is None:
        _buildings = pd.read_parquet(ROOT / "buildings.parquet").set_index("id")
        _cells = pd.read_parquet(ROOT / "building_cells.parquet")
    return _buildings, _cells


def load_grid(building_id: str):
    buildings, cells = _tables()
    meta = buildings.loc[str(building_id)]
    g = cells.loc[cells["id"] == str(building_id)]
    H, W = int(meta["n_rows"]), int(meta["n_cols"])
    grid = np.zeros((H, W, 64), dtype=np.float32)
    inside = np.zeros((H, W), dtype=bool)
    present = np.zeros((H, W), dtype=bool)
    vals = g[BANDS].to_numpy(dtype=np.float32)
    for k, (_, r) in enumerate(g.iterrows()):
        i, j = int(r["i"]), int(r["j"])
        grid[i, j] = vals[k]
        inside[i, j] = bool(r["inside_polygon"])
        present[i, j] = True
    return grid, meta, inside, present
'''
    )


def process_one_tile(payload: dict) -> dict:
    """Worker: open one COG, extract buildings, write per-tile parquet caches."""
    path = payload["path"]
    crs = payload["crs"]
    cache_dir = Path(payload["cache_dir"])
    force = bool(payload.get("force", False))
    key = tile_cache_key(path)
    b_path = cache_dir / f"{key}_buildings.parquet"
    c_path = cache_dir / f"{key}_cells.parquet"
    t0 = time.time()

    if b_path.exists() and c_path.exists() and not force:
        nb = len(pd.read_parquet(b_path, columns=["id"]))
        nc = len(pd.read_parquet(c_path, columns=["id"]))
        return {
            "path": path,
            "buildings_pq": str(b_path),
            "cells_pq": str(c_path),
            "elapsed_s": round(time.time() - t0, 1),
            "n_in": int(payload.get("n_in", 0)),
            "n_buildings": nb,
            "n_cells": nc,
            "cached": True,
        }

    subset = pd.DataFrame(payload["buildings"])
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(gs_to_https(path)) as src:
            b_rows, c_rows = extract_from_tile(src, crs, subset)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if b_rows:
        pd.DataFrame(b_rows).to_parquet(b_path, index=False)
    else:
        # empty schema placeholder — skip writing empty; signal via missing file
        b_path.unlink(missing_ok=True)
    if c_rows:
        pd.DataFrame(c_rows).to_parquet(c_path, index=False)
    else:
        c_path.unlink(missing_ok=True)

    return {
        "path": path,
        "buildings_pq": str(b_path) if b_path.exists() else None,
        "cells_pq": str(c_path) if c_path.exists() else None,
        "elapsed_s": round(time.time() - t0, 1),
        "n_in": len(subset),
        "n_buildings": len(b_rows),
        "n_cells": len(c_rows),
        "cached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional building cap (smoke test)")
    parser.add_argument("--rebuild", action="store_true", help="Overwrite final buildings/cells parquet")
    parser.add_argument(
        "--reuse-tiles",
        action="store_true",
        help="Reuse per-tile parquet caches under _cache/tiles/ (skip COG re-extract)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel tile + merge workers (default: min(n_tiles or cpu_count, cpu_count))",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TILE_CACHE.mkdir(parents=True, exist_ok=True)
    buildings_path = OUT_DIR / "buildings.parquet"
    cells_path = OUT_DIR / "building_cells.parquet"
    if buildings_path.exists() and cells_path.exists() and not args.rebuild and args.limit is None:
        print("Dataset already exists. Pass --rebuild to overwrite.", flush=True)
        return

    t0 = time.time()
    index = ensure_index()
    buildings = load_buildings(args.limit)
    print("Assigning AE tiles to buildings ...", flush=True)
    assigned = assign_tiles(buildings, index)
    missing = assigned[assigned["tile_paths"].map(len) == 0]
    if len(missing):
        print(f"WARNING: {len(missing)} buildings with no intersecting tile", flush=True)

    tile_to_ids: dict[str, list[str]] = defaultdict(list)
    by_id = assigned.set_index("id")
    for _, row in assigned.iterrows():
        for p in row["tile_paths"]:
            tile_to_ids[p].append(row["id"])

    tile_meta = index.drop_duplicates("path").set_index("path")
    force_tiles = not args.reuse_tiles
    payloads = []
    for path, ids in sorted(tile_to_ids.items()):
        subset = by_id.loc[ids].reset_index()
        payloads.append(
            {
                "path": path,
                "crs": str(tile_meta.loc[path, "crs"]),
                "buildings": subset.to_dict(orient="list"),
                "cache_dir": str(TILE_CACHE),
                "force": force_tiles,
                "n_in": len(subset),
            }
        )

    n_tiles = len(payloads)
    workers = args.workers or min(n_tiles, os.cpu_count() or 1)
    workers = max(1, min(workers, n_tiles if n_tiles else 1))
    print(
        f"Unique tiles: {n_tiles} · workers: {workers} · cpus: {os.cpu_count()} · "
        f"tile_cache={'reuse' if args.reuse_tiles else 'force-refresh'}",
        flush=True,
    )

    building_paths: list[Path] = []
    cell_paths: list[Path] = []
    done = 0
    n_cached = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one_tile, p): p["path"] for p in payloads}
        for fut in tqdm(as_completed(futures), total=n_tiles, desc="extract tiles"):
            path = futures[fut]
            result = fut.result()
            done += 1
            if result.get("cached"):
                n_cached += 1
            if result["buildings_pq"]:
                building_paths.append(Path(result["buildings_pq"]))
            if result["cells_pq"]:
                cell_paths.append(Path(result["cells_pq"]))
            tag = "cache" if result.get("cached") else "fetch"
            print(
                f"[{done}/{n_tiles}] {tag} {path.split('/')[-1]} · "
                f"in={result['n_in']} out_b={result['n_buildings']} "
                f"cells={result['n_cells']} · {result['elapsed_s']}s",
                flush=True,
            )

    print(
        f"Tile extract done · {n_cached}/{n_tiles} from cache · "
        f"merging with {os.cpu_count()} workers ...",
        flush=True,
    )
    merge_workers = args.workers or (os.cpu_count() or 1)
    bdf, cdf = merge_from_tile_parquets(building_paths, cell_paths, n_workers=merge_workers)
    if bdf.empty:
        raise RuntimeError("No buildings extracted")

    print(f"Writing {buildings_path} ...", flush=True)
    bdf.to_parquet(buildings_path, index=False)
    print(f"Writing {cells_path} ({len(cdf):,} rows) ...", flush=True)
    cdf.to_parquet(cells_path, index=False)
    write_loader(OUT_DIR / "load_example.py")

    manifest = {
        "name": "building_bbox_grid_v1",
        "label_definition": "resi_def_v1",
        "label_definition_doc": "docs/residential_definition_v1.md",
        "purpose": (
            "Per-building enclosing-rectangle grid of AlphaEarth 10 m embeddings "
            "(no averaging) for CNN models. Building y_res follows resi_def_v1 Building GT."
        ),
        "source_labels": str(
            (SRC_LABELS if SRC_LABELS.exists() else SRC_DATASET).relative_to(ROOT)
        ),
        "sampling_note": (
            "Buildings from unstratified Track-A labels (natural RES-majority mix). "
            "No per-class polygon stratification."
        ),
        "embedding": {
            "product": "AlphaEarth Foundations Satellite Embedding V1 (annual)",
            "year": EMBED_YEAR,
            "dim": 64,
            "resolution_m": 10,
            "source": "gs://alphaearth_foundations COGs (public, provider-pays)",
            "dequantization": "((v/127.5)**2)*sign(v); nodata=-128 dropped",
            "sample_method": "bbox_grid — all 10 m cells whose centers fall in the footprint AABB",
            "grid_indexing": "i=0 at north, j=0 at west; emb[i,j,:] is 64-D",
            "attribution": (
                "The AlphaEarth Foundations Satellite Embedding dataset is produced "
                "by Google and Google DeepMind."
            ),
        },
        "n_buildings": int(len(bdf)),
        "n_cells": int(len(cdf)),
        "n_regions": int(bdf["region"].nunique()),
        "regions": sorted(bdf["region"].unique().tolist()),
        "pct_res": float(bdf["y_res"].mean()) if "y_res" in bdf.columns else None,
        "label_counts": bdf["label"].value_counts().to_dict() if "label" in bdf.columns else {},
        "cells_per_building": {
            "min": int(bdf["n_cells"].min()),
            "median": float(bdf["n_cells"].median()),
            "p90": float(bdf["n_cells"].quantile(0.9)),
            "max": int(bdf["n_cells"].max()),
        },
        "merge": {
            "parallel_workers": merge_workers,
            "tile_cache": str(TILE_CACHE.relative_to(ROOT)),
            "tiles_from_cache": n_cached,
        },
        "paths": {
            "buildings": "datasets/building_bbox_grid/buildings.parquet",
            "cells": "datasets/building_bbox_grid/building_cells.parquet",
            "loader": "datasets/building_bbox_grid/load_example.py",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== bbox-grid dataset ready ===", flush=True)
    print(f"buildings: {len(bdf):,}  cells: {len(cdf):,}  pct_res={manifest['pct_res']}", flush=True)
    if "label" in bdf.columns:
        print(bdf.groupby("label")["n_cells"].median().sort_values(ascending=False).to_string(), flush=True)
    print(f"Wrote {buildings_path}", flush=True)
    print(f"Wrote {cells_path}", flush=True)
    print(f"Elapsed {manifest['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
