#!/usr/bin/env python3
"""Ingest USA week4 metros into PostgreSQL under the international contract.

Per city:
  1) Overture Track-A buildings in discovery disk around city center
  2) Compute AOI R = max geodesic distance of those buildings
  3) AlphaEarth 10 m / 2024 cells for building bboxes
  4) Intersections + versioned cell labels + week4 sidecar tables

Resumable via per-city stage DONE markers under RUN_DIR/checkpoints/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from build_building_bbox_grid_dataset import (  # noqa: E402
    BANDS,
    EMBED_YEAR,
    GDAL_ENV,
    assign_tiles,
    dequantize,
    ensure_index,
    extract_from_tile,
    gs_to_https,
    tile_cache_key,
    window_rc,
)
from build_building_type_dataset import (  # noqa: E402
    BRIEF_ROLLUP,
    BUILDINGS,
    CLASS_TO_LABEL,
    RELEASE,
    duck,
    primary_source,
)
from lib.db import connect, load_dotenv, wait_until_ready  # noqa: E402
from pg_batch import (  # noqa: E402
    constrained_label_fracs,
    copy_upsert_buildings,
    insert_cells_and_labels_parallel,
)
from sample_buildings import WEEK4_EQUAL_CITY_AREA_M2, stratified_area_balance  # noqa: E402

LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
RES_LABELS = {"single_family", "multi_family"}


def _ensure_mp_spawn() -> None:
    """Avoid fork-from-thread deadlocks (Linux + ThreadPool + ProcessPool)."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set by a parent / prior call in this interpreter.
        pass
    method = mp.get_start_method(allow_none=True)
    if method != "spawn":
        print(f"WARNING: multiprocessing start method is {method!r}, expected 'spawn'", flush=True)


def _embed_city_job(payload: dict) -> dict:
    """Top-level AEF worker for ProcessPoolExecutor (must be picklable under spawn).

    Runs in its own process so nested ProcessPools for tiles/shards/labels are safe.
    Never call embed_and_label_city from a ThreadPool thread.
    """
    return embed_and_label_city(
        payload["city"],
        Path(payload["run_dir"]),
        int(payload["workers"]),
        int(payload["aef_source_id"]),
        aef_shard_size=int(payload["aef_shard_size"]),
    )


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def update_progress(run_dir: Path, **kwargs) -> None:
    p = run_dir / "progress.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data.update(kwargs)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["last_progress_at"] = data["updated_at"]
    atomic_json(p, data)


def stage_done(run_dir: Path, city_id: str, stage: str) -> bool:
    return (run_dir / "checkpoints" / city_id / f"{stage}.DONE").exists()


def mark_stage(run_dir: Path, city_id: str, stage: str, payload: dict | None = None) -> None:
    d = run_dir / "checkpoints" / city_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}.DONE").write_text("ok\n")
    if payload is not None:
        atomic_json(d / f"{stage}.json", payload)


def prune_city_aef_temps(staging: Path, *, city_id: str | None = None) -> dict:
    """Drop COG windows + shard checkpoints once cells.parquet is durable.

    Keeps buildings/sample/cells/id_map; those are small and useful for audit/resume.
    """
    freed = 0
    removed: list[str] = []
    for name in ("cog_windows", "aef_shards"):
        path = staging / name
        if not path.exists():
            continue
        size = 0
        if path.is_dir():
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        size += p.stat().st_size
                    except OSError:
                        pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            path.unlink(missing_ok=True)
        freed += size
        removed.append(name)
    tag = city_id or staging.name
    if removed:
        print(
            f"[{tag}] pruned temps {removed} freed={freed / 1e9:.2f}GB",
            flush=True,
        )
    return {"removed": removed, "freed_bytes": int(freed)}


def discovery_bbox(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def pull_overture(city_id: str, lon: float, lat: float, discovery_radius_m: float) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = discovery_bbox(lon, lat, discovery_radius_m)
    class_list = ", ".join(f"'{c}'" for c in sorted(CLASS_TO_LABEL))
    sql = f"""
    WITH b AS (
      SELECT
        id, subtype, class, sources, height, num_floors,
        ST_X(ST_Centroid(geometry)) AS lon,
        ST_Y(ST_Centroid(geometry)) AS lat,
        ST_AsText(geometry) AS wkt,
        ST_Area(ST_Transform(geometry, 'EPSG:3857')) AS area_m2_approx
      FROM read_parquet('{BUILDINGS}', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
        AND bbox.ymin BETWEEN {ymin} AND {ymax}
        AND class IS NOT NULL
        AND class IN ({class_list})
        AND geometry IS NOT NULL
    )
    SELECT * FROM b
    WHERE lon BETWEEN {xmin} AND {xmax}
      AND lat BETWEEN {ymin} AND {ymax}
    """
    con = duck()
    df = con.execute(sql).df()
    if df.empty:
        return df
    # Keep buildings inside discovery disk (geodesic approx via haversine)
    df = df.loc[_within_radius_m(df["lon"].to_numpy(), df["lat"].to_numpy(), lon, lat, discovery_radius_m)].copy()
    df["id"] = df["id"].astype(str)
    df["label"] = df["class"].map(CLASS_TO_LABEL)
    df = df.loc[df["label"].notna()].copy()
    df["brief_category"] = df["label"].map(BRIEF_ROLLUP)
    df["y_res"] = df["label"].isin(RES_LABELS).astype(int)
    df["source_primary"] = df["sources"].map(primary_source)
    df["city_id"] = city_id
    df["harmonized_label"] = np.where(df["y_res"] == 1, "residential", "non_residential")
    return df.reset_index(drop=True)


def _within_radius_m(lons, lats, clon, clat, radius_m: float) -> np.ndarray:
    """Haversine filter."""
    r = 6371000.0
    lon1 = np.radians(lons.astype(float))
    lat1 = np.radians(lats.astype(float))
    lon2 = math.radians(clon)
    lat2 = math.radians(clat)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * r * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return dist <= radius_m


def ensure_data_sources(conn) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO data_sources (
              source_name, source_version, source_year, acquisition_date,
              processing_version, label_mapping_version, uri, metadata
            ) VALUES (
              'overture_buildings', %s, 2026, CURRENT_DATE,
              'international_ingest_v1', 'resi_def_v1', %s,
              '{"theme":"buildings","type":"building"}'::jsonb
            )
            ON CONFLICT (source_name, source_version, processing_version) DO UPDATE
              SET acquisition_date = EXCLUDED.acquisition_date
            RETURNING source_id
            """,
            (RELEASE, BUILDINGS),
        )
        overture_id = cur.fetchone()["source_id"]
        cur.execute(
            """
            INSERT INTO data_sources (
              source_name, source_version, source_year, acquisition_date,
              processing_version, label_mapping_version, uri, metadata
            ) VALUES (
              'alphaearth_foundations', 'v1', %s, CURRENT_DATE,
              'international_ingest_v1', NULL,
              'gs://alphaearth_foundations/satellite_embedding/v1/annual/',
              '{"bands":64,"pixel_size_m":10}'::jsonb
            )
            ON CONFLICT (source_name, source_version, processing_version) DO UPDATE
              SET acquisition_date = EXCLUDED.acquisition_date
            RETURNING source_id
            """,
            (EMBED_YEAR,),
        )
        aef_id = cur.fetchone()["source_id"]
    return int(overture_id), int(aef_id)


def insert_buildings(df: pd.DataFrame, overture_source_id: int) -> dict[str, int]:
    """Batched COPY upsert buildings + audit; return source_feature_id -> building_id."""
    return copy_upsert_buildings(
        str(df["city_id"].iloc[0]),
        df,
        overture_source_id,
        RELEASE,
    )


def compute_aoi(conn, city_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE cities SET aoi_center = COALESCE(aoi_center, centroid), updated_at = now()
            WHERE city_id = %s
            """,
            (city_id,),
        )
        cur.execute(
            """
            SELECT
              COUNT(*) AS n,
              MAX(ST_Distance(b.geom::geography, c.aoi_center::geography)) AS r_raw
            FROM buildings b
            JOIN cities c ON c.city_id = b.city_id
            WHERE b.city_id = %s
            """,
            (city_id,),
        )
        stats = cur.fetchone()
        if not stats["n"] or stats["r_raw"] is None:
            raise RuntimeError(f"No buildings for AOI on {city_id}")
        radius_m = float(stats["r_raw"])
        cur.execute(
            """
            UPDATE cities SET
              aoi_radius_m = %(r)s,
              aoi_buffer_m = 0,
              aoi_method = 'building_enclosing_radius',
              aoi_building_count = %(n)s,
              aoi_discovery_radius_m = %(disc)s,
              aoi_disk = city_aoi_disk_from_radius(aoi_center, %(r)s),
              aoi_computed_at = now(),
              updated_at = now()
            WHERE city_id = %(city_id)s
            RETURNING city_id, aoi_radius_m, aoi_building_count,
                      ST_Area(aoi_disk::geography)/1e6 AS area_km2
            """,
            {
                "r": radius_m,
                "n": int(stats["n"]),
                "disc": float(os.environ.get("DISCOVERY_RADIUS_M", "15000")),
                "city_id": city_id,
            },
        )
        return dict(cur.fetchone())



def _utm_bounds_for_buildings(buildings: pd.DataFrame, crs: str) -> tuple[float, float, float, float] | None:
    """Union of building footprint corners in tile CRS (UTM)."""
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs: list[float] = []
    ys: list[float] = []
    for w in buildings["wkt"].astype(str).tolist():
        try:
            g = shapely_wkt.loads(w)
            if not g.is_valid:
                g = g.buffer(0)
            minx, miny, maxx, maxy = g.bounds
        except Exception:
            continue
        for lon, lat in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
            x, y = tfm.transform(float(lon), float(lat))
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _materialize_cog_window(payload: dict) -> tuple[str, str]:
    """Fetch the buildings' covering window from a remote COG once → local GeoTIFF.

    Full AEF tiles are ~2GB; we only cache the city window so shard workers read disk.
    """
    remote = str(payload["path"])
    crs = str(payload["crs"])
    dest = Path(payload["dest"])
    buildings = pd.DataFrame(payload["buildings"])
    if dest.exists() and dest.stat().st_size > 0:
        return remote, str(dest)

    bounds = _utm_bounds_for_buildings(buildings, crs)
    if bounds is None:
        return remote, gs_to_https(remote)

    west, south, east, north = bounds
    pad = 5.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per PID avoids races when two workers target the same dest.
    tmp = dest.parent / f".{dest.name}.{os.getpid()}.tmp"
    os.environ.update(GDAL_ENV)
    url = gs_to_https(remote)
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(url) as src:
                win = window_rc(src, west - pad, south - pad, east + pad, north + pad)
                if win is None:
                    return remote, url
                (r0, r1), (c0, c1) = win
                data = src.read(window=((r0, r1), (c0, c1)))
                transform = src.window_transform(((r0, r1), (c0, c1)))
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=int(r1 - r0),
                    width=int(c1 - c0),
                    transform=transform,
                    compress="NONE",
                    tiled=True,
                    BIGTIFF="IF_SAFER",
                    NUM_THREADS="ALL_CPUS",
                )
                with rasterio.open(tmp, "w", **profile) as dst:
                    dst.write(data)
        try:
            os.replace(tmp, dest)
        except FileNotFoundError:
            # Another worker finished first, or NFS rename race — accept existing dest.
            if not (dest.exists() and dest.stat().st_size > 0):
                raise
        print(
            f"  cog_cache wrote {dest.name} shape={tuple(data.shape)} "
            f"bytes={dest.stat().st_size/1e6:.1f}MB",
            flush=True,
        )
        return remote, str(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _extract_tile_shard_job(payload: dict) -> tuple[str, list[dict]]:
    """Worker: extract AEF cells for one tile×building-shard; optional shard checkpoint."""
    path = payload["path"]
    crs = payload["crs"]
    shard_id = payload["shard_id"]
    shard_path = Path(payload["shard_path"]) if payload.get("shard_path") else None
    if shard_path is not None and shard_path.exists():
        df = pd.read_parquet(shard_path)
        return shard_id, df.to_dict(orient="records")

    buildings = pd.DataFrame(payload["buildings"])
    os.environ.update(GDAL_ENV)
    open_path = payload.get("local_path") or gs_to_https(path)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(open_path) as src:
            _br, cr = extract_from_tile(src, crs, buildings)
    if shard_path is not None and cr:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cr).to_parquet(shard_path, index=False)
    return shard_id, cr


def _assign_tiles_chunk(payload: dict) -> pd.DataFrame:
    buildings = pd.DataFrame(payload["buildings"])
    index = pd.DataFrame(payload["index"])
    return assign_tiles(buildings, index)


def extract_aef_cells(
    buildings_df: pd.DataFrame,
    workers: int,
    *,
    shard_size: int = 1000,
    shard_dir: Path | None = None,
    cog_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return unique cells; shard each COG tile across workers (fixes 1-tile idle)."""
    b = buildings_df.copy()
    b["region"] = b["city_id"]
    keep = [
        "id", "region", "lon", "lat", "wkt", "subtype", "class", "label",
        "brief_category", "y_res", "height", "num_floors", "area_m2_approx",
        "source_primary",
    ]
    for c in keep:
        if c not in b.columns:
            b[c] = None
    b["label_source"] = "overture_buildings.class"
    b["split_seed"] = 42
    b = b[keep + ["label_source", "split_seed"]].copy()

    # Prefer tile index intersecting this city's buildings (not CONUS-only).
    # ~0.05° (~5 km) pad covers footprint AABB + tile seam edge cases.
    pad = 0.05
    lon_min = float(b["lon"].min()) - pad
    lon_max = float(b["lon"].max()) + pad
    lat_min = float(b["lat"].min()) - pad
    lat_max = float(b["lat"].max()) + pad
    # Heuristic: CONUS lon stays on US cache; anything else uses regional bbox.
    if lon_max < -50.0:
        index = ensure_index()
    else:
        index = ensure_index(bbox=(lon_min, lon_max, lat_min, lat_max))
    if index.empty:
        raise RuntimeError(
            f"AEF index empty for buildings bbox=({lon_min:.3f},{lon_max:.3f},"
            f"{lat_min:.3f},{lat_max:.3f}); cannot extract cells"
        )
    assign_chunks = max(1, min(workers, max(1, math.ceil(len(b) / max(shard_size, 1)))))
    assigned: pd.DataFrame
    if assign_chunks <= 1 or len(b) < shard_size:
        assigned = assign_tiles(b, index)
    else:
        b_parts = np.array_split(b, assign_chunks)
        index_records = index.to_dict(orient="list")
        assigned_parts: list[pd.DataFrame] = []
        with ProcessPoolExecutor(max_workers=min(workers, assign_chunks)) as pool:
            futs = [
                pool.submit(
                    _assign_tiles_chunk,
                    {"buildings": part.to_dict(orient="records"), "index": index_records},
                )
                for part in b_parts
                if len(part)
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="assign_tiles_parallel"):
                assigned_parts.append(fut.result())
        assigned = pd.concat(assigned_parts, ignore_index=True) if assigned_parts else pd.DataFrame()

    exploded = assigned.explode("tile_paths").dropna(subset=["tile_paths"])
    if len(b) > 0 and exploded.empty:
        raise RuntimeError(
            f"No AEF tiles intersect {len(b)} buildings "
            f"(bbox≈{lon_min:.3f},{lon_max:.3f},{lat_min:.3f},{lat_max:.3f}; "
            f"index_tiles={len(index)}). Refusing empty-cell success."
        )
    cache_dir = cog_cache_dir
    if cache_dir is None and shard_dir is not None:
        cache_dir = shard_dir.parent / "cog_windows"
    if cache_dir is None:
        cache_dir = Path(os.environ.get("AEF_COG_CACHE", str(ROOT / "data" / "aef_cog_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Materialize one local window GeoTIFF per remote COG (parallel across tiles)
    tile_groups: list[tuple[str, pd.DataFrame]] = []
    for path, g in exploded.groupby("tile_paths"):
        bd = g.drop(columns=["tile_paths"], errors="ignore").reset_index(drop=True)
        tile_groups.append((str(path), bd))

    local_by_remote: dict[str, str] = {}
    mat_payloads = []
    for path, bd in tile_groups:
        key = tile_cache_key(path)
        dest = cache_dir / f"{key}_window.tif"
        mat_payloads.append(
            {
                "path": path,
                "crs": str(bd["crs"].iloc[0]),
                "buildings": bd[["wkt"]].to_dict(orient="list")
                if "wkt" in bd.columns
                else {"wkt": []},
                "dest": str(dest),
            }
        )
    n_mat = max(1, min(workers, len(mat_payloads)))
    print(
        f"  cog_cache: materializing {len(mat_payloads)} tile window(s) -> {cache_dir} "
        f"(workers={n_mat})",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=n_mat) as pool:
        futs = [pool.submit(_materialize_cog_window, mp) for mp in mat_payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cog_cache"):
            remote, local = fut.result()
            local_by_remote[remote] = local

    jobs: list[dict] = []
    for path, bd in tile_groups:
        n = len(bd)
        n_shards = max(1, min(workers, math.ceil(n / max(int(shard_size), 1))))
        parts = np.array_split(bd, n_shards)
        tile_key = tile_cache_key(str(path))
        local_path = local_by_remote.get(str(path), gs_to_https(str(path)))
        for si, part in enumerate(parts):
            if len(part) == 0:
                continue
            shard_id = f"{tile_key}_{si:03d}"
            shard_path = str(shard_dir / f"{shard_id}.parquet") if shard_dir is not None else None
            jobs.append(
                {
                    "path": path,
                    "local_path": local_path,
                    "crs": str(bd["crs"].iloc[0]),
                    "buildings": part.to_dict(orient="records"),
                    "shard_id": shard_id,
                    "shard_path": shard_path,
                }
            )

    meta = {
        "n_buildings": int(len(b)),
        "n_tiles": int(len(tile_groups)),
        "n_shards": int(len(jobs)),
        "shard_size": int(shard_size),
        "workers": int(workers),
        "cog_cache_dir": str(cache_dir),
        "cog_windows": len(local_by_remote),
    }
    print(
        f"  aef plan: buildings={meta['n_buildings']} tiles={meta['n_tiles']} "
        f"shards={meta['n_shards']} workers={workers} shard_size={shard_size} "
        f"cog_cache={cache_dir}",
        flush=True,
    )

    cell_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_tile_shard_job, j): j["shard_id"] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="aef_shards"):
            _sid, cr = fut.result()
            cell_rows.extend(cr)

    if not cell_rows:
        return pd.DataFrame(), meta
    cells = pd.DataFrame(cell_rows).drop_duplicates("pixel_wkt", keep="first").reset_index(drop=True)
    meta["n_cells"] = int(len(cells))
    return cells, meta


def insert_cells_and_labels(
    conn,
    city_id: str,
    buildings_df: pd.DataFrame,
    id_map: dict[str, int],
    cells_df: pd.DataFrame,
    aef_source_id: int,
) -> dict:
    if cells_df.empty:
        return {"n_cells": 0, "n_labels": 0, "n_intersections": 0}

    # Determine UTM CRS from first cell
    area_crs = str(cells_df["crs"].mode().iloc[0]) if "crs" in cells_df.columns else "EPSG:4326"

    # Insert unique cells
    cell_id_by_wkt: dict[str, int] = {}
    band_cols = [f"a{i:02d}" for i in range(64)]
    with conn.cursor() as cur:
        for i in range(0, len(cells_df), 200):
            batch = cells_df.iloc[i : i + 200]
            for _, r in batch.iterrows():
                emb = {f"a{k:02d}": float(r[f"A{k:02d}"]) for k in range(64)}
                sha = hashlib.sha1(str(r["pixel_wkt"]).encode()).hexdigest()[:16]
                cur.execute(
                    f"""
                    INSERT INTO alphaearth_cells (
                      city_id, geom, year, source, source_version, source_id, cell_key,
                      {", ".join(band_cols)}, metadata
                    ) VALUES (
                      %(city_id)s, ST_SetSRID(ST_GeomFromText(%(wkt)s), 4326),
                      %(year)s, 'alphaearth', 'v1', %(source_id)s, %(sha)s,
                      {", ".join(f"%({c})s" for c in band_cols)},
                      jsonb_build_object('pixel_wkt_sha1', %(sha)s::text)
                    )
                    ON CONFLICT (city_id, year, cell_key) WHERE cell_key IS NOT NULL DO UPDATE SET
                      updated_at = now()
                    RETURNING cell_id
                    """,
                    {
                        "city_id": city_id,
                        "wkt": r["pixel_wkt"],
                        "year": EMBED_YEAR,
                        "source_id": aef_source_id,
                        "sha": sha,
                        **emb,
                    },
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        """
                        SELECT cell_id FROM alphaearth_cells
                        WHERE city_id = %s AND year = %s AND cell_key = %s
                        LIMIT 1
                        """,
                        (city_id, EMBED_YEAR, sha),
                    )
                    row = cur.fetchone()
                if row is None:
                    continue
                cid = int(row["cell_id"])
                cell_id_by_wkt[str(r["pixel_wkt"])] = cid
                cur.execute(
                    """
                    INSERT INTO cell_spatial_metrics (
                      cell_id, pixel_lon, pixel_lat, pixel_size_m, pixel_area_m2,
                      area_crs, utm_x, utm_y
                    ) VALUES (
                      %(cid)s, %(lon)s, %(lat)s, %(size)s, NULL,
                      %(crs)s, %(ux)s, %(uy)s
                    )
                    ON CONFLICT (cell_id) DO UPDATE SET
                      pixel_lon = EXCLUDED.pixel_lon,
                      pixel_lat = EXCLUDED.pixel_lat,
                      area_crs = EXCLUDED.area_crs,
                      updated_at = now()
                    """,
                    {
                        "cid": cid,
                        "lon": float(r["pixel_lon"]),
                        "lat": float(r["pixel_lat"]),
                        "size": float(r.get("pixel_size_m") or 10),
                        "crs": str(r.get("crs") or area_crs),
                        "ux": float(r["utm_x"]) if pd.notna(r.get("utm_x")) else None,
                        "uy": float(r["utm_y"]) if pd.notna(r.get("utm_y")) else None,
                    },
                )
            conn.commit()

    # Fraction GT in projected CRS (same as week4)
    b = buildings_df.copy()
    b["id"] = b["id"].astype(str)
    b_geoms = []
    for w in b["wkt"].tolist():
        g = shapely_wkt.loads(w)
        if not g.is_valid:
            g = g.buffer(0)
        b_geoms.append(g)
    bg = gpd.GeoDataFrame(
        {"id": b["id"].tolist(), "y_res": b["y_res"].astype(int).tolist()},
        geometry=b_geoms,
        crs="EPSG:4326",
    ).to_crs(area_crs)

    pix = cells_df.drop_duplicates("pixel_wkt").copy()
    p_geoms = [shapely_wkt.loads(w) for w in pix["pixel_wkt"].tolist()]
    pg = gpd.GeoDataFrame(pix, geometry=p_geoms, crs="EPSG:4326").to_crs(area_crs)
    sindex = bg.sindex

    n_labels = 0
    n_intersections = 0
    with conn.cursor() as cur:
        for _, prow in tqdm(pg.iterrows(), total=len(pg), desc=f"labels:{city_id}"):
            wkt = str(prow["pixel_wkt"])
            cid = cell_id_by_wkt.get(wkt)
            if cid is None:
                continue
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
            res_parts, nres_parts = [], []
            res_ids, nres_ids = [], []
            for _, h in hits.iterrows():
                if not pixel.intersects(h.geometry):
                    continue
                inter = pixel.intersection(h.geometry)
                ia = float(inter.area)
                if ia <= 0:
                    continue
                bid = id_map.get(str(h["id"]))
                if bid is not None:
                    cur.execute(
                        """
                        INSERT INTO cell_building_intersections (
                          cell_id, building_id, intersection_area, cell_fraction,
                          frac_res_contrib, frac_nres_contrib
                        ) VALUES (
                          %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (cell_id, building_id) DO UPDATE SET
                          intersection_area = EXCLUDED.intersection_area,
                          cell_fraction = EXCLUDED.cell_fraction
                        """,
                        (
                            cid,
                            bid,
                            ia,
                            ia / pixel_area,
                            (ia / pixel_area) if int(h["y_res"]) == 1 else 0.0,
                            (ia / pixel_area) if int(h["y_res"]) == 0 else 0.0,
                        ),
                    )
                    n_intersections += 1
                if int(h["y_res"]) == 1:
                    res_parts.append(h.geometry)
                    res_ids.append(str(h["id"]))
                else:
                    nres_parts.append(h.geometry)
                    nres_ids.append(str(h["id"]))
            if not res_parts and not nres_parts:
                continue
            area_res = float(pixel.intersection(unary_union(res_parts)).area) if res_parts else 0.0
            area_nres = float(pixel.intersection(unary_union(nres_parts)).area) if nres_parts else 0.0
            frac_res_full = float(np.clip(area_res / pixel_area, 0.0, 1.0))
            frac_nres_full = float(np.clip(area_nres / pixel_area, 0.0, 1.0))
            frac_empty_full = float(np.clip(1.0 - frac_res_full - frac_nres_full, 0.0, 1.0))
            frac_res, frac_nres, frac_empty = constrained_label_fracs(frac_res_full, frac_nres_full)

            pure_res = frac_res_full > 0 and frac_nres_full == 0
            pure_nres = frac_nres_full > 0 and frac_res_full == 0
            if pure_res:
                binary_label, purity = 1, "pure_residential"
            elif pure_nres:
                binary_label, purity = 0, "pure_non_residential"
            else:
                binary_label, purity = None, "mixed_or_empty"

            cur.execute(
                """
                INSERT INTO cell_labels (
                  cell_id, label_version, frac_res, frac_nres, frac_empty,
                  binary_label, purity_status, ambiguity_status
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (cell_id, label_version) DO UPDATE SET
                  frac_res = EXCLUDED.frac_res,
                  frac_nres = EXCLUDED.frac_nres,
                  frac_empty = EXCLUDED.frac_empty,
                  binary_label = EXCLUDED.binary_label,
                  purity_status = EXCLUDED.purity_status
                """,
                (
                    cid,
                    LABEL_VERSION,
                    frac_res,
                    frac_nres,
                    frac_empty,
                    binary_label,
                    purity,
                    "ambiguous" if purity == "mixed_or_empty" else "clear",
                ),
            )
            cur.execute(
                """
                INSERT INTO cell_label_metrics (
                  cell_id, label_version, area_res_m2, area_nres_m2,
                  frac_res_full, frac_nres_full, frac_empty_full,
                  n_res_buildings, n_nres_buildings, n_buildings, area_crs
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (cell_id, label_version) DO UPDATE SET
                  area_res_m2 = EXCLUDED.area_res_m2,
                  area_nres_m2 = EXCLUDED.area_nres_m2,
                  frac_res_full = EXCLUDED.frac_res_full,
                  frac_nres_full = EXCLUDED.frac_nres_full,
                  frac_empty_full = EXCLUDED.frac_empty_full,
                  n_res_buildings = EXCLUDED.n_res_buildings,
                  n_nres_buildings = EXCLUDED.n_nres_buildings,
                  n_buildings = EXCLUDED.n_buildings
                """,
                (
                    cid,
                    LABEL_VERSION,
                    area_res,
                    area_nres,
                    frac_res_full,
                    frac_nres_full,
                    frac_empty_full,
                    len(res_ids),
                    len(nres_ids),
                    len(set(res_ids + nres_ids)),
                    area_crs,
                ),
            )
            cur.execute(
                """
                UPDATE cell_spatial_metrics
                SET pixel_area_m2 = %s, area_crs = %s, updated_at = now()
                WHERE cell_id = %s
                """,
                (pixel_area, area_crs, cid),
            )
            n_labels += 1
            if n_labels % 2000 == 0:
                conn.commit()
        conn.commit()

    return {
        "n_cells": len(cell_id_by_wkt),
        "n_labels": n_labels,
        "n_intersections": n_intersections,
        "area_crs": area_crs,
    }


def city_summary(conn, city_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM buildings WHERE city_id=%s", (city_id,))
        nb = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM alphaearth_cells WHERE city_id=%s", (city_id,))
        nc = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT COUNT(*) AS n FROM cell_labels cl
            JOIN alphaearth_cells c ON c.cell_id = cl.cell_id
            WHERE c.city_id=%s AND cl.label_version=%s
            """,
            (city_id, LABEL_VERSION),
        )
        nl = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE cl.purity_status='pure_residential') AS pure_res,
              COUNT(*) FILTER (WHERE cl.purity_status='pure_non_residential') AS pure_nres
            FROM cell_labels cl
            JOIN alphaearth_cells c ON c.cell_id = cl.cell_id
            WHERE c.city_id=%s AND cl.label_version=%s
            """,
            (city_id, LABEL_VERSION),
        )
        pure = cur.fetchone()
        cur.execute(
            "SELECT aoi_radius_m, aoi_building_count, aoi_method FROM cities WHERE city_id=%s",
            (city_id,),
        )
        aoi = cur.fetchone()
    return {
        "city_id": city_id,
        "buildings": nb,
        "cells": nc,
        "labels": nl,
        "pure_res": pure["pure_res"],
        "pure_nres": pure["pure_nres"],
        "aoi": dict(aoi) if aoi else None,
    }


def discover_city(
    city: dict,
    run_dir: Path,
    discovery_radius_m: float,
    overture_source_id: int,
    *,
    ring_km: float = 5.0,
    area_headroom: float = 1.10,
    target_area_m2: float | None = WEEK4_EQUAL_CITY_AREA_M2,
    sample_seed: int = 42,
) -> str:
    """Overture pull + buildings COPY + stratified AEF sample. Returns city_id."""
    city_id = city["city_id"]
    lon = float(city["longitude"])
    lat = float(city["latitude"])
    staging = run_dir / "artifacts" / city_id
    staging.mkdir(parents=True, exist_ok=True)

    if not stage_done(run_dir, city_id, "buildings"):
        buildings_path = staging / "buildings.parquet"
        if buildings_path.exists():
            df = pd.read_parquet(buildings_path)
            print(f"[{city_id}] resume buildings parquet n={len(df)}", flush=True)
        else:
            print(f"[{city_id}] pulling Overture discovery_r={discovery_radius_m}m", flush=True)
            df = pull_overture(city_id, lon, lat, discovery_radius_m)
            df.to_parquet(buildings_path, index=False)
            print(f"[{city_id}] buildings={len(df)}", flush=True)
        wait_until_ready()
        id_map = insert_buildings(df, overture_source_id)
        atomic_json(staging / "id_map.json", id_map)
        with connect() as conn:
            aoi = compute_aoi(conn, city_id)
        mark_stage(run_dir, city_id, "buildings", {"n": len(df), "aoi": aoi})
    else:
        df = pd.read_parquet(staging / "buildings.parquet")
        print(f"[{city_id}] resume buildings n={len(df)}", flush=True)

    sample_path = staging / "buildings_aef_sample.parquet"
    if not stage_done(run_dir, city_id, "aef_sample"):
        sample, manifest = stratified_area_balance(
            df,
            lon,
            lat,
            ring_km=ring_km,
            max_radius_m=discovery_radius_m,
            headroom=area_headroom,
            target_area_m2=target_area_m2,
            seed=sample_seed,
        )
        sample.to_parquet(sample_path, index=False)
        atomic_json(staging / "aef_sample_manifest.json", manifest)
        print(
            f"[{city_id}] aef_sample n={manifest['n_out']}/{manifest['n_in']} "
            f"keep={manifest['keep_frac']:.3f} area_km2={manifest['area_out_km2']:.2f} "
            f"target_km2={(manifest['target_area_km2'] or 0):.2f} "
            f"rings={len(manifest['rings'])} headroom={area_headroom}",
            flush=True,
        )
        mark_stage(run_dir, city_id, "aef_sample", manifest)
    else:
        print(
            f"[{city_id}] resume aef_sample n={len(pd.read_parquet(sample_path))}",
            flush=True,
        )
    return city_id


def embed_and_label_city(
    city: dict,
    run_dir: Path,
    workers: int,
    aef_source_id: int,
    *,
    aef_shard_size: int = 1000,
) -> dict:
    """AEF extract + labels/COPY. Uses full footprints for GT intersections."""
    city_id = city["city_id"]
    staging = run_dir / "artifacts" / city_id
    df = pd.read_parquet(staging / "buildings.parquet")
    id_map = {k: int(v) for k, v in json.loads((staging / "id_map.json").read_text()).items()}
    sample = pd.read_parquet(staging / "buildings_aef_sample.parquet")

    cells_path = staging / "cells.parquet"
    shard_dir = staging / "aef_shards"
    if not stage_done(run_dir, city_id, "aef"):
        print(
            f"[{city_id}] extracting AlphaEarth cells workers={workers} shard_size={aef_shard_size}",
            flush=True,
        )
        cells, aef_meta = extract_aef_cells(
            sample,
            workers=workers,
            shard_size=aef_shard_size,
            shard_dir=shard_dir,
            cog_cache_dir=staging / "cog_windows",
        )
        cells.to_parquet(cells_path, index=False)
        print(f"[{city_id}] unique_cells={len(cells)} shards={aef_meta.get('n_shards')}", flush=True)
        mark_stage(run_dir, city_id, "aef", aef_meta)
        # cells.parquet is the durable AEF checkpoint — free multi-GB temps now
        prune_city_aef_temps(staging, city_id=city_id)
    else:
        cells = pd.read_parquet(cells_path)
        print(f"[{city_id}] resume cells n={len(cells)}", flush=True)
        # Leftovers from pre-prune runs / interrupted cleanup
        prune_city_aef_temps(staging, city_id=city_id)

    if not stage_done(run_dir, city_id, "labels"):
        print(
            f"[{city_id}] batched cells + parallel labels/intersections workers={workers}",
            flush=True,
        )
        wait_until_ready()
        stats = insert_cells_and_labels_parallel(
            city_id,
            df,
            id_map,
            cells,
            aef_source_id,
            workers=workers,
        )
        with connect() as conn:
            summary = city_summary(conn, city_id)
        mark_stage(run_dir, city_id, "labels", {"stats": stats, "summary": summary})
    else:
        with connect() as conn:
            summary = city_summary(conn, city_id)

    mark_stage(run_dir, city_id, "CITY", summary)
    prune_city_aef_temps(staging, city_id=city_id)
    return summary


def process_city(
    city: dict,
    run_dir: Path,
    discovery_radius_m: float,
    workers: int,
    overture_source_id: int,
    aef_source_id: int,
    *,
    ring_km: float = 5.0,
    area_headroom: float = 1.10,
    target_area_m2: float = WEEK4_EQUAL_CITY_AREA_M2,
    aef_shard_size: int = 1000,
    sample_seed: int = 42,
) -> dict:
    """Full city path (discover → embed/label). Prefer pipelined main() for speed."""
    discover_city(
        city,
        run_dir,
        discovery_radius_m,
        overture_source_id,
        ring_km=ring_km,
        area_headroom=area_headroom,
        target_area_m2=target_area_m2,
        sample_seed=sample_seed,
    )
    return embed_and_label_city(
        city,
        run_dir,
        workers,
        aef_source_id,
        aef_shard_size=aef_shard_size,
    )


def main() -> int:
    _ensure_mp_spawn()
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery-radius-m", type=float, default=30_000.0)
    ap.add_argument("--ring-km", type=float, default=5.0)
    ap.add_argument("--area-headroom", type=float, default=1.10)
    ap.add_argument(
        "--target-area-m2",
        type=float,
        default=WEEK4_EQUAL_CITY_AREA_M2,
        help=(
            "Per-city footprint-area budget for AEF sampling (m²). "
            f"Default={WEEK4_EQUAL_CITY_AREA_M2:.0f} "
            "(week4 total Track-A footprint / 10 cities ≈ 24.60 km²). "
            "Use 0 to disable the city-wide cap (ring balance only)."
        ),
    )
    ap.add_argument("--aef-shard-size", type=int, default=1000)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 8) // 2))
    ap.add_argument(
        "--parallel-discovery",
        type=int,
        default=8,
        help="Max concurrent Overture/building pulls",
    )
    ap.add_argument(
        "--parallel-aef",
        type=int,
        default=2,
        help="Max concurrent AEF+label cities; workers split as workers // N",
    )
    ap.add_argument(
        "--parallel-cities",
        type=int,
        default=None,
        help="Deprecated alias: sets both --parallel-discovery and --parallel-aef",
    )
    ap.add_argument("--city-id", action="append", default=None, help="Optional subset of city_ids")
    ap.add_argument(
        "--country-id",
        default="USA",
        help="ISO3 country to ingest (selected cities in Postgres)",
    )
    args = ap.parse_args()
    target_area_m2 = None if float(args.target_area_m2) <= 0 else float(args.target_area_m2)

    if args.parallel_cities is not None:
        args.parallel_discovery = max(args.parallel_discovery, args.parallel_cities)
        args.parallel_aef = args.parallel_cities

    run_dir = Path(os.environ["RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(exist_ok=True)

    os.environ["DISCOVERY_RADIUS_M"] = str(args.discovery_radius_m)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    load_dotenv()
    wait_until_ready(timeout_s=60)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT city_id, city_name, latitude, longitude,
                       metadata->>'week4_region_id' AS week4_region_id
                FROM cities
                WHERE country_id=%s AND inclusion_status='selected'
                ORDER BY city_id
                """,
                (args.country_id,),
            )
            cities = list(cur.fetchall())
        overture_source_id, aef_source_id = ensure_data_sources(conn)

    if args.city_id:
        want = set(args.city_id)
        cities = [c for c in cities if c["city_id"] in want]
    if not cities:
        raise SystemExit(f"No selected cities for country_id={args.country_id}")

    order = {
        "USA_SAN_FRANCISCO": 0,
        "USA_SEATTLE": 1,
        "USA_NEW_YORK": 2,
        "USA_BOSTON": 3,
        "USA_CHICAGO": 4,
        "USA_HOUSTON": 5,
        "USA_DENVER": 6,
        "USA_PHOENIX": 7,
        "USA_ATLANTA": 8,
        "USA_LOS_ANGELES": 9,
        "IND_DELHI": 0,
        "IND_MUMBAI": 1,
        "IND_BENGALURU": 2,
        "IND_KOLKATA": 3,
        "IND_CHENNAI": 4,
        "JPN_TOKYO": 0,
        "JPN_OSAKA": 1,
        "JPN_NAGOYA": 2,
        "JPN_FUKUOKA": 3,
        "JPN_SAPPORO": 4,
        "JPN_YOKOHAMA": 5,
        "JPN_KOBE": 6,
        "JPN_KYOTO": 7,
        "JPN_HIROSHIMA": 8,
        "JPN_SENDAI": 9,
    }
    cities = sorted(cities, key=lambda c: order.get(c["city_id"], 99))
    pending = [c for c in cities if not stage_done(run_dir, c["city_id"], "CITY")]
    by_id = {c["city_id"]: c for c in cities}

    n_disc = max(1, int(args.parallel_discovery))
    n_aef = max(1, int(args.parallel_aef))
    workers_each = max(4, int(args.workers) // n_aef)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    already = len(cities) - len(pending)
    update_progress(
        run_dir,
        status="running",
        stage="pipeline",
        started_at=started,
        completed=already,
        failed=0,
        total=len(cities),
        percent_complete=round(100.0 * already / max(len(cities), 1), 2),
        country_id=args.country_id,
        resource_allocation={
            "country_id": args.country_id,
            "cpus": int(os.environ.get("CPUS", os.cpu_count() or 1)),
            "gpus": os.environ.get("GPUS", "all"),
            "workers": args.workers,
            "parallel_discovery": n_disc,
            "parallel_aef": n_aef,
            "workers_each_aef": workers_each,
            "discovery_radius_m": args.discovery_radius_m,
            "ring_km": args.ring_km,
            "area_headroom": args.area_headroom,
            "target_area_m2": target_area_m2,
            "target_area_km2": None if target_area_m2 is None else target_area_m2 / 1e6,
            "aef_shard_size": args.aef_shard_size,
        },
        output=str(run_dir),
    )
    print(
        f"pipeline country={args.country_id} pending={len(pending)} "
        f"discovery_parallel={n_disc} aef_parallel={n_aef} workers_each_aef={workers_each} "
        f"target_area_km2={None if target_area_m2 is None else target_area_m2/1e6:.2f}",
        flush=True,
    )

    summaries: list[dict] = []
    for c in cities:
        cid = c["city_id"]
        if stage_done(run_dir, cid, "CITY"):
            jp = run_dir / "checkpoints" / cid / "CITY.json"
            if jp.exists():
                summaries.append(json.loads(jp.read_text()))
        # Free leftovers from earlier runs once AEF cells are durable
        if stage_done(run_dir, cid, "aef"):
            prune_city_aef_temps(run_dir / "artifacts" / cid, city_id=cid)

    failed: list[dict] = []
    completed = already

    # Discovery = threads (I/O, no ProcessPools).
    # AEF = processes (spawn): nested tile/shard/label ProcessPools are safe there.
    # Never run embed_and_label_city on a ThreadPool — fork-from-thread deadlocks on Linux.
    disc_exec = ThreadPoolExecutor(max_workers=n_disc)
    aef_exec = ProcessPoolExecutor(max_workers=n_aef)
    disc_futs = {}
    aef_futs = {}

    def _submit_discover(city: dict):
        fut = disc_exec.submit(
            discover_city,
            city,
            run_dir,
            args.discovery_radius_m,
            overture_source_id,
            ring_km=args.ring_km,
            area_headroom=args.area_headroom,
            target_area_m2=target_area_m2,
            sample_seed=args.sample_seed,
        )
        disc_futs[fut] = city["city_id"]

    def _submit_aef(city_id: str):
        fut = aef_exec.submit(
            _embed_city_job,
            {
                "city": dict(by_id[city_id]),
                "run_dir": str(run_dir),
                "workers": workers_each,
                "aef_source_id": aef_source_id,
                "aef_shard_size": args.aef_shard_size,
            },
        )
        aef_futs[fut] = city_id
        print(
            f"[pipeline] AEF process queue <- {city_id} (queued_aef={len(aef_futs)} "
            f"workers_each={workers_each} mp={mp.get_start_method()})",
            flush=True,
        )

    for city in pending:
        # Already past sample → go straight to AEF queue
        if stage_done(run_dir, city["city_id"], "aef_sample"):
            _submit_aef(city["city_id"])
        else:
            _submit_discover(city)

    update_progress(
        run_dir,
        stage="pipeline",
        discovering=list(disc_futs.values()),
        embedding=list(aef_futs.values()),
    )

    while disc_futs or aef_futs:
        done_set, _ = wait(
            list(disc_futs.keys()) + list(aef_futs.keys()),
            return_when=FIRST_COMPLETED,
        )
        for fut in done_set:
            if fut in disc_futs:
                city_id = disc_futs.pop(fut)
                try:
                    fut.result()
                    print(f"[pipeline] discovery done {city_id}", flush=True)
                    _submit_aef(city_id)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"city_id": city_id, "error": repr(exc)})
                    (run_dir / "failures" / f"{city_id}.txt").write_text(repr(exc) + "\n")
                    print(f"FAILED discover {city_id}: {exc}", flush=True)
            elif fut in aef_futs:
                city_id = aef_futs.pop(fut)
                try:
                    summary = fut.result()
                    summaries.append(summary)
                    atomic_json(run_dir / "artifacts" / f"{city_id}_summary.json", summary)
                    completed += 1
                    print(json.dumps(summary), flush=True)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"city_id": city_id, "error": repr(exc)})
                    (run_dir / "failures" / f"{city_id}.txt").write_text(repr(exc) + "\n")
                    print(f"FAILED aef/labels {city_id}: {exc}", flush=True)
            update_progress(
                run_dir,
                completed=completed,
                failed=len(failed),
                percent_complete=round(100.0 * completed / max(len(cities), 1), 2),
                discovering=list(disc_futs.values()),
                embedding=list(aef_futs.values()),
            )

    disc_exec.shutdown(wait=True)
    aef_exec.shutdown(wait=True)

    final = {
        "status": "completed" if not failed else "completed_with_failures",
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "discovery_radius_m": args.discovery_radius_m,
        "ring_km": args.ring_km,
        "area_headroom": args.area_headroom,
        "country_id": args.country_id,
        "target_area_m2": target_area_m2,
        "aef_shard_size": args.aef_shard_size,
        "parallel_discovery": n_disc,
        "parallel_aef": n_aef,
        "label_version": LABEL_VERSION,
        "cities_ok_n": len(summaries),
        "cities_failed": failed,
    }
    atomic_json(run_dir / "summary.json", final)
    update_progress(
        run_dir,
        status=final["status"],
        stage="done",
        completed=completed,
        failed=len(failed),
        percent_complete=100.0,
    )
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(final, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    # Fix worker: extract_from_tile needs DataFrame not list of dicts in one code path
    raise SystemExit(main())
