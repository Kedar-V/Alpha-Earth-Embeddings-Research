#!/usr/bin/env python3
"""Embed AlphaEarth cells for suburban and rural buildings that have none yet.

Selects buildings already in `buildings` with no `cell_building_intersections`
row whose centroid falls in GHS-SMOD peri-urban (21) or rural (11, 12, 13).
Urban leftovers are left alone.

Per city:
  1. select targets and write a parquet checkpoint
  2. extract 2024 AlphaEarth 10 m cells (shard checkpoints, resumable)
  3. upsert cells, then label them from every building that intersects those
     cells so a shared cell is not relabeled from the new buildings only
  4. sample GHS-SMOD onto cells that still have no degree-of-urbanisation row

Cities run one at a time. AEF shards use the CPU pool; Postgres writers stay
low because WAL fsync on this ZFS pool stalls under many concurrent inserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from attach_degree_of_urbanisation import (  # noqa: E402
    DEFAULT_RASTER,
    _sample_city,
    ensure_source,
)
from ingest_usa_to_postgres import (  # noqa: E402
    EMBED_YEAR,
    _ensure_mp_spawn,
    ensure_data_sources,
    extract_aef_cells,
    prune_city_aef_temps,
)
from lib.db import connect, wait_until_ready  # noqa: E402
from pg_batch import copy_upsert_cells, insert_cells_and_labels_parallel  # noqa: E402

# Centroid class. Suburban in the census is smod 21 (peri_urban).
KEEP_DOU = {
    21: "peri_urban",
    13: "rural",
    12: "rural",
    11: "rural",
}

_state_lock = threading.Lock()
_state: dict = {}
_stop_heartbeat = threading.Event()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _resources() -> dict:
    usage = shutil.disk_usage(REPO)
    rss_mb = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_mb = round(int(line.split()[1]) / 1024.0, 1)
                break
    except OSError:
        pass
    load = os.getloadavg()
    return {
        "loadavg_1m": round(load[0], 2),
        "disk_free_gb": round(usage.free / 1e9, 1),
        "rss_mb": rss_mb,
    }


def set_progress(run_dir: Path, **kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        _state["updated_at"] = now
        _state["last_progress_at"] = now
        _state["resources"] = _resources()
        atomic_json(run_dir / "progress.json", dict(_state))


def _heartbeat(run_dir: Path) -> None:
    while not _stop_heartbeat.wait(20):
        with _state_lock:
            if not _state:
                continue
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            _state["updated_at"] = now
            _state["last_progress_at"] = now
            _state["resources"] = _resources()
            atomic_json(run_dir / "progress.json", dict(_state))


def stage_done(run_dir: Path, city_id: str, stage: str) -> bool:
    return (run_dir / "checkpoints" / city_id / f"{stage}.DONE").exists()


def mark_stage(run_dir: Path, city_id: str, stage: str, payload: dict | None = None) -> None:
    d = run_dir / "checkpoints" / city_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}.DONE").write_text("ok\n")
    if payload is not None:
        atomic_json(d / f"{stage}.json", payload)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def sample_smod_codes(raster_path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Containing 1 km GHS-SMOD cell. No interpolation."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    if len(lon) == 0:
        return np.array([], dtype=np.int16)
    with rasterio.open(raster_path) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pad = 2000.0
        window = from_bounds(
            float(x.min()) - pad,
            float(y.min()) - pad,
            float(x.max()) + pad,
            float(y.max()) + pad,
            transform=ds.transform,
        ).round_offsets().round_lengths()
        nodata = -200 if ds.nodata is None else int(ds.nodata)
        arr = ds.read(1, window=window, boundless=True, fill_value=nodata)
        win_transform = ds.window_transform(window)
    cols = np.floor((x - win_transform.c) / win_transform.a).astype(np.int64)
    rows = np.floor((y - win_transform.f) / win_transform.e).astype(np.int64)
    h, w = arr.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    codes = np.full(len(lon), nodata, dtype=np.int16)
    codes[inside] = arr[rows[inside], cols[inside]]
    return codes


def unembedded_centroids(city_id: str) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT b.building_id,
                   b.source_feature_id AS id,
                   ST_X(ST_Centroid(b.geom)) AS lon,
                   ST_Y(ST_Centroid(b.geom)) AS lat
            FROM buildings b
            WHERE b.city_id = %s
              AND b.source_feature_id IS NOT NULL
              AND NOT EXISTS (
                    SELECT 1
                    FROM cell_building_intersections i
                    WHERE i.building_id = b.building_id
              )
            """,
            (city_id,),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["building_id", "id", "lon", "lat"])
    return pd.DataFrame(rows)


def load_building_attrs(building_ids: list[int]) -> pd.DataFrame:
    if not building_ids:
        return pd.DataFrame()
    frames = []
    chunk = 4_000
    with connect() as conn:
        for start in range(0, len(building_ids), chunk):
            part = building_ids[start : start + chunk]
            rows = conn.execute(
                """
                SELECT b.building_id,
                       b.source_feature_id AS id,
                       b.city_id,
                       ST_X(ST_Centroid(b.geom)) AS lon,
                       ST_Y(ST_Centroid(b.geom)) AS lat,
                       ST_AsText(b.geom) AS wkt,
                       a.overture_class AS class,
                       a.overture_subtype AS subtype,
                       a.study_label AS label,
                       a.brief_category,
                       a.y_res,
                       a.height_m AS height,
                       a.num_floors,
                       a.area_m2_approx,
                       a.source_primary,
                       b.harmonized_label
                FROM buildings b
                JOIN building_label_audit a ON a.building_id = b.building_id
                WHERE b.building_id = ANY(%s)
                """,
                (part,),
            ).fetchall()
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def select_targets(city_id: str, raster_path: Path, staging: Path) -> pd.DataFrame:
    centroids = unembedded_centroids(city_id)
    n_unembedded = int(len(centroids))
    if centroids.empty:
        out = pd.DataFrame()
        out.to_parquet(staging / "targets.parquet", index=False)
        return out
    codes = sample_smod_codes(
        raster_path,
        centroids["lon"].to_numpy(dtype=np.float64),
        centroids["lat"].to_numpy(dtype=np.float64),
    )
    centroids = centroids.copy()
    centroids["smod_code"] = codes
    centroids["dou"] = [KEEP_DOU.get(int(c)) for c in codes]
    keep = centroids[centroids["dou"].notna()].copy()
    print(
        f"[{city_id}] unembedded={n_unembedded} peri_rural={len(keep)} "
        f"(peri={(keep['dou']=='peri_urban').sum()} rural={(keep['dou']=='rural').sum()})",
        flush=True,
    )
    if keep.empty:
        keep.to_parquet(staging / "targets.parquet", index=False)
        return keep
    attrs = load_building_attrs(keep["building_id"].astype(int).tolist())
    if attrs.empty:
        keep.iloc[0:0].to_parquet(staging / "targets.parquet", index=False)
        return attrs
    dou_map = dict(zip(keep["building_id"].tolist(), keep["dou"].tolist()))
    code_map = dict(zip(keep["building_id"].tolist(), keep["smod_code"].tolist()))
    attrs["dou"] = attrs["building_id"].map(dou_map)
    attrs["smod_code"] = attrs["building_id"].map(code_map)
    before = len(attrs)
    attrs = attrs[attrs["y_res"].notna() & attrs["wkt"].notna() & attrs["id"].notna()].copy()
    attrs["y_res"] = attrs["y_res"].astype(int)
    attrs["id"] = attrs["id"].astype(str)
    dropped = before - len(attrs)
    if dropped:
        print(f"[{city_id}] dropped {dropped} targets with null y_res/wkt/id", flush=True)
    staging.mkdir(parents=True, exist_ok=True)
    attrs.to_parquet(staging / "targets.parquet", index=False)
    return attrs


def cell_keys(cells: pd.DataFrame) -> list[str]:
    return [
        hashlib.sha1(str(w).encode()).hexdigest()[:16]
        for w in cells["pixel_wkt"].astype(str).tolist()
    ]


def intersecting_building_ids(city_id: str, keys: list[str]) -> list[int]:
    """Buildings whose footprint meets a newly written cell.

    Labels must see every building on a cell. Passing only the new suburban
    and rural buildings would overwrite a shared cell's residential fraction.
    """
    if not keys:
        return []
    with connect() as conn:
        conn.execute("SET work_mem TO '256MB'")
        conn.execute("SET statement_timeout TO 0")
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE stage_keys (cell_key TEXT PRIMARY KEY) ON COMMIT DROP"
            )
            with cur.copy("COPY stage_keys (cell_key) FROM STDIN") as cp:
                for key in keys:
                    cp.write_row((key,))
            cur.execute(
                """
                SELECT DISTINCT b.building_id
                FROM stage_keys k
                JOIN alphaearth_cells c
                  ON c.city_id = %s
                 AND c.year = %s
                 AND c.cell_key = k.cell_key
                JOIN buildings b
                  ON b.city_id = c.city_id
                 AND b.geom && c.geom
                 AND ST_Intersects(b.geom, c.geom)
                """,
                (city_id, EMBED_YEAR),
            )
            return [int(r["building_id"]) for r in cur.fetchall()]


def label_frame(city_id: str, targets: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    keys = cell_keys(cells)
    extra_ids = intersecting_building_ids(city_id, keys)
    target_ids = targets["building_id"].astype(int).tolist()
    all_ids = list(dict.fromkeys(target_ids + extra_ids))
    print(
        f"[{city_id}] label buildings: targets={len(target_ids)} "
        f"intersecting={len(extra_ids)} union={len(all_ids)}",
        flush=True,
    )
    frame = load_building_attrs(all_ids)
    if frame.empty:
        return targets[["id", "wkt", "y_res", "building_id"]].copy()
    frame = frame[frame["y_res"].notna() & frame["wkt"].notna() & frame["id"].notna()].copy()
    frame["y_res"] = frame["y_res"].astype(int)
    frame["id"] = frame["id"].astype(str)
    return frame


def id_map_for(frame: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for src, bid in zip(frame["id"].astype(str).tolist(), frame["building_id"].tolist()):
        out[src] = int(bid)
    return out


def _shard_size_for(n_buildings: int, workers: int, shard_size: int) -> int:
    """Split a city across its worker budget.

    The extractor only starts one process per shard, and a shard is at least
    `shard_size` buildings. A 150-building city with shard_size 1000 is one
    shard, so a 76-worker cap never starts. Shrink the shard until the city
    fills its workers, and leave the cap alone once a city is already large
    enough to fill them.
    """
    if n_buildings <= 0 or workers <= 1:
        return shard_size
    spread = max(1, math.ceil(n_buildings / workers))
    return max(32, min(shard_size, spread))


def extract_city_aef(
    city_id: str,
    raster_path: Path,
    run_dir: Path,
    aef_workers: int,
    shard_size: int,
) -> dict:
    """Download and read AlphaEarth cells. No Postgres. Safe to run in parallel."""
    staging = run_dir / "artifacts" / city_id
    staging.mkdir(parents=True, exist_ok=True)
    targets_path = staging / "targets.parquet"
    if not (stage_done(run_dir, city_id, "SELECT") and targets_path.exists()):
        raise RuntimeError(f"{city_id}: selection checkpoint missing")
    targets = pd.read_parquet(targets_path)
    n_targets = int(len(targets))
    if n_targets == 0 or (stage_done(run_dir, city_id, "AEF") and (staging / "cells.parquet").exists()):
        return {"city_id": city_id, "n_targets": n_targets, "n_cells": None, "extracted": False}

    use_shard = _shard_size_for(n_targets, aef_workers, shard_size)
    print(
        f"[{city_id}] extract start buildings={n_targets} workers={aef_workers} shard_size={use_shard}",
        flush=True,
    )
    cells, meta = extract_aef_cells(
        targets,
        aef_workers,
        shard_size=use_shard,
        shard_dir=staging / "aef_shards",
        cog_cache_dir=staging / "cog_windows",
    )
    if cells.empty:
        raise RuntimeError(f"{city_id}: AEF extract returned 0 cells for {n_targets} buildings")
    cells_path = staging / "cells.parquet"
    cells.to_parquet(cells_path, index=False)
    mark_stage(run_dir, city_id, "AEF", {"n_cells": int(len(cells)), "shard_size": use_shard, **meta})
    pruned = prune_city_aef_temps(staging, city_id=city_id)
    print(f"[{city_id}] cells={len(cells)} pruned={pruned}", flush=True)
    return {
        "city_id": city_id,
        "n_targets": n_targets,
        "n_cells": int(len(cells)),
        "extracted": True,
    }


def _extract_job(payload: dict) -> dict:
    """Process-pool entry. Must stay at module level so spawn can pickle it."""
    _ensure_mp_spawn()
    try:
        return extract_city_aef(
            payload["city_id"],
            Path(payload["raster"]),
            Path(payload["run_dir"]),
            int(payload["workers"]),
            int(payload["shard_size"]),
        )
    except Exception as exc:
        return {
            "city_id": payload["city_id"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def process_city(
    city_id: str,
    raster_path: Path,
    run_dir: Path,
    aef_source_id: int,
    smod_source_id: int,
    aef_workers: int,
    db_writers: int,
    shard_size: int,
) -> dict:
    t0 = time.time()
    staging = run_dir / "artifacts" / city_id
    staging.mkdir(parents=True, exist_ok=True)
    targets_path = staging / "targets.parquet"

    if stage_done(run_dir, city_id, "SELECT") and targets_path.exists():
        targets = pd.read_parquet(targets_path)
    else:
        targets = select_targets(city_id, raster_path, staging)
        mark_stage(
            run_dir,
            city_id,
            "SELECT",
            {
                "n_targets": int(len(targets)),
                "n_peri": int((targets["dou"] == "peri_urban").sum()) if len(targets) else 0,
                "n_rural": int((targets["dou"] == "rural").sum()) if len(targets) else 0,
            },
        )

    n_peri = int((targets["dou"] == "peri_urban").sum()) if len(targets) and "dou" in targets.columns else 0
    n_rural = int((targets["dou"] == "rural").sum()) if len(targets) and "dou" in targets.columns else 0
    summary = {
        "city_id": city_id,
        "n_targets": int(len(targets)),
        "n_peri": n_peri,
        "n_rural": n_rural,
        "n_cells": 0,
        "n_labels": 0,
        "n_intersections": 0,
    }
    if targets.empty:
        mark_stage(run_dir, city_id, "AEF", {"n_cells": 0})
        mark_stage(run_dir, city_id, "LABELS", {"n_labels": 0})
        mark_stage(run_dir, city_id, "DOU", {"n": 0})
        mark_stage(run_dir, city_id, "CITY", summary)
        summary["seconds"] = round(time.time() - t0, 1)
        return summary

    cells_path = staging / "cells.parquet"
    if not (stage_done(run_dir, city_id, "AEF") and cells_path.exists()):
        set_progress(run_dir, stage="extract_aef", current_city=city_id, current_city_buildings=int(len(targets)))
        extract_city_aef(city_id, raster_path, run_dir, aef_workers, shard_size)
    cells = pd.read_parquet(cells_path)

    summary["n_cells"] = int(len(cells))
    if not stage_done(run_dir, city_id, "LABELS"):
        set_progress(run_dir, stage="insert_cells_labels", current_city=city_id, current_city_cells=int(len(cells)))
        # Cells have to be in Postgres before we can find every building that
        # touches them. The label pass then upserts the same cells again and
        # writes fractions from that full building set.
        area_crs = str(cells["crs"].mode().iloc[0]) if "crs" in cells.columns and len(cells) else "EPSG:4326"
        copy_upsert_cells(
            city_id,
            cells,
            aef_source_id,
            area_crs,
            chunk_size=20_000,
            writers=max(1, min(db_writers, 4)),
        )
        buildings = label_frame(city_id, targets, cells)
        mapped = id_map_for(buildings)
        # Cells are already stored. This pass only writes labels and intersections.
        stats = insert_cells_and_labels_parallel(
            city_id,
            buildings,
            mapped,
            cells,
            aef_source_id,
            workers=db_writers,
            upsert_cells=False,
        )
        summary["n_labels"] = int(stats.get("n_labels", 0))
        summary["n_intersections"] = int(stats.get("n_intersections", 0))
        mark_stage(run_dir, city_id, "LABELS", stats)

    if not stage_done(run_dir, city_id, "DOU"):
        set_progress(run_dir, stage="attach_dou", current_city=city_id)
        dou = _sample_city(str(raster_path), city_id, smod_source_id)
        summary["dou"] = dou
        mark_stage(run_dir, city_id, "DOU", dou)
        print(f"[{city_id}] dou labeled {dou.get('n')} new cells", flush=True)

    summary["seconds"] = round(time.time() - t0, 1)
    mark_stage(run_dir, city_id, "CITY", summary)
    print(
        f"[{city_id}] done targets={summary['n_targets']} cells={summary['n_cells']} "
        f"labels={summary['n_labels']} seconds={summary['seconds']}",
        flush=True,
    )
    return summary


def selected_cities(only: list[str] | None) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT city_id, city_name, country_id
            FROM cities
            WHERE inclusion_status = 'selected'
            ORDER BY city_id
            """
        ).fetchall()
    cities = list(rows)
    if only:
        want = set(only)
        cities = [c for c in cities if c["city_id"] in want]
    return cities


def main() -> int:
    _ensure_mp_spawn()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", type=Path, default=DEFAULT_RASTER)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(8, (os.cpu_count() or 8) - 4))
    ap.add_argument(
        "--db-writers",
        type=int,
        default=4,
        help="Postgres writers for one city. Kept low so WAL fsync does not stall.",
    )
    ap.add_argument("--aef-shard-size", type=int, default=1000)
    ap.add_argument(
        "--parallel-cities",
        type=int,
        default=8,
        help=(
            "Cities extracted at once while each city is too small to fill the CPUs. "
            "A city large enough to fill --workers runs alone. Postgres stays at --db-writers."
        ),
    )
    ap.add_argument("--city-id", action="append", default=None)
    args = ap.parse_args()

    raster = args.raster.resolve()
    if not raster.is_file():
        raise SystemExit(f"GHS-SMOD raster not found: {raster}")

    run_dir = Path(args.run_dir or os.environ.get("RUN_DIR") or (ROOT / "runs" / f"embed_peri_rural_{time.strftime('%Y%m%d_%H%M%S')}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    wait_until_ready(timeout_s=60)

    with connect() as conn:
        _overture_id, aef_source_id = ensure_data_sources(conn)
        smod_source_id = ensure_source(conn)

    cities = selected_cities(args.city_id)
    if not cities:
        raise SystemExit("No selected cities")

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    set_progress(
        run_dir,
        status="running",
        stage="select",
        started_at=started,
        cities_completed=0,
        cities_failed=0,
        cities_total=len(cities),
        buildings_selected=0,
        buildings_embedded=0,
        buildings_remaining=None,
        cells_written=0,
        current_city=None,
        percent_complete=0.0,
        eta_seconds=None,
        output=str(run_dir),
        resource_allocation={
            "cpus": int(os.environ.get("CPUS", os.cpu_count() or 1)),
            "aef_workers": int(args.workers),
            "db_writers": int(args.db_writers),
            "aef_shard_size": int(args.aef_shard_size),
            "cities_parallel": int(args.parallel_cities),
            "keep_smod_codes": sorted(KEEP_DOU),
            "exclude": "urban leftovers and buildings that already have cells",
        },
    )
    hb = threading.Thread(target=_heartbeat, args=(run_dir,), daemon=True)
    hb.start()

    # Select every city first so the embed order is smallest-first and the
    # building total is known before the slow AlphaEarth pull.
    plans: list[tuple[dict, int]] = []
    for city in cities:
        cid = city["city_id"]
        if stage_done(run_dir, cid, "CITY"):
            payload = {}
            jp = run_dir / "checkpoints" / cid / "CITY.json"
            if jp.exists():
                payload = json.loads(jp.read_text())
            plans.append((city, int(payload.get("n_targets", 0))))
            continue
        set_progress(run_dir, stage="select", current_city=cid)
        staging = run_dir / "artifacts" / cid
        staging.mkdir(parents=True, exist_ok=True)
        if stage_done(run_dir, cid, "SELECT") and (staging / "targets.parquet").exists():
            targets = pd.read_parquet(staging / "targets.parquet")
        else:
            targets = select_targets(cid, raster, staging)
            mark_stage(
                run_dir,
                cid,
                "SELECT",
                {
                    "n_targets": int(len(targets)),
                    "n_peri": int((targets["dou"] == "peri_urban").sum()) if len(targets) else 0,
                    "n_rural": int((targets["dou"] == "rural").sum()) if len(targets) else 0,
                },
            )
        plans.append((city, int(len(targets))))

    total_targets = sum(n for _, n in plans)
    already_embedded = 0
    for city, _n in plans:
        if stage_done(run_dir, city["city_id"], "CITY"):
            already_embedded += _n
    set_progress(
        run_dir,
        stage="embed",
        buildings_selected=total_targets,
        buildings_embedded=already_embedded,
        buildings_remaining=total_targets - already_embedded,
        percent_complete=round(100.0 * already_embedded / max(total_targets, 1), 2),
    )
    print(
        f"selected peri+rural buildings={total_targets} already_done={already_embedded}",
        flush=True,
    )
    atomic_json(
        run_dir / "artifacts" / "selection.json",
        {
            "buildings_selected": total_targets,
            "cities": [{"city_id": c["city_id"], "n_targets": n} for c, n in plans],
        },
    )

    # A city can fill the CPU pool once it has at least one minimum-size shard
    # per worker. Smaller cities run together. Postgres writes stay on this
    # process, one city at a time.
    fill_workers_at = max(1, int(args.workers) * 32)
    pending = [(c, n) for c, n in plans if not stage_done(run_dir, c["city_id"], "CITY")]
    pending.sort(key=lambda item: (item[1], item[0]["city_id"]))
    small = [(c, n) for c, n in pending if n < fill_workers_at]
    large = [(c, n) for c, n in pending if n >= fill_workers_at]
    n_parallel = max(1, min(int(args.parallel_cities), len(small) or 1))
    workers_each = max(4, int(args.workers) // n_parallel) if n_parallel > 1 else int(args.workers)

    failed = 0
    completed = sum(1 for c, _n in plans if stage_done(run_dir, c["city_id"], "CITY"))
    buildings_embedded = already_embedded
    cells_written = 0
    for city, _n in plans:
        jp = run_dir / "checkpoints" / city["city_id"] / "CITY.json"
        if jp.exists():
            cells_written += int(json.loads(jp.read_text()).get("n_cells", 0))

    def account(summary: dict) -> None:
        nonlocal completed, buildings_embedded, cells_written
        append_jsonl(run_dir / "artifacts" / "manifest.jsonl", {**summary, "status": "done"})
        completed += 1
        buildings_embedded += int(summary.get("n_targets", 0))
        cells_written += int(summary.get("n_cells", 0))
        publish()

    def fail_city(cid: str, exc: BaseException, tb: str) -> None:
        nonlocal failed
        failed += 1
        err = {"city_id": cid, "error": str(exc), "traceback": tb}
        atomic_json(run_dir / "failures" / f"{cid}.json", err)
        append_jsonl(run_dir / "failures" / "failures.jsonl", err)
        append_jsonl(
            run_dir / "artifacts" / "manifest.jsonl",
            {"city_id": cid, "status": "failed", "error": str(exc)},
        )
        print(f"[{cid}] FAILED {exc}", flush=True)
        publish()

    def publish() -> None:
        elapsed = max(time.time() - t0, 1.0)
        rate = buildings_embedded / elapsed
        remaining = max(total_targets - buildings_embedded, 0)
        eta = int(remaining / rate) if rate > 0 and buildings_embedded > 0 else None
        set_progress(
            run_dir,
            stage="embed",
            cities_completed=completed,
            cities_failed=failed,
            cities_total=len(cities),
            buildings_selected=total_targets,
            buildings_embedded=buildings_embedded,
            buildings_remaining=remaining,
            cells_written=cells_written,
            rate_buildings_per_second=round(rate, 3),
            eta_seconds=eta,
            percent_complete=round(100.0 * buildings_embedded / max(total_targets, 1), 2),
            elapsed_seconds=int(elapsed),
            cities_parallel=n_parallel,
            workers_each=workers_each,
        )

    def label_city(city_id: str, n_targets: int, workers: int) -> None:
        set_progress(
            run_dir,
            stage="insert_cells_labels",
            current_city=city_id,
            current_city_buildings=n_targets,
            cities_parallel=n_parallel,
            workers_each=workers,
        )
        summary = process_city(
            city_id,
            raster,
            run_dir,
            aef_source_id,
            smod_source_id,
            workers,
            args.db_writers,
            args.aef_shard_size,
        )
        account(summary)

    print(
        f"schedule small={len(small)} parallel={n_parallel} workers_each={workers_each} "
        f"large={len(large)} large_workers={args.workers} fill_at={fill_workers_at}",
        flush=True,
    )

    if small:
        if n_parallel == 1:
            for city, n_targets in small:
                try:
                    label_city(city["city_id"], n_targets, workers_each)
                except Exception as exc:
                    fail_city(city["city_id"], exc, traceback.format_exc())
        else:
            with ProcessPoolExecutor(max_workers=n_parallel) as pool:
                futs = {
                    pool.submit(
                        _extract_job,
                        {
                            "city_id": city["city_id"],
                            "raster": str(raster),
                            "run_dir": str(run_dir),
                            "workers": workers_each,
                            "shard_size": int(args.aef_shard_size),
                        },
                    ): (city, n_targets)
                    for city, n_targets in small
                }
                set_progress(
                    run_dir,
                    stage="extract_aef",
                    current_city=f"{n_parallel} cities",
                    cities_parallel=n_parallel,
                    workers_each=workers_each,
                )
                for fut in as_completed(futs):
                    city, n_targets = futs[fut]
                    cid = city["city_id"]
                    result = fut.result()
                    if result.get("error"):
                        fail_city(cid, RuntimeError(result["error"]), result.get("traceback", ""))
                        continue
                    try:
                        label_city(cid, n_targets, workers_each)
                    except Exception as exc:
                        fail_city(cid, exc, traceback.format_exc())

    for city, n_targets in large:
        try:
            label_city(city["city_id"], n_targets, int(args.workers))
        except Exception as exc:
            fail_city(city["city_id"], exc, traceback.format_exc())

    status = "completed" if failed == 0 else "completed_with_failures"
    elapsed = int(time.time() - t0)
    summary = {
        "status": status,
        "buildings_selected": total_targets,
        "buildings_embedded": buildings_embedded,
        "cells_written": cells_written,
        "cities_completed": completed,
        "cities_failed": failed,
        "runtime_seconds": elapsed,
        "run_dir": str(run_dir),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text(status + "\n")
    set_progress(run_dir, status=status, stage="done", eta_seconds=0, **summary)
    _stop_heartbeat.set()
    print(json.dumps(summary), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
