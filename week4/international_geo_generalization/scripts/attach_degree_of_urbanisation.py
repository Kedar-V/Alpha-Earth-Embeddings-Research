#!/usr/bin/env python3
"""Attach GHS-SMOD Degree of Urbanisation to every AlphaEarth cell.

Samples the 1 km categorical raster at the cell center (containing cell,
no interpolation). Writes `cell_degree_of_urbanisation`:

  smod_code / smod_label  — native L2 class (the pixel label)
  dou                     — urban / peri_urban / rural study rollup
                            (null for water and no-data)

Idempotent. Cities already fully labeled are skipped. Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402
from lib.smod import DOU_ROLLUP, SMOD_LABEL  # noqa: E402

DEFAULT_RASTER = (
    REPO
    / "data"
    / "ghsl"
    / "smod"
    / "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif"
)
PRODUCT = "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0"
EPOCH = 2025
SOURCE_URI = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_SMOD_GLOBE_R2023A/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000/V2-0/"
    "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.zip"
)

# Integer sentinels so workers return arrays, not object arrays.
_LABEL_TO_ID = {name: i for i, name in enumerate(SMOD_LABEL.values())}
_LABEL_TO_ID["unclassified"] = len(_LABEL_TO_ID)
_ID_TO_LABEL = {i: name for name, i in _LABEL_TO_ID.items()}
_DOU_TO_ID = {None: -1, "urban": 0, "peri_urban": 1, "rural": 2}
_ID_TO_DOU = {v: k for k, v in _DOU_TO_ID.items()}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(path, prev)


def ensure_source(conn) -> int:
    row = conn.execute(
        """
        INSERT INTO data_sources (
            source_name, source_version, source_year, processing_version,
            label_mapping_version, uri, metadata
        )
        VALUES (
            'GHS-SMOD', 'R2023A', %s, 'V2-0', 'smod_l2_v1', %s,
            %s::jsonb
        )
        ON CONFLICT (source_name, source_version, processing_version)
        DO UPDATE SET
            source_year = EXCLUDED.source_year,
            label_mapping_version = EXCLUDED.label_mapping_version,
            uri = EXCLUDED.uri,
            metadata = EXCLUDED.metadata
        RETURNING source_id
        """,
        (
            EPOCH,
            SOURCE_URI,
            json.dumps(
                {
                    "product": PRODUCT,
                    "epoch": EPOCH,
                    "pixel_size_m": 1000,
                    "crs": "ESRI:54009",
                    "band": "smod_code",
                    "sample": "cell_center_containing_pixel",
                    "dou_rollup": {
                        "urban": [30, 23, 22],
                        "peri_urban": [21],
                        "rural": [13, 12, 11],
                    },
                }
            ),
        ),
    ).fetchone()
    return int(row["source_id"])


def pending_cities(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.city_id,
               count(*) AS n_cells,
               count(d.cell_id) AS n_labeled
        FROM alphaearth_cells c
        LEFT JOIN cell_degree_of_urbanisation d ON d.cell_id = c.cell_id
        GROUP BY c.city_id
        ORDER BY c.city_id
        """
    ).fetchall()
    return [r for r in rows if int(r["n_labeled"]) < int(r["n_cells"])]


def _codes_to_ids(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_id = np.full(codes.shape, _LABEL_TO_ID["unclassified"], dtype=np.int16)
    dou_id = np.full(codes.shape, _DOU_TO_ID[None], dtype=np.int8)
    for code, name in SMOD_LABEL.items():
        mask = codes == code
        if not mask.any():
            continue
        label_id[mask] = _LABEL_TO_ID[name]
        dou = DOU_ROLLUP.get(code)
        if dou is not None:
            dou_id[mask] = _DOU_TO_ID[dou]
    return label_id, dou_id


def _sample_city(raster_path: str, city_id: str, source_id: int) -> dict:
    """Sample one city and upsert. Runs in a worker process."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    from lib.db import connect

    t0 = time.time()
    with connect() as conn:
        pts = conn.execute(
            """
            SELECT sm.cell_id, sm.pixel_lon, sm.pixel_lat
            FROM alphaearth_cells c
            JOIN cell_spatial_metrics sm ON sm.cell_id = c.cell_id
            LEFT JOIN cell_degree_of_urbanisation d ON d.cell_id = c.cell_id
            WHERE c.city_id = %s
              AND d.cell_id IS NULL
            """,
            (city_id,),
        ).fetchall()
        if not pts:
            return {"city_id": city_id, "n": 0, "seconds": 0.0, "counts": {}}

        cell_id = np.fromiter((int(r["cell_id"]) for r in pts), dtype=np.int64, count=len(pts))
        lon = np.fromiter((float(r["pixel_lon"]) for r in pts), dtype=np.float64, count=len(pts))
        lat = np.fromiter((float(r["pixel_lat"]) for r in pts), dtype=np.float64, count=len(pts))

        with rasterio.open(raster_path) as ds:
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            pad = 2000.0  # 2 km, two pixels of the 1 km grid
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

        # Containing cell. Categorical — do not interpolate.
        cols = np.floor((x - win_transform.c) / win_transform.a).astype(np.int64)
        rows = np.floor((y - win_transform.f) / win_transform.e).astype(np.int64)
        h, w = arr.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        codes = np.full(len(cell_id), nodata, dtype=np.int16)
        codes[inside] = arr[rows[inside], cols[inside]]
        label_id, dou_id = _codes_to_ids(codes)

        # One city used to be one giant transaction: heap + 3 low-cardinality
        # indexes, fsynced on a 97% full ZFS pool. That stalled the checkpointer.
        # Commit in small batches, skip the extra indexes, and don't wait on fsync.
        conn.execute("SET synchronous_commit TO off")
        chunk = 20_000
        n_rows = len(cell_id)
        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            conn.execute(
                """
                CREATE TEMP TABLE stage_dou (
                    cell_id BIGINT,
                    smod_code SMALLINT,
                    smod_label TEXT,
                    dou TEXT
                ) ON COMMIT DROP
                """
            )
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY stage_dou (cell_id, smod_code, smod_label, dou) FROM STDIN"
                ) as cp:
                    for i in range(start, end):
                        cp.write_row(
                            (
                                int(cell_id[i]),
                                int(codes[i]),
                                _ID_TO_LABEL[int(label_id[i])],
                                _ID_TO_DOU[int(dou_id[i])],
                            )
                        )
                cur.execute(
                    """
                    INSERT INTO cell_degree_of_urbanisation (
                        cell_id, smod_code, smod_label, dou, epoch, product, source_id
                    )
                    SELECT cell_id, smod_code, smod_label, dou, %s, %s, %s
                    FROM stage_dou
                    ON CONFLICT (cell_id) DO NOTHING
                    """,
                    (EPOCH, PRODUCT, source_id),
                )
            conn.commit()

    # counts of native labels
    uniq, cnt = np.unique(label_id, return_counts=True)
    counts = {_ID_TO_LABEL[int(k)]: int(v) for k, v in zip(uniq, cnt)}
    return {
        "city_id": city_id,
        "n": int(len(cell_id)),
        "seconds": round(time.time() - t0, 2),
        "counts": counts,
    }


def drop_secondary_indexes(conn) -> None:
    """Keep only the primary key during the load. Rebuild the rest at the end."""
    conn.execute("DROP INDEX IF EXISTS idx_cell_dou_smod_code")
    conn.execute("DROP INDEX IF EXISTS idx_cell_dou_dou")
    conn.execute("DROP INDEX IF EXISTS idx_cell_dou_smod_label")


def create_secondary_indexes(conn) -> None:
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cell_dou_smod_code
            ON cell_degree_of_urbanisation (smod_code)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cell_dou_dou
            ON cell_degree_of_urbanisation (dou)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cell_dou_smod_label
            ON cell_degree_of_urbanisation (smod_label)
        """
    )


def label_summary(conn) -> list[dict]:
    return conn.execute(
        """
        SELECT smod_code, smod_label, dou, count(*) AS n
        FROM cell_degree_of_urbanisation
        GROUP BY smod_code, smod_label, dou
        ORDER BY smod_code
        """
    ).fetchall()


def main() -> int:
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", type=Path, default=DEFAULT_RASTER)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "City-parallel writers. Default 2: each writer is a Postgres connection, "
            "and more of them stall WAL fsync on this ZFS pool."
        ),
    )
    args = ap.parse_args()
    raster = args.raster.resolve()
    if not raster.is_file():
        raise SystemExit(f"GHS-SMOD raster not found: {raster}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "runs" / f"dou_attach_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    wait_until_ready()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    with connect() as conn:
        source_id = ensure_source(conn)
        cities = pending_cities(conn)
    n_cells = sum(int(c["n_cells"]) - int(c["n_labeled"]) for c in cities)
    write_progress(
        run_dir,
        status="running",
        stage="sample",
        started_at=started,
        raster=str(raster),
        source_id=source_id,
        epoch=EPOCH,
        cities_total=len(cities),
        cities_done=0,
        cells_remaining=n_cells,
        cells_done=0,
        workers=args.workers,
    )
    print(
        f"source_id={source_id} cities={len(cities)} cells_remaining={n_cells} "
        f"workers={args.workers}",
        flush=True,
    )
    with connect() as conn:
        drop_secondary_indexes(conn)
    print("secondary indexes dropped for the load", flush=True)

    if not cities:
        with connect() as conn:
            create_secondary_indexes(conn)
        summary = {"status": "completed", "cells_written": 0, "note": "already labeled"}
        atomic_json(run_dir / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0

    t0 = time.time()
    done_cities = 0
    done_cells = 0
    merged_counts: dict[str, int] = {}
    failures: list[dict] = []
    workers = max(1, min(args.workers, len(cities)))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_sample_city, str(raster), c["city_id"], source_id): c["city_id"]
            for c in cities
        }
        for fut in as_completed(futs):
            city_id = futs[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"city_id": city_id, "error": repr(exc)})
                (run_dir / "logs" / f"{city_id}.txt").write_text(repr(exc) + "\n")
                print(f"FAILED {city_id}: {exc}", flush=True)
                continue
            done_cities += 1
            done_cells += int(result["n"])
            for k, v in result["counts"].items():
                merged_counts[k] = merged_counts.get(k, 0) + int(v)
            elapsed = time.time() - t0
            rate = done_cells / elapsed if elapsed else 0.0
            remaining = max(n_cells - done_cells, 0)
            eta = remaining / rate if rate else None
            write_progress(
                run_dir,
                status="running",
                stage="sample",
                started_at=started,
                cities_done=done_cities,
                cities_total=len(cities),
                completed=done_cities,
                total=len(cities),
                failed=len(failures),
                cells_done=done_cells,
                cells_remaining=remaining,
                cells_per_sec=round(rate, 1),
                rate_per_second=round(rate, 1),
                percent_complete=round(100.0 * done_cells / max(n_cells, 1), 2),
                eta_seconds=None if eta is None else round(eta, 1),
                eta_sec=None if eta is None else round(eta, 1),
                failures=len(failures),
                last_city=city_id,
                last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            print(
                f"[{done_cities}/{len(cities)}] {city_id} n={result['n']} "
                f"{result['seconds']}s  {rate:.0f} cells/s",
                flush=True,
            )

    with connect() as conn:
        print("rebuilding secondary indexes", flush=True)
        create_secondary_indexes(conn)
        rows = label_summary(conn)
        n_labeled = conn.execute(
            "SELECT count(*) AS n FROM cell_degree_of_urbanisation"
        ).fetchone()["n"]
        n_all = conn.execute("SELECT count(*) AS n FROM alphaearth_cells").fetchone()["n"]

    final = {
        "status": "completed" if not failures else "completed_with_failures",
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raster": str(raster),
        "product": PRODUCT,
        "epoch": EPOCH,
        "source_id": source_id,
        "cells_written_this_run": done_cells,
        "cells_labeled": int(n_labeled),
        "cells_total": int(n_all),
        "cities_failed": failures,
        "label_counts": [
            {
                "smod_code": int(r["smod_code"]),
                "smod_label": r["smod_label"],
                "dou": r["dou"],
                "n": int(r["n"]),
            }
            for r in rows
        ],
    }
    atomic_json(run_dir / "summary.json", final)
    write_progress(
        run_dir,
        status=final["status"],
        stage="done",
        cities_done=done_cities,
        cells_done=done_cells,
        cells_labeled=int(n_labeled),
        cells_total=int(n_all),
        percent_complete=round(100.0 * int(n_labeled) / max(int(n_all), 1), 2),
    )
    (run_dir / "DONE").write_text(final["status"] + "\n")
    print(json.dumps(final, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
