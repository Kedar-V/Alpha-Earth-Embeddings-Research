#!/usr/bin/env python3
"""Export week4-parity balanced pure-exclusive cells from Postgres.

Per city:
  - label_version = resi_def_v1_pixel_frac_v1
  - pure exclusive + clear (binary_label set)
  - 1:1 undersample majority class (seed 42)
  - features a00..a63, year 2024

Writes under --out-dir:
  X.npy, y.npy, regions.npy, countries.npy, city_counts.csv, manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402

SEED = 42
LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
AE_COLS = [f"a{i:02d}" for i in range(64)]


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


def balance_city(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_per_class: int | None):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return None
    n = min(len(idx0), len(idx1))
    if max_per_class is not None:
        n = min(n, int(max_per_class))
    if len(idx0) > n:
        idx0 = rng.choice(idx0, size=n, replace=False)
    if len(idx1) > n:
        idx1 = rng.choice(idx1, size=n, replace=False)
    keep = np.concatenate([idx0, idx1])
    rng.shuffle(keep)
    return X[keep], y[keep], int(n)


def fetch_city(cur, city_id: str) -> tuple[np.ndarray, np.ndarray]:
    cols = ", ".join(f"c.{c}" for c in AE_COLS)
    cur.execute(
        f"""
        SELECT {cols}, cl.binary_label
        FROM alphaearth_cells c
        JOIN cell_labels cl
          ON cl.cell_id = c.cell_id
         AND cl.label_version = %s
        WHERE c.city_id = %s
          AND c.year = 2024
          AND cl.binary_label IS NOT NULL
          AND cl.ambiguity_status = 'clear'
          AND cl.purity_status IN ('pure_residential', 'pure_non_residential')
          AND c.a00 IS NOT NULL AND c.a63 IS NOT NULL
        """,
        (LABEL_VERSION, city_id),
    )
    rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 64), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    # dict rows
    X = np.empty((len(rows), 64), dtype=np.float32)
    y = np.empty((len(rows),), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, c in enumerate(AE_COLS):
            X[i, j] = float(r[c])
        y[i] = int(r["binary_label"])
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path, default=None, help="progress.json parent (defaults to out-dir)")
    ap.add_argument("--countries", nargs="+", default=["USA", "IND", "JPN"])
    ap.add_argument("--max-per-class", type=int, default=None, help="cap per class per city (smoke)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = args.out_dir
    run_dir = args.run_dir or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    done_marker = ckpt / "export.DONE"
    if done_marker.exists() and (out_dir / "X.npy").exists():
        print(f"export already DONE at {out_dir}; skipping", flush=True)
        return 0

    wait_until_ready(timeout_s=30)
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="export_balanced_extract",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        completed=0,
        failed=0,
        output_location=str(out_dir),
    )

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '0'")
            cur.execute(
                """
                SELECT city_id, country_id, city_name
                FROM cities
                WHERE country_id = ANY(%s)
                ORDER BY country_id, city_id
                """,
                (list(args.countries),),
            )
            cities = list(cur.fetchall())

            xs, ys, regs, countries = [], [], [], []
            counts = []
            for i, city in enumerate(cities, 1):
                cid = city["city_id"]
                t_city = time.time()
                print(f"[{i}/{len(cities)}] fetch {cid}", flush=True)
                X, y = fetch_city(cur, cid)
                n_raw = int(len(y))
                n0 = int((y == 0).sum()) if n_raw else 0
                n1 = int((y == 1).sum()) if n_raw else 0
                rng = np.random.default_rng(args.seed + (hash(cid) % 10_000))
                bal = balance_city(X, y, rng, args.max_per_class) if n_raw else None
                if bal is None:
                    print(f"  SKIP {cid}: insufficient classes raw={n_raw} n0={n0} n1={n1}", flush=True)
                    counts.append(
                        {
                            "city_id": cid,
                            "country_id": city["country_id"],
                            "city_name": city["city_name"],
                            "n_raw": n_raw,
                            "n_raw_nres": n0,
                            "n_raw_res": n1,
                            "n_balanced_per_class": 0,
                            "n_balanced_total": 0,
                            "status": "skipped",
                        }
                    )
                    write_progress(
                        run_dir,
                        completed=i,
                        total=len(cities),
                        current_city=cid,
                        failed=sum(1 for c in counts if c["status"] == "skipped"),
                        percent_complete=round(100.0 * i / len(cities), 2),
                        last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    continue
                Xb, yb, n_per = bal
                xs.append(Xb)
                ys.append(yb)
                regs.append(np.array([cid] * len(yb), dtype=object))
                countries.append(np.array([city["country_id"]] * len(yb), dtype=object))
                counts.append(
                    {
                        "city_id": cid,
                        "country_id": city["country_id"],
                        "city_name": city["city_name"],
                        "n_raw": n_raw,
                        "n_raw_nres": n0,
                        "n_raw_res": n1,
                        "n_balanced_per_class": n_per,
                        "n_balanced_total": int(len(yb)),
                        "status": "ok",
                        "seconds": round(time.time() - t_city, 2),
                    }
                )
                print(
                    f"  ok raw={n_raw} ({n1} res / {n0} nres) -> balanced {len(yb)} ({n_per}/class) "
                    f"in {time.time()-t_city:.1f}s",
                    flush=True,
                )
                write_progress(
                    run_dir,
                    completed=i,
                    total=len(cities),
                    current_city=cid,
                    success_count=sum(1 for c in counts if c["status"] == "ok"),
                    failed=sum(1 for c in counts if c["status"] == "skipped"),
                    percent_complete=round(100.0 * i / len(cities), 2),
                    last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    elapsed_seconds=round(time.time() - t0, 1),
                )

    if not xs:
        raise SystemExit("no cities exported")

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    regions = np.concatenate(regs, axis=0)
    country = np.concatenate(countries, axis=0)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "regions.npy", regions)
    np.save(out_dir / "countries.npy", country)
    pd.DataFrame(counts).to_csv(out_dir / "city_counts.csv", index=False)

    manifest = {
        "label_version": LABEL_VERSION,
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "countries": list(args.countries),
        "n_samples": int(len(y)),
        "n_res": int((y == 1).sum()),
        "n_nres": int((y == 0).sum()),
        "n_cities": int(len({r for r in regions.tolist()})),
        "feature_dim": 64,
        "year": 2024,
        "protocol": "pure_exclusive_clear + per-city 1:1 undersample",
        "paths": {
            "X": str(out_dir / "X.npy"),
            "y": str(out_dir / "y.npy"),
            "regions": str(out_dir / "regions.npy"),
            "countries": str(out_dir / "countries.npy"),
            "city_counts": str(out_dir / "city_counts.csv"),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    atomic_json(out_dir / "manifest.json", manifest)
    done_marker.write_text(json.dumps(manifest, indent=2) + "\n")
    write_progress(
        run_dir,
        status="export_done",
        stage="export_balanced_extract",
        percent_complete=100.0,
        n_samples=manifest["n_samples"],
        elapsed_seconds=manifest["runtime_seconds"],
        last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
