#!/usr/bin/env python3
"""Export a country-level Degree-of-Urbanisation balanced extract.

Same files the week-5 runners load (X.npy, y.npy, regions.npy, countries.npy),
plus dou.npy so the balance can be audited.

Protocol (seed 42):
  - label_version resi_def_v1_pixel_frac_v1, year 2024
  - pure exclusive + clear, a00..a63 present
  - dou in {urban, peri_urban, rural} (week-2 rollup, not official UN L1)
  - within each country, the same count of residential and non-residential
    cells in each of the three dou classes
  - that count is the country's scarcest (dou, binary-label) supply
  - cities are sampled in proportion to their supply in that stratum
    (a city with no rural cells, such as Tokyo, contributes urban only)

Does not force every city to be internally 1:1:1. That would drop Tokyo
and collapse cities whose rural pure-cell supply is a handful of rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402

SEED = 42
LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
AE_COLS = [f"a{i:02d}" for i in range(64)]
DOUS = ("urban", "peri_urban", "rural")
LABELS = (0, 1)


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
    prev["last_progress_at"] = prev["updated_at"]
    atomic_json(path, prev)


def city_seed(seed: int, city_id: str) -> int:
    return int(seed + (zlib.crc32(city_id.encode()) % 1_000_000))


def allocate(capacity: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation. Each quota is <= that city's capacity."""
    items = [(k, int(v)) for k, v in capacity.items() if int(v) > 0]
    total = sum(v for _, v in items)
    if n > total:
        raise ValueError(f"need {n}, supply is {total}")
    if n == 0 or not items:
        return {}
    if n == total:
        return {k: v for k, v in items}
    raw = [(k, n * v / total) for k, v in items]
    base = {k: int(np.floor(x)) for k, x in raw}
    rem = n - sum(base.values())
    cap = dict(items)
    order = sorted(raw, key=lambda kv: (kv[1] - np.floor(kv[1]), cap[kv[0]]), reverse=True)
    for k, _ in order:
        if rem <= 0:
            break
        if base[k] < cap[k]:
            base[k] += 1
            rem -= 1
    if rem != 0:
        raise RuntimeError(f"allocation remainder {rem} for n={n}")
    return {k: v for k, v in base.items() if v > 0}


def country_targets(counts: pd.DataFrame, countries: list[str]) -> dict[str, int]:
    """Per country, cells per (dou, binary label). Equal across the six strata."""
    targets = {}
    for country in countries:
        sub = counts[counts["country_id"] == country]
        n = None
        for dou in DOUS:
            for lab in LABELS:
                cell = sub[(sub["dou"] == dou) & (sub["binary_label"] == lab)]
                supply = int(cell["n"].sum()) if len(cell) else 0
                n = supply if n is None else min(n, supply)
        if not n:
            raise SystemExit(f"{country} has an empty dou stratum; cannot balance")
        targets[country] = int(n)
    return targets


def fetch_city(cur, city_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = ", ".join(f"c.{c}" for c in AE_COLS)
    cur.execute(
        f"""
        SELECT {cols}, cl.binary_label, d.dou
        FROM alphaearth_cells c
        JOIN cell_labels cl
          ON cl.cell_id = c.cell_id
         AND cl.label_version = %s
        JOIN cell_degree_of_urbanisation d
          ON d.cell_id = c.cell_id
        WHERE c.city_id = %s
          AND c.year = 2024
          AND cl.binary_label IS NOT NULL
          AND cl.ambiguity_status = 'clear'
          AND cl.purity_status IN ('pure_residential', 'pure_non_residential')
          AND c.a00 IS NOT NULL AND c.a63 IS NOT NULL
          AND d.dou = ANY(%s)
        """,
        (LABEL_VERSION, city_id, list(DOUS)),
    )
    rows = cur.fetchall()
    if not rows:
        return (
            np.zeros((0, 64), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )
    n = len(rows)
    X = np.empty((n, 64), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    dou = np.empty((n,), dtype=object)
    for i, r in enumerate(rows):
        for j, c in enumerate(AE_COLS):
            X[i, j] = float(r[c])
        y[i] = int(r["binary_label"])
        dou[i] = r["dou"]
    return X, y, dou


def sample_city(
    X: np.ndarray,
    y: np.ndarray,
    dou: np.ndarray,
    quotas: dict[tuple[str, int], int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = []
    for (d, lab), n in quotas.items():
        if n <= 0:
            continue
        idx = np.flatnonzero((dou == d) & (y == lab))
        if len(idx) < n:
            raise RuntimeError(f"stratum {d}/{lab} has {len(idx)} < quota {n}")
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        keep.append(idx)
    if not keep:
        return (
            np.zeros((0, 64), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )
    idx = np.concatenate(keep)
    rng.shuffle(idx)
    return X[idx], y[idx], dou[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--countries", nargs="+", default=["USA", "JPN", "IND"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = args.out_dir
    run_dir = args.run_dir or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "checkpoints" / "cities"
    ckpt.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "checkpoints" / "export.DONE"
    if done_marker.exists() and (out_dir / "X.npy").exists() and (out_dir / "dou.npy").exists():
        print(f"export already DONE at {out_dir}; skipping", flush=True)
        return 0

    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="count_strata",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        completed=0,
        failed=0,
        total=None,
        output_location=str(out_dir),
        resource_allocation={"db_writers": 1, "note": "one sequential read connection"},
    )

    wait_until_ready(timeout_s=60)
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '0'")
            cur.execute(
                """
                SELECT c.city_id, ci.country_id, d.dou, cl.binary_label::int AS binary_label,
                       count(*)::bigint AS n
                FROM alphaearth_cells c
                JOIN cities ci ON ci.city_id = c.city_id
                JOIN cell_labels cl
                  ON cl.cell_id = c.cell_id
                 AND cl.label_version = %s
                JOIN cell_degree_of_urbanisation d
                  ON d.cell_id = c.cell_id
                WHERE ci.country_id = ANY(%s)
                  AND c.year = 2024
                  AND cl.binary_label IS NOT NULL
                  AND cl.ambiguity_status = 'clear'
                  AND cl.purity_status IN ('pure_residential', 'pure_non_residential')
                  AND c.a00 IS NOT NULL AND c.a63 IS NOT NULL
                  AND d.dou = ANY(%s)
                GROUP BY 1, 2, 3, 4
                """,
                (LABEL_VERSION, list(args.countries), list(DOUS)),
            )
            counts = pd.DataFrame(cur.fetchall())
            if counts.empty:
                raise SystemExit("no eligible cells")
            counts["n"] = counts["n"].astype(int)
            counts["binary_label"] = counts["binary_label"].astype(int)

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

            targets = country_targets(counts, list(args.countries))
            quotas: dict[str, dict[tuple[str, int], int]] = {c["city_id"]: {} for c in cities}
            stratum_rows = []
            for country, n_star in targets.items():
                for dou in DOUS:
                    for lab in LABELS:
                        sub = counts[
                            (counts["country_id"] == country)
                            & (counts["dou"] == dou)
                            & (counts["binary_label"] == lab)
                        ]
                        cap = {r.city_id: int(r.n) for r in sub.itertuples(index=False)}
                        alloc = allocate(cap, n_star)
                        for city_id, q in alloc.items():
                            quotas[city_id][(dou, lab)] = q
                        stratum_rows.append(
                            {
                                "country_id": country,
                                "dou": dou,
                                "binary_label": lab,
                                "supply": int(sub["n"].sum()),
                                "sampled": n_star,
                            }
                        )
            atomic_json(
                run_dir / "checkpoints" / "quotas.json",
                {
                    "targets_per_stratum": targets,
                    "protocol": "country 1:1:1 dou, 1:1 res/nres inside each dou",
                },
            )
            print("targets per (dou, label):", targets, flush=True)

            total_cities = len(cities)
            sampled_rows = 0
            failed = 0
            for i, city in enumerate(cities, 1):
                cid = city["city_id"]
                city_quotas = quotas.get(cid, {})
                city_path = ckpt / f"{cid}.npz"
                expected = int(sum(city_quotas.values()))
                if city_path.exists():
                    with np.load(city_path, allow_pickle=True) as z:
                        n_saved = int(len(z["y"]))
                    if n_saved != expected:
                        city_path.unlink()
                        print(f"[{i}/{total_cities}] drop stale checkpoint {cid} ({n_saved} != {expected})", flush=True)
                    else:
                        sampled_rows += n_saved
                        write_progress(
                            run_dir,
                            stage="sample_cities",
                            status="running",
                            completed=i,
                            total=total_cities,
                            failed=failed,
                            current_city=cid,
                            cells_sampled=sampled_rows,
                            percent_complete=round(100.0 * i / total_cities, 2),
                            elapsed_seconds=round(time.time() - t0, 1),
                        )
                        print(f"[{i}/{total_cities}] skip {cid} (checkpoint)", flush=True)
                        continue
                if not city_quotas:
                    np.savez_compressed(
                        city_path,
                        X=np.zeros((0, 64), np.float32),
                        y=np.zeros((0,), np.int64),
                        dou=np.zeros((0,), dtype=object),
                    )
                    print(f"[{i}/{total_cities}] {cid} quota 0", flush=True)
                    continue

                write_progress(
                    run_dir,
                    stage="sample_cities",
                    status="running",
                    completed=i - 1,
                    total=total_cities,
                    failed=failed,
                    current_city=cid,
                    cells_sampled=sampled_rows,
                    percent_complete=round(100.0 * (i - 1) / total_cities, 2),
                    elapsed_seconds=round(time.time() - t0, 1),
                )
                t_city = time.time()
                print(f"[{i}/{total_cities}] fetch {cid}", flush=True)
                try:
                    X, y, dou = fetch_city(cur, cid)
                    rng = np.random.default_rng(city_seed(args.seed, cid))
                    Xs, ys, ds = sample_city(X, y, dou, city_quotas, rng)
                except Exception as exc:
                    failed += 1
                    fail_path = run_dir / "failures.jsonl"
                    with fail_path.open("a") as fh:
                        fh.write(json.dumps({"city_id": cid, "error": str(exc)}) + "\n")
                    write_progress(run_dir, status="failed", stage="sample_cities", failed=failed, current_city=cid)
                    raise
                np.savez_compressed(city_path, X=Xs, y=ys, dou=ds)
                sampled_rows += int(len(ys))
                rate = i / max(time.time() - t0, 1e-6)
                remaining = total_cities - i
                write_progress(
                    run_dir,
                    stage="sample_cities",
                    status="running",
                    completed=i,
                    total=total_cities,
                    failed=failed,
                    success=i - failed,
                    current_city=cid,
                    cells_sampled=sampled_rows,
                    city_seconds=round(time.time() - t_city, 1),
                    rate_per_second=round(rate, 4),
                    percent_complete=round(100.0 * i / total_cities, 2),
                    eta_seconds=round(remaining / rate, 1) if rate else None,
                    elapsed_seconds=round(time.time() - t0, 1),
                )
                print(
                    f"[{i}/{total_cities}] {cid} kept {len(ys)} / raw {len(y)} in {time.time() - t_city:.1f}s",
                    flush=True,
                )
                del X, y, dou, Xs, ys, ds

    write_progress(run_dir, stage="assemble", status="running", completed=total_cities, total=total_cities)
    xs, ys, dous, regs, countrys = [], [], [], [], []
    count_rows = []
    for city in cities:
        cid = city["city_id"]
        with np.load(ckpt / f"{cid}.npz", allow_pickle=True) as z:
            X = np.asarray(z["X"], dtype=np.float32)
            y = np.asarray(z["y"], dtype=np.int64)
            dou = np.asarray(z["dou"], dtype=object)
        if len(y) == 0:
            continue
        xs.append(X)
        ys.append(y)
        dous.append(dou)
        regs.append(np.full(len(y), cid, dtype=object))
        countrys.append(np.full(len(y), city["country_id"], dtype=object))
        for d in DOUS:
            n0 = int(((dou == d) & (y == 0)).sum())
            n1 = int(((dou == d) & (y == 1)).sum())
            if n0 or n1:
                count_rows.append(
                    {
                        "city_id": cid,
                        "country_id": city["country_id"],
                        "city_name": city["city_name"],
                        "dou": d,
                        "n_nonresidential": n0,
                        "n_residential": n1,
                        "n": n0 + n1,
                    }
                )

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    dou = np.concatenate(dous)
    regions = np.concatenate(regs)
    countries_arr = np.concatenate(countrys)
    # Shuffle once so country blocks are not contiguous on disk order only.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y))
    X, y, dou = X[order], y[order], dou[order]
    regions, countries_arr = regions[order], countries_arr[order]

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "dou.npy", dou)
    np.save(out_dir / "regions.npy", regions)
    np.save(out_dir / "countries.npy", countries_arr)
    pd.DataFrame(count_rows).to_csv(out_dir / "city_counts.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(out_dir / "stratum_counts.csv", index=False)

    balance = {}
    for country in args.countries:
        m = countries_arr == country
        balance[country] = {}
        for d in DOUS:
            md = m & (dou == d)
            balance[country][d] = {"y0": int((y[md] == 0).sum()), "y1": int((y[md] == 1).sum())}
            if balance[country][d]["y0"] != targets[country] or balance[country][d]["y1"] != targets[country]:
                raise SystemExit(f"balance check failed for {country} {d}: {balance[country][d]} != {targets[country]}")

    manifest = {
        "protocol": (
            "pure_exclusive_clear + country-level 1:1:1 dou "
            "(urban=peri_urban=rural) and 1:1 residential/non-residential within each dou. "
            "Cities sampled proportional to stratum supply. Not per-city 1:1:1."
        ),
        "label_version": LABEL_VERSION,
        "year": 2024,
        "dou_rollup": "week2 urban={30,23,22} peri_urban={21} rural={13,12,11}",
        "seed": args.seed,
        "n_cells": int(len(y)),
        "targets_per_stratum": targets,
        "balance": balance,
        "countries": list(args.countries),
        "features": AE_COLS,
        "files": ["X.npy", "y.npy", "dou.npy", "regions.npy", "countries.npy", "city_counts.csv", "stratum_counts.csv"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    atomic_json(out_dir / "manifest.json", manifest)
    done_marker.write_text("ok\n")
    write_progress(
        run_dir,
        status="running",
        stage="export_done",
        completed=total_cities,
        total=total_cities,
        failed=0,
        cells_sampled=int(len(y)),
        percent_complete=100.0,
        elapsed_seconds=round(time.time() - t0, 1),
        eta_seconds=0,
        output_location=str(out_dir),
    )
    print(json.dumps({"n_cells": int(len(y)), "targets": targets, "balance": balance}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
