#!/usr/bin/env python3
"""Sample DoU-balanced cells for the performance dashboard and score them.

Per city, draws the same number of cells from urban / peri_urban (suburban) /
rural, with 1:1 residential vs non-residential inside each DoU stratum.

  cells_per_city ≈ 2 * k * 3
  where k = min(target_per_class, bal_cap_urban, bal_cap_peri, bal_cap_rural)

Writes:
  viz/usa_embedding_globe/data/dashboard_points.json

Then rebuild metrics with:
  .venv/bin/python scripts/build_dashboard_metrics.py \\
      --points viz/.../dashboard_points.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DATA = ROOT / "viz" / "usa_embedding_globe" / "data"
OUT_POINTS = DATA / "dashboard_points.json"
INTL_FOLDS = DATA / "intl_folds"
A_COLS = [f"a{i:02d}" for i in range(64)]
DOU_ORDER = ("urban", "peri_urban", "rural")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def _caps_query(countries: list[str]) -> pd.DataFrame:
    """Lightweight city × DoU × class counts (no embeddings)."""
    from lib.db import connect

    prefixes = [f"{c}_" for c in countries]
    prefix_sql = " OR ".join(["starts_with(c.city_id, %s)"] * len(prefixes))
    sql = f"""
    SELECT
      c.city_id,
      ci.city_name,
      split_part(c.city_id, '_', 1) AS country,
      d.dou,
      cl.binary_label,
      count(*)::int AS n
    FROM alphaearth_cells c
    JOIN cities ci ON ci.city_id = c.city_id
    JOIN cell_labels cl
      ON cl.cell_id = c.cell_id
     AND cl.label_version = 'resi_def_v1_pixel_frac_v1'
    JOIN cell_degree_of_urbanisation d ON d.cell_id = c.cell_id
    WHERE ({prefix_sql})
      AND c.a00 IS NOT NULL
      AND cl.binary_label IN (0, 1)
      AND d.dou = ANY(%s)
    GROUP BY 1, 2, 3, 4, 5
    """
    with connect() as conn:
        rows = conn.execute(sql, (*prefixes, list(DOU_ORDER))).fetchall()
    return pd.DataFrame(rows)


def _fetch_city_stratum(
    city_id: str,
    dou: str,
    binary_label: int,
    k: int,
    seed: int,
) -> pd.DataFrame:
    """Fetch k random cells for one city × DoU × class (with embeddings)."""
    from lib.db import connect

    emb = ", ".join(f"c.{a}" for a in A_COLS)
    sql = f"""
    SELECT
      c.cell_id,
      c.city_id,
      ci.city_name,
      split_part(c.city_id, '_', 1) AS country,
      d.dou,
      cl.binary_label,
      cl.frac_res,
      cl.frac_nres,
      ST_X(ST_Centroid(c.geom)) AS lon,
      ST_Y(ST_Centroid(c.geom)) AS lat,
      {emb}
    FROM alphaearth_cells c
    JOIN cities ci ON ci.city_id = c.city_id
    JOIN cell_labels cl
      ON cl.cell_id = c.cell_id
     AND cl.label_version = 'resi_def_v1_pixel_frac_v1'
    JOIN cell_degree_of_urbanisation d ON d.cell_id = c.cell_id
    WHERE c.city_id = %s
      AND d.dou = %s
      AND cl.binary_label = %s
      AND c.a00 IS NOT NULL
    ORDER BY md5(c.cell_id::text || %s::text)
    LIMIT %s
    """
    with connect() as conn:
        rows = conn.execute(
            sql, (city_id, dou, int(binary_label), str(seed), int(k))
        ).fetchall()
    return pd.DataFrame(rows)


def _fetch_city_job(args: tuple) -> tuple[str, pd.DataFrame | None, str | None]:
    city_id, k, seed = args
    try:
        parts = []
        for dou in DOU_ORDER:
            for lab in (0, 1):
                part = _fetch_city_stratum(city_id, dou, lab, k, seed)
                if len(part) < k:
                    return city_id, None, f"short stratum {dou}/{lab}: {len(part)}<{k}"
                parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        return city_id, out, None
    except Exception as exc:  # noqa: BLE001
        return city_id, None, str(exc)


def sample_dou_balanced(
    *,
    n_per_dou_class: int,
    seed: int,
    countries: list[str],
    min_k: int = 50,
    workers: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Equal DoU × 1:1 class sample, with the same total N for every kept city.

    1. Measure balanced capacity per city × DoU (aggregate query).
    2. Drop cities with min(DoU caps) < min_k.
    3. Set k_global = min(city caps, n_per_dou_class) across kept cities.
    4. Parallel per-city fetches of exactly k_global per stratum.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    countries = [c.upper() for c in countries]
    t0 = time.time()
    print(
        f"capacity scan target≤{n_per_dou_class}/class/dou min_k={min_k} "
        f"countries={countries} …",
        flush=True,
    )
    caps_df = _caps_query(countries)
    print(f"  capacity rows={len(caps_df)} in {time.time()-t0:.1f}s", flush=True)
    if caps_df.empty:
        return pd.DataFrame(), {"cities": {}, "dropped": [], "k_global": 0}

    city_meta: dict[str, dict] = {}
    for city_id, g in caps_df.groupby("city_id", sort=True):
        caps = {}
        for dou in DOU_ORDER:
            gd = g[g["dou"] == dou]
            n0 = int(gd.loc[gd["binary_label"] == 0, "n"].sum()) if len(gd) else 0
            n1 = int(gd.loc[gd["binary_label"] == 1, "n"].sum()) if len(gd) else 0
            caps[dou] = min(n0, n1, n_per_dou_class)
        city_meta[str(city_id)] = {
            "caps": caps,
            "k_local": int(min(caps.values())) if caps else 0,
            "city_name": str(g["city_name"].iloc[0]),
            "country": str(g["country"].iloc[0]),
        }

    dropped: list[dict] = []
    eligible = {}
    for cid, info in city_meta.items():
        if info["k_local"] >= min_k:
            eligible[cid] = info
        else:
            dropped.append(
                {
                    "city_id": cid,
                    "reason": "min_dou_class_below_threshold",
                    "caps": info["caps"],
                    "k_local": info["k_local"],
                    "min_k": min_k,
                }
            )
            print(
                f"  DROP {cid}: caps={info['caps']} k_local={info['k_local']} < min_k={min_k}",
                flush=True,
            )

    if not eligible:
        return pd.DataFrame(), {"cities": {}, "dropped": dropped, "k_global": 0}

    k_global = int(min(min(info["k_local"] for info in eligible.values()), n_per_dou_class))
    n_city = 6 * k_global
    print(
        f"  k_global={k_global}/class/dou → n_per_city={n_city} "
        f"(urban=peri=rural={2*k_global}) cities={len(eligible)}",
        flush=True,
    )

    # Threads: I/O-bound Postgres fetches; processes were OOM-killing mid-run.
    n_workers = workers or min(len(eligible), max(8, (os.cpu_count() or 8)))
    print(f"  fetching strata with {n_workers} threads …", flush=True)
    jobs = [(cid, k_global, seed) for cid in sorted(eligible)]
    keep: dict[str, pd.DataFrame] = {}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_fetch_city_job, job): job[0] for job in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                city_id, city_df, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                city_id = futs[fut]
                city_df, err = None, str(exc)
            if err or city_df is None:
                dropped.append(
                    {
                        "city_id": city_id,
                        "reason": "fetch_failed",
                        "error": err,
                        "k_global": k_global,
                    }
                )
                print(f"  FAIL {city_id}: {err}", flush=True)
                eligible.pop(city_id, None)
                continue
            keep[city_id] = city_df
            print(f"  KEEP {city_id}: n={len(city_df)} ({i}/{len(jobs)})", flush=True)

    if not keep:
        return pd.DataFrame(), {"cities": {}, "dropped": dropped, "k_global": k_global}

    # If any city failed after k_global was set, re-trim survivors to equal n
    # (all successful fetches already have n_city).
    out = pd.concat([keep[c] for c in sorted(keep)], ignore_index=True)
    for a in A_COLS:
        out[a] = pd.to_numeric(out[a], errors="coerce")
    out["binary_label"] = pd.to_numeric(out["binary_label"], errors="coerce").astype(int)
    out = out.dropna(subset=A_COLS).reset_index(drop=True)

    by_city = out.groupby("city_id").size()
    if by_city.nunique() != 1:
        raise RuntimeError(f"city totals not equal: {by_city.to_dict()}")
    dou_c = Counter(out["dou"].tolist())
    if len(set(dou_c.values())) != 1:
        raise RuntimeError(f"DoU totals not equal: {dou_c}")

    city_info = {}
    for city_id in sorted(keep):
        info = eligible[city_id]
        city_info[city_id] = {
            "city_name": info["city_name"],
            "country": info["country"],
            "k_per_class": k_global,
            "n": int(n_city),
            "n_per_dou": {d: 2 * k_global for d in DOU_ORDER},
            "caps": info["caps"],
            "k_local": info["k_local"],
        }

    print(
        f"  fetched {len(out)} cells in {time.time()-t1:.1f}s "
        f"(total {time.time()-t0:.1f}s)",
        flush=True,
    )
    manifest = {
        "n_per_dou_class_target": n_per_dou_class,
        "min_k": min_k,
        "k_global": k_global,
        "n_per_city": n_city,
        "equal_city_totals": True,
        "seed": seed,
        "n_cities": len(city_info),
        "n_cells": int(len(out)),
        "dou_counts": dict(dou_c),
        "label_counts": {
            "res": int((out["binary_label"] == 1).sum()),
            "nres": int((out["binary_label"] == 0).sum()),
        },
        "cities": city_info,
        "dropped": dropped,
    }
    return out, manifest


def score_sample(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    from export_usa_embedding_globe_sample import week4_baseline_lr_predictions
    from intl_globe_folds import score_with_fold_dir

    X = df[A_COLS].to_numpy(dtype=np.float64)
    y = df["binary_label"].to_numpy(dtype=np.int64)
    cities = df["city_id"].astype(str).to_numpy()
    n = len(df)
    pred = np.full(n, -1, dtype=np.int64)
    proba = np.full(n, np.nan, dtype=np.float64)
    summaries = {}

    usa = np.asarray([c.startswith("USA_") for c in cities])
    intl = ~usa
    if usa.any():
        print(f"scoring USA n={int(usa.sum())} with week4 LORO lr64 …", flush=True)
        p, pr, s = week4_baseline_lr_predictions(X[usa], y[usa], cities[usa])
        pred[usa] = p
        proba[usa] = pr
        summaries["usa"] = s
    if intl.any():
        print(f"scoring IND/JPN n={int(intl.sum())} with intl LORO folds …", flush=True)
        if not INTL_FOLDS.exists():
            raise FileNotFoundError(f"missing intl folds: {INTL_FOLDS}")
        p, pr, _xyz, s = score_with_fold_dir(X[intl], y[intl], cities[intl], INTL_FOLDS)
        pred[intl] = p
        proba[intl] = pr
        summaries["intl"] = s
    if (pred < 0).any():
        bad = int((pred < 0).sum())
        raise RuntimeError(f"{bad} cells unscored")
    return pred, proba, summaries


def to_points(df: pd.DataFrame, pred: np.ndarray, proba: np.ndarray) -> list[dict]:
    points = []
    for i, r in df.iterrows():
        # i may not be positional after concat — use enumerate instead
        pass
    points = []
    for i in range(len(df)):
        r = df.iloc[i]
        gt = int(r["binary_label"])
        pr = int(pred[i])
        points.append(
            {
                "id": int(r["cell_id"]),
                "city_id": str(r["city_id"]),
                "city_name": str(r["city_name"]),
                "country": str(r["country"]),
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "gt": gt,
                "label": gt,
                "pred": pr,
                "proba_res": float(proba[i]),
                "correct": bool(pr == gt),
                "dou": str(r["dou"]),
                "smod_label": None,
                "frac_res": float(r["frac_res"]) if pd.notna(r["frac_res"]) else None,
                "frac_nres": float(r["frac_nres"]) if pd.notna(r["frac_nres"]) else None,
                "pred_model": "loro_lr64",
            }
        )
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-dou-class", type=int, default=200,
                    help="Upper bound on res/nres each, per DoU, per city (default 200)")
    ap.add_argument("--min-k", type=int, default=50,
                    help="Drop city if its DoU-balanced capacity < this; "
                         "remaining cities all use k_global=min(capacity)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel city fetch workers (0 = cpu_count)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--countries", nargs="+", default=["USA", "IND", "JPN"])
    ap.add_argument("--out", type=Path, default=OUT_POINTS)
    ap.add_argument("--rebuild-metrics", action="store_true", default=True)
    ap.add_argument("--no-rebuild-metrics", action="store_false", dest="rebuild_metrics")
    args = ap.parse_args()

    t0 = time.time()
    df, manifest = sample_dou_balanced(
        n_per_dou_class=args.n_per_dou_class,
        seed=args.seed,
        countries=args.countries,
        min_k=args.min_k,
        workers=args.workers or None,
    )
    if df.empty:
        print("ERROR: empty sample", flush=True)
        return 1

    # Verify global DoU balance
    dou_c = Counter(df["dou"].tolist())
    print(f"DoU counts: {dict(dou_c)}", flush=True)
    if len(set(dou_c.values())) != 1:
        print("WARN: global DoU counts not identical (city drops / caps)", flush=True)

    pred, proba, score_sum = score_sample(df)
    points = to_points(df, pred, proba)
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_design": {
            "type": "dou_balanced_1to1_class",
            "dou_order": list(DOU_ORDER),
            "n_per_dou_class": args.n_per_dou_class,
            "min_k": args.min_k,
            "seed": args.seed,
            "note": (
                "Drop cities with DoU-balanced capacity < min_k. "
                "Then k_global=min(capacity across kept cities, target); every kept city "
                "draws exactly k_global res + k_global nres from each of urban/peri_urban/rural "
                "→ identical city totals and identical global DoU totals. "
                "peri_urban shown as suburban in UI."
            ),
            "manifest": manifest,
            "scoring": {
                "usa": (score_sum.get("usa") or {}).get("method"),
                "intl": (score_sum.get("intl") or {}).get("method"),
                "usa_macro_f1": (score_sum.get("usa") or {}).get("macro_f1"),
                "intl_macro_f1": (score_sum.get("intl") or {}).get("macro_f1"),
            },
        },
        "n_points": len(points),
        "n_cities": int(df["city_id"].nunique()),
        "countries": sorted(df["country"].unique().tolist()),
        "dou_counts": dict(dou_c),
        "points": points,
    }
    atomic_json(args.out, payload)
    print(f"wrote {args.out} n={len(points)} cities={payload['n_cities']}", flush=True)

    if args.rebuild_metrics:
        from build_dashboard_metrics import main as metrics_main

        # Re-invoke metrics CLI against the new points file
        sys.argv = [
            "build_dashboard_metrics.py",
            "--points",
            str(args.out),
            "--out",
            str(DATA / "dashboard_metrics.json"),
        ]
        rc = metrics_main()
        if rc != 0:
            return rc

    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
