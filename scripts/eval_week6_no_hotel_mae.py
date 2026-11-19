#!/usr/bin/env python3
"""Week 6 sensitivity: exclude Overture hotels, rescore building soft + 100 m.

Same protocol as Section 3 baseline (20261110_041557 building soft +
20261110_044005 100 m tiles), but drops label=='hotel' then rebalances
1:1 y_res per metro (seed 42).

Outputs:
  outputs/week6_no_hotel_mae/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEEK6_RUN = ROOT / "outputs" / "week6_overture_polygon_mae" / "20261110_041557"
BUILDING_CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
AE_OOF = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
PLS_OOF = ROOT / "outputs" / "week4_pls_loro" / "oof_predictions.parquet"
TILE_CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
MIN_CELLS = 20
LONLAT_DECIMALS = 7
BALANCE_SEED = 42
EXCLUDE_LABELS = ("hotel",)

REGION_ORDER = [
    "Boston_NE",
    "NYC_NE",
    "Atlanta_SE",
    "Houston_S",
    "Chicago_MW",
    "Phoenix_SW",
    "Denver_W",
    "LA_W",
    "SF_W",
    "Seattle_NW",
]
PRETTY = {
    "Boston_NE": "Boston NE",
    "NYC_NE": "NYC NE",
    "Atlanta_SE": "Atlanta SE",
    "Houston_S": "Houston S",
    "Chicago_MW": "Chicago MW",
    "Phoenix_SW": "Phoenix SW",
    "Denver_W": "Denver W",
    "LA_W": "LA W",
    "SF_W": "SF W",
    "Seattle_NW": "Seattle NW",
}
BG = {
    "Boston_NE": "#ECE8F4",
    "NYC_NE": "#ECE8F4",
    "Atlanta_SE": "#E2EFE6",
    "Houston_S": "#F3EDD5",
    "Chicago_MW": "#F3E3E7",
    "Phoenix_SW": "#EFE4D9",
    "Denver_W": "#E1ECEE",
    "LA_W": "#E1ECEE",
    "SF_W": "#E1ECEE",
    "Seattle_NW": "#E9EDDC",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev:
            done = int(prev["completed"])
            total = max(1, int(prev["total"]))
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def lonlat_key(lon: np.ndarray, lat: np.ndarray, nd: int = LONLAT_DECIMALS) -> np.ndarray:
    return np.round(lon, nd).astype(str) + "_" + np.round(lat, nd).astype(str)


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    return float(mean_absolute_error(y[m], p[m]))


def metrics(df: pd.DataFrame, gt_res: str, gt_nres: str, pr: str, pn: str) -> dict[str, float]:
    return {
        "n": int(len(df)),
        "res": mae(df[gt_res].to_numpy(), df[pr].to_numpy()),
        "nres": mae(df[gt_nres].to_numpy(), df[pn].to_numpy()),
        "mean": 0.5
        * (
            mae(df[gt_res].to_numpy(), df[pr].to_numpy())
            + mae(df[gt_nres].to_numpy(), df[pn].to_numpy())
        ),
    }


def pooled_pack(df: pd.DataFrame, gt_res: str, gt_nres: str) -> dict[str, dict[str, float]]:
    return {
        "built_s": metrics(df, gt_res, gt_nres, "ghsl_frac_res", "ghsl_frac_nres"),
        "ridge": metrics(df, gt_res, gt_nres, "ae_frac_res" if "ae_frac_res" in df.columns else "ridge_frac_res",
                         "ae_frac_nres" if "ae_frac_nres" in df.columns else "ridge_frac_nres"),
        "pls": metrics(df, gt_res, gt_nres, "pls_frac_res", "pls_frac_nres"),
    }


def metro_pack(df: pd.DataFrame, gt_res: str, gt_nres: str, res_col: str, nres_col: str) -> list[dict[str, Any]]:
    rows = []
    for region in REGION_ORDER:
        sub = df[df["region"] == region]
        if not len(sub):
            continue
        bs = metrics(sub, gt_res, gt_nres, "ghsl_frac_res", "ghsl_frac_nres")
        if "ae_frac_res" in sub.columns:
            rg = metrics(sub, gt_res, gt_nres, "ae_frac_res", "ae_frac_nres")
        else:
            rg = metrics(sub, gt_res, gt_nres, res_col, nres_col)
        pl = metrics(sub, gt_res, gt_nres, "pls_frac_res", "pls_frac_nres")
        margin = min(rg["mean"], pl["mean"]) - bs["mean"]
        rows.append(
            {
                "region": region,
                "label": PRETTY[region],
                "bg": BG[region],
                "n": len(sub),
                "bs": bs,
                "rg": rg,
                "pl": pl,
                "margin": margin,
            }
        )
    return rows


def balance_yres(df: pd.DataFrame, seed: int = BALANCE_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    counts: dict[str, Any] = {}
    for region, sub in df.groupby("region", sort=True):
        res = sub.loc[sub["y_res"] == 1]
        nres = sub.loc[sub["y_res"] == 0]
        n = int(min(len(res), len(nres)))
        counts[str(region)] = {
            "res_before": int(len(res)),
            "nres_before": int(len(nres)),
            "n_per_class": n,
        }
        if n == 0:
            continue
        if len(res) > n:
            res = res.loc[rng.choice(res.index.to_numpy(), size=n, replace=False)]
        if len(nres) > n:
            nres = nres.loc[rng.choice(nres.index.to_numpy(), size=n, replace=False)]
        parts.append(pd.concat([res, nres], axis=0))
    out = pd.concat(parts, axis=0).reset_index(drop=True) if parts else df.iloc[0:0].copy()
    return out, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week6-run", type=Path, default=WEEK6_RUN)
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS)
    ap.add_argument("--seed", type=int, default=BALANCE_SEED)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week6_no_hotel_mae")
    args = ap.parse_args()

    n_cpus = os.cpu_count() or 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=7,
        n_cpus=n_cpus,
        exclude_labels=list(EXCLUDE_LABELS),
        week6_run=str(args.week6_run),
        min_cells=args.min_cells,
        seed=args.seed,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"cwd={Path.cwd()}\nargv={sys.argv}\ncpus={n_cpus}\nexclude={EXCLUDE_LABELS}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"python={sys.version}\npandas={pd.__version__}\nnumpy={np.__version__}\ncpus={n_cpus}\n"
    )
    log(run_dir, f"start no-hotel rescore · cpus={n_cpus} · exclude={EXCLUDE_LABELS}")

    # --- 1. load buildings ---
    write_progress(run_dir, stage="load_buildings", completed=0)
    bpath = args.week6_run / "artifacts" / "per_building_with_pls.parquet"
    buildings = pd.read_parquet(bpath)
    buildings["id"] = buildings["id"].astype(str)
    buildings["pls_covered"] = buildings["pls_covered"].fillna(False).astype(bool)
    base = buildings[buildings["shared_soft"] & buildings["pls_covered"]].copy()
    n_with_hotel = len(base)
    n_hotels = int((base["label"] == "hotel").sum())
    base = base[~base["label"].isin(EXCLUDE_LABELS)].copy()
    log(
        run_dir,
        f"shared∩PLS before exclude={n_with_hotel:,} · hotels dropped={n_hotels:,} · remain={len(base):,}",
    )
    write_progress(
        run_dir,
        stage="load_buildings",
        completed=1,
        n_before_exclude=n_with_hotel,
        n_hotels_dropped=n_hotels,
        n_after_exclude=len(base),
    )

    # --- 2. rebalance 1:1 y_res ---
    write_progress(run_dir, stage="balance", completed=1)
    bal, bal_counts = balance_yres(base, seed=args.seed)
    bal.to_parquet(run_dir / "artifacts" / "balanced_buildings_no_hotel.parquet", index=False)
    atomic_json(run_dir / "artifacts" / "balance_counts.json", bal_counts)
    log(
        run_dir,
        f"rebalanced 1:1 y_res · n={len(bal):,} · RES={int((bal.y_res==1).sum()):,} · "
        f"NRES={int((bal.y_res==0).sum()):,}",
    )
    write_progress(run_dir, stage="balance", completed=2, n_balanced=len(bal))

    # --- 3. building soft metrics ---
    write_progress(run_dir, stage="score_building", completed=2)
    # alias ridge cols for pooled_pack
    bscore = bal.rename(columns={"ae_frac_res": "ae_frac_res", "ae_frac_nres": "ae_frac_nres"})
    soft_b = {
        "built_s": metrics(bscore, "soft_gt_frac_res", "soft_gt_frac_nres", "ghsl_frac_res", "ghsl_frac_nres"),
        "ridge": metrics(bscore, "soft_gt_frac_res", "soft_gt_frac_nres", "ae_frac_res", "ae_frac_nres"),
        "pls": metrics(bscore, "soft_gt_frac_res", "soft_gt_frac_nres", "pls_frac_res", "pls_frac_nres"),
    }
    metro_b = metro_pack(bscore, "soft_gt_frac_res", "soft_gt_frac_nres", "ae_frac_res", "ae_frac_nres")
    wins_b = sum(1 for r in metro_b if r["bs"]["mean"] <= min(r["rg"]["mean"], r["pl"]["mean"]) + 1e-12)
    for name, d in soft_b.items():
        log(
            run_dir,
            f"building soft {name:8s} mean={d['mean']:.4f} res={d['res']:.4f} nres={d['nres']:.4f} n={d['n']}",
        )
    log(run_dir, f"building soft metro wins BUILT-S={wins_b}/{len(metro_b)}")
    write_progress(run_dir, stage="score_building", completed=3)

    # --- 4. load cells for 100m ---
    write_progress(run_dir, stage="load_cells", completed=3)
    want = set(bal["id"])
    cells = pd.read_parquet(
        BUILDING_CELLS,
        columns=["id", "pixel_lon", "pixel_lat", "utm_x", "utm_y", "inside_polygon"],
    )
    cells["id"] = cells["id"].astype(str)
    cells = cells[cells["inside_polygon"] & cells["id"].isin(want)].copy()
    cells = cells.merge(bal[["id", "region", "ghsl_frac_res", "ghsl_frac_nres"]], on="id", how="left")
    cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    log(run_dir, f"inside-polygon cells n={len(cells):,}")
    write_progress(run_dir, stage="load_cells", completed=4, n_cells=len(cells))

    # --- 5. join preds ---
    write_progress(run_dir, stage="join_preds", completed=4)
    ae = pd.read_parquet(
        AE_OOF,
        columns=[
            "pixel_id",
            "region",
            "pixel_lon",
            "pixel_lat",
            "frac_res_full",
            "frac_nres_full",
            "pred_frac_res_full",
            "pred_frac_nres_full",
        ],
    )
    ae["k"] = lonlat_key(ae["pixel_lon"].to_numpy(), ae["pixel_lat"].to_numpy())
    ae = ae.drop_duplicates("k")
    pls = pd.read_parquet(
        PLS_OOF,
        columns=["pixel_lon", "pixel_lat", "pred_frac_res_pls", "pred_frac_nres_pls"],
    )
    pls["k"] = lonlat_key(pls["pixel_lon"].to_numpy(), pls["pixel_lat"].to_numpy())
    pls = pls.drop_duplicates("k")
    j = cells.merge(
        ae[
            [
                "k",
                "pixel_id",
                "frac_res_full",
                "frac_nres_full",
                "pred_frac_res_full",
                "pred_frac_nres_full",
            ]
        ],
        on="k",
        how="left",
    )
    j = j.merge(pls[["k", "pred_frac_res_pls", "pred_frac_nres_pls"]], on="k", how="left")
    j = j.rename(
        columns={
            "frac_res_full": "soft_gt_frac_res",
            "frac_nres_full": "soft_gt_frac_nres",
            "pred_frac_res_full": "ridge_frac_res",
            "pred_frac_nres_full": "ridge_frac_nres",
            "pred_frac_res_pls": "pls_frac_res",
            "pred_frac_nres_pls": "pls_frac_nres",
        }
    )
    before = len(j)
    j = j[
        np.isfinite(j["soft_gt_frac_res"])
        & np.isfinite(j["ridge_frac_res"])
        & np.isfinite(j["pls_frac_res"])
        & np.isfinite(j["ghsl_frac_res"])
    ].copy()
    log(run_dir, f"cells joined n={len(j):,} (dropped {before - len(j):,})")
    j.to_parquet(run_dir / "checkpoints" / "cells_joined.parquet", index=False)
    write_progress(run_dir, stage="join_preds", completed=5, n_cells_joined=len(j))

    # --- 6. tiles ---
    write_progress(run_dir, stage="agg_100m", completed=5)
    tiles_map = pd.read_parquet(TILE_CELLS, columns=["pixel_id", "region", "tile_id", "tile_i", "tile_j"])
    d100 = j.merge(tiles_map, on=["pixel_id", "region"], how="left")
    miss = d100["tile_id"].isna()
    if miss.any():
        ti = np.floor(d100.loc[miss, "utm_x"].to_numpy(dtype=float) / 100.0).astype(np.int64)
        tj = np.floor(d100.loc[miss, "utm_y"].to_numpy(dtype=float) / 100.0).astype(np.int64)
        d100.loc[miss, "tile_i"] = ti
        d100.loc[miss, "tile_j"] = tj
        d100.loc[miss, "tile_id"] = (
            d100.loc[miss, "region"].astype(str) + "|" + ti.astype(str) + "|" + tj.astype(str)
        )
    d100 = d100[d100["tile_id"].notna()].copy()
    agg = {
        "n_cells": ("pixel_id", "size"),
        "n_buildings": ("id", "nunique"),
        "soft_gt_frac_res": ("soft_gt_frac_res", "mean"),
        "soft_gt_frac_nres": ("soft_gt_frac_nres", "mean"),
        "ridge_frac_res": ("ridge_frac_res", "mean"),
        "ridge_frac_nres": ("ridge_frac_nres", "mean"),
        "pls_frac_res": ("pls_frac_res", "mean"),
        "pls_frac_nres": ("pls_frac_nres", "mean"),
        "ghsl_frac_res": ("ghsl_frac_res", "mean"),
        "ghsl_frac_nres": ("ghsl_frac_nres", "mean"),
    }
    tiles = d100.groupby(["tile_id", "region"], sort=False).agg(**agg).reset_index()
    tiles = tiles.loc[tiles["n_cells"] >= args.min_cells].copy()
    tiles.to_parquet(run_dir / "artifacts" / "tiles_100m_no_hotel.parquet", index=False)
    log(run_dir, f"100 m tiles min>={args.min_cells}: n={len(tiles):,}")
    write_progress(run_dir, stage="agg_100m", completed=6, n_tiles=len(tiles))

    # --- 7. score 100m ---
    write_progress(run_dir, stage="score_100m", completed=6)
    soft_100 = {
        "built_s": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "ghsl_frac_res", "ghsl_frac_nres"),
        "ridge": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "ridge_frac_res", "ridge_frac_nres"),
        "pls": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "pls_frac_res", "pls_frac_nres"),
    }
    metro_100 = metro_pack(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "ridge_frac_res", "ridge_frac_nres")
    wins_100 = sum(1 for r in metro_100 if r["bs"]["mean"] <= min(r["rg"]["mean"], r["pl"]["mean"]) + 1e-12)
    for name, d in soft_100.items():
        log(
            run_dir,
            f"100m soft {name:8s} mean={d['mean']:.4f} res={d['res']:.4f} nres={d['nres']:.4f} n={d['n']}",
        )
    log(run_dir, f"100m soft metro wins BUILT-S={wins_100}/{len(metro_100)}")

    # baseline comparison (with hotels) from prior Section 3
    baseline_b = args.week6_run / "artifacts" / "section3_ridge_pls_metrics.json"
    baseline_100 = (
        ROOT
        / "outputs"
        / "week6_overture_polygon_mae_100m"
        / "20261110_044005"
        / "artifacts"
        / "section3_100m_metrics.json"
    )
    comparison: dict[str, Any] = {"exclude": list(EXCLUDE_LABELS)}
    if baseline_b.exists():
        bb = json.loads(baseline_b.read_text())
        comparison["building_with_hotel"] = bb.get("soft_pooled")
        comparison["building_with_hotel_n"] = bb.get("n_shared")
        comparison["building_with_hotel_wins"] = bb.get("soft_wins")
    if baseline_100.exists():
        b1 = json.loads(baseline_100.read_text())
        comparison["tiles100_with_hotel"] = b1.get("soft_pooled")
        comparison["tiles100_with_hotel_n"] = b1.get("n_tiles")
        comparison["tiles100_with_hotel_wins"] = b1.get("soft_wins")

    out = {
        "exclude_labels": list(EXCLUDE_LABELS),
        "n_hotels_dropped_from_shared": n_hotels,
        "n_shared_before_exclude": n_with_hotel,
        "n_buildings_balanced": len(bal),
        "n_tiles": len(tiles),
        "min_cells": args.min_cells,
        "seed": args.seed,
        "building_soft": soft_b,
        "building_wins": wins_b,
        "metro_building": metro_b,
        "tiles100_soft": soft_100,
        "tiles100_wins": wins_100,
        "metro_100": metro_100,
        "comparison_vs_with_hotel": comparison,
        "week6_run": str(args.week6_run),
    }
    atomic_json(run_dir / "artifacts" / "metrics_no_hotel.json", out)

    # CSV summary
    rows = []
    for scale, pack, n in [
        ("building_soft", soft_b, len(bal)),
        ("100m_min20", soft_100, len(tiles)),
    ]:
        for model, d in pack.items():
            rows.append(
                {
                    "scale": scale,
                    "model": model,
                    "n": d["n"],
                    "mae_res": d["res"],
                    "mae_nres": d["nres"],
                    "mae_mean": d["mean"],
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "artifacts" / "mae_summary.csv", index=False)

    summary = {
        "status": "completed",
        "exclude_labels": list(EXCLUDE_LABELS),
        "n_hotels_dropped": n_hotels,
        "n_buildings": len(bal),
        "n_tiles": len(tiles),
        "building_soft": soft_b,
        "building_wins": wins_b,
        "tiles100_soft": soft_100,
        "tiles100_wins": wins_100,
        "comparison_vs_with_hotel": comparison,
        "runtime_seconds": round(time.time() - t0, 1),
        "artifacts": {
            "metrics": "artifacts/metrics_no_hotel.json",
            "mae_summary": "artifacts/mae_summary.csv",
            "buildings": "artifacts/balanced_buildings_no_hotel.parquet",
            "tiles": "artifacts/tiles_100m_no_hotel.parquet",
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(run_dir, status="completed", stage="done", completed=7, total=7, percent_complete=100.0)
    log(
        run_dir,
        f"DONE building BS/Rg/Pl={soft_b['built_s']['mean']:.3f}/"
        f"{soft_b['ridge']['mean']:.3f}/{soft_b['pls']['mean']:.3f} · "
        f"100m BS/Rg/Pl={soft_100['built_s']['mean']:.3f}/"
        f"{soft_100['ridge']['mean']:.3f}/{soft_100['pls']['mean']:.3f}",
    )
    log(run_dir, f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
