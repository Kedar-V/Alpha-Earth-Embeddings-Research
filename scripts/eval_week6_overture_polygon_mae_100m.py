#!/usr/bin/env python3
"""Week 6: 100 m UTM tile MAE with soft-argmax 1:1 on cells AND tiles (Section 3).

Building universe: ALL Track-A buildings with soft ∩ Ridge ∩ PLS ∩ BUILT-S
coverage (no building y_res 1:1). Inside-polygon cells are labeled RES if
soft_gt_frac_res ≥ soft_gt_frac_nres and undersampled 1:1 per metro (seed 42),
then mean-pooled into per-metro UTM 100 m tiles (min ≥ 20 cells). Tiles are
again soft-argmax labeled and 1:1 balanced per metro (same seed).

  soft GT / Ridge / PLS  = cell-level week-4 values (frac_*_full / OOF)
  BUILT-S                = building-zonal GHSL fractions broadcast onto the
                           building's inside-polygon cells (polygon path), then
                           mean-pooled in the tile

Building-panel metrics stay on the separate y_res-balanced week-6 run.
Outputs under:
  outputs/week6_overture_polygon_mae_100m/<timestamp>/
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEEK6_BUILDING_RUN = ROOT / "outputs" / "week6_overture_polygon_mae" / "20261110_041557"
GHSL_FULL = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_034500"
    / "checkpoints"
    / "ghsl_by_building.parquet"
)
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
BUILDING_CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
AE_OOF = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
PLS_OOF = ROOT / "outputs" / "week4_pls_loro" / "oof_predictions.parquet"
TILE_CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
# Soft-argmax tile 1:1 needs min≥10: at min≥20, Atlanta/Phoenix/Seattle have 0 RES tiles
# after cell balance (tile-mean frac always tips NRES).
MIN_CELLS = 10
LONLAT_DECIMALS = 7
BALANCE_SEED = 42

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
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
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


def score_rows(df: pd.DataFrame, eval_name: str, scale: str) -> list[dict[str, Any]]:
    rows = []
    for model, pr, pn in [
        ("ghsl_built_s", "ghsl_frac_res", "ghsl_frac_nres"),
        ("ae_ridge", "ridge_frac_res", "ridge_frac_nres"),
        ("ae_pls", "pls_frac_res", "pls_frac_nres"),
    ]:
        m = metrics(df, "soft_gt_frac_res", "soft_gt_frac_nres", pr, pn)
        rows.append(
            {
                "eval": eval_name,
                "scale": scale,
                "model": model,
                "n": m["n"],
                "mae_frac_res": m["res"],
                "mae_frac_nres": m["nres"],
                "mae_mean_res_nres": m["mean"],
            }
        )
    return rows


def balance_soft_argmax(
    df: pd.DataFrame,
    *,
    label_col: str,
    seed: int = BALANCE_SEED,
    unit: str = "row",
) -> pd.DataFrame:
    """Per-metro 1:1 undersample by soft-argmax class (frac_res ≥ frac_nres → RES)."""
    d = df.copy()
    d[label_col] = (
        d["soft_gt_frac_res"].to_numpy(dtype=float) >= d["soft_gt_frac_nres"].to_numpy(dtype=float)
    ).astype(np.int8)
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    counts: dict[str, dict[str, int]] = {}
    for region, sub in d.groupby("region", sort=True):
        res = sub.loc[sub[label_col] == 1]
        nres = sub.loc[sub[label_col] == 0]
        n = int(min(len(res), len(nres)))
        counts[str(region)] = {"res_before": int(len(res)), "nres_before": int(len(nres)), "n_per_class": n}
        if n == 0:
            continue
        if len(res) > n:
            res = res.loc[rng.choice(res.index.to_numpy(), size=n, replace=False)]
        if len(nres) > n:
            nres = nres.loc[rng.choice(nres.index.to_numpy(), size=n, replace=False)]
        bal = pd.concat([res, nres], ignore_index=True)
        bal["balance_n_per_class"] = n
        parts.append(bal)
    if not parts:
        raise RuntimeError(f"No balanced {unit} regions produced")
    out = pd.concat(parts, ignore_index=True)
    out.attrs["balance_counts"] = counts
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def metro_block(tiles: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    metro_soft: list[dict[str, Any]] = []
    for region in REGION_ORDER:
        sub = tiles[tiles["region"] == region]
        if not len(sub):
            continue
        bs = metrics(sub, "soft_gt_frac_res", "soft_gt_frac_nres", "ghsl_frac_res", "ghsl_frac_nres")
        rg = metrics(sub, "soft_gt_frac_res", "soft_gt_frac_nres", "ridge_frac_res", "ridge_frac_nres")
        pl = metrics(sub, "soft_gt_frac_res", "soft_gt_frac_nres", "pls_frac_res", "pls_frac_nres")
        margin = min(rg["mean"], pl["mean"]) - bs["mean"]
        metro_soft.append(
            {
                "region": region,
                "label": PRETTY[region],
                "bg": BG[region],
                "n": len(sub),
                "n_res": int((sub["tile_y"] == 1).sum()) if "tile_y" in sub.columns else None,
                "n_nres": int((sub["tile_y"] == 0).sum()) if "tile_y" in sub.columns else None,
                "bs": bs,
                "rg": rg,
                "pl": pl,
                "margin": margin,
            }
        )
    wins = sum(1 for r in metro_soft if r["bs"]["mean"] <= min(r["rg"]["mean"], r["pl"]["mean"]) + 1e-12)
    return metro_soft, wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-run", type=Path, default=WEEK6_BUILDING_RUN, help="y_res-balanced building run (panel only)")
    ap.add_argument("--ghsl-from", type=Path, default=GHSL_FULL, help="full-building GHSL zonal cache")
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS)
    ap.add_argument("--balance-seed", type=int, default=BALANCE_SEED)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week6_overture_polygon_mae_100m")
    # back-compat alias
    ap.add_argument("--week6-run", type=Path, default=None)
    args = ap.parse_args()
    if args.week6_run is not None:
        args.building_run = args.week6_run

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
        total=8,
        n_cpus=n_cpus,
        min_cells=args.min_cells,
        balance_seed=args.balance_seed,
        building_run=str(args.building_run),
        ghsl_from=str(args.ghsl_from),
        protocol="cell_and_tile_soft_argmax_1to1_v1",
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"cwd={Path.cwd()}\nargv={sys.argv}\ncpus={n_cpus}\n"
        f"building_run={args.building_run}\nghsl={args.ghsl_from}\n"
        f"protocol=cell_and_tile_soft_argmax_1to1_v1\n"
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.executable}",
                f"cpus={n_cpus}",
                f"min_cells={args.min_cells}",
                f"balance_seed={args.balance_seed}",
                f"building_run={args.building_run}",
                f"ghsl_from={args.ghsl_from}",
            ]
        )
        + "\n"
    )
    log(
        run_dir,
        f"start week6 100m soft-argmax cell+tile 1:1 · cpus={n_cpus} · min_cells={args.min_cells} · seed={args.balance_seed}",
    )

    # --- 1. unbalanced buildings + full GHSL ---
    write_progress(run_dir, stage="load_buildings", completed=0)
    buildings = pd.read_parquet(BUILDINGS, columns=["id", "region"])
    buildings["id"] = buildings["id"].astype(str)
    ghsl = pd.read_parquet(args.ghsl_from)
    ghsl["id"] = ghsl["id"].astype(str)
    ghsl = ghsl.loc[ghsl["ghsl_covered"].fillna(False).astype(bool), ["id", "ghsl_frac_res", "ghsl_frac_nres"]].copy()
    buildings = buildings.merge(ghsl, on="id", how="inner")
    want = set(buildings["id"])
    log(run_dir, f"unbalanced GHSL-covered buildings n={len(buildings):,} (no y_res 1:1)")
    write_progress(run_dir, stage="load_buildings", completed=1, n_buildings=len(buildings))

    # --- 2. inside-polygon cells ---
    write_progress(run_dir, stage="load_cells", completed=1)
    cells = pd.read_parquet(
        BUILDING_CELLS,
        columns=["id", "pixel_lon", "pixel_lat", "utm_x", "utm_y", "inside_polygon"],
    )
    cells["id"] = cells["id"].astype(str)
    cells = cells[cells["inside_polygon"] & cells["id"].isin(want)].copy()
    cells = cells.merge(buildings[["id", "region", "ghsl_frac_res", "ghsl_frac_nres"]], on="id", how="left")
    cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    log(run_dir, f"inside-polygon cells n={len(cells):,}")
    write_progress(run_dir, stage="load_cells", completed=2, n_cells=len(cells))

    # --- 3. join AE ridge + soft GT + PLS ---
    write_progress(run_dir, stage="join_preds", completed=2)
    ae = pd.read_parquet(
        AE_OOF,
        columns=[
            "pixel_id",
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
    n_buildings_cov = int(j["id"].nunique())
    log(
        run_dir,
        f"cells soft∩ridge∩pls∩ghsl n={len(j):,} (dropped {before - len(j):,}) · buildings={n_buildings_cov:,}",
    )
    j.to_parquet(run_dir / "checkpoints" / "cells_joined.parquet", index=False)
    write_progress(
        run_dir,
        stage="join_preds",
        completed=3,
        n_cells_joined=len(j),
        n_buildings_covered=n_buildings_cov,
    )

    # --- 3b. soft-argmax 1:1 balance cells ---
    write_progress(run_dir, stage="balance_cells", completed=3)
    n_cells_pre = len(j)
    j = balance_soft_argmax(j, label_col="cell_y", seed=args.balance_seed, unit="cell")
    cell_bal_counts = j.attrs.get("balance_counts", {})
    atomic_json(
        run_dir / "artifacts" / "cell_balance_counts.json",
        {
            "seed": args.balance_seed,
            "rule": "soft_gt_frac_res >= soft_gt_frac_nres → RES",
            "n_before": n_cells_pre,
            "n_after": len(j),
            "regions": cell_bal_counts,
        },
    )
    for region, sub in j.groupby("region"):
        n1 = int((sub["cell_y"] == 1).sum())
        n0 = int((sub["cell_y"] == 0).sum())
        if n1 != n0:
            raise RuntimeError(f"cell balance failed {region}: RES={n1} NRES={n0}")
    log(
        run_dir,
        f"balanced cells {n_cells_pre:,} → {len(j):,} "
        f"(soft-argmax 1:1 per metro, seed={args.balance_seed})",
    )
    j.to_parquet(run_dir / "checkpoints" / "cells_balanced.parquet", index=False)
    write_progress(run_dir, stage="balance_cells", completed=4, n_cells_balanced=len(j))

    # --- 4. assign UTM 100 m tile_id ---
    write_progress(run_dir, stage="join_tiles", completed=4)
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
        log(run_dir, f"tile_id from utm floor for {int(miss.sum()):,} cells lacking week4 map")
    d100 = d100[d100["tile_id"].notna()].copy()
    log(run_dir, f"cells with tile_id n={len(d100):,}")
    write_progress(run_dir, stage="join_tiles", completed=5, n_cells_tiled=len(d100))

    # --- 5. aggregate tiles (pre-balance) ---
    write_progress(run_dir, stage="agg_100m", completed=5)
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
    tiles_all = d100.groupby(["tile_id", "region"], sort=False).agg(**agg).reset_index()
    tiles_pre = tiles_all.loc[tiles_all["n_cells"] >= args.min_cells].copy()
    log(
        run_dir,
        f"100 m tiles (from cell-balanced cells): all={len(tiles_all):,} · min>={args.min_cells} → {len(tiles_pre):,} "
        f"(mean cells/tile={tiles_pre['n_cells'].mean():.1f})",
    )
    tiles_all.to_parquet(run_dir / "artifacts" / "tiles_100m_all.parquet", index=False)
    tiles_pre.to_parquet(run_dir / "artifacts" / "tiles_100m_prebalance.parquet", index=False)
    write_progress(run_dir, stage="agg_100m", completed=6, n_tiles_prebalance=len(tiles_pre))

    # --- 6. soft-argmax 1:1 balance tiles ---
    write_progress(run_dir, stage="balance_tiles", completed=6)
    tiles = balance_soft_argmax(tiles_pre, label_col="tile_y", seed=args.balance_seed, unit="tile")
    bal_counts = tiles.attrs.get("balance_counts", {})
    atomic_json(
        run_dir / "artifacts" / "tile_balance_counts.json",
        {
            "seed": args.balance_seed,
            "rule": "soft_gt_frac_res >= soft_gt_frac_nres → RES",
            "regions": bal_counts,
            "upstream": "cells also soft-argmax 1:1 before pooling",
        },
    )
    for region, sub in tiles.groupby("region"):
        n1 = int((sub["tile_y"] == 1).sum())
        n0 = int((sub["tile_y"] == 0).sum())
        if n1 != n0:
            raise RuntimeError(f"tile balance failed {region}: RES={n1} NRES={n0}")
    log(
        run_dir,
        f"balanced tiles {len(tiles_pre):,} → {len(tiles):,} "
        f"(soft-argmax 1:1 per metro, seed={args.balance_seed})",
    )
    tiles.to_parquet(run_dir / "artifacts" / "tiles_100m.parquet", index=False)
    write_progress(run_dir, stage="balance_tiles", completed=7, n_tiles=len(tiles))

    # --- 7. score ---
    write_progress(run_dir, stage="score", completed=7)
    eval_name = f"100m_min{args.min_cells}_cell_tile_softargmax_1to1"
    rows = score_rows(tiles, eval_name, "100m")
    metro_rows = []
    for region, sub in tiles.groupby("region", sort=False):
        for r in score_rows(sub, eval_name, "100m"):
            metro_rows.append({**r, "region": region})
    summary = pd.DataFrame(rows)
    metro = pd.DataFrame(metro_rows)
    summary.to_csv(run_dir / "artifacts" / "mae_summary_100m.csv", index=False)
    metro.to_csv(run_dir / "artifacts" / "mae_by_metro_100m.csv", index=False)
    for r in summary.itertuples():
        log(
            run_dir,
            f"{r.scale} {r.eval} {r.model:14s} mae_rn={r.mae_mean_res_nres:.4f} "
            f"res={r.mae_frac_res:.4f} nres={r.mae_frac_nres:.4f} n={r.n}",
        )

    pooled = {
        "built_s": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "ghsl_frac_res", "ghsl_frac_nres"),
        "ridge": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "ridge_frac_res", "ridge_frac_nres"),
        "pls": metrics(tiles, "soft_gt_frac_res", "soft_gt_frac_nres", "pls_frac_res", "pls_frac_nres"),
    }
    metro_soft, wins = metro_block(tiles)
    out_metrics = {
        "n_tiles": len(tiles),
        "n_tiles_prebalance": len(tiles_pre),
        "n_buildings_covered": n_buildings_cov,
        "n_cells": len(d100),
        "min_cells": args.min_cells,
        "balance_seed": args.balance_seed,
        "balance_rule": "cell_and_tile_soft_argmax_1to1",
        "n_cells_balanced": len(j),
        "n_cells_prebalance": n_cells_pre,
        "soft_pooled": pooled,
        "soft_wins": wins,
        "metro_soft": metro_soft,
        "building_run": str(args.building_run),
        "ghsl_from": str(args.ghsl_from),
        "protocol": "cell_and_tile_soft_argmax_1to1_v1",
    }
    atomic_json(run_dir / "artifacts" / "section3_100m_metrics.json", out_metrics)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    order = ["ghsl_built_s", "ae_ridge", "ae_pls"]
    labels = {"ghsl_built_s": "GHS-BUILT-S", "ae_ridge": "AE Ridge", "ae_pls": "AE PLS"}
    colors = {"ghsl_built_s": "#C2691A", "ae_ridge": "#1F6FA8", "ae_pls": "#4AA0DB"}
    s = summary.set_index("model").loc[order]
    ax.barh([labels[m] for m in order], s["mae_mean_res_nres"], color=[colors[m] for m in order])
    for y, v in enumerate(s["mae_mean_res_nres"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=11)
    ax.set_xlabel("MAE mean(res, nres)")
    ax.set_title(f"Week 6 · 100 m cell+tile soft-argmax 1:1 (min {args.min_cells})")
    ax.set_xlim(0, max(0.4, float(s["mae_mean_res_nres"].max()) * 1.25))
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mae_100m_pooled.png", dpi=140)
    plt.close(fig)

    summary_payload = {
        "status": "completed",
        "protocol": "cell_and_tile_soft_argmax_1to1_v1",
        "n_tiles": len(tiles),
        "n_tiles_prebalance": len(tiles_pre),
        "n_buildings_covered": n_buildings_cov,
        "n_cells": int(len(d100)),
        "n_cells_balanced": len(j),
        "n_cells_prebalance": n_cells_pre,
        "min_cells": args.min_cells,
        "balance_seed": args.balance_seed,
        "soft_pooled": pooled,
        "soft_wins": wins,
        "runtime_seconds": round(time.time() - t0, 1),
        "artifacts": {
            "tiles": "artifacts/tiles_100m.parquet",
            "tiles_prebalance": "artifacts/tiles_100m_prebalance.parquet",
            "cell_balance_counts": "artifacts/cell_balance_counts.json",
            "tile_balance_counts": "artifacts/tile_balance_counts.json",
            "metrics": "artifacts/section3_100m_metrics.json",
            "mae_summary": "artifacts/mae_summary_100m.csv",
            "mae_by_metro": "artifacts/mae_by_metro_100m.csv",
        },
    }
    atomic_json(run_dir / "summary.json", summary_payload)
    write_progress(
        run_dir,
        status="completed",
        stage="done",
        completed=8,
        total=8,
        percent_complete=100.0,
        n_tiles=len(tiles),
        n_cells_balanced=len(j),
    )
    log(
        run_dir,
        f"DONE tiles={len(tiles):,} soft mean BS={pooled['built_s']['mean']:.3f} "
        f"Ridge={pooled['ridge']['mean']:.3f} PLS={pooled['pls']['mean']:.3f} wins={wins}/10",
    )
    log(run_dir, f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
