#!/usr/bin/env python3
"""Week 8: multi-scale AE rings + train-where-you-score.

Training: all soft-covered Track-A buildings (and tile soft GT available).
Eval holdout: locked Section 3/4 cohorts only
  buildings n=71,497 (ridge_section3_building_ids.parquet)
  tiles     n=25,220 (ridge_section3_tile_ids.parquet)

Per building (or tile centroid):
  - center = mean AE of inside cells / tile cells (64)
  - rings (0,50], (50,100], (100,200] m: neighbor mean+std (3×128)

Ridge LORO by metro on soft fractions; Section 5 metrics use Section 3/4 IDs only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
W6_BUILD = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)
W6_100M = ROOT / "outputs" / "week6_overture_polygon_mae_100m" / "20261110_062739"
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
COHORT_TILE = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_tile_ids.parquet"

BANDS = [f"A{i:02d}" for i in range(64)]
RINGS_M = (50.0, 100.0, 200.0)
N_FEATURES = 64 + 3 * (64 + 64)
LONLAT_DECIMALS = 7

METRO_META = [
    ("Boston_NE", "Boston NE", "#ECE8F4"),
    ("NYC_NE", "NYC NE", "#ECE8F4"),
    ("Atlanta_SE", "Atlanta SE", "#E2EFE6"),
    ("Houston_S", "Houston S", "#F3EDD5"),
    ("Chicago_MW", "Chicago MW", "#F3E3E7"),
    ("Phoenix_SW", "Phoenix SW", "#EFE4D9"),
    ("Denver_W", "Denver W", "#E1ECEE"),
    ("LA_W", "LA W", "#E1ECEE"),
    ("SF_W", "SF W", "#E1ECEE"),
    ("Seattle_NW", "Seattle NW", "#E9EDDC"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    if "_t0" not in prev and "_t0" not in kwargs:
        kwargs["_t0"] = time.time()
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0.0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def lonlat_key(lon, lat, nd: int = LONLAT_DECIMALS) -> np.ndarray:
    lon_s = np.round(np.asarray(lon, dtype=np.float64), nd).astype(str)
    lat_s = np.round(np.asarray(lat, dtype=np.float64), nd).astype(str)
    return np.char.add(lon_s.astype(object), np.char.add("_", lat_s.astype(object)))


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if int(m.sum()) == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    r = float(mean_absolute_error(y_res[m], p_res[m]))
    n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": r, "nres": n, "mean": 0.5 * (r + n)}


def feat_colnames() -> list[str]:
    cols = [f"c_{b}" for b in BANDS]
    for r in (50, 100, 200):
        cols += [f"r{r}_mean_{b}" for b in BANDS]
        cols += [f"r{r}_std_{b}" for b in BANDS]
    return cols


def ring_features_for_points(
    query_xy: np.ndarray,
    cell_xy: np.ndarray,
    cell_emb: np.ndarray,
    chunk: int = 4096,
) -> np.ndarray:
    """query_xy (N,2), cell_xy (M,2), cell_emb (M,64) → (N, N_FEATURES)."""
    n = query_xy.shape[0]
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    if n == 0 or cell_xy.shape[0] == 0:
        return out
    tree = cKDTree(cell_xy)
    bounds = (0.0,) + RINGS_M
    # nearest cell embedding as center proxy when no explicit center passed
    # (overwritten by caller for buildings with inside mean)
    _, nn = tree.query(query_xy, k=1)
    out[:, :64] = cell_emb[nn]

    # Chunk query_ball_point so peak neighbor-list memory stays bounded.
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        neighbors = tree.query_ball_point(query_xy[start:stop], r=RINGS_M[-1])
        for j, pts in enumerate(neighbors):
            if not pts:
                continue
            i = start + j
            pts_a = np.asarray(pts, dtype=np.int64)
            d = np.linalg.norm(cell_xy[pts_a] - query_xy[i], axis=1)
            e = cell_emb[pts_a]
            off = 64
            for ri in range(3):
                lo, hi = bounds[ri], bounds[ri + 1]
                m = (d > lo) & (d <= hi)
                if np.any(m):
                    block = e[m]
                    out[i, off : off + 64] = block.mean(axis=0)
                    if block.shape[0] > 1:
                        out[i, off + 64 : off + 128] = block.std(axis=0)
                off += 128
    return out


CELL_SHARD_DIR = ROOT / "outputs" / "week8_ae_rings_loro" / "_shared" / "cell_shards"
CELL_SHARD_COLS = ["id", "utm_x", "utm_y", "inside_polygon", "crs", "pixel_lon", "pixel_lat", *BANDS]


def ensure_cell_shards(run_dir: Path, regions: list[str]) -> Path:
    """One-time per-metro cell shards so workers never reload the 653MB cells table."""
    CELL_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    missing = [r for r in regions if not (CELL_SHARD_DIR / f"{r}.parquet").exists()]
    if not missing:
        log(run_dir, f"cell shards ready · {CELL_SHARD_DIR}")
        return CELL_SHARD_DIR

    write_progress(
        run_dir,
        status="running",
        stage="shard_cells",
        completed=len(regions) - len(missing),
        total=len(regions),
        failed=0,
    )
    log(run_dir, f"Sharding cells for {len(missing)} metros (serial, memory-safe)")
    bmeta = pd.read_parquet(BUILDINGS, columns=["id", "region"], filters=[("region", "in", missing)])
    bmeta["id"] = bmeta["id"].astype(str)
    id_to_region = dict(zip(bmeta["id"], bmeta["region"]))
    wanted = set(id_to_region)

    # Stream row-groups so we never hold the full 4.6M×64 matrix.
    import pyarrow.parquet as pq

    tables: dict[str, list] = {r: [] for r in missing}
    pf = pq.ParquetFile(CELLS)
    keep = [c for c in CELL_SHARD_COLS if c in pf.schema_arrow.names]
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i, columns=keep).to_pandas()
        rg["id"] = rg["id"].astype(str)
        rg = rg[rg["id"].isin(wanted)]
        if rg.empty:
            continue
        rg["region"] = rg["id"].map(id_to_region)
        for region, part in rg.groupby("region", sort=False):
            tables[str(region)].append(part.drop(columns=["region"]))
        log(run_dir, f"shard cells row_group {i + 1}/{pf.num_row_groups}")

    for region in missing:
        parts = tables[region]
        if not parts:
            raise RuntimeError(f"no cells for shard {region}")
        out = pd.concat(parts, ignore_index=True)
        path = CELL_SHARD_DIR / f"{region}.parquet"
        tmp = path.with_suffix(".tmp.parquet")
        out.to_parquet(tmp, index=False)
        tmp.replace(path)
        log(run_dir, f"shard {region}: cells={len(out):,}")
        write_progress(
            run_dir,
            status="running",
            stage="shard_cells",
            completed=sum(1 for r in regions if (CELL_SHARD_DIR / f"{r}.parquet").exists()),
            total=len(regions),
            failed=0,
        )
    return CELL_SHARD_DIR


def build_metro_building_features(region: str, out_path: Path, shard_dir: str) -> dict[str, Any]:
    t0 = time.time()
    bmeta = pd.read_parquet(
        BUILDINGS,
        columns=["id", "region", "lon", "lat", "crs"],
        filters=[("region", "==", region)],
    )
    bmeta["id"] = bmeta["id"].astype(str)
    if bmeta.empty:
        raise RuntimeError(f"no buildings {region}")

    # Train on all soft-covered buildings in this metro; Section 3/4 cohort
    # is applied only at scoring time (score_buildings).
    w6 = pd.read_parquet(
        W6_BUILD,
        columns=[
            "id",
            "region",
            "soft_gt_frac_res",
            "soft_gt_frac_nres",
            "soft_gt_covered",
            "ae_frac_res",
            "ae_frac_nres",
            "ghsl_frac_res",
            "ghsl_frac_nres",
            "pls_frac_res",
            "pls_frac_nres",
            "y_res",
        ],
        filters=[("region", "==", region), ("soft_gt_covered", "==", True)],
    )
    w6["id"] = w6["id"].astype(str)

    cells = pd.read_parquet(Path(shard_dir) / f"{region}.parquet")
    cells["id"] = cells["id"].astype(str)
    if cells.empty:
        raise RuntimeError(f"no cells {region}")

    crs = str(cells["crs"].iloc[0])
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    bx, by = to_utm.transform(bmeta["lon"].to_numpy(), bmeta["lat"].to_numpy())
    bmeta = bmeta.assign(utm_x=bx, utm_y=by)

    cell_xy = cells[["utm_x", "utm_y"]].to_numpy(np.float64)
    cell_emb = cells[BANDS].to_numpy(np.float32)

    # center = mean emb of inside_polygon cells per building
    inside = cells[cells["inside_polygon"]]
    center = inside.groupby("id", sort=False)[BANDS].mean().reset_index()

    # only buildings with soft GT + center emb
    panel = w6.merge(bmeta[["id", "utm_x", "utm_y"]], on="id", how="inner")
    panel = panel.merge(center, on="id", how="inner")
    query_xy = panel[["utm_x", "utm_y"]].to_numpy(np.float64)
    feats = ring_features_for_points(query_xy, cell_xy, cell_emb)
    # overwrite center with true inside mean
    feats[:, :64] = panel[BANDS].to_numpy(np.float32)

    cols = feat_colnames()
    feat_df = pd.DataFrame(feats, columns=cols)
    out = pd.concat(
        [
            panel[
                [
                    "id",
                    "region",
                    "soft_gt_frac_res",
                    "soft_gt_frac_nres",
                    "ae_frac_res",
                    "ae_frac_nres",
                    "ghsl_frac_res",
                    "ghsl_frac_nres",
                    "pls_frac_res",
                    "pls_frac_nres",
                    "y_res",
                ]
            ].reset_index(drop=True),
            feat_df,
        ],
        axis=1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return {
        "region": region,
        "n_buildings": int(len(out)),
        "n_cells": int(len(cells)),
        "seconds": round(time.time() - t0, 1),
        "path": str(out_path),
    }


def _metro_job(args: tuple[str, str, str]) -> dict[str, Any]:
    region, out_s, shard_dir = args
    return build_metro_building_features(region, Path(out_s), shard_dir)


def stage_building_features(run_dir: Path, regions: list[str], workers: int) -> Path:
    shard_dir = ensure_cell_shards(run_dir, regions)
    feat_dir = run_dir / "checkpoints" / "building_ring_features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "failures").mkdir(parents=True, exist_ok=True)
    jobs = []
    for region in regions:
        out = feat_dir / f"{region}.parquet"
        if out.exists() and out.stat().st_size > 1000:
            log(run_dir, f"features skip {region}")
            continue
        jobs.append((region, str(out), str(shard_dir)))

    # Cap concurrency: each metro still holds ~hundreds of MB of emb + KDTree.
    # Africa ingest often co-resides on this box; 2 workers is the safe default.
    feat_workers = max(1, min(workers, 2))
    write_progress(
        run_dir,
        status="running",
        stage="building_features",
        completed=len(regions) - len(jobs),
        total=len(regions),
        failed=0,
        workers=feat_workers,
    )
    done = len(regions) - len(jobs)
    failed = 0
    if jobs:
        log(run_dir, f"Building ring features · {len(jobs)} metros · workers={feat_workers}")
        with ProcessPoolExecutor(max_workers=feat_workers) as pool:
            futs = {pool.submit(_metro_job, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                region = futs[fut]
                try:
                    s = fut.result()
                    done += 1
                    log(
                        run_dir,
                        f"features {s['region']}: buildings={s['n_buildings']:,} "
                        f"cells={s['n_cells']:,} in {s['seconds']}s",
                    )
                    atomic_json(run_dir / "checkpoints" / f"feat_{region}.json", s)
                except Exception as e:
                    failed += 1
                    log(run_dir, f"FEATURES FAIL {region}: {e}")
                    with (run_dir / "failures" / "features.jsonl").open("a") as f:
                        f.write(json.dumps({"region": region, "error": str(e), "at": utc_now()}) + "\n")
                write_progress(
                    run_dir,
                    status="running",
                    stage="building_features",
                    completed=done,
                    total=len(regions),
                    failed=failed,
                    workers=feat_workers,
                )
    if failed:
        raise RuntimeError(f"{failed} feature builds failed")
    return feat_dir


def load_building_table(feat_dir: Path, regions: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(feat_dir / f"{r}.parquet") for r in regions]
    return pd.concat(parts, ignore_index=True)


def train_loro_buildings(df: pd.DataFrame, run_dir: Path, alpha: float) -> pd.DataFrame:
    cols = feat_colnames()
    regions = sorted(df["region"].unique())
    oof = np.full((len(df), 2), np.nan, dtype=np.float32)
    fold_metrics = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_progress(run_dir, status="running", stage="train_loro", completed=0, total=len(regions), failed=0)

    for i, hold in enumerate(regions):
        t0 = time.time()
        tr = df["region"] != hold
        te = ~tr
        Xtr = df.loc[tr, cols].to_numpy(np.float32)
        Ytr = df.loc[tr, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
        Xte = df.loc[te, cols].to_numpy(np.float32)
        Yte = df.loc[te, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
        scaler = StandardScaler()
        model = Ridge(alpha=alpha)
        model.fit(scaler.fit_transform(Xtr), Ytr)
        pred = np.clip(model.predict(scaler.transform(Xte)), 0, 1).astype(np.float32)
        oof[te.to_numpy()] = pred
        mae_r = float(mean_absolute_error(Yte[:, 0], pred[:, 0]))
        mae_n = float(mean_absolute_error(Yte[:, 1], pred[:, 1]))
        fold_metrics.append(
            {
                "holdout": hold,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "mae_res": mae_r,
                "mae_nres": mae_n,
                "mae_mean": 0.5 * (mae_r + mae_n),
                "seconds": round(time.time() - t0, 1),
            }
        )
        np.savez_compressed(
            ckpt / f"fold_{hold}.npz",
            coef=model.coef_.astype(np.float32),
            intercept=model.intercept_.astype(np.float32),
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
        )
        log(
            run_dir,
            f"LORO hold={hold}: mean={0.5*(mae_r+mae_n):.4f} res={mae_r:.4f} "
            f"nres={mae_n:.4f} n_te={int(te.sum()):,}",
        )
        write_progress(
            run_dir, status="running", stage="train_loro", completed=i + 1, total=len(regions), failed=0
        )

    out = df[
        [
            "id",
            "region",
            "soft_gt_frac_res",
            "soft_gt_frac_nres",
            "ae_frac_res",
            "ae_frac_nres",
            "ghsl_frac_res",
            "ghsl_frac_nres",
            "pls_frac_res",
            "pls_frac_nres",
            "y_res",
        ]
    ].copy()
    out["ring_frac_res"] = oof[:, 0]
    out["ring_frac_nres"] = oof[:, 1]
    out.to_parquet(run_dir / "artifacts" / "buildings_oof_ring.parquet", index=False)
    pd.DataFrame(fold_metrics).to_csv(run_dir / "artifacts" / "fold_metrics.csv", index=False)
    atomic_json(run_dir / "artifacts" / "fold_metrics.json", fold_metrics)
    return out


def build_tile_features_and_loro(run_dir: Path, regions: list[str], alpha: float, workers: int) -> pd.DataFrame:
    """Tile-centroid rings + LORO on tile soft GT.

    Prefer prebalance tiles for training if present; Section 3/4 cohort is
    applied only when scoring (score_tiles).
    """
    pre = W6_100M / "artifacts" / "tiles_100m_prebalance.parquet"
    tiles_all = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    tiles = pd.read_parquet(pre) if pre.exists() else tiles_all
    # ensure soft GT + baseline cols exist on prebalance; else fall back
    need = {"soft_gt_frac_res", "soft_gt_frac_nres", "ridge_frac_res", "ghsl_frac_res"}
    if not need.issubset(tiles.columns):
        tiles = tiles_all
    log(run_dir, f"Tile ring features for training n={len(tiles):,} (eval cohort is Section 3/4)")
    shard_dir = ensure_cell_shards(run_dir, regions)

    # map tile -> region, soft gt, existing preds from tiles parquet
    # Build features per metro
    feat_parts = []
    for region in regions:
        t0 = time.time()
        tsub = tiles[tiles["region"] == region].copy()
        if tsub.empty:
            continue
        metro_cells = pd.read_parquet(shard_dir / f"{region}.parquet")
        metro_cells["id"] = metro_cells["id"].astype(str)

        # Assign metro cells to 100 m tiles; use as neighborhood + tile center emb.
        metro_cells = metro_cells.copy()
        metro_cells["tile_id"] = (
            region
            + "|"
            + (metro_cells["utm_x"] // 100).astype(int).astype(str)
            + "|"
            + (metro_cells["utm_y"] // 100).astype(int).astype(str)
        )
        centers = (
            metro_cells.groupby("tile_id", sort=False)
            .agg(utm_x=("utm_x", "mean"), utm_y=("utm_y", "mean"))
            .reset_index()
        )
        cemb = metro_cells.groupby("tile_id", sort=False)[BANDS].mean().reset_index()
        panel = tsub.merge(centers, on="tile_id", how="inner").merge(cemb, on="tile_id", how="inner")
        if len(panel) < len(tsub):
            # parse tile_id → nominal center for any missing
            miss = tsub[~tsub["tile_id"].isin(panel["tile_id"])].copy()
            parts = miss["tile_id"].str.rsplit("|", n=2, expand=True)
            miss["utm_x"] = parts[1].astype(float) * 100 + 50
            miss["utm_y"] = parts[2].astype(float) * 100 + 50
            # nearest-cell emb as center
            tree_xy = metro_cells[["utm_x", "utm_y"]].to_numpy(np.float64)
            from scipy.spatial import cKDTree as _KD

            nn = _KD(tree_xy).query(miss[["utm_x", "utm_y"]].to_numpy(np.float64), k=1)[1]
            for i, b in enumerate(BANDS):
                miss[b] = metro_cells.iloc[nn][b].to_numpy()
            panel = pd.concat([panel, miss], ignore_index=True)

        cell_xy = metro_cells[["utm_x", "utm_y"]].to_numpy(np.float64)
        cell_emb = metro_cells[BANDS].to_numpy(np.float32)
        query_xy = panel[["utm_x", "utm_y"]].to_numpy(np.float64)
        feats = ring_features_for_points(query_xy, cell_xy, cell_emb)
        feats[:, :64] = panel[BANDS].to_numpy(np.float32)
        feat_df = pd.DataFrame(feats, columns=feat_colnames())
        base_cols = [c for c in panel.columns if c not in BANDS and c not in ("utm_x", "utm_y")]
        out = pd.concat([panel[base_cols].reset_index(drop=True), feat_df], axis=1)
        feat_parts.append(out)
        log(run_dir, f"tile features {region}: n={len(out):,} in {time.time()-t0:.1f}s")

    df = pd.concat(feat_parts, ignore_index=True)
    log(run_dir, f"tile training table n={len(df):,}")
    cols = feat_colnames()
    oof = np.full((len(df), 2), np.nan, dtype=np.float32)
    for hold in regions:
        tr = df["region"] != hold
        te = ~tr
        if int(te.sum()) == 0:
            continue
        Xtr = df.loc[tr, cols].to_numpy(np.float32)
        Ytr = df.loc[tr, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
        Xte = df.loc[te, cols].to_numpy(np.float32)
        scaler = StandardScaler()
        model = Ridge(alpha=alpha)
        model.fit(scaler.fit_transform(Xtr), Ytr)
        oof[te.to_numpy()] = np.clip(model.predict(scaler.transform(Xte)), 0, 1).astype(np.float32)
    df["ring_frac_res"] = oof[:, 0]
    df["ring_frac_nres"] = oof[:, 1]
    # aliases for scoring
    df["ae_frac_res"] = df["ridge_frac_res"]
    df["ae_frac_nres"] = df["ridge_frac_nres"]
    df.to_parquet(run_dir / "artifacts" / "tiles_oof_ring.parquet", index=False)
    return df


def decorate_metro(rows: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in rows}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        r["bs"] = r["ghsl_built_s"]
        r["rg"] = r["ae_ridge"]
        r["pl"] = r["ae_pls"]
        r["xg"] = r.get("ae_xgb", {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")})
        r["rk"] = r["ae_ring"]
        r["margin"] = float(r["ae_ring"]["mean"] - r["ae_ridge"]["mean"])
        out.append(r)
    return out


def score_buildings(run_dir: Path, oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    oof = oof.copy()
    oof["id"] = oof["id"].astype(str)
    m = cohort.merge(oof, on=["id", "region"], how="left", suffixes=("", "_oof"))

    # XGB from week7
    xgb_path = (
        ROOT
        / "outputs"
        / "week7_section3_xgb_mae"
        / "20261113_033609"
        / "artifacts"
        / "buildings_section3_with_xgb.parquet"
    )
    if xgb_path.exists():
        xb = pd.read_parquet(xgb_path, columns=["id", "xgb_frac_res", "xgb_frac_nres"])
        xb["id"] = xb["id"].astype(str)
        m = m.merge(xb, on="id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ae_frac_res", "ae_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_ring": ("ring_frac_res", "ring_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in m.columns:
            soft_pooled[name] = {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan"), "model": name, "gt": "soft"}
            continue
        soft_pooled[name] = {
            **mae_pair(
                m["soft_gt_frac_res"].to_numpy(float),
                m["soft_gt_frac_nres"].to_numpy(float),
                m[pr].to_numpy(float),
                m[pn].to_numpy(float),
            ),
            "model": name,
            "gt": "soft",
        }

    metro = []
    for region, sub in m.groupby("region", sort=False):
        row: dict[str, Any] = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            if pr not in sub.columns:
                continue
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    metro_soft = decorate_metro(metro)
    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_ring.parquet", index=False)
    for name in ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb", "ae_ring"]:
        r = soft_pooled[name]
        log(run_dir, f"building {name:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f}")
    return {
        "n_shared": len(m),
        "soft_pooled": soft_pooled,
        "metro_soft": metro_soft,
        "ring_beats_ridge": sum(1 for r in metro_soft if r["rk"]["mean"] + 1e-12 < r["rg"]["mean"]),
    }


def score_tiles(run_dir: Path, tile_oof: pd.DataFrame) -> dict[str, Any]:
    """Score locked Section 3/4 tiles only (same 25,220 IDs)."""
    cohort = pd.read_parquet(COHORT_TILE)
    base = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    # cohort already has region; drop duplicate from base to avoid region_x/region_y
    base_cols = [c for c in base.columns if c != "region" or "region" not in cohort.columns]
    t = cohort.merge(base[base_cols], on="tile_id", how="left")
    if "region" not in t.columns:
        # parse from tile_id if needed
        t["region"] = t["tile_id"].astype(str).str.split("|").str[0]
    assert len(t) == len(cohort)
    ring = tile_oof[["tile_id", "ring_frac_res", "ring_frac_nres"]].drop_duplicates("tile_id")
    t = t.merge(ring, on="tile_id", how="left")
    t["ae_frac_res"] = t["ridge_frac_res"]
    t["ae_frac_nres"] = t["ridge_frac_nres"]

    w7 = (
        ROOT
        / "outputs"
        / "week7_section3_xgb_mae"
        / "20261113_033609"
        / "artifacts"
        / "tiles_section3_with_xgb.parquet"
    )
    if w7.exists():
        xb = pd.read_parquet(w7)
        cols = ["tile_id"] + [c for c in xb.columns if c.startswith("xgb_")]
        t = t.merge(xb[cols], on="tile_id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_ring": ("ring_frac_res", "ring_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in t.columns:
            soft_pooled[name] = {
                "n": 0,
                "res": float("nan"),
                "nres": float("nan"),
                "mean": float("nan"),
                "model": name,
                "gt": "soft",
            }
            continue
        soft_pooled[name] = {
            **mae_pair(
                t["soft_gt_frac_res"].to_numpy(float),
                t["soft_gt_frac_nres"].to_numpy(float),
                t[pr].to_numpy(float),
                t[pn].to_numpy(float),
            ),
            "model": name,
            "gt": "soft",
        }

    metro = []
    for region, sub in t.groupby("region", sort=False):
        row: dict[str, Any] = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            if pr not in sub.columns:
                continue
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    metro_soft = decorate_metro(metro)
    t.to_parquet(run_dir / "artifacts" / "tiles_section3_with_ring.parquet", index=False)
    for name in ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb", "ae_ring"]:
        r = soft_pooled[name]
        log(run_dir, f"tile {name:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f}")
    return {
        "n_tiles": len(t),
        "soft_pooled": soft_pooled,
        "metro_soft": metro_soft,
        "ring_beats_ridge": sum(1 for r in metro_soft if r["rk"]["mean"] + 1e-12 < r["rg"]["mean"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(10, os.cpu_count() or 4)))
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-tiles", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "week8_ae_rings_loro" / stamp)
    for sub in ("checkpoints", "artifacts", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"start={utc_now()}\nworkers={args.workers}\nalpha={args.alpha}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpu_count={os.cpu_count()}\nn_gpus={os.popen('nvidia-smi -L 2>/dev/null | wc -l').read().strip()}\n"
    )
    latest = ROOT / "outputs" / "week8_ae_rings_loro" / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    regions = [r for r, _, _ in METRO_META]
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        completed=0,
        total=len(regions),
        failed=0,
        output=str(run_dir),
        workers=args.workers,
        _t0=time.time(),
    )
    log(run_dir, f"Week8 AE rings · workers={args.workers} cpus={os.cpu_count()} regions={len(regions)}")

    if args.skip_features:
        feat_dir = run_dir / "checkpoints" / "building_ring_features"
        if not any(feat_dir.glob("*.parquet")):
            prev = sorted((ROOT / "outputs" / "week8_ae_rings_loro").glob("*/checkpoints/building_ring_features"))
            feat_dir = prev[-1] if prev else feat_dir
        log(run_dir, f"reuse features {feat_dir}")
    else:
        feat_dir = stage_building_features(run_dir, regions, args.workers)

    df = load_building_table(feat_dir, regions)
    n_cohort = len(pd.read_parquet(COHORT_BUILD))
    log(
        run_dir,
        f"building training table n={len(df):,} · eval cohort n={n_cohort:,} "
        f"(train may be larger; metrics use Section 3/4 IDs only)",
    )
    oof = train_loro_buildings(df, run_dir, alpha=args.alpha)
    building = score_buildings(run_dir, oof)

    if args.skip_tiles:
        tile = {
            "n_tiles": 0,
            "soft_pooled": {},
            "metro_soft": [],
            "ring_beats_ridge": 0,
        }
    else:
        write_progress(run_dir, status="running", stage="tile_features_loro")
        tile_oof = build_tile_features_and_loro(run_dir, regions, args.alpha, args.workers)
        tile = score_tiles(run_dir, tile_oof)

    metrics = {
        "building": building,
        "tile_100m": tile,
        "protocol": (
            "Building-level AE mean embedding + mean/std in 50/100/200 m rings; "
            "Ridge LORO trained on all soft-covered buildings (and available tiles); "
            "Section 5 metrics scored only on locked Section 3/4 holdout cohorts "
            "(buildings n=71,497; tiles n=25,220) — same eval set as Section 3/4"
        ),
        "cohort_buildings": str(COHORT_BUILD),
        "cohort_tiles": str(COHORT_TILE),
        "rings_m": list(RINGS_M),
        "n_features": N_FEATURES,
        "alpha": args.alpha,
    }
    atomic_json(run_dir / "artifacts" / "section5_ring_metrics.json", metrics)
    summary = {
        "status": "completed",
        "finished_at": utc_now(),
        "building_soft_mean": {k: v["mean"] for k, v in building["soft_pooled"].items()},
        "tile_soft_mean": {k: v["mean"] for k, v in tile["soft_pooled"].items()},
        "ring_beats_ridge_buildings": building["ring_beats_ridge"],
        "ring_beats_ridge_tiles": tile["ring_beats_ridge"],
        "metrics": str(run_dir / "artifacts" / "section5_ring_metrics.json"),
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(run_dir, status="completed", stage="done", percent_complete=100.0, completed=len(regions), total=len(regions))
    b = building["soft_pooled"]
    log(
        run_dir,
        f"DONE buildings: GHSL={b['ghsl_built_s']['mean']:.3f} "
        f"Ridge={b['ae_ridge']['mean']:.3f} Ring={b['ae_ring']['mean']:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
