#!/usr/bin/env python3
"""Week 15: Rings+S2 (464-d) with XGB *scaled* for ~6× feature width.

Week 10 used the same thin AE+S2 XGB recipe on 464-d rings+S2 and diluted S2
(pooled MAE 0.201 vs AE+S2 0.195). This run keeps the same features and LORO
protocol but scales capacity + per-tree feature sample:

  AE+S2 ≈ 80 dims · recipe n_estimators=300, colsample=0.8
  Rings+S2 = 464 dims ≈ 5.8× → n_estimators=1800, colsample≈0.17, lr=0.025

Apples-to-apples (docs/eval_protocol_apples_to_apples.md / balanced_eval).
Reuses Week 8 rings + Week 9 S2 (+ Week 10 train table when present).
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
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
W6_100M = ROOT / "outputs" / "week6_overture_polygon_mae_100m" / "20261110_062739"
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
COHORT_TILE = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_tile_ids.parquet"
XGB_BUILD = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "buildings_section3_with_xgb.parquet"
)
XGB_TILE = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "tiles_section3_with_xgb.parquet"
)
RING_FEAT_DIR = (
    ROOT
    / "outputs"
    / "week8_ae_rings_loro"
    / "20261113_050936"
    / "checkpoints"
    / "building_ring_features"
)
RING_OOF = ROOT / "outputs" / "week8_ae_rings_loro" / "latest" / "artifacts" / "buildings_oof_ring.parquet"
S2_TRAIN = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "train_buildings_ae_s2.parquet"
S2_OOF = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "buildings_oof_ae_s2.parquet"
W10_TRAIN = (
    ROOT / "outputs" / "week10_ae_rings_s2_loro" / "latest" / "artifacts" / "train_buildings_rings_s2.parquet"
)
W10_OOF = (
    ROOT / "outputs" / "week10_ae_rings_s2_loro" / "latest" / "artifacts" / "buildings_oof_rings_s2.parquet"
)
CNN_OOF = (
    ROOT
    / "outputs"
    / "week13_ae_grid_cnn_s2_loro"
    / "latest"
    / "artifacts"
    / "buildings_oof_ae_grid_cnn_s2.parquet"
)

# Thin AE+S2 recipe (Week 9/10) vs scaled for 464-d
BASE_FEAT_DIM = 80  # AE 64 + S2 16
BASE_N_ESTIMATORS = 300
BASE_COLSAMPLE = 0.8
BASE_LR = 0.05
SCALED_N_ESTIMATORS = 1800  # ~6× trees
SCALED_COLSAMPLE = round(BASE_FEAT_DIM / 464, 3)  # ≈0.172 → ~80 feats/tree
SCALED_LR = 0.025

BANDS = [f"A{i:02d}" for i in range(64)]
RINGS_M = (50, 100, 200)
S2_FEATS = [
    "s2_blue",
    "s2_green",
    "s2_red",
    "s2_nir",
    "s2_brightness",
    "s2_blue_dom",
    "s2_ndvi",
    "s2_contrast_small",
    "s2_contrast_large",
    "s2_contrast_ratio",
    "fp_area_m2",
    "fp_equiv_diam_m",
    "fp_compactness",
    "s2_nres_score",
    "s2_n_pix",
    "s2_ok",
]
SEED = 42

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


def detect_gpus() -> list[str]:
    """Prefer nvidia-smi (fast); avoid importing torch at startup (can hang on CUDA init)."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


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


def ring_colnames() -> list[str]:
    cols = [f"c_{b}" for b in BANDS]
    for r in RINGS_M:
        cols += [f"r{r}_mean_{b}" for b in BANDS]
        cols += [f"r{r}_std_{b}" for b in BANDS]
    return cols


def feat_colnames() -> list[str]:
    return ring_colnames() + S2_FEATS


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if int(m.sum()) == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    mae_r = float(mean_absolute_error(y_res[m], p_res[m]))
    mae_n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": mae_r, "nres": mae_n, "mean": 0.5 * (mae_r + mae_n)}


def decorate_metro(rows: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in rows}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        out.append(r)
    return out


def build_training_table(run_dir: Path, ring_dir: Path, s2_path: Path) -> pd.DataFrame:
    regions = sorted(p.stem for p in ring_dir.glob("*.parquet"))
    if not regions:
        raise FileNotFoundError(f"no ring feature parquets in {ring_dir}")
    rings = pd.concat([pd.read_parquet(ring_dir / f"{r}.parquet") for r in regions], ignore_index=True)
    rings["id"] = rings["id"].astype(str)
    s2 = pd.read_parquet(s2_path)
    s2["id"] = s2["id"].astype(str)
    s2_cols = ["id", "region"] + [c for c in S2_FEATS if c in s2.columns]
    s2 = s2[s2_cols].copy()
    for c in S2_FEATS:
        if c not in s2.columns:
            s2[c] = np.nan
    df = rings.merge(s2, on=["id", "region"], how="inner")
    missing = [c for c in feat_colnames() if c not in df.columns]
    if missing:
        raise RuntimeError(f"training table missing cols: {missing[:8]}…")
    for c in feat_colnames():
        df[c] = df[c].astype(np.float32)
    out = run_dir / "artifacts" / "train_buildings_rings_s2.parquet"
    df.to_parquet(out, index=False)
    log(
        run_dir,
        f"training table n={len(df):,} features={len(feat_colnames())} "
        f"s2_ok={int((df['s2_ok'] > 0.5).sum()):,} ring_dir={ring_dir}",
    )
    return df


def xgb_params(n_jobs: int, device: str) -> dict[str, Any]:
    """Scaled recipe for ~6× feature width vs AE+S2 thin XGB."""
    kwargs: dict[str, Any] = dict(
        n_estimators=SCALED_N_ESTIMATORS,
        max_depth=6,
        learning_rate=SCALED_LR,
        subsample=0.8,
        colsample_bytree=SCALED_COLSAMPLE,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
    )
    if device.startswith("cuda"):
        kwargs["device"] = device
    return kwargs


def _loro_fold(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker: one LORO holdout fold. Loads features from shared parquet path."""
    from xgboost import XGBRegressor

    hold = payload["hold"]
    train_path = Path(payload["train_path"])
    cols = payload["cols"]
    n_jobs = payload["n_jobs"]
    device = payload.get("device", "cpu")
    t0 = time.time()

    df = pd.read_parquet(train_path, columns=["region", "soft_gt_frac_res", "soft_gt_frac_nres", *cols])
    te = (df["region"] == hold).to_numpy()
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)
    Xtr = np.nan_to_num(df.loc[tr, cols].to_numpy(np.float32), nan=0.0)
    Ytr = df.loc[tr, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
    Xte = np.nan_to_num(df.loc[te, cols].to_numpy(np.float32), nan=0.0)
    Yte = df.loc[te, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    base = XGBRegressor(**xgb_params(n_jobs, device))
    model = MultiOutputRegressor(base)
    model.fit(Xtr_s, Ytr)
    pred = np.clip(model.predict(Xte_s), 0, 1).astype(np.float32)
    mae_r = float(mean_absolute_error(Yte[:, 0], pred[:, 0]))
    mae_n = float(mean_absolute_error(Yte[:, 1], pred[:, 1]))
    return {
        "hold": hold,
        "te_idx": te_idx,
        "pred": pred,
        "mae_res": mae_r,
        "mae_nres": mae_n,
        "mae_mean": 0.5 * (mae_r + mae_n),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "n_estimators": SCALED_N_ESTIMATORS,
        "colsample_bytree": SCALED_COLSAMPLE,
        "learning_rate": SCALED_LR,
    }


def train_loro_xgb(df: pd.DataFrame, run_dir: Path, fold_workers: int) -> pd.DataFrame:
    import multiprocessing as mp

    cols = feat_colnames()
    regions = sorted(df["region"].unique())
    train_path = run_dir / "artifacts" / "train_buildings_rings_s2.parquet"
    oof = np.full((len(df), 2), np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)

    n_gpus = len(detect_gpus())
    log(
        run_dir,
        f"LORO folds={len(regions)} fold_workers={fold_workers} n_gpus={n_gpus} "
        f"n_features={len(cols)} n_estimators={SCALED_N_ESTIMATORS} "
        f"colsample={SCALED_COLSAMPLE} lr={SCALED_LR}",
    )

    payloads = []
    for i, hold in enumerate(regions):
        device = f"cuda:{i % n_gpus}" if n_gpus > 0 else "cpu"
        if n_gpus > 0:
            n_jobs = 1
        else:
            n_jobs = max(1, (os.cpu_count() or 8) // max(1, fold_workers))
        payloads.append(
            {
                "hold": hold,
                "train_path": str(train_path),
                "cols": cols,
                "n_jobs": n_jobs,
                "device": device,
            }
        )

    write_progress(run_dir, status="running", stage="train_loro", completed=0, total=len(regions), failed=0)
    max_w = min(fold_workers, n_gpus) if n_gpus > 0 else fold_workers
    max_w = max(1, min(max_w, len(payloads)))
    done = 0
    failed = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
        futs = {pool.submit(_loro_fold, p): p["hold"] for p in payloads}
        for fut in as_completed(futs):
            hold = futs[fut]
            try:
                res = fut.result()
                oof[res["te_idx"]] = res["pred"]
                fold_metrics.append(
                    {
                        "holdout": res["hold"],
                        "n_train": res["n_train"],
                        "n_test": res["n_test"],
                        "mae_res": res["mae_res"],
                        "mae_nres": res["mae_nres"],
                        "mae_mean": res["mae_mean"],
                        "seconds": res["seconds"],
                        "device": res["device"],
                    }
                )
                np.savez_compressed(
                    ckpt / f"fold_{hold}.npz",
                    pred=res["pred"],
                    te_idx=res["te_idx"],
                    scaler_mean=res["scaler_mean"],
                    scaler_scale=res["scaler_scale"],
                )
                log(
                    run_dir,
                    f"LORO hold={hold}: mean={res['mae_mean']:.4f} res={res['mae_res']:.4f} "
                    f"nres={res['mae_nres']:.4f} n_te={res['n_test']:,} "
                    f"device={res['device']} in {res['seconds']}s",
                )
            except Exception as e:
                failed += 1
                log(run_dir, f"LORO FAIL {hold}: {e}")
                atomic_json(run_dir / "failures" / f"fold_{hold}.json", {"hold": hold, "error": str(e)})
            done += 1
            write_progress(
                run_dir,
                status="running",
                stage="train_loro",
                completed=done,
                total=len(regions),
                failed=failed,
            )
    if failed:
        raise RuntimeError(f"{failed} LORO folds failed")
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF preds incomplete")

    keep = [
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
    ]
    if "y_res" in df.columns:
        keep.append("y_res")
    out = df[keep + S2_FEATS].copy()
    out["rs2x_frac_res"] = oof[:, 0]
    out["rs2x_frac_nres"] = oof[:, 1]

    if RING_OOF.exists():
        rk = pd.read_parquet(RING_OOF.resolve(), columns=["id", "ring_frac_res", "ring_frac_nres"])
        rk["id"] = rk["id"].astype(str)
        out = out.merge(rk, on="id", how="left")
    if S2_OOF.exists():
        s2p = pd.read_parquet(S2_OOF.resolve(), columns=["id", "s2_frac_res", "s2_frac_nres"])
        s2p["id"] = s2p["id"].astype(str)
        out = out.merge(s2p, on="id", how="left")
    if W10_OOF.exists():
        w10 = pd.read_parquet(W10_OOF.resolve(), columns=["id", "rs2_frac_res", "rs2_frac_nres"])
        w10["id"] = w10["id"].astype(str)
        w10 = w10.rename(columns={"rs2_frac_res": "rs2_thin_frac_res", "rs2_frac_nres": "rs2_thin_frac_nres"})
        out = out.merge(w10, on="id", how="left")
    if CNN_OOF.exists():
        cn = pd.read_parquet(CNN_OOF.resolve(), columns=["id", "cnn_frac_res", "cnn_frac_nres"])
        cn["id"] = cn["id"].astype(str)
        out = out.merge(cn, on="id", how="left")

    out.to_parquet(run_dir / "artifacts" / "buildings_oof_rings_s2_scaled.parquet", index=False)
    pd.DataFrame(fold_metrics).sort_values("holdout").to_csv(run_dir / "artifacts" / "fold_metrics.csv", index=False)
    atomic_json(run_dir / "artifacts" / "fold_metrics.json", sorted(fold_metrics, key=lambda x: x["holdout"]))
    return out


def score_buildings(run_dir: Path, oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    oof = oof.copy()
    oof["id"] = oof["id"].astype(str)
    m = cohort.merge(oof, on=["id", "region"], how="left")
    assert len(m) == 71497, f"building cohort size {len(m)} != 71497"

    if XGB_BUILD.exists():
        xb = pd.read_parquet(XGB_BUILD, columns=["id", "xgb_frac_res", "xgb_frac_nres"])
        xb["id"] = xb["id"].astype(str)
        m = m.merge(xb, on="id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ae_frac_res", "ae_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_ring": ("ring_frac_res", "ring_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
        "ae_cnn_s2": ("cnn_frac_res", "cnn_frac_nres"),
        "ae_ring_s2_thin": ("rs2_thin_frac_res", "rs2_thin_frac_nres"),
        "ae_ring_s2_scaled": ("rs2x_frac_res", "rs2x_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in m.columns:
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
    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_rings_s2_scaled.parquet", index=False)
    for name in models:
        r = soft_pooled[name]
        log(run_dir, f"building {name:18s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")

    def _beats(a: str, b: str) -> int:
        return sum(
            1
            for r in metro_soft
            if a in r
            and b in r
            and np.isfinite(r[a]["mean"])
            and np.isfinite(r[b]["mean"])
            and r[a]["mean"] + 1e-12 < r[b]["mean"]
        )

    return {
        "n_shared": len(m),
        "coverage": float(np.mean(np.isfinite(m["rs2x_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "metro_soft": metro_soft,
        "scaled_beats_s2": _beats("ae_ring_s2_scaled", "ae_s2"),
        "scaled_beats_thin": _beats("ae_ring_s2_scaled", "ae_ring_s2_thin"),
        "scaled_beats_cnn": _beats("ae_ring_s2_scaled", "ae_cnn_s2"),
        "scaled_beats_ghsl": _beats("ae_ring_s2_scaled", "ghsl_built_s"),
    }


def score_tiles_from_buildings(run_dir: Path, building_oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_TILE)
    base = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    base_cols = [c for c in base.columns if not (c == "region" and "region" in cohort.columns)]
    t = cohort.merge(base[base_cols], on="tile_id", how="left")
    assert len(t) == 25220, f"tile cohort size {len(t)} != 25220"

    if XGB_TILE.exists():
        xb = pd.read_parquet(XGB_TILE)
        cols = ["tile_id"] + [c for c in xb.columns if c.startswith("xgb_")]
        t = t.merge(xb[cols], on="tile_id", how="left")

    # prior tile ring preds if available
    ring_tile = ROOT / "outputs" / "week8_ae_rings_loro" / "latest" / "artifacts" / "tiles_section3_with_ring.parquet"
    if ring_tile.exists():
        rt = pd.read_parquet(ring_tile.resolve(), columns=["tile_id", "ring_frac_res", "ring_frac_nres"])
        t = t.merge(rt, on="tile_id", how="left")

    s2_tile = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "tiles_section3_with_ae_s2.parquet"
    if s2_tile.exists():
        st = pd.read_parquet(s2_tile.resolve(), columns=["tile_id", "s2_frac_res", "s2_frac_nres"])
        t = t.merge(st, on="tile_id", how="left")

    # Fuse tile preds = mean of building OOF in tile; keep Week8/9 tile baselines as-is.
    bo = building_oof[["id", "region", "rs2x_frac_res", "rs2x_frac_nres"]].copy()
    bo["id"] = bo["id"].astype(str)
    cell_xy = pd.read_parquet(
        CELLS,
        columns=["id", "utm_x", "utm_y", "inside_polygon"],
        filters=[("inside_polygon", "==", True)],
    )
    cell_xy["id"] = cell_xy["id"].astype(str)
    centers = cell_xy.groupby("id", sort=False).agg(utm_x=("utm_x", "mean"), utm_y=("utm_y", "mean")).reset_index()
    bm = bo.merge(centers, on="id", how="left")
    bm = bm[np.isfinite(bm["utm_x"]) & np.isfinite(bm["utm_y"])].copy()
    bm["tile_id"] = (
        bm["region"]
        + "|"
        + (bm["utm_x"] // 100).astype(int).astype(str)
        + "|"
        + (bm["utm_y"] // 100).astype(int).astype(str)
    )
    agg = (
        bm.groupby("tile_id", sort=False)
        .agg(rs2x_frac_res=("rs2x_frac_res", "mean"), rs2x_frac_nres=("rs2x_frac_nres", "mean"))
        .reset_index()
    )
    t = t.drop(columns=["rs2x_frac_res", "rs2x_frac_nres"], errors="ignore").merge(agg, on="tile_id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_ring": ("ring_frac_res", "ring_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
        "ae_ring_s2_scaled": ("rs2x_frac_res", "rs2x_frac_nres"),
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
    t.to_parquet(run_dir / "artifacts" / "tiles_section3_with_rings_s2_scaled.parquet", index=False)
    for name in models:
        r = soft_pooled[name]
        log(run_dir, f"tile {name:18s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    return {
        "n_shared": len(t),
        "coverage": float(np.mean(np.isfinite(t["rs2x_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "metro_soft": decorate_metro(metro),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week15_rings_s2_xgb_scaled_loro")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument("--ring-feat-dir", type=Path, default=RING_FEAT_DIR)
    ap.add_argument("--s2-train", type=Path, default=S2_TRAIN)
    ap.add_argument(
        "--fold-workers",
        type=int,
        default=2,
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.resume_dir or (args.out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    for sub in ("artifacts", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    latest = args.out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    gpu_names = detect_gpus()
    n_gpus = len(gpu_names)
    (run_dir / "launch.txt").write_text(
        f"started={utc_now()}\npid={os.getpid()}\nfold_workers={args.fold_workers}\n"
        f"n_gpus={n_gpus}\ngpus={gpu_names}\n"
        f"n_features={len(feat_colnames())}\n"
        f"n_estimators={SCALED_N_ESTIMATORS}\ncolsample_bytree={SCALED_COLSAMPLE}\nlr={SCALED_LR}\n"
        f"ring_feat_dir={args.ring_feat_dir}\ns2_train={args.s2_train}\n"
        f"protocol=docs/eval_protocol_apples_to_apples.md\n"
        f"note=scaled XGB for 6x feature width vs Week10 thin recipe\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpus={os.cpu_count()}\nn_gpus={n_gpus}\ngpus={gpu_names}\n"
        f"python={os.sys.executable}\ncwd={os.getcwd()}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="init",
        completed=0,
        total=5,
        failed=0,
        _t0=time.time(),
        output=str(run_dir),
        n_cpus=os.cpu_count(),
        n_gpus=n_gpus,
        fold_workers=args.fold_workers,
    )
    log(
        run_dir,
        f"Week15 Rings+S2 scaled-XGB LORO start → {run_dir} · GPUs={n_gpus} {gpu_names} · "
        f"est={SCALED_N_ESTIMATORS} colsample={SCALED_COLSAMPLE} lr={SCALED_LR}",
    )

    train_dst = run_dir / "artifacts" / "train_buildings_rings_s2.parquet"
    if W10_TRAIN.exists():
        import shutil

        shutil.copy2(W10_TRAIN.resolve(), train_dst)
        df = pd.read_parquet(train_dst)
        log(run_dir, f"Reused Week10 train table n={len(df):,} features={len(feat_colnames())}")
    else:
        ring_dir = args.ring_feat_dir.resolve()
        s2_path = args.s2_train.resolve()
        if not ring_dir.exists():
            raise FileNotFoundError(ring_dir)
        if not s2_path.exists():
            raise FileNotFoundError(s2_path)
        df = build_training_table(run_dir, ring_dir, s2_path)
    write_progress(run_dir, stage="train_table", completed=1, total=5, train_n=len(df))

    oof = train_loro_xgb(df, run_dir, fold_workers=args.fold_workers)
    write_progress(run_dir, stage="score", completed=3, total=5)

    b_metrics = score_buildings(run_dir, oof)
    t_metrics = score_tiles_from_buildings(run_dir, oof)

    metrics = {
        "protocol": "docs/eval_protocol_apples_to_apples.md",
        "train_n_buildings": int(len(df)),
        "eval_n_buildings": 71497,
        "eval_n_tiles": 25220,
        "features": feat_colnames(),
        "n_features": len(feat_colnames()),
        "rings_m": list(RINGS_M),
        "xgb": {
            "n_estimators": SCALED_N_ESTIMATORS,
            "max_depth": 6,
            "learning_rate": SCALED_LR,
            "colsample_bytree": SCALED_COLSAMPLE,
            "subsample": 0.8,
            "note": "scaled vs Week9/10 thin recipe (300 trees, colsample=0.8) for ~6x feature width",
            "thin_recipe": {
                "n_estimators": BASE_N_ESTIMATORS,
                "colsample_bytree": BASE_COLSAMPLE,
                "learning_rate": BASE_LR,
                "approx_feature_dim": BASE_FEAT_DIM,
            },
        },
        "buildings": b_metrics,
        "tiles": t_metrics,
        "checklist": {
            "eval_ids_locked": True,
            "soft_gt_from_week6": True,
            "spatial_split": "LORO",
            "metric": "0.5*(MAE_res+MAE_nres)",
            "baselines_joined": True,
            "rings_reused_from_week8": True,
            "s2_reused_from_week9": True,
            "train_table_reused_from_week10": True,
            "xgb_scaled_for_6x_features": True,
        },
    }
    atomic_json(run_dir / "artifacts" / "section12_rings_s2_xgb_scaled_metrics.json", metrics)
    scaled_mean = b_metrics["soft_pooled"]["ae_ring_s2_scaled"]["mean"]
    thin_mean = b_metrics["soft_pooled"].get("ae_ring_s2_thin", {}).get("mean")
    s2_mean = b_metrics["soft_pooled"].get("ae_s2", {}).get("mean")
    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "finished_at": utc_now(),
            "output": str(run_dir),
            "building_mean_ae_ring_s2_scaled": scaled_mean,
            "building_mean_ae_ring_s2_thin": thin_mean,
            "building_mean_ae_s2": s2_mean,
            "building_delta_scaled_minus_thin": None if thin_mean is None else float(scaled_mean - thin_mean),
            "building_delta_scaled_minus_s2": None if s2_mean is None else float(scaled_mean - s2_mean),
            "building_mean_ghsl": b_metrics["soft_pooled"]["ghsl_built_s"]["mean"],
            "tile_mean_ae_ring_s2_scaled": t_metrics["soft_pooled"]["ae_ring_s2_scaled"]["mean"],
            "tile_coverage": t_metrics["coverage"],
            "metrics": str(run_dir / "artifacts" / "section12_rings_s2_xgb_scaled_metrics.json"),
            "n_features": len(feat_colnames()),
            "n_estimators": SCALED_N_ESTIMATORS,
            "colsample_bytree": SCALED_COLSAMPLE,
        },
    )
    write_progress(run_dir, status="completed", stage="done", completed=5, total=5, failed=0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, "DONE")

    try:
        import subprocess

        board_script = ROOT / "scripts" / "rebuild_section12_rings_s2_xgb_scaled_board.py"
        if board_script.exists():
            subprocess.check_call(
                [
                    os.sys.executable,
                    str(board_script),
                    "--metrics",
                    str(run_dir / "artifacts" / "section12_rings_s2_xgb_scaled_metrics.json"),
                ],
                cwd=str(ROOT),
            )
            log(run_dir, "Section 12 canvas board injected")
    except Exception as e:
        log(run_dir, f"board inject skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
