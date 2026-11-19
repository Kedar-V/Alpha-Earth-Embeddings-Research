#!/usr/bin/env python3
"""Week 11: AE+S2 features → XGB LORO with 3-head (res, nres, empty).

Saves building soft_gt_frac_empty = mean(week4 frac_empty_full) over
inside_polygon AE cells (same aggregation path as soft_gt_frac_res/nres).

Headline metric unchanged (eval_protocol_apples_to_apples.md):
  0.5*(MAE_res + MAE_nres) on locked Section 3/4 IDs.
Side metric: MAE_empty. Compare vs Week 9 2-head AE+S2.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
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
PIXEL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
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
S2_TRAIN = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "train_buildings_ae_s2.parquet"
S2_OOF = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "buildings_oof_ae_s2.parquet"
RING_OOF = ROOT / "outputs" / "week8_ae_rings_loro" / "latest" / "artifacts" / "buildings_oof_ring.parquet"

BANDS = [f"A{i:02d}" for i in range(64)]
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
TARGET_COLS = ["soft_gt_frac_res", "soft_gt_frac_nres", "soft_gt_frac_empty"]
PRED_COLS = ["s2e_frac_res", "s2e_frac_nres", "s2e_frac_empty"]
SEED = 42
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


def detect_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def lonlat_key(lon, lat, nd: int = LONLAT_DECIMALS) -> np.ndarray:
    lon_s = np.round(np.asarray(lon, dtype=np.float64), nd).astype(str)
    lat_s = np.round(np.asarray(lat, dtype=np.float64), nd).astype(str)
    return np.char.add(lon_s.astype(object), np.char.add("_", lat_s.astype(object)))


def feat_colnames() -> list[str]:
    return BANDS + S2_FEATS


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if int(m.sum()) == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    mae_r = float(mean_absolute_error(y_res[m], p_res[m]))
    mae_n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": mae_r, "nres": mae_n, "mean": 0.5 * (mae_r + mae_n)}


def mae_one(y, p) -> dict[str, float]:
    m = np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) == 0:
        return {"n": 0, "mae": float("nan")}
    return {"n": int(m.sum()), "mae": float(mean_absolute_error(y[m], p[m]))}


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


def save_building_soft_gt_empty(run_dir: Path) -> Path:
    """Mean week4 frac_empty_full over inside_polygon cells → building soft_gt_frac_empty."""
    out = run_dir / "artifacts" / "buildings_soft_gt_with_empty.parquet"
    if out.exists() and out.stat().st_size > 1000:
        log(run_dir, f"Reusing saved soft GT+empty {out}")
        return out

    t0 = time.time()
    write_progress(run_dir, status="running", stage="save_empty_gt", completed=0, total=4, failed=0)
    w6 = pd.read_parquet(W6_BUILD)
    w6 = w6[w6["soft_gt_covered"] == True].copy()  # noqa: E712
    w6["id"] = w6["id"].astype(str)
    want = set(w6["id"])

    log(run_dir, f"Loading pixel GT empty from {PIXEL_GT}")
    pix = pd.read_parquet(
        PIXEL_GT,
        columns=["pixel_lon", "pixel_lat", "frac_res_full", "frac_nres_full", "frac_empty_full"],
    )
    pix["k"] = lonlat_key(pix["pixel_lon"].to_numpy(), pix["pixel_lat"].to_numpy())
    pix = pix.drop_duplicates("k", keep="first")
    write_progress(run_dir, stage="save_empty_gt", completed=1, total=4)

    log(run_dir, f"Loading inside_polygon cells for {len(want):,} soft-covered buildings")
    cells = pd.read_parquet(CELLS, columns=["id", "pixel_lon", "pixel_lat", "inside_polygon"])
    cells = cells[cells["inside_polygon"] == True].copy()  # noqa: E712
    cells["id"] = cells["id"].astype(str)
    cells = cells[cells["id"].isin(want)].copy()
    cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    write_progress(run_dir, stage="save_empty_gt", completed=2, total=4)

    joined = cells.merge(
        pix[["k", "frac_res_full", "frac_nres_full", "frac_empty_full"]],
        on="k",
        how="left",
    )
    agg = (
        joined.groupby("id", sort=False)
        .agg(
            soft_gt_frac_res_cellmean=("frac_res_full", "mean"),
            soft_gt_frac_nres_cellmean=("frac_nres_full", "mean"),
            soft_gt_frac_empty=("frac_empty_full", "mean"),
            soft_gt_n_cells_empty=("frac_empty_full", "count"),
        )
        .reset_index()
    )
    write_progress(run_dir, stage="save_empty_gt", completed=3, total=4)

    # Join onto week6 soft GT (canonical res/nres) + residual audit column
    out_df = w6[
        [
            "id",
            "region",
            "soft_gt_frac_res",
            "soft_gt_frac_nres",
            "soft_gt_n_cells",
            "soft_gt_covered",
            "ae_frac_res",
            "ae_frac_nres",
            "ghsl_frac_res",
            "ghsl_frac_nres",
            "pls_frac_res",
            "pls_frac_nres",
            "y_res",
        ]
    ].merge(agg, on="id", how="left")
    out_df["soft_gt_frac_empty"] = out_df["soft_gt_frac_empty"].astype(np.float32)
    out_df["soft_gt_empty_residual"] = np.clip(
        1.0 - out_df["soft_gt_frac_res"].to_numpy(float) - out_df["soft_gt_frac_nres"].to_numpy(float),
        0.0,
        1.0,
    ).astype(np.float32)
    out_df["soft_gt_empty_covered"] = out_df["soft_gt_n_cells_empty"].fillna(0).astype(int) > 0
    out_df.to_parquet(out, index=False)

    # also write a compact empty-only sidecar for reuse
    side = run_dir / "artifacts" / "soft_gt_frac_empty_by_building.parquet"
    out_df[["id", "region", "soft_gt_frac_res", "soft_gt_frac_nres", "soft_gt_frac_empty", "soft_gt_empty_residual", "soft_gt_n_cells_empty"]].to_parquet(
        side, index=False
    )

    corr = float(
        np.corrcoef(
            out_df["soft_gt_frac_empty"].to_numpy(float),
            out_df["soft_gt_empty_residual"].to_numpy(float),
        )[0, 1]
    )
    log(
        run_dir,
        f"Saved soft_gt_frac_empty n={len(out_df):,} covered={int(out_df.soft_gt_empty_covered.sum()):,} "
        f"mean_empty={out_df.soft_gt_frac_empty.mean():.4f} residual_mean={out_df.soft_gt_empty_residual.mean():.4f} "
        f"corr(empty,residual)={corr:.4f} in {time.time()-t0:.1f}s → {out}",
    )
    write_progress(run_dir, stage="save_empty_gt_done", completed=4, total=4)
    atomic_json(
        run_dir / "artifacts" / "soft_gt_empty_manifest.json",
        {
            "definition": "mean(week4 frac_empty_full) over inside_polygon AE cells",
            "frac_empty_cell": "clip(1 - frac_res - frac_nres, 0, 1) at 10 m",
            "n_buildings": int(len(out_df)),
            "n_covered": int(out_df.soft_gt_empty_covered.sum()),
            "mean_soft_gt_frac_empty": float(out_df.soft_gt_frac_empty.mean()),
            "mean_soft_gt_empty_residual": float(out_df.soft_gt_empty_residual.mean()),
            "corr_empty_vs_residual": corr,
            "path": str(out),
            "sidecar": str(side),
        },
    )
    return out


def build_training_table(run_dir: Path, empty_path: Path, s2_path: Path) -> pd.DataFrame:
    s2 = pd.read_parquet(s2_path)
    s2["id"] = s2["id"].astype(str)
    empty = pd.read_parquet(empty_path, columns=["id", "soft_gt_frac_empty", "soft_gt_empty_residual"])
    empty["id"] = empty["id"].astype(str)
    df = s2.merge(empty, on="id", how="inner")
    # Prefer week6 soft res/nres already on s2 table; ensure empty finite
    df = df[np.isfinite(df["soft_gt_frac_empty"])].copy()
    for c in feat_colnames():
        if c not in df.columns:
            raise RuntimeError(f"missing feature {c}")
        df[c] = df[c].astype(np.float32)
    for c in TARGET_COLS:
        df[c] = df[c].astype(np.float32)
    out = run_dir / "artifacts" / "train_buildings_ae_s2_empty.parquet"
    df.to_parquet(out, index=False)
    log(
        run_dir,
        f"training table n={len(df):,} features={len(feat_colnames())} "
        f"targets={TARGET_COLS} mean_empty={df['soft_gt_frac_empty'].mean():.4f}",
    )
    return df


def _loro_fold(payload: dict[str, Any]) -> dict[str, Any]:
    from xgboost import XGBRegressor

    hold = payload["hold"]
    train_path = Path(payload["train_path"])
    cols = payload["cols"]
    targets = payload["targets"]
    n_jobs = payload["n_jobs"]
    device = payload.get("device", "cpu")
    t0 = time.time()

    df = pd.read_parquet(train_path, columns=["region", *targets, *cols])
    te = (df["region"] == hold).to_numpy()
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)
    Xtr = np.nan_to_num(df.loc[tr, cols].to_numpy(np.float32), nan=0.0)
    Ytr = df.loc[tr, targets].to_numpy(np.float32)
    Xte = np.nan_to_num(df.loc[te, cols].to_numpy(np.float32), nan=0.0)
    Yte = df.loc[te, targets].to_numpy(np.float32)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    kwargs: dict[str, Any] = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
    )
    if device.startswith("cuda"):
        kwargs["device"] = device
    model = MultiOutputRegressor(XGBRegressor(**kwargs))
    model.fit(Xtr_s, Ytr)
    pred = np.clip(model.predict(Xte_s), 0, 1).astype(np.float32)

    mae_r = float(mean_absolute_error(Yte[:, 0], pred[:, 0]))
    mae_n = float(mean_absolute_error(Yte[:, 1], pred[:, 1]))
    mae_e = float(mean_absolute_error(Yte[:, 2], pred[:, 2]))
    return {
        "hold": hold,
        "te_idx": te_idx,
        "pred": pred,
        "mae_res": mae_r,
        "mae_nres": mae_n,
        "mae_empty": mae_e,
        "mae_mean": 0.5 * (mae_r + mae_n),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }


def train_loro_xgb(df: pd.DataFrame, run_dir: Path, fold_workers: int) -> pd.DataFrame:
    cols = feat_colnames()
    regions = sorted(df["region"].unique())
    train_path = run_dir / "artifacts" / "train_buildings_ae_s2_empty.parquet"
    oof = np.full((len(df), 3), np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)

    n_gpus = len(detect_gpus())
    log(run_dir, f"LORO 3-head folds={len(regions)} fold_workers={fold_workers} n_gpus={n_gpus}")

    payloads = []
    for i, hold in enumerate(regions):
        device = f"cuda:{i % n_gpus}" if n_gpus > 0 else "cpu"
        n_jobs = 1 if n_gpus > 0 else max(1, (os.cpu_count() or 8) // max(1, fold_workers))
        payloads.append(
            {
                "hold": hold,
                "train_path": str(train_path),
                "cols": cols,
                "targets": TARGET_COLS,
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
                        "mae_empty": res["mae_empty"],
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
                    f"nres={res['mae_nres']:.4f} empty={res['mae_empty']:.4f} "
                    f"n_te={res['n_test']:,} device={res['device']} in {res['seconds']}s",
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
        "soft_gt_frac_empty",
        "ae_frac_res",
        "ae_frac_nres",
        "ghsl_frac_res",
        "ghsl_frac_nres",
        "pls_frac_res",
        "pls_frac_nres",
    ]
    if "y_res" in df.columns:
        keep.append("y_res")
    out = df[[c for c in keep if c in df.columns] + [c for c in S2_FEATS if c in df.columns]].copy()
    out["s2e_frac_res"] = oof[:, 0]
    out["s2e_frac_nres"] = oof[:, 1]
    out["s2e_frac_empty"] = oof[:, 2]

    # also renormalized variant (optional side metric): soft-max style from 3 preds
    s = np.clip(oof.sum(axis=1, keepdims=True), 1e-6, None)
    out["s2e_renorm_frac_res"] = (oof[:, 0:1] / s).ravel()
    out["s2e_renorm_frac_nres"] = (oof[:, 1:2] / s).ravel()
    out["s2e_renorm_frac_empty"] = (oof[:, 2:3] / s).ravel()

    if S2_OOF.exists():
        s2p = pd.read_parquet(S2_OOF.resolve(), columns=["id", "s2_frac_res", "s2_frac_nres"])
        s2p["id"] = s2p["id"].astype(str)
        out = out.merge(s2p, on="id", how="left")
    if RING_OOF.exists():
        rk = pd.read_parquet(RING_OOF.resolve(), columns=["id", "ring_frac_res", "ring_frac_nres"])
        rk["id"] = rk["id"].astype(str)
        out = out.merge(rk, on="id", how="left")

    out.to_parquet(run_dir / "artifacts" / "buildings_oof_ae_s2_empty.parquet", index=False)
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
        "ae_s2_empty": ("s2e_frac_res", "s2e_frac_nres"),
        "ae_s2_empty_renorm": ("s2e_renorm_frac_res", "s2e_renorm_frac_nres"),
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

    empty_mae = {}
    if "s2e_frac_empty" in m.columns and "soft_gt_frac_empty" in m.columns:
        empty_mae["ae_s2_empty"] = mae_one(
            m["soft_gt_frac_empty"].to_numpy(float), m["s2e_frac_empty"].to_numpy(float)
        )
        empty_mae["ae_s2_empty_renorm"] = mae_one(
            m["soft_gt_frac_empty"].to_numpy(float), m["s2e_renorm_frac_empty"].to_numpy(float)
        )

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
        if "s2e_frac_empty" in sub.columns:
            row["empty_mae"] = mae_one(
                sub["soft_gt_frac_empty"].to_numpy(float), sub["s2e_frac_empty"].to_numpy(float)
            )
        metro.append(row)

    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_ae_s2_empty.parquet", index=False)
    for name in models:
        r = soft_pooled[name]
        log(run_dir, f"building {name:18s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    for name, r in empty_mae.items():
        log(run_dir, f"building empty {name:12s} mae={r['mae']:.4f} n={r['n']}")

    return {
        "n_shared": len(m),
        "coverage": float(np.mean(np.isfinite(m["s2e_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "empty_mae": empty_mae,
        "metro_soft": decorate_metro(metro),
        "s2e_beats_s2": sum(
            1
            for r in decorate_metro(metro)
            if "ae_s2_empty" in r
            and "ae_s2" in r
            and np.isfinite(r["ae_s2_empty"]["mean"])
            and np.isfinite(r["ae_s2"]["mean"])
            and r["ae_s2_empty"]["mean"] + 1e-12 < r["ae_s2"]["mean"]
        ),
    }


def score_tiles_from_buildings(run_dir: Path, building_oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_TILE)
    base = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    base_cols = [c for c in base.columns if not (c == "region" and "region" in cohort.columns)]
    t = cohort.merge(base[base_cols], on="tile_id", how="left")
    assert len(t) == 25220

    if XGB_TILE.exists():
        xb = pd.read_parquet(XGB_TILE)
        cols = ["tile_id"] + [c for c in xb.columns if c.startswith("xgb_")]
        t = t.merge(xb[cols], on="tile_id", how="left")

    s2_tile = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "tiles_section3_with_ae_s2.parquet"
    if s2_tile.exists():
        st = pd.read_parquet(s2_tile.resolve(), columns=["tile_id", "s2_frac_res", "s2_frac_nres"])
        t = t.merge(st, on="tile_id", how="left")

    bo = building_oof[
        ["id", "region", "s2e_frac_res", "s2e_frac_nres", "s2e_frac_empty", "s2e_renorm_frac_res", "s2e_renorm_frac_nres"]
    ].copy()
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
        .agg(
            s2e_frac_res=("s2e_frac_res", "mean"),
            s2e_frac_nres=("s2e_frac_nres", "mean"),
            s2e_frac_empty=("s2e_frac_empty", "mean"),
            s2e_renorm_frac_res=("s2e_renorm_frac_res", "mean"),
            s2e_renorm_frac_nres=("s2e_renorm_frac_nres", "mean"),
        )
        .reset_index()
    )
    drop = [c for c in agg.columns if c != "tile_id" and c in t.columns]
    t = t.drop(columns=drop, errors="ignore").merge(agg, on="tile_id", how="left")

    # tile soft empty residual for side metric if soft res/nres exist
    if "soft_gt_frac_res" in t.columns:
        t["soft_gt_frac_empty"] = np.clip(
            1.0 - t["soft_gt_frac_res"].to_numpy(float) - t["soft_gt_frac_nres"].to_numpy(float),
            0.0,
            1.0,
        )

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
        "ae_s2_empty": ("s2e_frac_res", "s2e_frac_nres"),
        "ae_s2_empty_renorm": ("s2e_renorm_frac_res", "s2e_renorm_frac_nres"),
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
    t.to_parquet(run_dir / "artifacts" / "tiles_section3_with_ae_s2_empty.parquet", index=False)
    for name in models:
        r = soft_pooled[name]
        log(run_dir, f"tile {name:18s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    return {
        "n_shared": len(t),
        "coverage": float(np.mean(np.isfinite(t["s2e_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "metro_soft": decorate_metro(metro),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week11_ae_s2_empty_head_loro")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument("--s2-train", type=Path, default=S2_TRAIN)
    ap.add_argument("--fold-workers", type=int, default=2)
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
        f"n_gpus={n_gpus}\ngpus={gpu_names}\ns2_train={args.s2_train}\n"
        f"protocol=docs/eval_protocol_apples_to_apples.md\n"
        f"targets={TARGET_COLS}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpus={os.cpu_count()}\nn_gpus={n_gpus}\ngpus={gpu_names}\npython={os.sys.executable}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="init",
        completed=0,
        total=6,
        failed=0,
        _t0=time.time(),
        output=str(run_dir),
        n_cpus=os.cpu_count(),
        n_gpus=n_gpus,
        fold_workers=args.fold_workers,
    )
    log(run_dir, f"Week11 AE+S2 empty-head LORO start → {run_dir} · GPUs={n_gpus} {gpu_names}")

    empty_path = save_building_soft_gt_empty(run_dir)
    write_progress(run_dir, stage="empty_saved", completed=1, total=6)

    s2_path = args.s2_train.resolve()
    if not s2_path.exists():
        raise FileNotFoundError(s2_path)
    df = build_training_table(run_dir, empty_path, s2_path)
    write_progress(run_dir, stage="train_table", completed=2, total=6, train_n=len(df))

    oof = train_loro_xgb(df, run_dir, fold_workers=args.fold_workers)
    write_progress(run_dir, stage="score", completed=4, total=6)

    b_metrics = score_buildings(run_dir, oof)
    t_metrics = score_tiles_from_buildings(run_dir, oof)

    metrics = {
        "protocol": "docs/eval_protocol_apples_to_apples.md",
        "train_n_buildings": int(len(df)),
        "eval_n_buildings": 71497,
        "eval_n_tiles": 25220,
        "features": feat_colnames(),
        "targets": TARGET_COLS,
        "empty_definition": "mean(week4 frac_empty_full) over inside_polygon AE cells",
        "buildings": b_metrics,
        "tiles": t_metrics,
        "checklist": {
            "eval_ids_locked": True,
            "soft_gt_from_week6": True,
            "soft_gt_empty_saved": True,
            "spatial_split": "LORO",
            "metric": "0.5*(MAE_res+MAE_nres)",
            "baselines_joined": True,
            "s2_features_reused_from_week9": True,
        },
    }
    atomic_json(run_dir / "artifacts" / "section8_ae_s2_empty_metrics.json", metrics)

    s2_mean = b_metrics["soft_pooled"].get("ae_s2", {}).get("mean")
    s2e_mean = b_metrics["soft_pooled"]["ae_s2_empty"]["mean"]
    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "finished_at": utc_now(),
            "output": str(run_dir),
            "building_mean_ae_s2_empty": s2e_mean,
            "building_mean_ae_s2": s2_mean,
            "building_delta_s2e_minus_s2": None if s2_mean is None else float(s2e_mean - s2_mean),
            "building_mae_empty": b_metrics.get("empty_mae", {}).get("ae_s2_empty", {}).get("mae"),
            "building_mean_ghsl": b_metrics["soft_pooled"]["ghsl_built_s"]["mean"],
            "tile_mean_ae_s2_empty": t_metrics["soft_pooled"]["ae_s2_empty"]["mean"],
            "tile_coverage": t_metrics["coverage"],
            "metrics": str(run_dir / "artifacts" / "section8_ae_s2_empty_metrics.json"),
            "soft_gt_empty_path": str(empty_path),
        },
    )
    write_progress(run_dir, status="completed", stage="done", completed=6, total=6, failed=0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, "DONE")

    try:
        board = ROOT / "scripts" / "rebuild_section8_ae_s2_empty_board.py"
        if board.exists():
            subprocess.check_call(
                [
                    os.sys.executable,
                    str(board),
                    "--metrics",
                    str(run_dir / "artifacts" / "section8_ae_s2_empty_metrics.json"),
                ],
                cwd=str(ROOT),
            )
            log(run_dir, "Section 8 canvas board injected")
    except Exception as e:
        log(run_dir, f"board inject skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
