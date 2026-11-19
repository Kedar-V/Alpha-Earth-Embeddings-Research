#!/usr/bin/env python3
"""Week 7 XGB LORO on GPU via *isolated subprocesses per fold*.

Does not use ProcessPoolExecutor (CUDA teardown crashes). Each fold is a
fresh Python process bound to one GPU. Up to n_gpus folds run concurrently.

Writes under outputs/week7_xgb_loro_gpu/ so the CPU train in
outputs/week7_xgb_loro/ is undisturbed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relpath(path: Path) -> str:
    """Prefer path relative to ROOT; fall back to absolute if symlink roots differ."""
    path = Path(path).resolve()
    root = ROOT.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def _xgb_regressor(device: str):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        multi_strategy="one_output_per_tree",
        tree_method="hist",
        device=device,
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


def _xgb_classifier(device: str):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


def run_fold_worker(args: argparse.Namespace) -> int:
    """Single-fold entrypoint (fresh process, one GPU)."""
    run_dir = Path(args.run_dir)
    te_region = args.fold
    device = args.device
    cache = run_dir / "_cache"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    X = np.load(cache / "X.npy", mmap_mode="r")
    Y = np.load(cache / "Y.npy", mmap_mode="r")
    y_bin = np.load(cache / "y_bin.npy", mmap_mode="r")
    regions = np.load(cache / "regions.npy", allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = np.asarray(X[tr], dtype=np.float32)
    Y_tr = np.asarray(Y[tr], dtype=np.float32)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float32)
    y_te = np.asarray(Y[te], dtype=np.float32)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    reg = _xgb_regressor(device)
    reg.fit(X_tr_s, Y_tr)
    pred = np.clip(np.asarray(reg.predict(X_te_s), dtype=np.float64), 0.0, 1.0)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    clf = _xgb_classifier(device)
    clf.fit(X_tr_s, y_tr)
    y_pred = np.asarray(clf.predict(X_te_s), dtype=np.int64)
    p_res = np.asarray(clf.predict_proba(X_te_s)[:, 1], dtype=np.float64)

    pipe_reg = Pipeline([("scaler", scaler), ("xgb", reg)])
    joblib.dump(
        {
            "region_held_out": te_region,
            "regressor": pipe_reg,
            "scaler": scaler,
            "classifier": clf,
            "device": device,
            "seed": SEED,
            "n_train": int(tr.sum()),
            "n_train_res": int((y_tr == 1).sum()),
            "n_train_nres": int((y_tr == 0).sum()),
            "model": "xgboost_gpu_subprocess_v1",
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    np.save(cache / f"oof_pred_{te_region}.npy", pred)
    np.save(cache / f"oof_idx_{te_region}.npy", np.where(te)[0].astype(np.int64))
    np.save(cache / f"oof_ypred_{te_region}.npy", y_pred)
    np.save(cache / f"oof_pres_{te_region}.npy", p_res)

    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(3)]
    metrics = {
        "region": te_region,
        "device": device,
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
        "n_train_res": int((y_tr == 1).sum()),
        "n_train_nres": int((y_tr == 0).sum()),
        "mae_frac_res": mae[0],
        "mae_frac_nres": mae[1],
        "mae_frac_empty": mae[2],
        "mae_mean": float(np.mean(mae)),
        "macro_f1": float(f1_score(yb_te, y_pred, average="macro", zero_division=0)),
    }
    atomic_json(ckpt / f"fold_{te_region}_metrics.json", metrics)
    print(json.dumps(metrics), flush=True)
    return 0


def score_full_gt(run_dir: Path) -> dict:
    log(run_dir, "Scoring full GT pure cells with fold checkpoints")
    full = pd.read_parquet(FULL_GT)
    pure = full.loc[
        ((full["frac_res_full"] > 0) & (full["frac_nres_full"] == 0))
        | ((full["frac_nres_full"] > 0) & (full["frac_res_full"] == 0))
    ].copy()
    pure["y"] = (pure["frac_res_full"] > 0).astype(np.int64)
    pure["stratum"] = np.where(pure["y"] == 1, "residential", "non_residential")
    rows = []
    ckpt = run_dir / "checkpoints"
    for region in sorted(pure["region"].astype(str).unique()):
        ck = ckpt / f"fold_{region}.joblib"
        if not ck.exists():
            log(run_dir, f"  skip {region}: missing checkpoint")
            continue
        bundle = joblib.load(ck)
        sub = pure.loc[pure["region"].astype(str) == region]
        X = sub[BANDS].to_numpy(dtype=np.float32)
        pred = np.clip(bundle["regressor"].predict(X), 0.0, 1.0)
        Xs = bundle["scaler"].transform(X)
        p_res = bundle["classifier"].predict_proba(Xs)[:, 1]
        y_pred = bundle["classifier"].predict(Xs).astype(np.int64)
        part = sub[
            [
                "pixel_id",
                "region",
                "stratum",
                "y",
                "frac_res",
                "frac_nres",
                "frac_empty",
                "frac_res_full",
                "frac_nres_full",
                "frac_empty_full",
                "pixel_lon",
                "pixel_lat",
            ]
        ].copy()
        part["pred_frac_res"] = np.round(pred[:, 0], 2)
        part["pred_frac_nres"] = np.round(pred[:, 1], 2)
        part["pred_frac_empty"] = np.round(pred[:, 2], 2)
        part["pred_frac_res_full"] = pred[:, 0]
        part["pred_frac_nres_full"] = pred[:, 1]
        part["pred_frac_empty_full"] = pred[:, 2]
        part["y_pred"] = y_pred
        part["p_res"] = p_res
        rows.append(part)
        log(run_dir, f"  {region:12s} n={len(part):,}")
    oof = pd.concat(rows, ignore_index=True)
    out = run_dir / "artifacts" / "oof_predictions_full_gt.parquet"
    oof.to_parquet(out, index=False)
    return {
        "n": int(len(oof)),
        "macro_f1": float(f1_score(oof["y"], oof["y_pred"], average="macro", zero_division=0)),
        "path": str(out),
    }


def _launch_fold(py: str, script: Path, run_dir: Path, region: str, device: str) -> dict:
    log_path = run_dir / "logs" / f"fold_{region}.log"
    env = os.environ.copy()
    # Pin one physical GPU per subprocess; inside it always address cuda:0.
    worker_device = "cuda:0"
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    cmd = [
        py,
        "-u",
        str(script),
        "--fold-worker",
        "--run-dir",
        str(run_dir),
        "--fold",
        region,
        "--device",
        worker_device,
    ]
    with log_path.open("w") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    metrics_path = run_dir / "checkpoints" / f"fold_{region}_metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        raise RuntimeError(f"fold {region} failed rc={proc.returncode}; see {log_path}")
    metrics = json.loads(metrics_path.read_text())
    metrics["device"] = device  # physical GPU assignment
    return metrics


def main_orchestrator(args: argparse.Namespace) -> int:
    import torch
    import xgboost

    n_gpus = int(torch.cuda.device_count())
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures", "_cache"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_parquet(PIXELS).reset_index(drop=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float32)
    y_bin = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].to_numpy()
    region_list = sorted(df["region"].astype(str).unique().tolist())

    cache = run_dir / "_cache"
    np.save(cache / "X.npy", X)
    np.save(cache / "Y.npy", Y)
    np.save(cache / "y_bin.npy", y_bin)
    np.save(cache / "regions.npy", regions)
    meta_cols = [
        "pixel_id",
        "region",
        "stratum",
        "y",
        "frac_res",
        "frac_nres",
        "frac_empty",
        "frac_res_full",
        "frac_nres_full",
        "frac_empty_full",
        "pixel_lon",
        "pixel_lat",
    ]
    df[meta_cols].to_parquet(cache / "meta.parquet", index=False)

    n_parallel = args.workers or n_gpus
    total = len(region_list) + (0 if args.skip_full_gt else 1) + 1
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total,
        n_gpus=n_gpus,
        workers=n_parallel,
        backend="subprocess_per_fold",
        xgboost_version=xgboost.__version__,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"argv={sys.argv}\nn_gpus={n_gpus}\nparallel={n_parallel}\n")
    log(
        run_dir,
        f"GPU XGB LORO · n={len(df):,} folds={len(region_list)} "
        f"parallel={n_parallel} gpus={n_gpus} (subprocess isolation)",
    )

    script = Path(__file__).resolve()
    py = sys.executable
    jobs = [(r, f"cuda:{i % n_gpus}") for i, r in enumerate(region_list)]
    rows = []
    done = 0
    failed = 0

    # Thread pool only schedules subprocesses; each fold is its own process.
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futs = {
            pool.submit(_launch_fold, py, script, run_dir, region, device): region
            for region, device in jobs
        }
        for fut in as_completed(futs):
            region = futs[fut]
            try:
                metrics = fut.result()
                rows.append(metrics)
                done += 1
                log(
                    run_dir,
                    f"  {metrics['region']:12s} device={metrics['device']} "
                    f"mae_mean={metrics['mae_mean']:.3f} macroF1={metrics['macro_f1']:.3f}",
                )
                write_progress(
                    run_dir,
                    stage=f"fold_{region}",
                    completed=done,
                    failed=failed,
                    last_region=region,
                )
            except Exception as e:
                failed += 1
                done += 1
                log(run_dir, f"FAIL {region}: {e}")
                (run_dir / "failures" / f"fold_{region}.txt").write_text(str(e))
                write_progress(run_dir, stage=f"fail_{region}", completed=done, failed=failed)

    if not rows:
        write_progress(run_dir, status="failed", stage="no_folds")
        return 1

    metrics_df = pd.DataFrame(rows).sort_values("region")
    metrics_df.to_csv(run_dir / "artifacts" / "loro_metrics.csv", index=False)

    # assemble balanced OOF
    meta = pd.read_parquet(cache / "meta.parquet")
    oof_pred = np.full((len(meta), 3), np.nan)
    oof_ypred = np.full(len(meta), -1, dtype=np.int64)
    oof_pres = np.full(len(meta), np.nan)
    for r in rows:
        region = r["region"]
        idx = np.load(cache / f"oof_idx_{region}.npy")
        oof_pred[idx] = np.load(cache / f"oof_pred_{region}.npy")
        oof_ypred[idx] = np.load(cache / f"oof_ypred_{region}.npy")
        oof_pres[idx] = np.load(cache / f"oof_pres_{region}.npy")
    oof = meta.copy()
    oof["pred_frac_res"] = np.round(oof_pred[:, 0], 2)
    oof["pred_frac_nres"] = np.round(oof_pred[:, 1], 2)
    oof["pred_frac_empty"] = np.round(oof_pred[:, 2], 2)
    oof["pred_frac_res_full"] = oof_pred[:, 0]
    oof["pred_frac_nres_full"] = oof_pred[:, 1]
    oof["pred_frac_empty_full"] = oof_pred[:, 2]
    oof["y_pred"] = oof_ypred
    oof["p_res"] = oof_pres
    oof.to_parquet(run_dir / "artifacts" / "oof_predictions.parquet", index=False)

    yt = Y
    pooled = {
        "oof_mae_mean": float(
            np.mean([mean_absolute_error(yt[:, i], oof_pred[:, i]) for i in range(3)])
        ),
        "oof_macro_f1": float(f1_score(y_bin, oof_ypred, average="macro", zero_division=0)),
        "loro_mae_mean_mean": float(metrics_df["mae_mean"].mean()),
        "loro_macro_f1_mean": float(metrics_df["macro_f1"].mean()),
    }
    for i, name in enumerate(FRAC_COLS):
        pooled[f"oof_mae_{name}"] = float(mean_absolute_error(yt[:, i], oof_pred[:, i]))
        pooled[f"oof_r2_{name}"] = float(r2_score(yt[:, i], oof_pred[:, i]))

    full_sum = None
    if not args.skip_full_gt and failed == 0:
        write_progress(run_dir, stage="full_gt_oof", completed=done)
        full_sum = score_full_gt(run_dir)
        done += 1

    summary = {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "name": "week7_xgb_loro_gpu_subprocess_v1",
        "backend": "subprocess_per_fold",
        "n_gpus": n_gpus,
        "workers": n_parallel,
        "failed_folds": failed,
        **pooled,
        "runtime_seconds": round(time.time() - t0, 1),
        "paths": {
            "oof": _relpath(run_dir / "artifacts" / "oof_predictions.parquet"),
            "run_dir": _relpath(run_dir),
        },
    }
    if full_sum:
        summary["full_gt_oof"] = full_sum
        summary["paths"]["oof_full_gt"] = _relpath(Path(full_sum["path"]))
    atomic_json(run_dir / "summary.json", summary)
    write_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        completed=total,
        total=total,
        percent_complete=100.0,
        failed=failed,
    )
    (run_dir / "DONE").write_text(utc_now() + "\n")
    latest = args.out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    log(run_dir, f"DONE · oof_mae_mean={pooled['oof_mae_mean']:.4f} · {run_dir}")
    return 0 if failed == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-worker", action="store_true")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week7_xgb_loro_gpu")
    ap.add_argument("--workers", type=int, default=0, help="concurrent fold subprocesses (default=n_gpus)")
    ap.add_argument("--skip-full-gt", action="store_true")
    args = ap.parse_args()
    if args.fold_worker:
        if not args.run_dir or not args.fold:
            raise SystemExit("--fold-worker requires --run-dir and --fold")
        return run_fold_worker(args)
    return main_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
