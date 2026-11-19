#!/usr/bin/env python3
"""Week 7: LORO XGBoost soft-fraction regression (Ridge drop-in).

Same panel / targets / LORO folds as week4 Ridge:
  data   = datasets/week4_pixel_gt_balanced/pixels.parquet
  X      = A00–A63
  Y      = (frac_res_full, frac_nres_full, frac_empty_full)
  + binary XGBClassifier for side-car macro-F1

Also scores ALL pure cells in week4_pixel_gt → oof_predictions_full_gt.parquet
for Section 3 polygon joins.

Outputs: outputs/week7_xgb_loro/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# CUDA + fork is unsafe with XGBoost GPU; workers must spawn.
mp.set_start_method("spawn", force=True)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")


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
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done = int(prev["completed"])
            total = int(prev["total"])
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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def _xgb_regressor(device: str):
    from xgboost import XGBRegressor

    # Native multi-output (one booster) — avoids MultiOutputRegressor GPU teardown crashes
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
        n_jobs=max(1, (os.cpu_count() or 4) // max(2, 1)),
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
        n_jobs=max(1, (os.cpu_count() or 4) // 4),
        verbosity=0,
    )


def _fit_fold(payload: dict[str, Any]) -> dict[str, Any]:
    te_region = payload["te_region"]
    device = payload["device"]
    X = np.load(payload["x_path"], mmap_mode="r")
    Y = np.load(payload["y_path"], mmap_mode="r")
    y_bin = np.load(payload["ybin_path"], mmap_mode="r")
    regions = np.load(payload["reg_path"], allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = np.asarray(X[tr], dtype=np.float32)
    Y_tr = np.asarray(Y[tr], dtype=np.float32)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float32)
    y_te = np.asarray(Y[te], dtype=np.float32)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    # Scale on CPU; trees on assigned device
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

    cache = Path(payload["cache_dir"])
    ckpt = Path(payload["ckpt_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    pred_path = cache / f"oof_pred_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    ypred_path = cache / f"oof_ypred_{te_region}.npy"
    pres_path = cache / f"oof_pres_{te_region}.npy"
    np.save(pred_path, pred.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))
    np.save(ypred_path, y_pred)
    np.save(pres_path, p_res.astype(np.float64))

    # Wrap for Ridge-compatible predict API used in full-GT scoring
    pipe_reg = Pipeline([("scaler", scaler), ("xgb", reg)])
    # classifier needs transform then predict — store as dict
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
            "model": "xgboost_multioutput_v1",
        },
        ckpt / f"fold_{te_region}.joblib",
    )

    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(3)]
    return {
        "region": te_region,
        "device": device,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "y_pred_path": str(ypred_path),
        "p_res_path": str(pres_path),
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


def score_full_gt_oof(ckpt_dir: Path, out_path: Path, run_dir: Path) -> dict:
    log(run_dir, f"Scoring full GT pure cells → {out_path}")
    full = pd.read_parquet(FULL_GT)
    pure = full.loc[
        ((full["frac_res_full"] > 0) & (full["frac_nres_full"] == 0))
        | ((full["frac_nres_full"] > 0) & (full["frac_res_full"] == 0))
    ].copy()
    pure["y"] = (pure["frac_res_full"] > 0).astype(np.int64)
    pure["stratum"] = np.where(pure["y"] == 1, "residential", "non_residential")

    rows = []
    regions = sorted(pure["region"].astype(str).unique())
    for i, region in enumerate(regions):
        ck = ckpt_dir / f"fold_{region}.joblib"
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
        f1 = float(f1_score(part["y"], y_pred, average="macro", zero_division=0))
        log(run_dir, f"  {region:12s} n={len(part):,}  macroF1={f1:.3f}")
        write_progress(
            run_dir,
            stage="full_gt_oof",
            completed=i + 1,
            total=len(regions),
            last_region=region,
        )

    if not rows:
        raise RuntimeError("No full-GT OOF rows produced")
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(out_path, index=False)
    summary = {
        "n": int(len(oof)),
        "n_res": int((oof["y"] == 1).sum()),
        "n_nres": int((oof["y"] == 0).sum()),
        "macro_f1": float(f1_score(oof["y"], oof["y_pred"], average="macro", zero_division=0)),
        "path": str(out_path),
    }
    log(
        run_dir,
        f"Full-GT OOF n={summary['n']:,} RES={summary['n_res']:,} NRES={summary['n_nres']:,} "
        f"macroF1={summary['macro_f1']:.3f}",
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week7_xgb_loro")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu (default, parallel folds) or cuda / cuda:0 (sequential folds)",
    )
    ap.add_argument("--skip-full-gt", action="store_true")
    args = ap.parse_args()

    try:
        import torch

        n_gpus = int(torch.cuda.device_count())
    except Exception:
        n_gpus = 0
    import xgboost  # noqa: F401

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures", "_cache"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if not PIXELS.exists():
        raise FileNotFoundError(PIXELS)

    df = pd.read_parquet(PIXELS).reset_index(drop=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float32)
    y_bin = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].to_numpy()
    region_list = sorted(df["region"].astype(str).unique().tolist())

    # GPU hist + ProcessPool has been unstable here (cudaErrorIllegalAddress on
    # worker teardown). Default to CPU hist across all cores; --device cuda
    # keeps single-process sequential GPU if explicitly requested.
    use_gpu = bool(args.device.startswith("cuda") and n_gpus > 0)
    if args.workers:
        n_workers = args.workers
    elif use_gpu:
        n_workers = 1  # sequential folds on one GPU — avoids multi-proc CUDA crashes
    else:
        n_workers = min(len(region_list), os.cpu_count() or 1)
    total_steps = len(region_list) + (0 if args.skip_full_gt else 1) + 1

    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total_steps,
        n_gpus=n_gpus,
        workers=n_workers,
        device_mode=args.device,
        regions=region_list,
        xgboost_version=xgboost.__version__,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"argv={sys.argv}\nworkers={n_workers}\nn_gpus={n_gpus}\nn={len(df)}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"python={sys.executable}\nxgboost={xgboost.__version__}\nn_gpus={n_gpus}\n"
        f"cpus={os.cpu_count()}\n"
    )
    log(
        run_dir,
        f"Week7 XGB LORO · n={len(df):,} folds={len(region_list)} "
        f"workers={n_workers} gpus={n_gpus}",
    )

    cache = run_dir / "_cache"
    x_path = str(cache / "X.npy")
    y_path = str(cache / "Y.npy")
    ybin_path = str(cache / "y_bin.npy")
    reg_path = str(cache / "regions.npy")
    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(ybin_path, y_bin)
    np.save(reg_path, regions)

    jobs = []
    for i, r in enumerate(region_list):
        if use_gpu:
            if args.device == "cuda":
                device = f"cuda:{i % n_gpus}"
            else:
                device = args.device
        else:
            device = "cpu"
        jobs.append(
            {
                "te_region": r,
                "device": device,
                "x_path": x_path,
                "y_path": y_path,
                "ybin_path": ybin_path,
                "reg_path": reg_path,
                "cache_dir": str(cache),
                "ckpt_dir": str(run_dir / "checkpoints"),
            }
        )

    oof_pred = np.full((len(df), 3), np.nan, dtype=np.float64)
    oof_ypred = np.full(len(df), -1, dtype=np.int64)
    oof_pres = np.full(len(df), np.nan, dtype=np.float64)
    rows: list[dict] = []
    done = 0
    failed = 0

    write_progress(run_dir, stage="loro_folds", completed=0)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit_fold, job): job["te_region"] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="XGB LORO"):
            region = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                failed += 1
                log(run_dir, f"FAIL fold {region}: {e}")
                (run_dir / "failures" / f"fold_{region}.txt").write_text(str(e))
                done += 1
                write_progress(
                    run_dir, stage=f"fail_{region}", completed=done, failed=failed
                )
                continue
            te = np.load(r["idx_path"])
            oof_pred[te] = np.load(r["pred_path"])
            oof_ypred[te] = np.load(r["y_pred_path"])
            oof_pres[te] = np.load(r["p_res_path"])
            rows.append(r)
            done += 1
            log(
                run_dir,
                f"  {r['region']:12s} device={r['device']} mae_mean={r['mae_mean']:.3f} "
                f"macroF1={r['macro_f1']:.3f}",
            )
            write_progress(
                run_dir,
                stage=f"fold_{region}",
                completed=done,
                failed=failed,
                last_region=region,
            )

    if not rows:
        write_progress(run_dir, status="failed", stage="no_folds")
        return 1

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(run_dir / "artifacts" / "loro_metrics.csv", index=False)

    oof = df[
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
    oof["pred_frac_res"] = np.round(oof_pred[:, 0], 2)
    oof["pred_frac_nres"] = np.round(oof_pred[:, 1], 2)
    oof["pred_frac_empty"] = np.round(oof_pred[:, 2], 2)
    oof["pred_frac_res_full"] = oof_pred[:, 0]
    oof["pred_frac_nres_full"] = oof_pred[:, 1]
    oof["pred_frac_empty_full"] = oof_pred[:, 2]
    oof["y_pred"] = oof_ypred
    oof["p_res"] = oof_pres
    oof_path = run_dir / "artifacts" / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)
    # stable pointer for eval scripts
    latest = args.out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    yt, yp = Y, oof_pred
    pooled: dict[str, float] = {}
    for i, name in enumerate(FRAC_COLS):
        pooled[f"oof_mae_{name}"] = float(mean_absolute_error(yt[:, i], yp[:, i]))
        pooled[f"oof_rmse_{name}"] = _rmse(yt[:, i], yp[:, i])
        pooled[f"oof_r2_{name}"] = float(r2_score(yt[:, i], yp[:, i]))
    pooled["oof_mae_mean"] = float(np.mean([pooled[f"oof_mae_{c}"] for c in FRAC_COLS]))
    pooled["oof_macro_f1"] = float(f1_score(y_bin, oof_ypred, average="macro", zero_division=0))
    pooled["loro_mae_mean_mean"] = float(metrics["mae_mean"].mean())
    pooled["loro_mae_mean_std"] = float(metrics["mae_mean"].std(ddof=0))
    pooled["loro_macro_f1_mean"] = float(metrics["macro_f1"].mean())
    pooled["loro_macro_f1_std"] = float(metrics["macro_f1"].std(ddof=0))

    # figures
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    m = metrics.sort_values("mae_mean", ascending=False)
    ax.barh(m["region"].str.replace("_", " "), m["mae_mean"], color="#8E44AD")
    ax.axvline(pooled["loro_mae_mean_mean"], color="#333", ls="--", lw=1)
    ax.set_xlabel("Mean MAE (3 fractions)")
    ax.set_title("Week 7 XGB LORO · fraction MAE by held-out metro")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "loro_mae_by_region.png", dpi=140)
    plt.close(fig)

    full_sum = None
    if not args.skip_full_gt:
        write_progress(run_dir, stage="full_gt_oof", completed=done)
        full_path = run_dir / "artifacts" / "oof_predictions_full_gt.parquet"
        try:
            full_sum = score_full_gt_oof(run_dir / "checkpoints", full_path, run_dir)
            done += 1
        except Exception as e:
            failed += 1
            log(run_dir, f"FULL GT FAIL: {e}")
            (run_dir / "failures" / "full_gt.txt").write_text(str(e))

    summary = {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "name": "week7_xgb_loro_v1",
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "protocol": "leave-one-region-out",
        "regressor": "StandardScaler + XGBRegressor multi_strategy=one_output_per_tree",
        "classifier": "XGBClassifier hist",
        "seed": SEED,
        "n_gpus": n_gpus,
        "workers": n_workers,
        "n_pixels": int(len(df)),
        "regions": region_list,
        "failed_folds": failed,
        **pooled,
        "runtime_seconds": round(time.time() - t0, 1),
        "paths": {
            "oof": str(oof_path.relative_to(ROOT)),
            "metrics": "artifacts/loro_metrics.csv",
            "checkpoints": "checkpoints/",
            "figures": "figures/",
            "run_dir": str(run_dir.relative_to(ROOT)),
        },
    }
    if full_sum:
        summary["full_gt_oof"] = full_sum
        summary["paths"]["oof_full_gt"] = str(
            Path(full_sum["path"]).relative_to(ROOT)
        )
    atomic_json(run_dir / "summary.json", summary)
    write_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        completed=total_steps,
        total=total_steps,
        percent_complete=100.0,
        failed=failed,
    )
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(
        run_dir,
        f"DONE · oof_mae_mean={pooled['oof_mae_mean']:.4f} · "
        f"LORO F1={pooled['loro_macro_f1_mean']:.4f} · run_dir={run_dir}",
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
