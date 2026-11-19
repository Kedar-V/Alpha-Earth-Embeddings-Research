#!/usr/bin/env python3
"""Score ALL week4 full-GT cells with Week7 XGB fold checkpoints on GPU.

Uses isolated subprocesses per fold (same pattern as train_week7_xgb_loro_gpu.py)
so CUDA teardown stays safe. Up to n_gpus folds run concurrently.

Writes:
  <ckpt-run>/artifacts/oof_predictions_full_gt_all_cells.parquet
  and replaces oof_predictions_full_gt.parquet (pure-only archived if present).
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
import torch

ROOT = Path(__file__).resolve().parents[1]
FULL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
BANDS = [f"A{i:02d}" for i in range(64)]
META_COLS = [
    "pixel_id",
    "region",
    "frac_res",
    "frac_nres",
    "frac_empty",
    "frac_res_full",
    "frac_nres_full",
    "frac_empty_full",
    "pixel_lon",
    "pixel_lat",
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


def _force_xgb_device(est, device: str) -> None:
    """Ensure loaded booster / sklearn wrapper uses the assigned GPU."""
    try:
        est.set_params(device=device, tree_method="hist")
    except Exception:
        pass
    # Booster may still be on old device after joblib load
    booster = getattr(est, "get_booster", lambda: None)()
    if booster is not None:
        try:
            booster.set_param({"device": device, "tree_method": "hist"})
        except Exception:
            pass


def run_fold_worker(args: argparse.Namespace) -> int:
    region = args.fold
    device = args.device  # always cuda:0 after CUDA_VISIBLE_DEVICES remap
    ckpt_run = Path(args.ckpt_run)
    out_part = Path(args.out_part)
    out_part.parent.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(FULL_GT)
    sub = full.loc[full["region"].astype(str) == region].copy()
    del full
    n = len(sub)
    print(f"[{utc_now()}] fold={region} n={n:,} device={device}", flush=True)

    bundle = joblib.load(ckpt_run / "checkpoints" / f"fold_{region}.joblib")
    reg = bundle["regressor"]
    clf = bundle["classifier"]
    # Pipeline: scaler + xgb
    if hasattr(reg, "named_steps") and "xgb" in reg.named_steps:
        _force_xgb_device(reg.named_steps["xgb"], device)
    else:
        _force_xgb_device(reg, device)
    _force_xgb_device(clf, device)

    X = sub[BANDS].to_numpy(dtype=np.float32)
    pred = np.clip(np.asarray(reg.predict(X), dtype=np.float64), 0.0, 1.0)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    Xs = bundle["scaler"].transform(X)
    p_res = np.asarray(clf.predict_proba(Xs)[:, 1], dtype=np.float64)
    y_pred = np.asarray(clf.predict(Xs), dtype=np.int64)

    pure_res = (sub["frac_res_full"] > 0) & (sub["frac_nres_full"] == 0)
    pure_nres = (sub["frac_nres_full"] > 0) & (sub["frac_res_full"] == 0)
    y = np.where(pure_res, 1, np.where(pure_nres, 0, -1)).astype(np.int64)
    stratum = np.where(pure_res, "residential", np.where(pure_nres, "non_residential", "mixed"))

    part = sub[META_COLS].copy()
    part["stratum"] = stratum
    part["y"] = y
    part["pred_frac_res"] = np.round(pred[:, 0], 2)
    part["pred_frac_nres"] = np.round(pred[:, 1], 2)
    part["pred_frac_empty"] = np.round(pred[:, 2], 2)
    part["pred_frac_res_full"] = pred[:, 0]
    part["pred_frac_nres_full"] = pred[:, 1]
    part["pred_frac_empty_full"] = pred[:, 2]
    part["y_pred"] = y_pred
    part["p_res"] = p_res
    part.to_parquet(out_part, index=False)
    metrics = {
        "region": region,
        "n": int(n),
        "n_mixed": int((stratum == "mixed").sum()),
        "n_res": int((stratum == "residential").sum()),
        "n_nres": int((stratum == "non_residential").sum()),
        "device": device,
        "path": str(out_part),
    }
    print(json.dumps(metrics), flush=True)
    return 0


def _launch_fold(
    py: str,
    script: Path,
    job_dir: Path,
    ckpt_run: Path,
    region: str,
    device: str,
    parts_dir: Path,
) -> dict:
    log_path = job_dir / "logs" / f"fold_{region}.log"
    out_part = parts_dir / f"region_{region}.parquet"
    env = os.environ.copy()
    worker_device = "cuda:0"
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    cmd = [
        py,
        "-u",
        str(script),
        "--fold-worker",
        "--ckpt-run",
        str(ckpt_run),
        "--fold",
        region,
        "--device",
        worker_device,
        "--out-part",
        str(out_part),
    ]
    with log_path.open("w") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0 or not out_part.exists():
        raise RuntimeError(f"fold {region} failed rc={proc.returncode}; see {log_path}")
    # last JSON line from worker
    metrics = {"region": region, "device": device, "path": str(out_part)}
    for line in reversed(log_path.read_text().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                metrics.update(json.loads(line))
            except Exception:
                pass
            break
    metrics["device"] = device
    return metrics


def main_orchestrator(args: argparse.Namespace) -> int:
    n_gpus = int(torch.cuda.device_count())
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible")

    ckpt_run = args.ckpt_run.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures", "_parts"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    regions = sorted(
        p.name[len("fold_") : -len(".joblib")]
        for p in (ckpt_run / "checkpoints").glob("fold_*.joblib")
        if not p.name.endswith("_metrics.json")
    )
    if not regions:
        raise SystemExit(f"No fold checkpoints in {ckpt_run / 'checkpoints'}")

    t0 = time.time()
    n_parallel = args.workers or n_gpus
    total = len(regions) + 1  # folds + assemble
    write_progress(
        job_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total,
        n_gpus=n_gpus,
        workers=n_parallel,
        ckpt_run=str(ckpt_run),
        backend="subprocess_per_fold_gpu",
    )
    (job_dir / "job.pid").write_text(str(os.getpid()))
    (job_dir / "launch.txt").write_text(
        f"argv={sys.argv}\nn_gpus={n_gpus}\nparallel={n_parallel}\nckpt={ckpt_run}\n"
    )
    log(
        job_dir,
        f"Score ALL full-GT cells · folds={len(regions)} parallel={n_parallel} "
        f"gpus={n_gpus} ckpt={ckpt_run.name}",
    )
    print(f"GPUs: {n_gpus} · {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}", flush=True)

    script = Path(__file__).resolve()
    py = sys.executable
    jobs = [(r, f"cuda:{i % n_gpus}") for i, r in enumerate(regions)]
    parts_dir = job_dir / "_parts"
    rows = []
    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futs = {
            pool.submit(
                _launch_fold, py, script, job_dir, ckpt_run, region, device, parts_dir
            ): region
            for region, device in jobs
        }
        for fut in as_completed(futs):
            region = futs[fut]
            try:
                metrics = fut.result()
                rows.append(metrics)
                done += 1
                log(
                    job_dir,
                    f"  {metrics['region']:12s} device={metrics['device']} "
                    f"n={metrics.get('n', '?'):,} mixed={metrics.get('n_mixed', '?')}",
                )
                write_progress(
                    job_dir,
                    stage=f"fold_{region}",
                    completed=done,
                    failed=failed,
                    last_region=region,
                )
            except Exception as e:
                failed += 1
                done += 1
                log(job_dir, f"FAIL {region}: {e}")
                (job_dir / "failures" / f"fold_{region}.txt").write_text(str(e))
                write_progress(job_dir, stage=f"fail_{region}", completed=done, failed=failed)

    if failed:
        write_progress(job_dir, status="failed", stage="folds_failed", failed=failed)
        return 1

    write_progress(job_dir, stage="assemble", completed=done)
    log(job_dir, "Assembling all-cell OOF parquet")
    parts = [pd.read_parquet(r["path"]) for r in sorted(rows, key=lambda x: x["region"])]
    oof = pd.concat(parts, ignore_index=True)
    out_all = job_dir / "artifacts" / "oof_predictions_full_gt_all_cells.parquet"
    oof.to_parquet(out_all, index=False)

    # Promote into ckpt run artifacts for eval resolve path
    art = ckpt_run / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    canonical = art / "oof_predictions_full_gt.parquet"
    pure_archive = art / "oof_predictions_full_gt_pure_only.parquet"
    if canonical.exists() and not pure_archive.exists():
        # only archive if it looks like pure-only (much smaller than all cells)
        try:
            old_n = len(pd.read_parquet(canonical, columns=["pixel_id"]))
        except Exception:
            old_n = -1
        if old_n > 0 and old_n < len(oof) * 0.9:
            canonical.rename(pure_archive)
            log(job_dir, f"Archived pure-only OOF ({old_n:,}) → {pure_archive.name}")
    oof.to_parquet(canonical, index=False)
    # also keep a copy under job_dir name in ckpt artifacts
    oof.to_parquet(art / "oof_predictions_full_gt_all_cells.parquet", index=False)

    summary = {
        "status": "completed",
        "name": "week7_xgb_score_full_gt_all_cells_gpu",
        "n_rows": int(len(oof)),
        "n_regions": len(rows),
        "n_gpus": n_gpus,
        "workers": n_parallel,
        "ckpt_run": str(ckpt_run),
        "stratum_counts": oof["stratum"].value_counts().to_dict(),
        "runtime_seconds": round(time.time() - t0, 1),
        "paths": {
            "oof_all_cells_job": str(out_all),
            "oof_all_cells_ckpt": str(art / "oof_predictions_full_gt_all_cells.parquet"),
            "oof_full_gt_canonical": str(canonical),
            "run_dir": str(job_dir),
        },
    }
    atomic_json(job_dir / "summary.json", summary)
    atomic_json(art / "all_cells_score_summary.json", summary)
    write_progress(
        job_dir,
        status="completed",
        stage="done",
        completed=total,
        total=total,
        percent_complete=100.0,
        failed=0,
        n_rows=len(oof),
    )
    (job_dir / "DONE").write_text(utc_now() + "\n")
    latest = args.out_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(job_dir.name)
    log(job_dir, f"DONE · n={len(oof):,} · {out_all}")
    log(job_dir, f"Canonical OOF → {canonical}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-worker", action="store_true")
    ap.add_argument("--ckpt-run", type=Path, required=True)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out-part", type=Path, default=None)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs" / "week7_xgb_score_full_gt_all_cells",
    )
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if args.fold_worker:
        if not args.fold or not args.out_part:
            raise SystemExit("--fold-worker requires --fold and --out-part")
        return run_fold_worker(args)
    return main_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
