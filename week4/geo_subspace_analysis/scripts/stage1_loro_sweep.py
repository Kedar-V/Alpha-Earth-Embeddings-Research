#!/usr/bin/env python3
"""Stage 1: LORO 64-D logistic regression, plus PCA-k and PLS-k logistic regression.

Scaler, PCA, and PLS are fit on the training metros only. Threshold is 0.5.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import lib

_STATE: dict = {}


def _fold(held_out: str) -> dict:
    t0 = time.time()
    X = _STATE["X"]
    y = _STATE["y"]
    regions = _STATE["regions"]
    ks = _STATE["ks"]
    fold_dir = Path(_STATE["fold_root"]) / held_out
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "train.log"
    lib.log(log_path, f"start held_out={held_out} n={len(y)} n_train={int((regions!=held_out).sum())} n_test={int((regions==held_out).sum())}")
    test_idx = np.where(regions == held_out)[0]
    train_idx = np.where(regions != held_out)[0]
    lib.assert_fold_split(regions, train_idx, test_idx, held_out)
    y_tr = y[train_idx]
    y_te = y[test_idx]
    X_tr = np.asarray(X[train_idx], dtype=np.float64)
    X_te = np.asarray(X[test_idx], dtype=np.float64)
    X_tr_s, X_te_s, scaler = lib.scale_train(X_tr, X_te)
    # Leak check: scaler must match the training block only.
    if not np.allclose(scaler.mean_, X_tr.mean(axis=0), atol=1e-5):
        raise AssertionError("scaler mean is not the training mean")

    rows = []
    lib.log(log_path, "fitting lr64")
    pred_b, proba_b, clf = lib.fit_lr(X_tr_s, y_tr, X_te_s)
    m = lib.binary_metrics(y_te, pred_b, proba_b)
    m.update({"region": held_out, "model": "lr64", "k": 64})
    rows.append(m)
    lib.log(log_path, f"lr64 macro_f1={m['macro_f1']:.4f} auroc={m['auroc']:.4f}")

    basis_pca = lib.pca_basis(X_tr_s)
    rot_pls = lib.pls_rotations(X_tr_s, y_tr, n_components=max(ks))
    pred_store = {"y": y_te.astype(np.int64), "idx": test_idx.astype(np.int64), "p_lr64": proba_b.astype(np.float32)}
    coefs = {"lr64_coef": clf.coef_.ravel().astype(np.float64), "lr64_intercept": np.array([clf.intercept_[0]])}

    for kind, basis in (("pca", basis_pca), ("pls", rot_pls)):
        Z_tr_full = X_tr_s @ basis
        Z_te_full = X_te_s @ basis
        for k in ks:
            kk = min(k, basis.shape[1])
            pred, proba, clf_k = lib.fit_lr(Z_tr_full[:, :kk], y_tr, Z_te_full[:, :kk])
            mk = lib.binary_metrics(y_te, pred, proba)
            mk.update({"region": held_out, "model": kind, "k": int(kk)})
            rows.append(mk)
            pred_store[f"p_{kind}_{kk}"] = proba.astype(np.float32)
            coefs[f"{kind}_{kk}_coef"] = clf_k.coef_.ravel().astype(np.float64)
            coefs[f"{kind}_{kk}_intercept"] = np.array([float(clf_k.intercept_[0])])
        lib.log(log_path, f"{kind} done k={list(ks)} last_f1={rows[-1]['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    lib.atomic_df(fold_dir / "metrics.csv", df)
    np.savez_compressed(fold_dir / "predictions.npz", **pred_store)
    np.savez_compressed(
        fold_dir / "bases.npz",
        pca=basis_pca.astype(np.float32),
        pls=rot_pls.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        **coefs,
    )
    elapsed = time.time() - t0
    lib.atomic_json(
        fold_dir / "summary.json",
        {
            "status": "completed",
            "region": held_out,
            "best_metric": "macro_f1",
            "best_metric_value": m["macro_f1"],
            "higher_is_better": True,
            "runtime_seconds": elapsed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
    )
    (fold_dir / "DONE").write_text("ok\n")
    lib.log(log_path, f"DONE seconds={elapsed:.1f}")
    return {"region": held_out, "rows": rows, "seconds": elapsed}


def _aggregate(run_dir: Path, frames: list[pd.DataFrame]) -> None:
    df = pd.concat(frames, ignore_index=True)
    lib.atomic_df(run_dir / "metrics.csv", df)
    base = df[df["model"] == "lr64"].set_index("region")["macro_f1"]
    rows = []
    for model in ("pca", "pls"):
        sub = df[df["model"] == model]
        for k, g in sub.groupby("k"):
            aligned = g.set_index("region")["macro_f1"].reindex(base.index)
            retention = aligned / base
            rows.append(
                {
                    "model": model,
                    "k": int(k),
                    "macro_f1_mean": float(aligned.mean()),
                    "macro_f1_std": float(aligned.std(ddof=1)),
                    "retention_mean": float(retention.mean()),
                    "retention_of_mean": float(aligned.mean() / base.mean()),
                    "auroc_mean": float(g["auroc"].mean()),
                }
            )
    summary_k = pd.DataFrame(rows).sort_values(["model", "k"])
    lib.atomic_df(run_dir / "k_summary.csv", summary_k)
    base_mean = float(base.mean())
    picks = {"baseline_macro_f1_mean": base_mean, "baseline_macro_f1_std": float(base.std(ddof=1))}
    for model in ("pca", "pls"):
        sub = summary_k[summary_k["model"] == model].sort_values("k")
        picks[model] = {
            "smallest_k_retention_ge_0.95": _first_k(sub, sub["retention_of_mean"] >= 0.95),
            "smallest_k_within_0.01": _first_k(sub, sub["macro_f1_mean"] >= base_mean - 0.01),
            "smallest_k_within_0.02": _first_k(sub, sub["macro_f1_mean"] >= base_mean - 0.02),
        }
    # Metro-level bootstrap of the mean F1 (the geographic unit is the metro).
    rng = np.random.default_rng(lib.SEED)
    boot = []
    values = base.to_numpy()
    for _ in range(2000):
        boot.append(float(rng.choice(values, size=len(values), replace=True).mean()))
    boot = np.asarray(boot)
    picks["baseline_metro_bootstrap_mean_ci95"] = [
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]
    lib.atomic_json(run_dir / "k_picks.json", picks)
    per = df[df["model"] == "lr64"][
        [
            "region",
            "macro_f1",
            "f1_residential",
            "f1_nonresidential",
            "accuracy",
            "balanced_accuracy",
            "auroc",
            "n",
            "n_residential",
            "n_nonresidential",
        ]
    ].sort_values("region")
    lib.atomic_df(run_dir / "baseline_loro.csv", per)


def _first_k(sub: pd.DataFrame, mask: pd.Series):
    hit = sub.loc[mask.values]
    if hit.empty:
        return None
    return int(hit["k"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "baseline_loro",
            "smoke": args.smoke,
            "workers": args.workers,
            "k": list(lib.K_SWEEP),
            "command": " ".join(sys.argv),
            "gpus": 0,
        },
    )
    log_path = run_dir / "train.log"
    try:
        lib.log(log_path, f"loading cache workers={args.workers} smoke={args.smoke}")
        X, y, regions = lib.load_arrays()
        metros = lib.metro_list(regions)
        ks = list(lib.K_SWEEP)
        if args.smoke:
            rng = np.random.default_rng(lib.SEED)
            keep = []
            for region in metros:
                for lab in (0, 1):
                    idx = np.where((regions == region) & (y == lab))[0]
                    keep.append(rng.choice(idx, size=min(1500, len(idx)), replace=False))
            sel = np.concatenate(keep)
            X = np.asarray(X[sel])
            y = y[sel]
            regions = regions[sel]
            metros = metros[:2]
            ks = [1, 2, 64]
            lib.log(log_path, f"smoke n={len(y)} metros={metros} k={ks}")
        _STATE["X"] = X
        _STATE["y"] = y
        _STATE["regions"] = regions
        _STATE["ks"] = ks
        _STATE["fold_root"] = str(run_dir / "folds")
        frames = []
        t0 = time.time()
        workers = min(args.workers, len(metros))
        lib.log(log_path, f"CPUs={os.cpu_count()} OMP={os.environ.get('OMP_NUM_THREADS')} fold_workers={workers}")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_fold, region): region for region in metros}
            done = 0
            for fut in as_completed(futs):
                region = futs[fut]
                result = fut.result()
                frames.append(pd.DataFrame(result["rows"]))
                done += 1
                lib.atomic_df(run_dir / "metrics.csv", pd.concat(frames, ignore_index=True))
                lib.log(log_path, f"fold {done}/{len(metros)} {region} seconds={result['seconds']:.1f}")
        _aggregate(run_dir, frames)
        lib.atomic_json(
            run_dir / "summary.json",
            {
                "status": "completed",
                "runtime_seconds": time.time() - t0,
                "n_metros": len(metros),
                "baseline_csv": "baseline_loro.csv",
                "k_summary": "k_summary.csv",
            },
        )
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
