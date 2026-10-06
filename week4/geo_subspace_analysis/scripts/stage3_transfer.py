#!/usr/bin/env python3
"""Stage 3: directional transfer. Train on metro A only, test on metro B.

Off-diagonal uses every pixel in A. The diagonal trains on an 80% stratified
split of B and tests on the other 20%. Threshold stays at 0.5.
"""

from __future__ import annotations

import argparse
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
MODELS = (("lr64", "lr64", 64), ("pca", "pca", 2), ("pca", "pca", 9), ("pls", "pls", 2), ("pls", "pls", 9))


def _split(y_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(lib.SEED)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y_local == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(0.2 * len(idx))))
        te.append(idx[:n_te])
        tr.append(idx[n_te:])
    train = np.concatenate(tr)
    test = np.concatenate(te)
    if np.intersect1d(train, test).size:
        raise AssertionError("diagonal split overlap")
    return train, test


def _fit_bundle(X_tr: np.ndarray, y_tr: np.ndarray):
    X_tr_s, _, scaler = lib.scale_train(X_tr, X_tr[:1])
    pca = lib.pca_basis(X_tr_s)
    pls = lib.pls_rotations(X_tr_s, y_tr, n_components=9)
    Zp = X_tr_s @ pca
    Zl = X_tr_s @ pls
    models = {}
    pred, proba, clf = lib.fit_lr(X_tr_s, y_tr, X_tr_s[:2])
    models[("lr64", 64)] = clf
    for k in (2, 9):
        _, _, clf_p = lib.fit_lr(Zp[:, :k], y_tr, Zp[:2, :k])
        _, _, clf_l = lib.fit_lr(Zl[:, :k], y_tr, Zl[:2, :k])
        models[("pca", k)] = clf_p
        models[("pls", k)] = clf_l
    return scaler, pca, pls, models


def _score(bundle, X_te: np.ndarray, y_te: np.ndarray) -> list[dict]:
    scaler, pca, pls, models = bundle
    Xs = scaler.transform(X_te)
    rows = []
    for kind, k in (("lr64", 64), ("pca", 2), ("pca", 9), ("pls", 2), ("pls", 9)):
        if kind == "lr64":
            Z = Xs
        elif kind == "pca":
            Z = Xs @ pca[:, :k]
        else:
            Z = Xs @ pls[:, :k]
        clf = models[(kind, k)]
        proba = clf.predict_proba(Z)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        m = lib.binary_metrics(y_te, pred, proba)
        m.update({"model": kind, "k": k})
        rows.append(m)
    return rows


def _source(region: str) -> dict:
    t0 = time.time()
    X = _STATE["X"]
    y = _STATE["y"]
    regions = _STATE["regions"]
    metros = _STATE["metros"]
    src = np.where(regions == region)[0]
    if set(regions[src].tolist()) != {region}:
        raise AssertionError("source slice leaked")
    local_tr, local_te = _split(y[src])
    full = _fit_bundle(np.asarray(X[src], dtype=np.float64), y[src])
    diag = _fit_bundle(np.asarray(X[src][local_tr], dtype=np.float64), y[src][local_tr])
    rows = []
    for target in metros:
        if target == region:
            te = src[local_te]
            bundle = diag
            protocol = "within_metro_20pct"
        else:
            te = np.where(regions == target)[0]
            bundle = full
            protocol = "train_all_source"
            if region in set(regions[te].tolist()):
                raise AssertionError("target contains source")
        for m in _score(bundle, np.asarray(X[te], dtype=np.float64), y[te]):
            m.update({"source": region, "target": target, "protocol": protocol})
            rows.append(m)
    return {"region": region, "rows": rows, "seconds": time.time() - t0, "n_diag_test": int(len(local_te))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "transfer_matrix",
            "smoke": args.smoke,
            "workers": args.workers,
            "command": " ".join(sys.argv),
            "gpus": 0,
            "diagonal": "stratified 80/20 within the source metro, seed 42",
            "off_diagonal": "train on all of A, test on all of B",
        },
    )
    log_path = run_dir / "train.log"
    try:
        X, y, regions = lib.load_arrays()
        metros = lib.metro_list(regions)
        if args.smoke:
            rng = np.random.default_rng(lib.SEED)
            keep = []
            for region in metros[:3]:
                for lab in (0, 1):
                    idx = np.where((regions == region) & (y == lab))[0]
                    keep.append(rng.choice(idx, size=1000, replace=False))
            sel = np.concatenate(keep)
            X = np.asarray(X[sel])
            y = y[sel]
            regions = regions[sel]
            metros = metros[:3]
        _STATE.update({"X": X, "y": y, "regions": regions, "metros": metros})
        lib.log(log_path, f"sources={metros} workers={args.workers}")
        t0 = time.time()
        frames = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(metros))) as pool:
            futs = {pool.submit(_source, region): region for region in metros}
            done = 0
            for fut in as_completed(futs):
                result = fut.result()
                frames.append(pd.DataFrame(result["rows"]))
                done += 1
                lib.atomic_df(run_dir / "metrics.csv", pd.concat(frames, ignore_index=True))
                lib.log(log_path, f"source {done}/{len(metros)} {result['region']} seconds={result['seconds']:.1f}")
        df = pd.concat(frames, ignore_index=True)
        lib.atomic_df(run_dir / "metrics.csv", df)
        for (model, k), g in df.groupby(["model", "k"]):
            mat = g.pivot(index="source", columns="target", values="macro_f1").reindex(index=metros, columns=metros)
            mat.to_csv(run_dir / f"transfer_f1_{model}_k{k}.csv")
        lib.atomic_json(
            run_dir / "summary.json",
            {"status": "completed", "runtime_seconds": time.time() - t0, "metros": metros},
        )
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
