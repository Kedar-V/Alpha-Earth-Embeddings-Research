#!/usr/bin/env python3
"""Paper curves: metro vs building probe + PLS/PCA transfer for every k.

Fills the missing widths needed for the three-question paper notebook.
Does not overwrite prior transfer_matrix or statistical_tests runs.
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

# Every width the notebook asks for. PLS/PCA fit once at 64, then slice.
K_ALL = tuple(range(1, 65))
K_PROBE = lib.K_SWEEP  # denser grid is enough for the Pareto scatter


_STATE: dict = {}


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


def _fit_bundle(X_tr: np.ndarray, y_tr: np.ndarray, ks: tuple[int, ...], do_pca: bool):
    X_tr_s, _, scaler = lib.scale_train(X_tr, X_tr[:1])
    pca = lib.pca_basis(X_tr_s) if do_pca else None
    pls = lib.pls_rotations(X_tr_s, y_tr, n_components=max(ks))
    Zp = X_tr_s @ pca if do_pca else None
    Zl = X_tr_s @ pls
    models = {("lr64", 64): lib.fit_lr(X_tr_s, y_tr, X_tr_s[:2])[2]}
    for k in ks:
        models[("pls", k)] = lib.fit_lr(Zl[:, :k], y_tr, Zl[:2, :k])[2]
        if do_pca:
            models[("pca", k)] = lib.fit_lr(Zp[:, :k], y_tr, Zp[:2, :k])[2]
    return scaler, pca, pls, models


def _score(bundle, X_te, y_te, ks, do_pca: bool):
    scaler, pca, pls, models = bundle
    Xs = scaler.transform(X_te)
    rows = []
    kinds = [("lr64", 64)] + [("pls", k) for k in ks]
    if do_pca:
        kinds += [("pca", k) for k in ks]
    for kind, k in kinds:
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
        m.update({"model": kind, "k": int(k)})
        rows.append(m)
    return rows


def _source(region: str) -> dict:
    t0 = time.time()
    X, y, regions = _STATE["X"], _STATE["y"], _STATE["regions"]
    metros, ks, do_pca = _STATE["metros"], _STATE["ks"], _STATE["do_pca"]
    src = np.where(regions == region)[0]
    local_tr, local_te = _split(y[src])
    full = _fit_bundle(np.asarray(X[src], dtype=np.float64), y[src], ks, do_pca)
    diag = _fit_bundle(np.asarray(X[src][local_tr], dtype=np.float64), y[src][local_tr], ks, do_pca)
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
        for m in _score(bundle, np.asarray(X[te], dtype=np.float64), y[te], ks, do_pca):
            m.update({"source": region, "target": target, "protocol": protocol})
            rows.append(m)
    return {"region": region, "rows": rows, "seconds": time.time() - t0}


def _metro_probe(X, y, regions, metros, ks) -> pd.DataFrame:
    rng = np.random.default_rng(lib.SEED)
    idx = []
    for region in metros:
        for lab in (0, 1):
            pool = np.where((regions == region) & (y == lab))[0]
            idx.append(rng.choice(pool, size=min(2000, len(pool)), replace=False))
    sel = np.concatenate(idx)
    Xs = np.asarray(X[sel], dtype=np.float64)
    ys = y[sel]
    rs = regions[sel]
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    tr, te = train_test_split(np.arange(len(sel)), test_size=0.2, random_state=lib.SEED, stratify=rs)
    Xtr, Xte = Xs[tr], Xs[te]
    scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    pca = lib.pca_basis(Xtr_s)
    pls = lib.pls_rotations(Xtr_s, ys[tr], n_components=max(ks))
    enc = LabelEncoder().fit(rs)
    rows = []
    reps = {"raw64": (Xtr_s, Xte_s)}
    for k in ks:
        reps[f"pca{k}"] = (Xtr_s @ pca[:, :k], Xte_s @ pca[:, :k])
        reps[f"pls{k}"] = (Xtr_s @ pls[:, :k], Xte_s @ pls[:, :k])
    for name, (A, B) in reps.items():
        pred, proba, _ = lib.fit_lr(A, ys[tr], B)
        b = lib.binary_metrics(ys[te], pred, proba)
        clf = __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
            solver="lbfgs", max_iter=400, random_state=lib.SEED
        )
        clf.fit(A, enc.transform(rs[tr]))
        metro_acc = float((clf.predict(B) == enc.transform(rs[te])).mean())
        kind = "raw" if name == "raw64" else name[:3]
        k = 64 if name == "raw64" else int(name[3:])
        rows.append(
            {
                "representation": name,
                "kind": kind,
                "k": k,
                "building_macro_f1": b["macro_f1"],
                "building_auroc": b["auroc"],
                "metro_accuracy": metro_acc,
                "n_test": int(len(te)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_transfer(metrics: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    rows = []
    for (model, k), g in metrics.groupby(["model", "k"]):
        pivot = g.pivot(index="source", columns="target", values="macro_f1")
        pivot.to_csv(run_dir / f"transfer_f1_{model}_k{int(k)}.csv")
        a = pivot.to_numpy(dtype=float)
        diag = np.diag(a).copy()
        off = a[~np.eye(len(a), dtype=bool)]
        ret = a / diag[None, :]
        rows.append(
            {
                "model": model,
                "k": int(k),
                "local_f1_mean": float(diag.mean()),
                "local_f1_std": float(diag.std(ddof=1)),
                "cross_f1_mean": float(off.mean()),
                "cross_f1_std": float(off.std(ddof=1)),
                "cross_retention_mean": float(ret[~np.eye(len(a), dtype=bool)].mean()),
                "n_pairs": int(off.size),
            }
        )
        if model == "lr64" or (model == "pls" and int(k) in (2, 9)) or (model == "pca" and int(k) in (2, 9)):
            R = pd.DataFrame(ret, index=pivot.index, columns=pivot.columns)
            R.to_csv(run_dir / f"transfer_retention_{model}_k{int(k)}.csv")
    summary = pd.DataFrame(rows).sort_values(["model", "k"])
    lib.atomic_df(run_dir / "transfer_by_k.csv", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pls-only", action="store_true", help="Skip PCA transfer widths (Pareto probe still includes PCA)")
    ap.add_argument("--probe-csv", default="", help="Reuse an existing metro_vs_building_probe.csv")
    args = ap.parse_args()
    ks = (1, 2, 4, 9, 16, 64) if args.smoke else K_ALL
    probe_ks = (1, 2, 9) if args.smoke else K_PROBE
    do_pca = not args.pls_only and not args.smoke
    if args.smoke:
        do_pca = False
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "paper_curves",
            "ks_transfer": list(ks),
            "ks_probe": list(probe_ks),
            "pls_only": args.pls_only or args.smoke,
            "workers": args.workers,
            "smoke": args.smoke,
            "command": " ".join(sys.argv),
            "gpus": 0,
        },
    )
    log = run_dir / "train.log"
    try:
        t0 = time.time()
        X, y, regions = lib.load_arrays()
        metros = lib.metro_list(regions)
        if args.smoke:
            metros = metros[:3]
            rng = np.random.default_rng(lib.SEED)
            keep = []
            for m in metros:
                idx = np.where(regions == m)[0]
                keep.append(rng.choice(idx, size=min(4000, len(idx)), replace=False))
            keep = np.concatenate(keep)
            X = np.asarray(X[keep])
            y = y[keep]
            regions = regions[keep]
        lib.log(log, f"metros={metros} transfer_k={ks} probe_k={probe_ks} pls_only={not do_pca} workers={args.workers}")

        if args.probe_csv:
            probe = pd.read_csv(args.probe_csv)
            lib.atomic_df(run_dir / "metro_vs_building_probe.csv", probe)
            lib.log(log, f"reused probe from {args.probe_csv}")
        else:
            lib.log(log, "metro vs building probe")
            probe = _metro_probe(X, y, regions, metros, probe_ks)
            lib.atomic_df(run_dir / "metro_vs_building_probe.csv", probe)

        global _STATE
        _STATE = {"X": X, "y": y, "regions": regions, "metros": metros, "ks": ks, "do_pca": do_pca}
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_source, m): m for m in metros}
            done = 0
            for fut in as_completed(futs):
                out = fut.result()
                done += 1
                rows.extend(out["rows"])
                lib.log(log, f"source {done}/{len(metros)} {out['region']} seconds={out['seconds']:.1f}")
                lib.atomic_df(run_dir / "metrics.csv", pd.DataFrame(rows))
        metrics = pd.DataFrame(rows)
        lib.atomic_df(run_dir / "metrics.csv", metrics)
        summary = _summarize_transfer(metrics, run_dir)
        # delta T for PLS-2 vs AEF-64
        t64 = metrics[(metrics.model == "lr64")].pivot(index="source", columns="target", values="macro_f1")
        t2 = metrics[(metrics.model == "pls") & (metrics.k == 2)].pivot(index="source", columns="target", values="macro_f1")
        (t2 - t64).to_csv(run_dir / "transfer_delta_pls2_minus_lr64.csv")
        lib.atomic_json(
            run_dir / "summary.json",
            {
                "status": "completed",
                "runtime_seconds": time.time() - t0,
                "n_metros": len(metros),
                "ks": list(ks),
                "pls2_cross": float(summary[(summary.model == "pls") & (summary.k == 2)]["cross_f1_mean"].iloc[0]),
                "pls9_cross": float(summary[(summary.model == "pls") & (summary.k == 9)]["cross_f1_mean"].iloc[0]),
            },
        )
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
