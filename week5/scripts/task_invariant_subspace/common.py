#!/usr/bin/env python3
"""Shared helpers for Experiment 05 (task-invariant vs geography-specific subspaces)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
_GEO = Path(__file__).resolve().parents[1] / "three_country_falsification"
_PRIOR = Path(__file__).resolve().parents[1] / "global_local_subspace"
sys.path.insert(0, str(_GEO))
sys.path.insert(0, str(_PRIOR))

from geometry_ext import (  # noqa: E402
    average_projection_basis,
    average_projection_spectrum,
    balance_cap_by_city,
    binary_metrics,
    concat_orthonormal,
    fit_lr,
    global_overlap,
    loco_bases,
    orthonormalize,
    pca_basis,
    pls_rotations,
    principal_angles_rad,
    project_eval,
    random_orthogonal,
    residual_basis,
    stratified_split_idx,
    subspace_distances,
)
from geometry import grassmann_karcher_mean  # noqa: E402

COUNTRIES = ("USA", "JPN", "IND")
PAIRS = [
    ("USA", "JPN"),
    ("USA", "IND"),
    ("JPN", "USA"),
    ("JPN", "IND"),
    ("IND", "USA"),
    ("IND", "JPN"),
]
KS = list(range(1, 21))
R_GRID = (1, 2, 3, 5)
TI_LAMBDAS = (0.0, 0.5, 1.0, 2.0)

_CACHE: dict = {}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(run_dir: Path, msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kw) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    prev["updated_at"] = now()
    prev["last_progress_at"] = prev["updated_at"]
    atomic_json(path, prev)


def stage_done(run_dir: Path, name: str) -> bool:
    return (run_dir / "checkpoints" / f"{name}.DONE").exists()


def mark_done(run_dir: Path, name: str, payload: dict | None = None) -> None:
    p = run_dir / "checkpoints" / f"{name}.DONE"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload or {"status": "ok", "completed_at": now()}, indent=2) + "\n")


def load_extract(extract_dir: Path):
    X = np.load(extract_dir / "X.npy", mmap_mode="r")
    y = np.load(extract_dir / "y.npy").astype(np.int64)
    regions = np.load(extract_dir / "regions.npy", allow_pickle=True).astype(str)
    countries = np.load(extract_dir / "countries.npy", allow_pickle=True).astype(str)
    return X, y, regions, countries


def init_cache(extract_dir: Path):
    global _CACHE
    X, y, regions, countries = load_extract(extract_dir)
    _CACHE = {"X": X, "y": y, "regions": regions, "countries": countries}


def country_slice(country: str, max_per_class: int, seed: int):
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    mask = countries == country
    Xm = np.asarray(X[mask], dtype=np.float32)
    ym = y[mask]
    rm = regions[mask]
    rng = np.random.default_rng(seed)
    return balance_cap_by_city(Xm, ym, rm, max_per_class, rng)


def load_prior_bases(prior_run: Path, seed: int) -> dict[str, np.ndarray]:
    sub = prior_run / "artifacts" / "subspaces"
    return {c: orthonormalize(np.load(sub / f"U_pls_{c}_seed{seed}.npy")) for c in COUNTRIES}


def fit_country_pls(country: str, max_per: int, seed: int, k_max: int = 20, shuffle: bool = False) -> np.ndarray:
    Xm, ym, _ = country_slice(country, max_per, seed)
    if shuffle:
        ym = np.random.default_rng(seed + 991).permutation(ym)
    rng = np.random.default_rng(seed)
    tr, _ = stratified_split_idx(ym, 0.2, rng)
    sc = StandardScaler().fit(Xm[tr])
    R = pls_rotations(sc.transform(Xm[tr]), ym[tr], n_components=k_max)
    return orthonormalize(R)


def direction_stats(z: np.ndarray, y: np.ndarray, fit_logistic: bool = False) -> dict:
    """1D predictive utilities + signed separation.

    By default skips LogisticRegression (dominant cost); uses median-threshold
    predictions for F1 / balanced accuracy. Set fit_logistic=True for coef.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    m0 = float(z[y == 0].mean()) if np.any(y == 0) else 0.0
    m1 = float(z[y == 1].mean()) if np.any(y == 1) else 0.0
    s0 = float(z[y == 0].std(ddof=1)) if np.sum(y == 0) > 1 else 1.0
    s1 = float(z[y == 1].std(ddof=1)) if np.sum(y == 1) > 1 else 1.0
    pooled = np.sqrt(0.5 * (s0 * s0 + s1 * s1)) + 1e-12
    cohens_d = (m1 - m0) / pooled
    zy = (z - z.mean()) / (z.std(ddof=1) + 1e-12)
    yy = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
    r_pb = float(np.mean(zy * yy))
    try:
        auroc = float(roc_auc_score(y, z))
    except Exception:
        auroc = 0.5
    auroc_abs = max(auroc, 1.0 - auroc)
    # orient so higher z -> class 1 for threshold metrics
    z_use = z if (m1 >= m0) else -z
    thr = float(np.median(z_use))
    pred = (z_use >= thr).astype(np.int64)
    # soft scores for AUROC-compatible proba proxy
    zr = (z_use - z_use.min()) / (z_use.max() - z_use.min() + 1e-12)
    m = binary_metrics(y, pred, zr)
    coef = 0.0
    if fit_logistic:
        try:
            clf = LogisticRegression(max_iter=200, solver="lbfgs")
            clf.fit(z.reshape(-1, 1), y)
            coef = float(clf.coef_.ravel()[0])
            proba = clf.predict_proba(z.reshape(-1, 1))[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = binary_metrics(y, pred, proba)
        except Exception:
            pass
    return {
        "mean_diff": m1 - m0,
        "cohens_d": float(cohens_d),
        "point_biserial": r_pb,
        "auroc": auroc,
        "auroc_abs": float(auroc_abs),
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "logistic_coef": coef,
        "sign_res_positive": int((m1 - m0) > 0),
    }


def eval_direction_on_country(
    u: np.ndarray,
    country: str,
    max_per: int,
    seed: int,
    scaler: StandardScaler | None = None,
    flip: bool = False,
) -> dict:
    """Evaluate direction u on a country's held-out 20% (or full if scaler given from source)."""
    Xm, ym, _ = country_slice(country, max_per, seed)
    rng = np.random.default_rng(seed)
    tr, te = stratified_split_idx(ym, 0.2, rng)
    if scaler is None:
        scaler = StandardScaler().fit(Xm[tr])
        Xte = scaler.transform(Xm[te])
        yte = ym[te]
    else:
        # transfer: evaluate on full capped country (consistent with Exp04 transfer)
        Xte = scaler.transform(Xm)
        yte = ym
    u = np.asarray(u, dtype=np.float64).ravel()
    u = u / (np.linalg.norm(u) + 1e-12)
    if flip:
        u = -u
    z = Xte @ u
    stats = direction_stats(z, yte)
    stats["n"] = int(len(yte))
    return stats


def task_invariance_score(utils: list[float], lam: float = 1.0) -> float:
    arr = np.asarray(utils, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean() - lam * arr.std(ddof=0))


def n_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


__all__ = [
    "COUNTRIES",
    "PAIRS",
    "KS",
    "R_GRID",
    "TI_LAMBDAS",
    "ROOT",
    "average_projection_basis",
    "average_projection_spectrum",
    "balance_cap_by_city",
    "binary_metrics",
    "concat_orthonormal",
    "country_slice",
    "direction_stats",
    "eval_direction_on_country",
    "fit_country_pls",
    "fit_lr",
    "global_overlap",
    "grassmann_karcher_mean",
    "init_cache",
    "load_extract",
    "load_prior_bases",
    "loco_bases",
    "log",
    "mark_done",
    "n_gpus",
    "now",
    "orthonormalize",
    "pca_basis",
    "pls_rotations",
    "principal_angles_rad",
    "project_eval",
    "random_orthogonal",
    "residual_basis",
    "stage_done",
    "stratified_split_idx",
    "subspace_distances",
    "task_invariance_score",
    "write_progress",
    "atomic_json",
]
