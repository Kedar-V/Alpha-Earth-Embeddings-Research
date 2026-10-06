"""Shared IO, metrics, and CPU PLS/PCA for the geographic subspace suite."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parents[1]
CACHE_X = ROOT / "outputs" / "week4_pls_loro" / "_cache" / "X.npy"
CACHE_Y = ROOT / "outputs" / "week4_pls_loro" / "_cache" / "y_bin.npy"
CACHE_R = ROOT / "outputs" / "week4_pls_loro" / "_cache" / "regions.npy"
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
SEED = 42
K_SWEEP = (1, 2, 3, 4, 5, 8, 9, 12, 15, 16, 24, 32, 48, 64)
K_SUBSPACE = (1, 2, 3, 5, 9)
K_TRANSFER = (2, 9)

NAMES = {
    "Atlanta_SE": "Atlanta",
    "Boston_NE": "Boston",
    "Chicago_MW": "Chicago",
    "Denver_W": "Denver",
    "Houston_S": "Houston",
    "LA_W": "Los Angeles",
    "NYC_NE": "New York",
    "Phoenix_SW": "Phoenix",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(path: Path, msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    import numpy
    import sklearn

    out = {"python": os.sys.version.split()[0], "numpy": numpy.__version__, "sklearn": sklearn.__version__}
    try:
        import scipy

        out["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:
        pass
    return out


def init_run(run_dir: Path, config: dict) -> Path:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite existing run {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = dict(config)
    config["created"] = now()
    config["git"] = git_hash()
    config["packages"] = package_versions()
    config["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    config["cpus"] = os.cpu_count()
    config["seed"] = SEED
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return run_dir


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CACHE_X.exists():
        raise FileNotFoundError(CACHE_X)
    X = np.load(CACHE_X, mmap_mode="r")
    y = np.load(CACHE_Y)
    regions = np.load(CACHE_R, allow_pickle=True).astype(str)
    if X.ndim != 2 or X.shape[1] != 64:
        raise RuntimeError(f"unexpected X shape {X.shape}")
    if len(y) != len(X) or len(regions) != len(X):
        raise RuntimeError("X/y/region length mismatch")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise RuntimeError(f"unexpected labels {np.unique(y)}")
    return X, y.astype(np.int64), regions


def binary_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_residential": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_nonresidential": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
        "n_residential": int((y == 1).sum()),
        "n_nonresidential": int((y == 0).sum()),
    }
    if len(np.unique(y)) > 1 and np.isfinite(proba).all():
        out["auroc"] = float(roc_auc_score(y, proba))
    else:
        out["auroc"] = float("nan")
    return out


def fit_lr(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray):
    clf = LogisticRegression(
        solver="newton-cholesky",
        C=1.0,
        max_iter=200,
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba, clf


def pca_basis(X_tr: np.ndarray) -> np.ndarray:
    """Eigenvectors of the train covariance, columns, largest eigenvalue first. X is already scaled."""
    cov = (X_tr.T @ X_tr) / max(X_tr.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return evecs[:, order].astype(np.float64)


def pls_rotations(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    """PLS x-rotations (p, k). X must already be column-centered. Fit uses train rows only.

    Gram-form NIPALS: one XtX, then rank-1 deflation in p-space. Matches the
    rotation formula R = W @ pinv(P.T @ W) used by sklearn PLSRegression.
    """
    X = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    yv = yv - yv.mean()
    n_comp = int(min(n_components, X.shape[1], max(X.shape[0] - 1, 1)))
    XtX = X.T @ X
    XtY = X.T @ yv  # (p,)
    p = X.shape[1]
    W = np.zeros((p, n_comp))
    P = np.zeros((p, n_comp))
    for k in range(n_comp):
        w = XtY.copy()
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            W = W[:, :k]
            P = P[:, :k]
            break
        w = w / nrm
        for _ in range(80):
            t_var = float(w @ (XtX @ w))
            if t_var < 1e-18:
                break
            c_scalar = float(XtY @ w) / t_var
            if abs(c_scalar) < 1e-18:
                break
            c_unit = 1.0 if c_scalar >= 0 else -1.0
            w_new = XtY * c_unit
            nrm = np.linalg.norm(w_new)
            if nrm < 1e-12:
                break
            w_new = w_new / nrm
            if np.linalg.norm(w_new - w) < 1e-8:
                w = w_new
                break
            w = w_new
        t_var = float(w @ (XtX @ w))
        t_var = max(t_var, 1e-18)
        p_k = (XtX @ w) / t_var
        c_scalar = float(XtY @ w) / t_var
        Xtw = XtX @ w
        # X <- X - t p', Y <- Y - t c, with t = X w
        XtX = XtX - np.outer(p_k, Xtw) - np.outer(Xtw, p_k) + t_var * np.outer(p_k, p_k)
        XtY = XtY - c_scalar * Xtw - p_k * float(w @ XtY) + t_var * c_scalar * p_k
        W[:, k] = w
        P[:, k] = p_k
    if W.shape[1] == 0:
        raise RuntimeError("PLS produced no components")
    R = W @ np.linalg.pinv(P.T @ W)
    return R


def orthonormalize(W: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(W, mode="reduced")
    return Q


def principal_angles_deg(Qi: np.ndarray, Qj: np.ndarray) -> np.ndarray:
    M = Qi.T @ Qj
    sig = np.linalg.svd(M, compute_uv=False)
    sig = np.clip(sig, -1.0, 1.0)
    return np.degrees(np.arccos(sig))


def projection_similarity(Qi: np.ndarray, Qj: np.ndarray) -> float:
    M = Qi.T @ Qj
    k = M.shape[0]
    return float(np.sum(M * M) / k)


def assert_fold_split(regions: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, held_out: str) -> None:
    if np.intersect1d(train_idx, test_idx).size:
        raise AssertionError("train/test index overlap")
    te_regions = set(regions[test_idx].tolist())
    tr_regions = set(regions[train_idx].tolist())
    if te_regions != {held_out}:
        raise AssertionError(f"test regions {te_regions} != {{{held_out}}}")
    if held_out in tr_regions:
        raise AssertionError("held-out metro leaked into training rows")


def scale_train(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), scaler


def metro_list(regions: np.ndarray) -> list[str]:
    return sorted(np.unique(regions).tolist())
