#!/usr/bin/env python3
"""Subspace geometry helpers for the global⊕local AlphaEarth study."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SEED_DEFAULT = 42


def orthonormalize(U: np.ndarray) -> np.ndarray:
    """QR orthonormalization; drop near-zero columns."""
    if U.size == 0:
        return U.reshape(U.shape[0], 0)
    Q, R = np.linalg.qr(U, mode="reduced")
    # keep columns with sufficient R diagonal
    diag = np.abs(np.diag(R))
    keep = diag > 1e-8 * (diag.max() if diag.size else 1.0)
    if not np.any(keep):
        return Q[:, :0]
    return Q[:, keep]


def pls_rotations(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    """PLS x-rotations (p, k). X already column-scaled/centered."""
    X = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    yv = yv - yv.mean()
    n_comp = int(min(n_components, X.shape[1], max(X.shape[0] - 1, 1)))
    XtX = X.T @ X
    XtY = X.T @ yv
    p = X.shape[1]
    W = np.zeros((p, n_comp))
    P = np.zeros((p, n_comp))
    for k in range(n_comp):
        w = XtY.copy()
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            W, P = W[:, :k], P[:, :k]
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
        t_var = max(float(w @ (XtX @ w)), 1e-18)
        p_k = (XtX @ w) / t_var
        c_scalar = float(XtY @ w) / t_var
        Xtw = XtX @ w
        XtX = XtX - np.outer(p_k, Xtw) - np.outer(Xtw, p_k) + t_var * np.outer(p_k, p_k)
        XtY = XtY - c_scalar * Xtw - p_k * float(w @ XtY) + t_var * c_scalar * p_k
        W[:, k] = w
        P[:, k] = p_k
    if W.shape[1] == 0:
        raise RuntimeError("PLS produced no components")
    return W @ np.linalg.pinv(P.T @ W)


def pca_basis(X: np.ndarray, n_components: int | None = None) -> np.ndarray:
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    U = evecs[:, order]
    if n_components is not None:
        U = U[:, :n_components]
    return U.astype(np.float64)


def fit_lr(Z_tr: np.ndarray, y_tr: np.ndarray, Z_te: np.ndarray, seed: int = SEED_DEFAULT):
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=500, random_state=seed)
    clf.fit(Z_tr, y_tr)
    proba = clf.predict_proba(Z_te)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba, clf


def binary_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_residential": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_nonresidential": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "precision_residential": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_residential": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "precision_nonresidential": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_nonresidential": float(recall_score(y, pred, pos_label=0, zero_division=0)),
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


def principal_angles_rad(Ui: np.ndarray, Uj: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column-spaces of Ui, Uj."""
    A = orthonormalize(Ui)
    B = orthonormalize(Uj)
    k = min(A.shape[1], B.shape[1])
    if k == 0:
        return np.array([])
    A = A[:, :k]
    B = B[:, :k]
    # SVD of A.T @ B
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def subspace_distances(Ui: np.ndarray, Uj: np.ndarray) -> dict:
    ang = principal_angles_rad(Ui, Uj)
    if ang.size == 0:
        return {
            "mean_principal_angle_rad": float("nan"),
            "mean_principal_angle_deg": float("nan"),
            "max_principal_angle_rad": float("nan"),
            "max_principal_angle_deg": float("nan"),
            "min_principal_angle_rad": float("nan"),
            "chordal_distance": float("nan"),
            "grassmann_geodesic": float("nan"),
            "projection_frobenius": float("nan"),
            "n_angles": 0,
            "principal_angles_deg": [],
        }
    chordal = float(np.linalg.norm(np.sin(ang)))
    geo = float(np.linalg.norm(ang))
    # projection distance: ||Pi - Pj||_F / sqrt(2) related to chordal
    Ai = orthonormalize(Ui)
    Bj = orthonormalize(Uj)
    k = min(Ai.shape[1], Bj.shape[1])
    Pi = Ai[:, :k] @ Ai[:, :k].T
    Pj = Bj[:, :k] @ Bj[:, :k].T
    proj = float(np.linalg.norm(Pi - Pj, ord="fro"))
    return {
        "mean_principal_angle_rad": float(ang.mean()),
        "mean_principal_angle_deg": float(np.degrees(ang.mean())),
        "max_principal_angle_rad": float(ang.max()),
        "max_principal_angle_deg": float(np.degrees(ang.max())),
        "min_principal_angle_rad": float(ang.min()),
        "chordal_distance": chordal,
        "grassmann_geodesic": geo,
        "projection_frobenius": proj,
        "n_angles": int(len(ang)),
        "principal_angles_deg": [float(x) for x in np.degrees(ang)],
    }


def average_projection_basis(bases: list[np.ndarray], k: int) -> np.ndarray:
    """Consensus subspace via average of projection matrices + top-k evecs."""
    p = bases[0].shape[0]
    P = np.zeros((p, p), dtype=np.float64)
    for U in bases:
        Uo = orthonormalize(U)[:, : min(k, U.shape[1])]
        if Uo.shape[1] == 0:
            continue
        P += Uo @ Uo.T
    P /= max(len(bases), 1)
    evals, evecs = np.linalg.eigh(P)
    order = np.argsort(evals)[::-1]
    return orthonormalize(evecs[:, order[:k]])


def grassmann_karcher_mean(bases: list[np.ndarray], k: int, n_iter: int = 30, step: float = 0.5) -> np.ndarray:
    """Approximate Karcher mean on Grassmann via iterative projection averaging.

    Starts from average-projection basis; iteratively recenters.
    """
    U = average_projection_basis(bases, k)
    for _ in range(n_iter):
        # tangent average of logs
        H = np.zeros_like(U)
        for B in bases:
            Bo = orthonormalize(B)[:, :k]
            # horizontal lift of log: (I - UU^T) Bo pinv(U^T Bo)  — approx via SVD alignment
            M = U.T @ Bo
            Wv, s, Vh = np.linalg.svd(M, full_matrices=False)
            s = np.clip(s, 0.0, 1.0)
            # Procrustes alignment of Bo to U
            Bo_al = Bo @ (Vh.T @ Wv.T)
            # residual in ambient space
            R = Bo_al - U @ (U.T @ Bo_al)
            H += R
        H /= max(len(bases), 1)
        U = orthonormalize(U + step * H)[:, :k]
    return U


def residual_basis(U_local: np.ndarray, U_global: np.ndarray) -> np.ndarray:
    """Orthonormal basis for local residual after removing global projection."""
    Ug = orthonormalize(U_global)
    Ul = orthonormalize(U_local)
    R = Ul - Ug @ (Ug.T @ Ul)
    return orthonormalize(R)


def random_orthogonal(p: int, k: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(p, k))
    return orthonormalize(A)


def stratified_split_idx(y: np.ndarray, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(test_frac * len(idx))))
        te.append(idx[:n_te])
        tr.append(idx[n_te:])
    return np.concatenate(tr), np.concatenate(te)


def balance_cap_by_city(
    X: np.ndarray,
    y: np.ndarray,
    regions: np.ndarray,
    max_per_class: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_per_class is None:
        return X, y, regions
    keep = []
    for city in np.unique(regions):
        for lab in (0, 1):
            idx = np.where((regions == city) & (y == lab))[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep], regions[keep]
