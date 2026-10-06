#!/usr/bin/env python3
"""Extensions for three-country global/local falsification (spectrum, LOCO, overlap)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Reuse prior geometry helpers
_PRIOR = Path(__file__).resolve().parents[1] / "global_local_subspace"
sys.path.insert(0, str(_PRIOR))
from geometry import (  # noqa: E402
    SEED_DEFAULT,
    average_projection_basis,
    balance_cap_by_city,
    binary_metrics,
    fit_lr,
    orthonormalize,
    pca_basis,
    pls_rotations,
    principal_angles_rad,
    random_orthogonal,
    residual_basis,
    stratified_split_idx,
    subspace_distances,
)

__all__ = [
    "SEED_DEFAULT",
    "average_projection_basis",
    "average_projection_spectrum",
    "balance_cap_by_city",
    "binary_metrics",
    "concat_orthonormal",
    "fit_lr",
    "global_overlap",
    "loco_bases",
    "orthonormalize",
    "pca_basis",
    "pls_rotations",
    "principal_angles_rad",
    "project_eval",
    "random_orthogonal",
    "residual_basis",
    "stratified_split_idx",
    "subspace_distances",
]


def average_projection_spectrum(bases: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (evals descending, evecs) of average projection of k-D bases."""
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
    return evals[order], evecs[:, order]


def global_overlap(evals: np.ndarray, k: int) -> float:
    """Sum of top-k eigenvalues of average projection (shared energy)."""
    return float(np.sum(evals[:k]))


def loco_bases(bases: dict[str, np.ndarray], hold_out: str) -> list[np.ndarray]:
    """Country bases excluding hold_out (for LOCO S_G)."""
    return [bases[c] for c in bases if c != hold_out]


def concat_orthonormal(Ua: np.ndarray, Ub: np.ndarray) -> np.ndarray:
    if Ua.size == 0 or Ua.shape[1] == 0:
        return orthonormalize(Ub)
    if Ub.size == 0 or Ub.shape[1] == 0:
        return orthonormalize(Ua)
    return orthonormalize(np.concatenate([Ua, Ub], axis=1))


def project_eval(
    U: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> dict:
    """StandardScaler on train → project by U → LR → metrics on test."""
    from sklearn.preprocessing import StandardScaler

    if U.ndim != 2 or U.shape[1] == 0:
        return {
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "auroc": float("nan"),
            "n": int(len(y_te)),
        }
    sc = StandardScaler().fit(X_tr)
    Ztr = sc.transform(X_tr) @ U
    Zte = sc.transform(X_te) @ U
    pred, proba, _ = fit_lr(Ztr, y_tr, Zte, seed=seed)
    return binary_metrics(y_te, pred, proba)
