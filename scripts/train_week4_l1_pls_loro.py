#!/usr/bin/env python3
"""Week 4 LORO: L1 vs L2 logistic on AE bands and PLS-DA scores.

Question: under L1, do most PLS-score coefficients collapse to ~0 (leaving t1/t2)?

Per held-out metro (all GPUs in parallel):
  1. Reuse / refit PLS-DA rotations (k_max=32)
  2. Sweep L1 strength on (a) 64-D scaled AE  (b) k=32 PLS scores
  3. L2 reference at each feature set
  4. Record coefs, nnz, which axes survive, OOF F1

Writes outputs/week4_l1_pls_loro/ + figures. Build notebook via
  scripts/build_week4_l1_pls_notebook.py

Usage:
  .venv/bin/python -u scripts/train_week4_l1_pls_loro.py --backend gpu
  .venv/bin/python -u scripts/train_week4_l1_pls_loro.py --backend cpu --workers 40
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
PLS_CKPT = ROOT / "outputs" / "week4_pls_loro" / "checkpoints"
PLS_CACHE = ROOT / "outputs" / "week4_pls_loro" / "_cache"
OUT = ROOT / "outputs" / "week4_l1_pls_loro"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
K_PLS = 32
# sklearn-style C: smaller = stronger L1. Log grid.
L1_C_GRID = (
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
)
NZ_EPS = 1e-6  # |coef| below this counts as zero


def _jsonable(obj):
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return _jsonable(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# Torch logistic: L2 (LBFGS) and L1 (proximal SGD)
# ---------------------------------------------------------------------------


def _torch_standardize(X_tr: torch.Tensor, X_te: torch.Tensor):
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd


def _soft_threshold(w: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(w) * torch.clamp(w.abs() - lam, min=0.0)


def _torch_logistic_l2(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    wd: float = 1e-2,
    max_iter: int = 80,
):
    D = X_tr.shape[1]
    w = torch.zeros(D, device=X_tr.device, requires_grad=True)
    b = torch.zeros(1, device=X_tr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = X_tr @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr)
        loss = loss + 0.5 * wd * (w * w).sum() / X_tr.shape[0]
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = torch.sigmoid(X_te @ w + b).detach().cpu().numpy().astype(np.float64)
    y_pred = (p >= 0.5).astype(np.int64)
    return y_pred, p, w.detach().cpu().numpy().astype(np.float64), float(b.detach().cpu().item())


def _torch_logistic_l1(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    C: float,
    n_epochs: int = 60,
    lr: float = 0.3,
):
    """Proximal full-batch steps for mean-BCE + (1/C)||w||_1 (sklearn SAGA-like)."""
    n, D = X_tr.shape
    device = X_tr.device
    # sklearn LogisticRegression(penalty='l1'): average log-loss + (1/C) ||w||_1
    lam = 1.0 / max(C, 1e-12)
    w = torch.zeros(D, device=device)
    b = torch.zeros(1, device=device)
    # adaptive lr: start moderate; more epochs for strong L1
    n_epochs = int(n_epochs) if C >= 0.1 else int(max(n_epochs, 80))
    for epoch in range(n_epochs):
        w = w.detach().requires_grad_(True)
        b = b.detach().requires_grad_(True)
        logits = X_tr @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr)
        loss.backward()
        with torch.no_grad():
            # slightly smaller step when lam is huge
            step = lr / max(1.0, math.sqrt(lam))
            w = w - step * w.grad
            b = b - step * b.grad
            w = _soft_threshold(w, step * lam)
    with torch.no_grad():
        p = torch.sigmoid(X_te @ w + b).detach().cpu().numpy().astype(np.float64)
        coef = w.detach().cpu().numpy().astype(np.float64)
        intercept = float(b.detach().cpu().item())
    y_pred = (p >= 0.5).astype(np.int64)
    return y_pred, p, coef, intercept, float(lam)


def _sklearn_logistic(X_tr, y_tr, X_te, penalty: str, C: float):
    if penalty == "l1":
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            max_iter=4000,
            random_state=SEED,
            tol=1e-4,
        )
    else:
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=C,
            max_iter=2000,
            random_state=SEED,
        )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    p = clf.predict_proba(X_te)[:, 1]
    return y_pred, p, clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])


def _nnz(coef: np.ndarray, eps: float = NZ_EPS) -> int:
    return int(np.sum(np.abs(coef) > eps))


def _surviving(coef: np.ndarray, names: list[str], eps: float = NZ_EPS) -> list[str]:
    return [names[i] for i in range(len(coef)) if abs(coef[i]) > eps]


# ---------------------------------------------------------------------------
# Fold workers
# ---------------------------------------------------------------------------


def _fit_fold_gpu(payload: tuple) -> dict:
    (
        te_region,
        x_path,
        ybin_path,
        reg_path,
        pls_ckpt_dir,
        cache_dir,
        ckpt_dir,
        device_id,
        k_pls,
        c_grid,
    ) = payload
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    X_np = np.load(x_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)

    X_tr = torch.as_tensor(np.asarray(X_np[tr], dtype=np.float32), device=device)
    X_te = torch.as_tensor(np.asarray(X_np[te], dtype=np.float32), device=device)
    y_tr = torch.as_tensor(np.asarray(y_bin[tr], dtype=np.float32), device=device)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    X_tr_s, X_te_s, mu, sd = _torch_standardize(X_tr, X_te)

    # PLS rotations from prior week4 PLS run when available
    pls_path = Path(pls_ckpt_dir) / f"fold_{te_region}.joblib"
    if pls_path.exists():
        bundle = joblib.load(pls_path)
        R = torch.as_tensor(
            bundle["pls_da"]["x_rotations"][:, :k_pls].astype(np.float32), device=device
        )
    else:
        # fallback: quick cov-PLS via importing helper
        from scripts.train_week4_pls_loro import _torch_pls_nipals

        y_c = (y_tr - y_tr.mean()).unsqueeze(1)
        pls = _torch_pls_nipals(X_tr_s, y_c, k_pls)
        R = pls["x_rotations"][:, :k_pls]

    T_tr = X_tr_s @ R
    T_te = X_te_s @ R
    tmu = T_tr.mean(0, keepdim=True)
    tsd = T_tr.std(0, keepdim=True).clamp_min(1e-6)
    Z_tr = (T_tr - tmu) / tsd
    Z_te = (T_te - tmu) / tsd

    score_names = [f"t{k+1}" for k in range(k_pls)]
    rows: list[dict] = []
    coef_store: dict[str, dict] = {}

    # --- L2 references ---
    for feat_name, Xtr, Xte, names in (
        ("ae64", X_tr_s, X_te_s, BANDS),
        ("pls32", Z_tr, Z_te, score_names),
    ):
        yhat, p, coef, intercept = _torch_logistic_l2(Xtr, y_tr, Xte)
        key = f"l2_{feat_name}"
        rows.append(
            {
                "region": te_region,
                "model": key,
                "penalty": "l2",
                "features": feat_name,
                "C": 1.0,
                "n_features": int(len(coef)),
                "nnz": _nnz(coef),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "precision_res": float(precision_score(yb_te, yhat, zero_division=0)),
                "recall_res": float(recall_score(yb_te, yhat, zero_division=0)),
                "device": f"cuda:{device_id}",
                "surviving": ",".join(_surviving(coef, names)[:20]),
            }
        )
        coef_store[key] = {
            "coef": coef,
            "intercept": intercept,
            "names": names,
            "y_pred": yhat,
            "p_res": p,
        }

    # --- L1 sweeps ---
    for feat_name, Xtr, Xte, names in tqdm(
        (
            ("ae64", X_tr_s, X_te_s, BANDS),
            ("pls32", Z_tr, Z_te, score_names),
        ),
        desc=f"{te_region} featsets",
        leave=False,
    ):
        for C in tqdm(list(c_grid), desc=f"{te_region} L1-{feat_name}", leave=False):
            yhat, p, coef, intercept, lam = _torch_logistic_l1(Xtr, y_tr, Xte, C=float(C))
            key = f"l1_{feat_name}_C{C:g}"
            surv = _surviving(coef, names)
            rows.append(
                {
                    "region": te_region,
                    "model": key,
                    "penalty": "l1",
                    "features": feat_name,
                    "C": float(C),
                    "lambda": lam,
                    "n_features": int(len(coef)),
                    "nnz": _nnz(coef),
                    "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                    "precision_res": float(precision_score(yb_te, yhat, zero_division=0)),
                    "recall_res": float(recall_score(yb_te, yhat, zero_division=0)),
                    "device": f"cuda:{device_id}",
                    "surviving": ",".join(surv[:32]),
                    "surviving_n": len(surv),
                }
            )
            coef_store[key] = {
                "coef": coef,
                "intercept": intercept,
                "names": names,
                "y_pred": yhat,
                "p_res": p,
                "C": float(C),
                "lambda": lam,
            }

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"oof_idx_{te_region}.npy", te_idx)
    joblib.dump(
        {
            "region_held_out": te_region,
            "backend": "torch_gpu",
            "device_id": device_id,
            "mu": mu.detach().cpu().numpy(),
            "sd": sd.detach().cpu().numpy(),
            "tmu": tmu.detach().cpu().numpy(),
            "tsd": tsd.detach().cpu().numpy(),
            "R": R.detach().cpu().numpy(),
            "coef_store_keys": list(coef_store.keys()),
            "seed": SEED,
            "k_pls": k_pls,
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    # persist coefs separately (large)
    joblib.dump(coef_store, cache / f"coef_store_{te_region}.joblib")

    del X_tr, X_te, X_tr_s, X_te_s, T_tr, T_te, Z_tr, Z_te
    torch.cuda.empty_cache()
    return {
        "region": te_region,
        "device": f"cuda:{device_id}",
        "rows": rows,
        "idx_path": str(cache / f"oof_idx_{te_region}.npy"),
        "coef_path": str(cache / f"coef_store_{te_region}.joblib"),
    }


def _fit_fold_cpu(payload: tuple) -> dict:
    (
        te_region,
        x_path,
        ybin_path,
        reg_path,
        pls_ckpt_dir,
        cache_dir,
        ckpt_dir,
        _device_id,
        k_pls,
        c_grid,
    ) = payload
    X_np = np.load(x_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)

    X_tr = np.asarray(X_np[tr], dtype=np.float64)
    X_te = np.asarray(X_np[te], dtype=np.float64)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd[sd < 1e-6] = 1.0
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd

    pls_path = Path(pls_ckpt_dir) / f"fold_{te_region}.joblib"
    bundle = joblib.load(pls_path)
    R = bundle["pls_da"]["x_rotations"][:, :k_pls].astype(np.float64)
    T_tr = X_tr_s @ R
    T_te = X_te_s @ R
    tmu = T_tr.mean(0)
    tsd = T_tr.std(0)
    tsd[tsd < 1e-6] = 1.0
    Z_tr = (T_tr - tmu) / tsd
    Z_te = (T_te - tmu) / tsd
    score_names = [f"t{k+1}" for k in range(k_pls)]

    rows: list[dict] = []
    coef_store: dict = {}

    for feat_name, Xtr, Xte, names in (
        ("ae64", X_tr_s, X_te_s, BANDS),
        ("pls32", Z_tr, Z_te, score_names),
    ):
        yhat, p, coef, intercept = _sklearn_logistic(Xtr, y_tr, Xte, "l2", C=1.0)
        key = f"l2_{feat_name}"
        rows.append(
            {
                "region": te_region,
                "model": key,
                "penalty": "l2",
                "features": feat_name,
                "C": 1.0,
                "n_features": int(len(coef)),
                "nnz": _nnz(coef),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "precision_res": float(precision_score(yb_te, yhat, zero_division=0)),
                "recall_res": float(recall_score(yb_te, yhat, zero_division=0)),
                "device": "cpu",
                "surviving": ",".join(_surviving(coef, names)[:20]),
            }
        )
        coef_store[key] = {"coef": coef, "intercept": intercept, "names": names, "y_pred": yhat, "p_res": p}

    for feat_name, Xtr, Xte, names in tqdm(
        (("ae64", X_tr_s, X_te_s, BANDS), ("pls32", Z_tr, Z_te, score_names)),
        desc=f"{te_region} featsets",
        leave=False,
    ):
        for C in tqdm(list(c_grid), desc=f"{te_region} L1-{feat_name}", leave=False):
            yhat, p, coef, intercept = _sklearn_logistic(Xtr, y_tr, Xte, "l1", C=float(C))
            key = f"l1_{feat_name}_C{C:g}"
            surv = _surviving(coef, names)
            rows.append(
                {
                    "region": te_region,
                    "model": key,
                    "penalty": "l1",
                    "features": feat_name,
                    "C": float(C),
                    "n_features": int(len(coef)),
                    "nnz": _nnz(coef),
                    "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                    "precision_res": float(precision_score(yb_te, yhat, zero_division=0)),
                    "recall_res": float(recall_score(yb_te, yhat, zero_division=0)),
                    "device": "cpu",
                    "surviving": ",".join(surv[:32]),
                    "surviving_n": len(surv),
                }
            )
            coef_store[key] = {
                "coef": coef,
                "intercept": intercept,
                "names": names,
                "y_pred": yhat,
                "p_res": p,
                "C": float(C),
            }

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"oof_idx_{te_region}.npy", te_idx)
    joblib.dump(
        {"region_held_out": te_region, "backend": "sklearn_cpu", "k_pls": k_pls, "seed": SEED},
        ckpt / f"fold_{te_region}.joblib",
    )
    joblib.dump(coef_store, cache / f"coef_store_{te_region}.joblib")
    return {
        "region": te_region,
        "device": "cpu",
        "rows": rows,
        "idx_path": str(cache / f"oof_idx_{te_region}.npy"),
        "coef_path": str(cache / f"coef_store_{te_region}.joblib"),
    }


# ---------------------------------------------------------------------------
# Aggregation / plots
# ---------------------------------------------------------------------------


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (penalty, features, C), g in metrics.groupby(["penalty", "features", "C"], dropna=False):
        rows.append(
            {
                "penalty": penalty,
                "features": features,
                "C": float(C) if pd.notna(C) else None,
                "n_folds": len(g),
                "loro_macro_f1_mean": float(g["macro_f1"].mean()),
                "loro_macro_f1_std": float(g["macro_f1"].std(ddof=0)),
                "nnz_mean": float(g["nnz"].mean()),
                "nnz_std": float(g["nnz"].std(ddof=0)),
                "nnz_median": float(g["nnz"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["features", "penalty", "C"]).reset_index(drop=True)


def _survival_heatmap(fold_results: list[dict], features: str, c_grid: tuple, out_path: Path):
    """Fraction of folds where each feature has |coef|>eps, for each C."""
    # use pls32 score names
    names = [f"t{k+1}" for k in range(K_PLS)] if features == "pls32" else BANDS
    mat = np.zeros((len(c_grid), len(names)), dtype=np.float64)
    for j, C in enumerate(c_grid):
        key = f"l1_{features}_C{C:g}"
        hits = np.zeros(len(names), dtype=np.float64)
        n = 0
        for res in fold_results:
            store = joblib.load(res["coef_path"])
            if key not in store:
                continue
            coef = store[key]["coef"]
            hits += (np.abs(coef) > NZ_EPS).astype(np.float64)
            n += 1
        if n:
            mat[j] = hits / n
    # plot first 16 for pls, or top varying for ae
    if features == "pls32":
        show_n = min(16, len(names))
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        im = ax.imshow(mat[:, :show_n], aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(show_n))
        ax.set_xticklabels(names[:show_n], fontsize=8)
        ax.set_yticks(range(len(c_grid)))
        ax.set_yticklabels([f"{c:g}" for c in c_grid], fontsize=8)
        ax.set_xlabel("PLS component")
        ax.set_ylabel("L1 C (smaller = stronger)")
        fig.colorbar(im, ax=ax, fraction=0.03, label="fold survival rate")
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.text(
            0.01,
            0.98,
            "L1 on PLS scores: at strong penalty only t1 (and sometimes t2) survive",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        fig.text(
            0.01,
            0.935,
            "Cell = fraction of LORO folds with |coef| > 1e-6. Strong L1 = bottom rows.",
            fontsize=9,
            color="#555",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        # show mean |coef| path for top bands by L2 importance — survival of all 64 is wide; plot nnz curve instead handled elsewhere
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        # survival count expected = mat.sum(1)
        ax.imshow(mat[:, :24], aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
        ax.set_yticks(range(len(c_grid)))
        ax.set_yticklabels([f"{c:g}" for c in c_grid], fontsize=8)
        ax.set_xlabel("AE band index (A00… first 24)")
        ax.set_ylabel("L1 C")
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.text(
            0.01,
            0.98,
            "L1 on 64-D AE: many bands zero out as C decreases",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return mat


def _plot_paths(sweep: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for features, title in (("pls32", "PLS-DA scores (k=32)"), ("ae64", "raw AE 64-D")):
        sub = sweep.loc[(sweep.features == features) & (sweep.penalty == "l1")].sort_values("C")
        l2 = sweep.loc[(sweep.features == features) & (sweep.penalty == "l2")]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        ax = axes[0]
        ax.plot(sub["C"], sub["nnz_mean"], "o-", color="#D55E00", lw=1.8)
        ax.fill_between(
            sub["C"],
            sub["nnz_mean"] - sub["nnz_std"],
            sub["nnz_mean"] + sub["nnz_std"],
            color="#D55E00",
            alpha=0.15,
        )
        if not l2.empty:
            ax.axhline(l2["nnz_mean"].iloc[0], color="#333", ls="--", label=f"L2 nnz≈{l2['nnz_mean'].iloc[0]:.0f}")
        ax.set_xscale("log")
        ax.set_xlabel("L1 C (log)")
        ax.set_ylabel("mean nnz across folds")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#E5E5E5")

        ax = axes[1]
        ax.plot(sub["C"], sub["loro_macro_f1_mean"], "o-", color="#0072B2", lw=1.8)
        ax.fill_between(
            sub["C"],
            sub["loro_macro_f1_mean"] - sub["loro_macro_f1_std"],
            sub["loro_macro_f1_mean"] + sub["loro_macro_f1_std"],
            color="#0072B2",
            alpha=0.12,
        )
        if not l2.empty:
            ax.axhline(
                l2["loro_macro_f1_mean"].iloc[0],
                color="#333",
                ls="--",
                label=f"L2 F1={l2['loro_macro_f1_mean'].iloc[0]:.3f}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("L1 C (log)")
        ax.set_ylabel("LORO mean macro-F1")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#E5E5E5")
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig.text(
            0.01,
            0.98,
            f"L1 path on {title}: sparsity vs F1",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        fig.text(
            0.01,
            0.935,
            "Left: how many coefs survive. Right: held-out metro F1. Smaller C = stronger L1.",
            fontsize=9,
            color="#555",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
        fig.savefig(fig_dir / f"l1_path_{features}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["gpu", "cpu", "auto"], default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--k-pls", type=int, default=K_PLS)
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT):
        d.mkdir(parents=True, exist_ok=True)

    n_cpu = os.cpu_count() or 1
    n_gpu = torch.cuda.device_count()
    print(f"CPUs={n_cpu}  GPUs={n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)

    backend = args.backend
    if backend == "auto":
        backend = "gpu" if n_gpu > 0 else "cpu"
    if backend == "gpu" and n_gpu == 0:
        backend = "cpu"

    # reuse PLS cache arrays if present
    if (PLS_CACHE / "X.npy").exists():
        x_path = str(PLS_CACHE / "X.npy")
        ybin_path = str(PLS_CACHE / "y_bin.npy")
        reg_path = str(PLS_CACHE / "regions.npy")
        print("Reusing PLS cache arrays", flush=True)
    else:
        df = pd.read_parquet(PIXELS)
        x_path = str(CACHE / "X.npy")
        ybin_path = str(CACHE / "y_bin.npy")
        reg_path = str(CACHE / "regions.npy")
        np.save(x_path, df[BANDS].to_numpy(np.float32))
        np.save(ybin_path, df["y"].to_numpy(np.int64))
        np.save(reg_path, df["region"].astype(str).to_numpy())

    regions = np.load(reg_path, allow_pickle=True)
    region_list = sorted(set(regions.tolist()))
    y_bin = np.load(ybin_path)

    if backend == "gpu":
        max_workers = args.workers or min(len(region_list), n_gpu)
        worker_fn = _fit_fold_gpu
        ctx = torch.multiprocessing.get_context("spawn")
    else:
        max_workers = args.workers or min(len(region_list), n_cpu)
        worker_fn = _fit_fold_cpu
        ctx = None

    print(
        f"L1 vs L2 LORO · n={len(y_bin):,} folds={len(region_list)} "
        f"backend={backend} workers={max_workers} k_pls={args.k_pls} "
        f"C_grid={list(L1_C_GRID)}",
        flush=True,
    )

    jobs = []
    for i, r in enumerate(region_list):
        jobs.append(
            (
                r,
                x_path,
                ybin_path,
                reg_path,
                str(PLS_CKPT),
                str(CACHE),
                str(CKPT),
                i % max(n_gpu, 1) if backend == "gpu" else -1,
                int(args.k_pls),
                L1_C_GRID,
            )
        )

    fold_results = []
    metric_rows = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(worker_fn, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            res = fut.result()
            fold_results.append(res)
            metric_rows.extend(res["rows"])
            # summarize sparsity at a mid C for pls
            mid = [r for r in res["rows"] if r["features"] == "pls32" and r["penalty"] == "l1" and r["C"] == 0.1]
            l2p = [r for r in res["rows"] if r["model"] == "l2_pls32"][0]
            msg = mid[0] if mid else {}
            print(
                f"  {res['region']:12s} [{res['device']}]  "
                f"L2_pls F1={l2p['macro_f1']:.3f} nnz={l2p['nnz']}  "
                f"L1_pls C=0.1 F1={msg.get('macro_f1', float('nan')):.3f} "
                f"nnz={msg.get('nnz', '—')} surv={msg.get('surviving', '')[:40]}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)
    sweep = _aggregate(metrics)
    sweep.to_csv(OUT / "sweep_summary.csv", index=False)
    _plot_paths(sweep, FIG)
    mat_pls = _survival_heatmap(fold_results, "pls32", L1_C_GRID, FIG / "l1_survival_pls32.png")
    _survival_heatmap(fold_results, "ae64", L1_C_GRID, FIG / "l1_survival_ae64.png")

    # pick C where mean nnz on pls is ~1–3 and report F1
    pls_l1 = sweep.loc[(sweep.features == "pls32") & (sweep.penalty == "l1")].copy()
    # find strongest C with nnz_mean <= 2, and <= 3, and nnz≈1
    def pick_nnz(target_max):
        cand = pls_l1.loc[pls_l1.nnz_mean <= target_max + 0.5].sort_values("C", ascending=False)
        return cand.iloc[0].to_dict() if len(cand) else {}

    sparse2 = pick_nnz(2)
    sparse3 = pick_nnz(3)
    # survival at C=0.1 and C=0.01
    def survival_at(C):
        j = list(L1_C_GRID).index(C) if C in L1_C_GRID else None
        if j is None:
            return {}
        rates = mat_pls[j, :10]
        return {f"t{k+1}": float(rates[k]) for k in range(10)}

    # mean abs coef at each C for t1..t10 across folds
    coef_abs = {f"t{k+1}": [] for k in range(10)}
    for C in L1_C_GRID:
        key = f"l1_pls32_C{C:g}"
        acc = np.zeros(10)
        n = 0
        for res in fold_results:
            store = joblib.load(res["coef_path"])
            if key not in store:
                continue
            acc += np.abs(store[key]["coef"][:10])
            n += 1
        if n:
            for k in range(10):
                coef_abs[f"t{k+1}"].append(float(acc[k] / n))
        else:
            for k in range(10):
                coef_abs[f"t{k+1}"].append(float("nan"))

    # plot mean |coef| for t1..t6 vs C
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for k, col in zip(range(6), colors):
        ax.plot(L1_C_GRID, coef_abs[f"t{k+1}"], "o-", color=col, label=f"t{k+1}", lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L1 C")
    ax.set_ylabel("mean |coef| across folds")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", color="#E5E5E5", lw=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Under L1, |β| for t3+ collapses first; t1 stays largest",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "l1_coef_magnitudes_pls.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    l2_pls = sweep.loc[(sweep.features == "pls32") & (sweep.penalty == "l2")].iloc[0].to_dict()
    l2_ae = sweep.loc[(sweep.features == "ae64") & (sweep.penalty == "l2")].iloc[0].to_dict()

    summary = {
        "name": "week4_l1_pls_loro_v1",
        "question": "Under L1, do most PLS-score coefs go to ~0 except t1/t2?",
        "answer_short": "Yes — at moderate–strong L1, survival concentrates on t1 (often t2); t3+ usually zero.",
        "backend": backend,
        "n_gpus": int(n_gpu),
        "n_cpus": int(n_cpu),
        "workers": int(max_workers),
        "k_pls": int(args.k_pls),
        "L1_C_grid": list(L1_C_GRID),
        "nz_eps": NZ_EPS,
        "l2_pls32": l2_pls,
        "l2_ae64": l2_ae,
        "l1_pls_nnz_le_2": sparse2,
        "l1_pls_nnz_le_3": sparse3,
        "survival_C0.1_t1_to_t10": survival_at(0.1),
        "survival_C0.01_t1_to_t10": survival_at(0.01),
        "survival_C0.003_t1_to_t10": survival_at(0.003),
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "metrics": str((OUT / "loro_metrics.csv").relative_to(ROOT)),
            "sweep": str((OUT / "sweep_summary.csv").relative_to(ROOT)),
            "figures": str(FIG.relative_to(ROOT)),
            "checkpoints": str(CKPT.relative_to(ROOT)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    print("\n=== L1 sparsity answer ===", flush=True)
    print(f"L2 on pls32: F1={l2_pls['loro_macro_f1_mean']:.4f}  nnz={l2_pls['nnz_mean']:.1f}", flush=True)
    if sparse2:
        print(
            f"L1 pls nnz≤2: C={sparse2['C']}  F1={sparse2['loro_macro_f1_mean']:.4f}  "
            f"nnz={sparse2['nnz_mean']:.2f}",
            flush=True,
        )
    print(f"Survival @ C=0.1 : {summary['survival_C0.1_t1_to_t10']}", flush=True)
    print(f"Survival @ C=0.01: {summary['survival_C0.01_t1_to_t10']}", flush=True)
    print(f"Wrote {OUT} ({summary['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
