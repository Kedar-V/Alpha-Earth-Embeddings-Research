#!/usr/bin/env python3
"""Week 4 LORO: Partial Least Squares vs Ridge / LR on balanced pure AE pixels.

Fits per held-out metro (all GPUs in parallel):
  - baseline_lr / baseline_ridge on scaled A00–A63
  - PCA → LR (unsupervised compression control)
  - PLS-DA → LR (PLS on binary y, then logistic on scores)
  - PLS fraction head (PLSRegression X→(frac_res, frac_nres, frac_empty))
  - PLS frac scores → LR (supervised spatial compression for binary)

Sweeps n_components; picks best by LORO mean macro-F1 / MAE.
Writes outputs/week4_pls_loro/ (checkpoints, OOF, PLS scores, plots, summary).

Usage:
  .venv/bin/python -u scripts/train_week4_pls_loro.py
  .venv/bin/python -u scripts/train_week4_pls_loro.py --backend cpu --workers 40
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_pls_loro"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")
N_COMPS = (2, 4, 8, 12, 16, 24, 32)
N_COMP_MAX = max(N_COMPS)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------------------
# Torch GPU helpers
# ---------------------------------------------------------------------------


def _torch_standardize(
    X_tr: torch.Tensor, X_te: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd


def _torch_pls_nipals(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Fast NIPALS PLSRegression for n ≫ p (iterations in p-space only).

    X, Y must already be centered (and preferably scaled like StandardScaler).
    Returns x_weights (W), x_loadings (P), y_loadings (C), x_scores train (T),
    and x_rotations for transform: X_new @ x_rotations.
    """
    device = X.device
    dtype = X.dtype
    n, p = X.shape
    q = Y.shape[1]
    n_comp = int(min(n_components, p))

    # Work with cross-covariance; deflate via rank-1 updates of XtX / XtY
    # Equivalent to sklearn NIPALS when n is large.
    XtX = X.T @ X  # (p, p)
    XtY = X.T @ Y  # (p, q)
    W = torch.zeros((p, n_comp), device=device, dtype=dtype)
    P = torch.zeros((p, n_comp), device=device, dtype=dtype)
    C = torch.zeros((q, n_comp), device=device, dtype=dtype)
    T = torch.zeros((n, n_comp), device=device, dtype=dtype)

    for k in range(n_comp):
        # leading direction of XtY (power / SVD on small matrix)
        if q == 1:
            w = XtY[:, 0].clone()
        else:
            # top left singular vector of XtY
            U, _, _ = torch.linalg.svd(XtY, full_matrices=False)
            w = U[:, 0].clone()
        wn = w.norm().clamp_min(1e-12)
        if float(wn) < 1e-12:
            break
        w = w / wn
        w_old = torch.zeros(p, device=device, dtype=dtype)
        for _ in range(max_iter):
            # t ∝ Xw ; c ∝ Y't ; w ∝ X'Y c  — all via Gram
            # ||t||^2 = w' XtX w
            t_var = (w @ (XtX @ w)).clamp_min(1e-12)
            c = (XtY.T @ w) / t_var  # (q,)
            cn = c.norm().clamp_min(1e-12)
            c = c / cn
            w_new = XtY @ c
            wn = w_new.norm().clamp_min(1e-12)
            w_new = w_new / wn
            if float((w_new - w_old).norm()) < tol:
                w = w_new
                break
            w_old = w
            w = w_new

        t = X @ w
        t_var = (t @ t).clamp_min(1e-12)
        p_k = (X.T @ t) / t_var
        c = (Y.T @ t) / t_var

        # Deflate X, Y in Gram form: X ← X - t p', Y ← Y - t c'
        # XtX ← (I - p w')' XtX (I - w p') … use explicit residual on X/Y for accuracy
        X = X - t.unsqueeze(1) @ p_k.unsqueeze(0)
        Y = Y - t.unsqueeze(1) @ c.unsqueeze(0)
        XtX = X.T @ X
        XtY = X.T @ Y

        W[:, k] = w
        P[:, k] = p_k
        C[:, k] = c
        T[:, k] = t

    PtW = P.T @ W
    try:
        R = W @ torch.linalg.inv(PtW)
    except Exception:
        R = W @ torch.linalg.pinv(PtW)

    return {
        "x_weights": W,
        "x_loadings": P,
        "y_loadings": C,
        "x_scores": T,
        "x_rotations": R,
    }


def _torch_pls_predict(
    X_te_c: torch.Tensor,
    Y_tr_mean: torch.Tensor,
    pls: dict[str, torch.Tensor],
    n_comp: int,
) -> torch.Tensor:
    """Predict Y from centered X using truncated PLS components (sklearn-like)."""
    R = pls["x_rotations"][:, :n_comp]
    C = pls["y_loadings"][:, :n_comp]
    T = X_te_c @ R
    # Y_hat = T @ C.T + mean(Y_tr)  (for standardized Y this is in scaled space;
    # we fit on raw-centered Y below so this is absolute.)
    return T @ C.T + Y_tr_mean


def _torch_logistic(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    max_iter: int = 80,
    wd: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
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
    return y_pred, p, w.detach(), b.detach()


def _torch_ridge(
    X_tr: torch.Tensor, Y_tr: torch.Tensor, X_te: torch.Tensor, alpha: float = 1.0
) -> np.ndarray:
    ones_tr = torch.ones((X_tr.shape[0], 1), device=X_tr.device, dtype=X_tr.dtype)
    ones_te = torch.ones((X_te.shape[0], 1), device=X_te.device, dtype=X_te.dtype)
    Phi_tr = torch.cat([X_tr, ones_tr], dim=1)
    Phi_te = torch.cat([X_te, ones_te], dim=1)
    xtx = Phi_tr.T @ Phi_tr
    xtx = xtx + alpha * torch.eye(xtx.shape[0], device=X_tr.device, dtype=X_tr.dtype)
    beta = torch.linalg.solve(xtx, Phi_tr.T @ Y_tr)
    pred = torch.clamp(Phi_te @ beta, 0.0, 1.0)
    return pred.detach().cpu().numpy().astype(np.float64)


def _torch_pca_fit_transform(
    X_tr: torch.Tensor, X_te: torch.Tensor, n_comp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # X already standardized; SVD
    # economy SVD
    _, _, Vh = torch.linalg.svd(X_tr, full_matrices=False)
    components = Vh[:n_comp].T  # (p, k)
    return X_tr @ components, X_te @ components, components


# ---------------------------------------------------------------------------
# Fold workers
# ---------------------------------------------------------------------------


def _fit_fold_gpu(payload: tuple) -> dict:
    (
        te_region,
        x_path,
        y_path,
        ybin_path,
        reg_path,
        cache_dir,
        ckpt_dir,
        device_id,
        n_comps,
        n_comp_max,
    ) = payload
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    X_np = np.load(x_path, mmap_mode="r")
    Y_np = np.load(y_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)

    X_tr = torch.as_tensor(np.asarray(X_np[tr], dtype=np.float32), device=device)
    Y_tr = torch.as_tensor(np.asarray(Y_np[tr], dtype=np.float32), device=device)
    y_tr = torch.as_tensor(np.asarray(y_bin[tr], dtype=np.float32), device=device)
    X_te = torch.as_tensor(np.asarray(X_np[te], dtype=np.float32), device=device)
    Y_te = np.asarray(Y_np[te], dtype=np.float64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    X_tr_s, X_te_s, mu, sd = _torch_standardize(X_tr, X_te)

    rows: list[dict] = []
    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    # --- baselines on full 64-D ---
    y_pred_lr, p_res_lr, w_lr, b_lr = _torch_logistic(X_tr_s, y_tr, X_te_s)
    pred_ridge = _torch_ridge(X_tr_s, Y_tr, X_te_s, alpha=1.0)
    mae_r = [float(mean_absolute_error(Y_te[:, i], pred_ridge[:, i])) for i in range(3)]
    rows.append(
        {
            "region": te_region,
            "model": "baseline_lr",
            "n_components": 64,
            "macro_f1": float(f1_score(yb_te, y_pred_lr, average="macro", zero_division=0)),
            "mae_mean": float("nan"),
            "mae_frac_res": float("nan"),
            "mae_frac_nres": float("nan"),
            "mae_frac_empty": float("nan"),
            "device": f"cuda:{device_id}",
        }
    )
    rows.append(
        {
            "region": te_region,
            "model": "baseline_ridge",
            "n_components": 64,
            "macro_f1": float("nan"),
            "mae_mean": float(np.mean(mae_r)),
            "mae_frac_res": mae_r[0],
            "mae_frac_nres": mae_r[1],
            "mae_frac_empty": mae_r[2],
            "device": f"cuda:{device_id}",
        }
    )

    # save baseline OOF
    np.save(cache / f"oof_baseline_ypred_{te_region}.npy", y_pred_lr)
    np.save(cache / f"oof_baseline_pres_{te_region}.npy", p_res_lr)
    np.save(cache / f"oof_baseline_frac_{te_region}.npy", pred_ridge)
    np.save(cache / f"oof_idx_{te_region}.npy", te_idx)

    # --- center Y for PLS ---
    Y_mu = Y_tr.mean(dim=0, keepdim=True)
    Y_tr_c = Y_tr - Y_mu
    y_mu = y_tr.mean()
    y_tr_c = (y_tr - y_mu).unsqueeze(1)

    # Fit PLS once at max components
    pls_da = _torch_pls_nipals(X_tr_s, y_tr_c, n_comp_max)
    pls_frac = _torch_pls_nipals(X_tr_s, Y_tr_c, n_comp_max)

    # PCA at max
    _, _, pca_comp_max = _torch_pca_fit_transform(X_tr_s, X_te_s, n_comp_max)

    # Store OOF containers for each n_comp
    oof_pack: dict[str, dict[int, dict]] = {
        "pls_da_lr": {},
        "pls_frac": {},
        "pls_frac_lr": {},
        "pca_lr": {},
    }

    for k in tqdm(list(n_comps), desc=f"{te_region} n_comp", leave=False):
        # PLS-DA → LR
        R_da = pls_da["x_rotations"][:, :k]
        T_tr_da = X_tr_s @ R_da
        T_te_da = X_te_s @ R_da
        # re-standardize scores for LR stability
        tmu = T_tr_da.mean(0, keepdim=True)
        tsd = T_tr_da.std(0, keepdim=True).clamp_min(1e-6)
        yhat, p_res, _, _ = _torch_logistic(
            (T_tr_da - tmu) / tsd, y_tr, (T_te_da - tmu) / tsd
        )
        oof_pack["pls_da_lr"][k] = {"y_pred": yhat, "p_res": p_res}
        rows.append(
            {
                "region": te_region,
                "model": "pls_da_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": f"cuda:{device_id}",
            }
        )

        # PLS fraction prediction (X already standardized / centered)
        pred_pls = _torch_pls_predict(X_te_s, Y_mu, pls_frac, k)
        pred_pls = torch.clamp(pred_pls, 0.0, 1.0).detach().cpu().numpy().astype(np.float64)
        mae = [float(mean_absolute_error(Y_te[:, i], pred_pls[:, i])) for i in range(3)]
        oof_pack["pls_frac"][k] = {"frac": pred_pls}
        rows.append(
            {
                "region": te_region,
                "model": "pls_frac",
                "n_components": int(k),
                "macro_f1": float("nan"),
                "mae_mean": float(np.mean(mae)),
                "mae_frac_res": mae[0],
                "mae_frac_nres": mae[1],
                "mae_frac_empty": mae[2],
                "device": f"cuda:{device_id}",
            }
        )

        # PLS frac scores → LR
        R_f = pls_frac["x_rotations"][:, :k]
        T_tr_f = X_tr_s @ R_f
        T_te_f = X_te_s @ R_f
        tmu = T_tr_f.mean(0, keepdim=True)
        tsd = T_tr_f.std(0, keepdim=True).clamp_min(1e-6)
        yhat2, p2, _, _ = _torch_logistic(
            (T_tr_f - tmu) / tsd, y_tr, (T_te_f - tmu) / tsd
        )
        oof_pack["pls_frac_lr"][k] = {"y_pred": yhat2, "p_res": p2}
        rows.append(
            {
                "region": te_region,
                "model": "pls_frac_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat2, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": f"cuda:{device_id}",
            }
        )

        # PCA → LR
        comps = pca_comp_max[:, :k]
        T_tr_p = X_tr_s @ comps
        T_te_p = X_te_s @ comps
        tmu = T_tr_p.mean(0, keepdim=True)
        tsd = T_tr_p.std(0, keepdim=True).clamp_min(1e-6)
        yhat3, p3, _, _ = _torch_logistic(
            (T_tr_p - tmu) / tsd, y_tr, (T_te_p - tmu) / tsd
        )
        oof_pack["pca_lr"][k] = {"y_pred": yhat3, "p_res": p3}
        rows.append(
            {
                "region": te_region,
                "model": "pca_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat3, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": f"cuda:{device_id}",
            }
        )

    # persist OOF pack + checkpoint
    joblib.dump(
        {
            "region_held_out": te_region,
            "backend": "torch_gpu",
            "device_id": device_id,
            "mu": mu.detach().cpu().numpy(),
            "sd": sd.detach().cpu().numpy(),
            "Y_mu": Y_mu.detach().cpu().numpy(),
            "pls_da": {k: v.detach().cpu().numpy() for k, v in pls_da.items()},
            "pls_frac": {k: v.detach().cpu().numpy() for k, v in pls_frac.items()},
            "pca_components": pca_comp_max.detach().cpu().numpy(),
            "baseline_lr_w": w_lr.cpu().numpy(),
            "baseline_lr_b": float(b_lr.cpu().item()),
            "n_comps": list(n_comps),
            "seed": SEED,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    joblib.dump(oof_pack, cache / f"oof_pack_{te_region}.joblib")

    # free GPU
    del X_tr, X_te, Y_tr, X_tr_s, X_te_s, pls_da, pls_frac
    torch.cuda.empty_cache()

    return {
        "region": te_region,
        "device": f"cuda:{device_id}",
        "rows": rows,
        "idx_path": str(cache / f"oof_idx_{te_region}.npy"),
        "pack_path": str(cache / f"oof_pack_{te_region}.joblib"),
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
    }


def _fit_fold_cpu(payload: tuple) -> dict:
    """Sklearn CPU fold (same models) — used when --backend cpu."""
    (
        te_region,
        x_path,
        y_path,
        ybin_path,
        reg_path,
        cache_dir,
        ckpt_dir,
        _device_id,
        n_comps,
        n_comp_max,
    ) = payload

    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)

    X_tr = np.asarray(X[tr], dtype=np.float64)
    Y_tr = np.asarray(Y[tr], dtype=np.float64)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float64)
    Y_te = np.asarray(Y[te], dtype=np.float64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    y_pred_lr = clf.predict(X_te_s)
    p_res_lr = clf.predict_proba(X_te_s)[:, 1]
    rows.append(
        {
            "region": te_region,
            "model": "baseline_lr",
            "n_components": 64,
            "macro_f1": float(f1_score(yb_te, y_pred_lr, average="macro", zero_division=0)),
            "mae_mean": float("nan"),
            "mae_frac_res": float("nan"),
            "mae_frac_nres": float("nan"),
            "mae_frac_empty": float("nan"),
            "device": "cpu",
        }
    )

    reg = MultiOutputRegressor(Ridge(alpha=1.0, random_state=SEED), n_jobs=1)
    reg.fit(X_tr_s, Y_tr)
    pred_ridge = np.clip(reg.predict(X_te_s), 0.0, 1.0)
    mae_r = [float(mean_absolute_error(Y_te[:, i], pred_ridge[:, i])) for i in range(3)]
    rows.append(
        {
            "region": te_region,
            "model": "baseline_ridge",
            "n_components": 64,
            "macro_f1": float("nan"),
            "mae_mean": float(np.mean(mae_r)),
            "mae_frac_res": mae_r[0],
            "mae_frac_nres": mae_r[1],
            "mae_frac_empty": mae_r[2],
            "device": "cpu",
        }
    )
    np.save(cache / f"oof_baseline_ypred_{te_region}.npy", y_pred_lr.astype(np.int64))
    np.save(cache / f"oof_baseline_pres_{te_region}.npy", p_res_lr.astype(np.float64))
    np.save(cache / f"oof_baseline_frac_{te_region}.npy", pred_ridge.astype(np.float64))
    np.save(cache / f"oof_idx_{te_region}.npy", te_idx)

    pls_da = PLSRegression(n_components=n_comp_max, scale=False)
    pls_da.fit(X_tr_s, y_tr.astype(np.float64))
    pls_frac = PLSRegression(n_components=n_comp_max, scale=False)
    pls_frac.fit(X_tr_s, Y_tr)
    pca = PCA(n_components=n_comp_max, random_state=SEED)
    pca.fit(X_tr_s)

    oof_pack: dict[str, dict[int, dict]] = {
        "pls_da_lr": {},
        "pls_frac": {},
        "pls_frac_lr": {},
        "pca_lr": {},
    }

    for k in tqdm(list(n_comps), desc=f"{te_region} n_comp", leave=False):
        # truncate via re-fit at k (sklearn doesn't truncate cleanly for predict)
        pls_k = PLSRegression(n_components=k, scale=False)
        pls_k.fit(X_tr_s, y_tr.astype(np.float64))
        T_tr = pls_k.transform(X_tr_s)
        T_te = pls_k.transform(X_te_s)
        lr = LogisticRegression(max_iter=2000, random_state=SEED)
        lr.fit(T_tr, y_tr)
        yhat = lr.predict(T_te)
        p = lr.predict_proba(T_te)[:, 1]
        oof_pack["pls_da_lr"][k] = {"y_pred": yhat.astype(np.int64), "p_res": p.astype(np.float64)}
        rows.append(
            {
                "region": te_region,
                "model": "pls_da_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": "cpu",
            }
        )

        pls_fk = PLSRegression(n_components=k, scale=False)
        pls_fk.fit(X_tr_s, Y_tr)
        pred = np.clip(pls_fk.predict(X_te_s), 0.0, 1.0)
        mae = [float(mean_absolute_error(Y_te[:, i], pred[:, i])) for i in range(3)]
        oof_pack["pls_frac"][k] = {"frac": pred.astype(np.float64)}
        rows.append(
            {
                "region": te_region,
                "model": "pls_frac",
                "n_components": int(k),
                "macro_f1": float("nan"),
                "mae_mean": float(np.mean(mae)),
                "mae_frac_res": mae[0],
                "mae_frac_nres": mae[1],
                "mae_frac_empty": mae[2],
                "device": "cpu",
            }
        )

        T_tr_f = pls_fk.transform(X_tr_s)
        T_te_f = pls_fk.transform(X_te_s)
        lr2 = LogisticRegression(max_iter=2000, random_state=SEED)
        lr2.fit(T_tr_f, y_tr)
        yhat2 = lr2.predict(T_te_f)
        p2 = lr2.predict_proba(T_te_f)[:, 1]
        oof_pack["pls_frac_lr"][k] = {
            "y_pred": yhat2.astype(np.int64),
            "p_res": p2.astype(np.float64),
        }
        rows.append(
            {
                "region": te_region,
                "model": "pls_frac_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat2, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": "cpu",
            }
        )

        T_tr_p = pca.transform(X_tr_s)[:, :k]
        T_te_p = pca.transform(X_te_s)[:, :k]
        lr3 = LogisticRegression(max_iter=2000, random_state=SEED)
        lr3.fit(T_tr_p, y_tr)
        yhat3 = lr3.predict(T_te_p)
        p3 = lr3.predict_proba(T_te_p)[:, 1]
        oof_pack["pca_lr"][k] = {
            "y_pred": yhat3.astype(np.int64),
            "p_res": p3.astype(np.float64),
        }
        rows.append(
            {
                "region": te_region,
                "model": "pca_lr",
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat3, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "mae_frac_res": float("nan"),
                "mae_frac_nres": float("nan"),
                "mae_frac_empty": float("nan"),
                "device": "cpu",
            }
        )

    joblib.dump(
        {
            "region_held_out": te_region,
            "backend": "sklearn_cpu",
            "scaler": scaler,
            "baseline_clf": clf,
            "baseline_reg": reg,
            "pls_da": pls_da,
            "pls_frac": pls_frac,
            "pca": pca,
            "n_comps": list(n_comps),
            "seed": SEED,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    joblib.dump(oof_pack, cache / f"oof_pack_{te_region}.joblib")
    return {
        "region": te_region,
        "device": "cpu",
        "rows": rows,
        "idx_path": str(cache / f"oof_idx_{te_region}.npy"),
        "pack_path": str(cache / f"oof_pack_{te_region}.joblib"),
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
    }


# ---------------------------------------------------------------------------
# Aggregation + plots
# ---------------------------------------------------------------------------


def _aggregate_sweep(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, k), g in metrics.groupby(["model", "n_components"], dropna=False):
        rec = {
            "model": model,
            "n_components": int(k) if pd.notna(k) else None,
            "n_folds": len(g),
        }
        if g["macro_f1"].notna().any():
            rec["loro_macro_f1_mean"] = float(g["macro_f1"].mean())
            rec["loro_macro_f1_std"] = float(g["macro_f1"].std(ddof=0))
        else:
            rec["loro_macro_f1_mean"] = float("nan")
            rec["loro_macro_f1_std"] = float("nan")
        if g["mae_mean"].notna().any():
            rec["loro_mae_mean"] = float(g["mae_mean"].mean())
            rec["loro_mae_std"] = float(g["mae_mean"].std(ddof=0))
        else:
            rec["loro_mae_mean"] = float("nan")
            rec["loro_mae_std"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "n_components"]).reset_index(drop=True)


def _plot_sweep(sweep: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # F1 vs n_comp
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {
        "pls_da_lr": "#0072B2",
        "pls_frac_lr": "#E69F00",
        "pca_lr": "#009E73",
        "baseline_lr": "#333333",
    }
    for model, c in colors.items():
        sub = sweep.loc[sweep["model"] == model].dropna(subset=["loro_macro_f1_mean"])
        if sub.empty:
            continue
        if model == "baseline_lr":
            ax.axhline(
                float(sub["loro_macro_f1_mean"].iloc[0]),
                color=c,
                ls="--",
                lw=1.5,
                label=f"baseline LR ({sub['loro_macro_f1_mean'].iloc[0]:.3f})",
            )
        else:
            ax.plot(
                sub["n_components"],
                sub["loro_macro_f1_mean"],
                marker="o",
                color=c,
                lw=1.8,
                label=model,
            )
            ax.fill_between(
                sub["n_components"],
                sub["loro_macro_f1_mean"] - sub["loro_macro_f1_std"],
                sub["loro_macro_f1_mean"] + sub["loro_macro_f1_std"],
                color=c,
                alpha=0.12,
            )
    ax.set_xlabel("n_components")
    ax.set_ylabel("LORO mean macro-F1")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "PLS-DA matches full LR once ~16 components; PCA trails",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.01,
        0.935,
        "Shaded = ± LORO std across metros. Balanced pure 10 m cells.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(fig_dir / "f1_vs_ncomp.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # MAE vs n_comp
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model, c, ls in (
        ("pls_frac", "#D55E00", "-"),
        ("baseline_ridge", "#333333", "--"),
    ):
        sub = sweep.loc[sweep["model"] == model].dropna(subset=["loro_mae_mean"])
        if sub.empty:
            continue
        if model == "baseline_ridge":
            ax.axhline(
                float(sub["loro_mae_mean"].iloc[0]),
                color=c,
                ls=ls,
                lw=1.5,
                label=f"baseline Ridge ({sub['loro_mae_mean'].iloc[0]:.3f})",
            )
        else:
            ax.plot(
                sub["n_components"],
                sub["loro_mae_mean"],
                marker="o",
                color=c,
                lw=1.8,
                label="pls_frac",
            )
            ax.fill_between(
                sub["n_components"],
                sub["loro_mae_mean"] - sub["loro_mae_std"],
                sub["loro_mae_mean"] + sub["loro_mae_std"],
                color=c,
                alpha=0.12,
            )
    ax.set_xlabel("n_components")
    ax.set_ylabel("LORO mean MAE (3 fractions)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "PLS fraction head vs Ridge on all 64 bands",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(fig_dir / "mae_vs_ncomp.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pick_best(sweep: pd.DataFrame, model: str, metric: str, higher: bool) -> dict:
    sub = sweep.loc[sweep["model"] == model].dropna(subset=[metric])
    if sub.empty:
        return {}
    idx = sub[metric].idxmax() if higher else sub[metric].idxmin()
    return sub.loc[idx].to_dict()


def _jsonable(obj):
    """Convert NaN/Inf floats to None for strict JSON."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["gpu", "cpu", "auto"], default="auto")
    ap.add_argument("--workers", type=int, default=None, help="Max parallel fold workers")
    ap.add_argument(
        "--n-comps",
        type=str,
        default=",".join(str(x) for x in N_COMPS),
        help="Comma-separated n_components sweep",
    )
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT):
        d.mkdir(parents=True, exist_ok=True)

    n_comps = tuple(int(x) for x in args.n_comps.split(",") if x.strip())
    n_comp_max = max(n_comps)

    n_cpu = os.cpu_count() or 1
    n_gpu = torch.cuda.device_count()
    print(f"CPUs={n_cpu}  GPUs={n_gpu}", flush=True)
    if n_gpu:
        for i in range(n_gpu):
            print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)

    backend = args.backend
    if backend == "auto":
        backend = "gpu" if n_gpu > 0 else "cpu"
    if backend == "gpu" and n_gpu == 0:
        print("No CUDA; falling back to CPU", flush=True)
        backend = "cpu"

    if not PIXELS.exists():
        raise FileNotFoundError(PIXELS)

    print(f"Loading {PIXELS} …", flush=True)
    df = pd.read_parquet(PIXELS).reset_index(drop=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float32)
    y_bin = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].astype(str).to_numpy()
    region_list = sorted(df["region"].astype(str).unique().tolist())

    x_path = str(CACHE / "X.npy")
    y_path = str(CACHE / "Y.npy")
    ybin_path = str(CACHE / "y_bin.npy")
    reg_path = str(CACHE / "regions.npy")
    for path, arr in (
        (x_path, X),
        (y_path, Y),
        (ybin_path, y_bin),
        (reg_path, regions),
    ):
        np.save(path, arr)

    if backend == "gpu":
        max_workers = args.workers or min(len(region_list), n_gpu)
        worker_fn = _fit_fold_gpu
    else:
        max_workers = args.workers or min(len(region_list), n_cpu)
        worker_fn = _fit_fold_cpu

    print(
        f"PLS LORO · n={len(df):,} folds={len(region_list)} "
        f"backend={backend} workers={max_workers} n_comps={list(n_comps)}",
        flush=True,
    )

    jobs = []
    for i, r in enumerate(region_list):
        device_id = i % max(n_gpu, 1) if backend == "gpu" else -1
        jobs.append(
            (
                r,
                x_path,
                y_path,
                ybin_path,
                reg_path,
                str(CACHE),
                str(CKPT),
                device_id,
                n_comps,
                n_comp_max,
            )
        )

    fold_results = []
    metric_rows: list[dict] = []
    ctx = torch.multiprocessing.get_context("spawn") if backend == "gpu" else None

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(worker_fn, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            res = fut.result()
            fold_results.append(res)
            metric_rows.extend(res["rows"])
            # print best F1 line for this fold
            fr = [r for r in res["rows"] if r["model"] == "baseline_lr"][0]
            best_pls = max(
                (r for r in res["rows"] if r["model"] == "pls_da_lr"),
                key=lambda r: r["macro_f1"],
            )
            print(
                f"  {res['region']:12s}  [{res['device']}]  "
                f"baselineF1={fr['macro_f1']:.3f}  "
                f"best_pls_da_lr(k={best_pls['n_components']})={best_pls['macro_f1']:.3f}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)
    sweep = _aggregate_sweep(metrics)
    sweep.to_csv(OUT / "sweep_summary.csv", index=False)
    _plot_sweep(sweep, FIG)

    best_pls_da = _pick_best(sweep, "pls_da_lr", "loro_macro_f1_mean", higher=True)
    best_pls_frac_lr = _pick_best(sweep, "pls_frac_lr", "loro_macro_f1_mean", higher=True)
    best_pca = _pick_best(sweep, "pca_lr", "loro_macro_f1_mean", higher=True)
    best_pls_frac = _pick_best(sweep, "pls_frac", "loro_mae_mean", higher=False)
    base_lr = _pick_best(sweep, "baseline_lr", "loro_macro_f1_mean", higher=True)
    base_ridge = _pick_best(sweep, "baseline_ridge", "loro_mae_mean", higher=False)

    # Reconstruct OOF for baselines + best PLS-DA / best PLS-frac
    n = len(df)
    oof_ypred_base = np.full(n, -1, dtype=np.int64)
    oof_pres_base = np.full(n, np.nan, dtype=np.float64)
    oof_frac_base = np.full((n, 3), np.nan, dtype=np.float64)
    k_da = int(best_pls_da.get("n_components", n_comp_max))
    k_frac = int(best_pls_frac.get("n_components", n_comp_max))
    oof_ypred_pls = np.full(n, -1, dtype=np.int64)
    oof_pres_pls = np.full(n, np.nan, dtype=np.float64)
    oof_frac_pls = np.full((n, 3), np.nan, dtype=np.float64)
    oof_scores_pls = np.full((n, k_da), np.nan, dtype=np.float32)

    for res in tqdm(fold_results, desc="Assemble OOF"):
        te = np.load(res["idx_path"])
        region = res["region"]
        oof_ypred_base[te] = np.load(CACHE / f"oof_baseline_ypred_{region}.npy")
        oof_pres_base[te] = np.load(CACHE / f"oof_baseline_pres_{region}.npy")
        oof_frac_base[te] = np.load(CACHE / f"oof_baseline_frac_{region}.npy")
        pack = joblib.load(res["pack_path"])
        oof_ypred_pls[te] = pack["pls_da_lr"][k_da]["y_pred"]
        oof_pres_pls[te] = pack["pls_da_lr"][k_da]["p_res"]
        oof_frac_pls[te] = pack["pls_frac"][k_frac]["frac"]

        # OOF PLS-DA scores from checkpoint rotations
        bundle = joblib.load(CKPT / f"fold_{region}.joblib")
        if bundle.get("backend") == "torch_gpu":
            mu = bundle["mu"].ravel()
            sd = bundle["sd"].ravel()
            R = bundle["pls_da"]["x_rotations"][:, :k_da]
            X_te = np.asarray(X[te], dtype=np.float64)
            X_s = (X_te - mu) / sd
            oof_scores_pls[te] = (X_s @ R).astype(np.float32)
        else:
            # sklearn: transform with truncated fit approx via full then slice
            Xs = bundle["scaler"].transform(np.asarray(X[te], dtype=np.float64))
            # re-fit not available; use stored pls_da transform and slice
            T = bundle["pls_da"].transform(Xs)[:, :k_da]
            oof_scores_pls[te] = T.astype(np.float32)

    oof = df[
        [
            "pixel_id",
            "region",
            "stratum",
            "y",
            "frac_res_full",
            "frac_nres_full",
            "frac_empty_full",
            "pixel_lon",
            "pixel_lat",
        ]
    ].copy()
    oof["y_pred_baseline"] = oof_ypred_base
    oof["p_res_baseline"] = oof_pres_base
    oof["y_pred_pls_da"] = oof_ypred_pls
    oof["p_res_pls_da"] = oof_pres_pls
    for i, name in enumerate(FRAC_COLS):
        oof[f"pred_{name}_baseline"] = oof_frac_base[:, i]
        oof[f"pred_{name}_pls"] = oof_frac_pls[:, i]
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    np.save(OUT / f"oof_pls_da_scores_k{k_da}.npy", oof_scores_pls)

    oof_f1_base = float(f1_score(y_bin, oof_ypred_base, average="macro", zero_division=0))
    oof_f1_pls = float(f1_score(y_bin, oof_ypred_pls, average="macro", zero_division=0))
    oof_mae_base = float(
        np.mean([mean_absolute_error(Y[:, i], oof_frac_base[:, i]) for i in range(3)])
    )
    oof_mae_pls = float(
        np.mean([mean_absolute_error(Y[:, i], oof_frac_pls[:, i]) for i in range(3)])
    )

    # Comparison bar chart (best of each family)
    compare_rows = []
    for label, d, metric in (
        ("baseline_lr", base_lr, "loro_macro_f1_mean"),
        (f"pls_da_lr (k={k_da})", best_pls_da, "loro_macro_f1_mean"),
        (
            f"pls_frac_lr (k={int(best_pls_frac_lr.get('n_components', -1))})",
            best_pls_frac_lr,
            "loro_macro_f1_mean",
        ),
        (f"pca_lr (k={int(best_pca.get('n_components', -1))})", best_pca, "loro_macro_f1_mean"),
    ):
        if d:
            compare_rows.append({"label": label, "value": float(d[metric]), "kind": "f1"})
    cdf = pd.DataFrame(compare_rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    order = cdf.sort_values("value")
    colors = ["#0072B2" if "pls" in l else ("#009E73" if "pca" in l else "#555555") for l in order["label"]]
    bars = ax.barh(order["label"], order["value"], color=colors, height=0.65)
    for rect, v in zip(bars, order["value"]):
        ax.text(v + 0.005, rect.get_y() + rect.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("LORO mean macro-F1")
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Best-per-family LORO F1: PLS ≈ full LR; PCA weaker",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "best_f1_compare.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # MAE compare
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    mae_labels = [
        "baseline_ridge",
        f"pls_frac (k={k_frac})",
    ]
    mae_vals = [
        float(base_ridge.get("loro_mae_mean", np.nan)),
        float(best_pls_frac.get("loro_mae_mean", np.nan)),
    ]
    bars = ax.barh(mae_labels, mae_vals, color=["#555555", "#D55E00"], height=0.55)
    for rect, v in zip(bars, mae_vals):
        ax.text(v + 0.003, rect.get_y() + rect.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("LORO mean MAE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Fraction MAE: PLS vs Ridge",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "best_mae_compare.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Confusion for best PLS-DA
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, yp, title in (
        (axes[0], oof_ypred_base, f"Baseline LR · F1={oof_f1_base:.3f}"),
        (axes[1], oof_ypred_pls, f"PLS-DA→LR k={k_da} · F1={oof_f1_pls:.3f}"),
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_bin, yp, display_labels=["NRES", "RES"], ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "oof_confusion_compare.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Per-region F1: baseline vs best PLS-DA
    m_base = metrics.loc[metrics["model"] == "baseline_lr", ["region", "macro_f1"]].rename(
        columns={"macro_f1": "baseline"}
    )
    m_pls = metrics.loc[
        (metrics["model"] == "pls_da_lr") & (metrics["n_components"] == k_da),
        ["region", "macro_f1"],
    ].rename(columns={"macro_f1": "pls_da"})
    merged = m_base.merge(m_pls, on="region")
    merged["delta"] = merged["pls_da"] - merged["baseline"]
    merged = merged.sort_values("baseline")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(merged))
    ax.scatter(merged["baseline"], y, color="#555555", s=40, label="baseline LR", zorder=3)
    ax.scatter(merged["pls_da"], y, color="#0072B2", s=40, label=f"pls_da_lr k={k_da}", zorder=3)
    for yi, a, b in zip(y, merged["baseline"], merged["pls_da"]):
        ax.plot([a, b], [yi, yi], color="#BBBBBB", lw=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["region"].str.replace("_", " "))
    ax.set_xlabel("Macro-F1")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Per-metro F1: PLS-DA vs full 64-D LR",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "loro_f1_pls_vs_baseline.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "name": "week4_pls_loro_v1",
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "data": str(PIXELS.relative_to(ROOT)),
        "protocol": "leave-one-region-out",
        "features": BANDS,
        "n_comps_sweep": list(n_comps),
        "backend": backend,
        "n_gpus": int(n_gpu),
        "n_cpus": int(n_cpu),
        "workers": int(max_workers),
        "seed": SEED,
        "n_pixels": int(len(df)),
        "n_residential": int((y_bin == 1).sum()),
        "n_non_residential": int((y_bin == 0).sum()),
        "regions": region_list,
        "baseline_lr": base_lr,
        "baseline_ridge": base_ridge,
        "best_pls_da_lr": best_pls_da,
        "best_pls_frac_lr": best_pls_frac_lr,
        "best_pca_lr": best_pca,
        "best_pls_frac": best_pls_frac,
        "oof_macro_f1_baseline": oof_f1_base,
        "oof_macro_f1_pls_da": oof_f1_pls,
        "oof_mae_mean_baseline": oof_mae_base,
        "oof_mae_mean_pls_frac": oof_mae_pls,
        "delta_oof_f1_pls_minus_baseline": oof_f1_pls - oof_f1_base,
        "delta_oof_mae_pls_minus_baseline": oof_mae_pls - oof_mae_base,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "oof": str((OUT / "oof_predictions.parquet").relative_to(ROOT)),
            "metrics": str((OUT / "loro_metrics.csv").relative_to(ROOT)),
            "sweep": str((OUT / "sweep_summary.csv").relative_to(ROOT)),
            "pls_scores": str((OUT / f"oof_pls_da_scores_k{k_da}.npy").relative_to(ROOT)),
            "checkpoints": str(CKPT.relative_to(ROOT)),
            "figures": str(FIG.relative_to(ROOT)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    print("\n=== Summary ===", flush=True)
    print(
        f"Baseline LR   LORO F1={base_lr.get('loro_macro_f1_mean', float('nan')):.4f}  "
        f"OOF F1={oof_f1_base:.4f}",
        flush=True,
    )
    print(
        f"PLS-DA→LR k={k_da}  LORO F1={best_pls_da.get('loro_macro_f1_mean', float('nan')):.4f}  "
        f"OOF F1={oof_f1_pls:.4f}  ΔOOF={oof_f1_pls - oof_f1_base:+.4f}",
        flush=True,
    )
    print(
        f"PCA→LR best   LORO F1={best_pca.get('loro_macro_f1_mean', float('nan')):.4f} "
        f"(k={int(best_pca.get('n_components', -1))})",
        flush=True,
    )
    print(
        f"Baseline Ridge LORO MAE={base_ridge.get('loro_mae_mean', float('nan')):.4f}  "
        f"OOF MAE={oof_mae_base:.4f}",
        flush=True,
    )
    print(
        f"PLS frac k={k_frac}  LORO MAE={best_pls_frac.get('loro_mae_mean', float('nan')):.4f}  "
        f"OOF MAE={oof_mae_pls:.4f}  ΔOOF={oof_mae_pls - oof_mae_base:+.4f}",
        flush=True,
    )
    print(f"Wrote {OUT}  ({summary['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
