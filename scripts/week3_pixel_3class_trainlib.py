# Week 3 — shared helpers for notebook 05 (imported by the notebook).
# Kept small so the notebook remains the orchestration / training entrypoint.

from __future__ import annotations

import os
from pathlib import Path

# Keep BLAS single-threaded inside each fold worker (SMOTE + sklearn).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

SEED = 42
SMOTE_K = 5
EPS = 1e-6
# Building share below this → empty (noise margin on soft fraction heads).
# Tuned on soft-reg OOF (~0.15 maximizes macro-F1 and empty F1).
BUILDING_SHARE_MAX = 0.15
CLASSES = ["empty", "non_residential", "residential"]


def decode_fractions(
    frac_res,
    frac_nres,
    frac_empty,
    *,
    building_share_max: float = BUILDING_SHARE_MAX,
) -> np.ndarray:
    """Decode soft fractions → class label.

    Rule (configurable noise margin):
      if frac_res + frac_nres < building_share_max → empty
      else → residential if frac_res >= frac_nres else non_residential

    ``frac_empty`` is accepted for API compatibility / logging; the empty call
    is driven by building share so soft residual mass on RES/NRES below the
    margin does not veto empty.
    """
    frac_res = np.asarray(frac_res, dtype=np.float64)
    frac_nres = np.asarray(frac_nres, dtype=np.float64)
    _ = frac_empty  # kept in signature; empty decided by building share
    building = frac_res + frac_nres
    out = np.where(
        building < building_share_max,
        "empty",
        np.where(frac_res >= frac_nres, "residential", "non_residential"),
    )
    return out.astype(object)


def smote_xy(X_tr, y_tr, seed=SEED):
    """Classifier path: SMOTE on hard labels."""
    counts = np.bincount(y_tr, minlength=3)
    k = min(SMOTE_K, max(1, int(counts[counts > 0].min()) - 1))
    return SMOTE(random_state=seed, k_neighbors=k).fit_resample(
        np.asarray(X_tr, dtype=np.float32), np.asarray(y_tr, dtype=np.int64)
    )


def oversample_soft_xy(X_tr, y_tr, Y_tr, seed=SEED):
    """Regression path: balance by hard label but keep soft fraction targets.

    RandomOverSampler duplicates minority rows (with their true
    frac_res/nres/empty) — no synthetic X, no hard one-hot targets.
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    Y_tr = np.asarray(Y_tr, dtype=np.float32)
    idx = np.arange(len(y_tr)).reshape(-1, 1)
    idx_bal, _ = RandomOverSampler(random_state=seed).fit_resample(idx, y_tr)
    idx_bal = idx_bal.ravel()
    return X_tr[idx_bal], Y_tr[idx_bal], y_tr[idx_bal]


def _device_for(gpu_id) -> str:
    if gpu_id is None:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    return f"cuda:{int(gpu_id) % n}"


def fit_clf_torch(X_tr, y_tr, X_te, device: str, seed=SEED, epochs=80, lr=0.05):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr).astype(np.float32)
    Xt = scaler.transform(X_te).astype(np.float32)
    torch.manual_seed(seed)
    model = nn.Linear(Xs.shape[1], 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xb = torch.from_numpy(Xs).to(device, non_blocking=True)
    yb = torch.from_numpy(y_tr.astype(np.int64)).to(device, non_blocking=True)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xt).to(device, non_blocking=True))
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        proba = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
    return pred, proba


def fit_reg_torch(X_tr, Y_tr, X_te, device: str, seed=SEED, epochs=80, lr=0.05):
    """Softmax head + MSE to soft GT fractions (simplex-valued targets)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr).astype(np.float32)
    Xt = scaler.transform(X_te).astype(np.float32)
    # renormalize targets in case of float noise
    Y_tr = np.asarray(Y_tr, dtype=np.float32)
    Y_tr = np.clip(Y_tr, 0.0, 1.0)
    s = Y_tr.sum(axis=1, keepdims=True)
    Y_tr = Y_tr / np.where(s > 0, s, 1.0)

    torch.manual_seed(seed)
    model = nn.Linear(Xs.shape[1], 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    xb = torch.from_numpy(Xs).to(device, non_blocking=True)
    yb = torch.from_numpy(Y_tr).to(device, non_blocking=True)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = torch.softmax(model(xb), dim=1)
        loss_fn(pred, yb).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.softmax(model(torch.from_numpy(Xt).to(device, non_blocking=True)), dim=1)
        return pred.cpu().numpy().astype(np.float64)


def clf_fold_worker(payload: dict) -> dict:
    device = _device_for(payload.get("gpu_id"))
    gpu_id = payload.get("gpu_id")

    X = np.load(payload["x_path"], mmap_mode="r")
    y = np.load(payload["y_path"], mmap_mode="r")
    regions = np.load(payload["reg_path"], allow_pickle=True)
    te_region = payload["te_region"]
    cache = Path(payload["cache_dir"])
    te = regions == te_region
    tr = ~te
    print(f"[clf] {te_region}: SMOTE on {int(tr.sum())} → … [{device}]", flush=True)
    X_bal, y_bal = smote_xy(X[tr], y[tr])
    print(f"[clf] {te_region}: train {len(y_bal)} · fit", flush=True)
    pred, proba = fit_clf_torch(X_bal, y_bal, np.asarray(X[te], dtype=np.float32), device)
    y_te = np.asarray(y[te], dtype=np.int64)
    np.save(cache / f"clf_pred_{te_region}.npy", pred)
    np.save(cache / f"clf_proba_{te_region}.npy", proba)
    np.save(cache / f"clf_idx_{te_region}.npy", np.where(te)[0].astype(np.int64))
    f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    print(f"[clf] {te_region}: F1={f1:.3f}", flush=True)
    return {
        "region": te_region,
        "n_test": int(te.sum()),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_macro": f1,
        "n_train_before": int(tr.sum()),
        "n_train_after_smote": int(len(y_bal)),
        "device": device if gpu_id is not None else "cpu",
    }


def reg_fold_worker(payload: dict) -> dict:
    """Regression on soft GT fractions (frac_res, frac_nres, frac_empty)."""
    device = _device_for(payload.get("gpu_id"))
    gpu_id = payload.get("gpu_id")

    X = np.load(payload["x_path"], mmap_mode="r")
    y = np.load(payload["y_path"], mmap_mode="r")
    Y = np.load(payload["yf_path"], mmap_mode="r")
    regions = np.load(payload["reg_path"], allow_pickle=True)
    te_region = payload["te_region"]
    cache = Path(payload["cache_dir"])
    te = regions == te_region
    tr = ~te
    X_tr = np.asarray(X[tr], dtype=np.float32)
    y_tr = np.asarray(y[tr], dtype=np.int64)
    Y_tr = np.asarray(Y[tr], dtype=np.float32)  # soft GT
    print(
        f"[reg] {te_region}: soft-frac ROS on {len(y_tr)} "
        f"(Y mean empty={Y_tr[:, 2].mean():.3f}) → … [{device}]",
        flush=True,
    )
    X_bal, Y_bal, y_bal = oversample_soft_xy(X_tr, y_tr, Y_tr)
    print(
        f"[reg] {te_region}: train {len(Y_bal)} · "
        f"Y mean=[{Y_bal[:, 0].mean():.3f},{Y_bal[:, 1].mean():.3f},{Y_bal[:, 2].mean():.3f}] · fit",
        flush=True,
    )
    pred = fit_reg_torch(X_bal, Y_bal, np.asarray(X[te], dtype=np.float32), device)
    y_te = np.asarray(y[te], dtype=np.int64)
    y_te_frac = np.asarray(Y[te], dtype=np.float64)
    np.save(cache / f"reg_pred_{te_region}.npy", pred)
    np.save(cache / f"reg_idx_{te_region}.npy", np.where(te)[0].astype(np.int64))
    decoded = decode_fractions(pred[:, 0], pred[:, 1], pred[:, 2])
    class_to_i = {"empty": 0, "non_residential": 1, "residential": 2}
    y_hat = np.array([class_to_i[s] for s in decoded], dtype=np.int64)
    f1 = float(f1_score(y_te, y_hat, average="macro", zero_division=0))
    print(f"[reg] {te_region}: F1={f1:.3f}", flush=True)
    return {
        "region": te_region,
        "n_test": int(te.sum()),
        "f1_macro": f1,
        "accuracy": float(accuracy_score(y_te, y_hat)),
        "mae_res": float(mean_absolute_error(y_te_frac[:, 0], pred[:, 0])),
        "mae_nres": float(mean_absolute_error(y_te_frac[:, 1], pred[:, 1])),
        "mae_empty": float(mean_absolute_error(y_te_frac[:, 2], pred[:, 2])),
        "n_train_before": int(tr.sum()),
        "n_train_after_ros": int(len(Y_bal)),
        "targets": "soft_frac_res_nres_empty",
        "device": device if gpu_id is not None else "cpu",
    }
