# Week 3 notebook 06 — k×k neighborhood Conv encoder helpers.

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from week3_pixel_3class_trainlib import BUILDING_SHARE_MAX, CLASSES, decode_fractions

SEED = 42
EMB_DIM = 128
EPOCHS = 40
BATCH_SIZE = 512
LR = 1e-3


class NbCNN(nn.Module):
    """Conv encoder over (65, k, k) — AE bands + validity mask."""

    def __init__(self, n_out: int = 3, in_ch: int = 65, emb: int = EMB_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, emb, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(emb),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(emb, n_out)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, -1:, :, :]
        feats = self.backbone(x)
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return (feats * mask).sum(dim=(2, 3)) / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def pack_input(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,64,k,k) + (N,1,k,k) → (N,65,k,k) float32."""
    return np.concatenate(
        [np.asarray(X, dtype=np.float32), np.asarray(mask, dtype=np.float32)], axis=1
    )


def oversample_indices(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    idx = np.arange(len(y)).reshape(-1, 1)
    idx_bal, _ = RandomOverSampler(random_state=seed).fit_resample(idx, y)
    return idx_bal.ravel()


def _device_for(gpu_id) -> torch.device:
    if gpu_id is None or not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    return torch.device(f"cuda:{int(gpu_id) % n}")


def _make_loader(X, y_or_Y, shuffle: bool, batch_size: int = BATCH_SIZE):
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(y_or_Y)
    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fit_clf_cnn(X_tr, y_tr, X_te, device, seed=SEED, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = NbCNN(n_out=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = _make_loader(X_tr, y_tr.astype(np.int64), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    pred_chunks, proba_chunks = [], []
    with torch.no_grad():
        for xb, _ in _make_loader(X_te, np.zeros(len(X_te), dtype=np.int64), shuffle=False):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred_chunks.append(logits.argmax(1).cpu().numpy())
            proba_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (
        np.concatenate(pred_chunks).astype(np.int64),
        np.concatenate(proba_chunks).astype(np.float64),
        model,
    )


def fit_reg_cnn(X_tr, Y_tr, X_te, device, seed=SEED, epochs=EPOCHS):
    Y_tr = np.asarray(Y_tr, dtype=np.float32)
    Y_tr = np.clip(Y_tr, 0.0, 1.0)
    s = Y_tr.sum(axis=1, keepdims=True)
    Y_tr = Y_tr / np.where(s > 0, s, 1.0)

    torch.manual_seed(seed)
    model = NbCNN(n_out=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = _make_loader(X_tr, Y_tr, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = torch.softmax(model(xb), dim=1)
            loss_fn(pred, yb).backward()
            opt.step()
    model.eval()
    chunks = []
    with torch.no_grad():
        for xb, _ in _make_loader(X_te, np.zeros((len(X_te), 3), dtype=np.float32), shuffle=False):
            xb = xb.to(device, non_blocking=True)
            chunks.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64), model


def clf_fold_worker(payload: dict) -> dict:
    device = _device_for(payload.get("gpu_id"))
    gpu_id = payload.get("gpu_id")
    cache = Path(payload["cache_dir"])
    te_region = payload["te_region"]

    X = np.load(payload["x_path"], mmap_mode="r")
    mask = np.load(payload["mask_path"], mmap_mode="r")
    y = np.load(payload["y_path"], mmap_mode="r")
    regions = np.load(payload["reg_path"], allow_pickle=True)

    te = regions == te_region
    tr = ~te
    X_pack = pack_input(X, mask)
    idx_bal = oversample_indices(np.asarray(y[tr], dtype=np.int64))
    X_bal = X_pack[np.where(tr)[0][idx_bal]]
    y_bal = np.asarray(y[tr], dtype=np.int64)[idx_bal]
    X_te = X_pack[te]

    print(
        f"[clf] {te_region}: ROS {int(tr.sum())}→{len(y_bal)} · CNN fit [{device}]",
        flush=True,
    )
    pred, proba, model = fit_clf_cnn(X_bal, y_bal, X_te, device)
    y_te = np.asarray(y[te], dtype=np.int64)
    ckpt = cache / f"clf_ckpt_{te_region}.pt"
    torch.save(
        {"state_dict": model.state_dict(), "region": te_region, "task": "clf", "k": payload.get("k")},
        ckpt,
    )
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
        "n_train_after_ros": int(len(y_bal)),
        "device": str(device),
        "ckpt": str(ckpt),
    }


def reg_fold_worker(payload: dict) -> dict:
    device = _device_for(payload.get("gpu_id"))
    cache = Path(payload["cache_dir"])
    te_region = payload["te_region"]

    X = np.load(payload["x_path"], mmap_mode="r")
    mask = np.load(payload["mask_path"], mmap_mode="r")
    y = np.load(payload["y_path"], mmap_mode="r")
    Y = np.load(payload["yf_path"], mmap_mode="r")
    regions = np.load(payload["reg_path"], allow_pickle=True)

    te = regions == te_region
    tr = ~te
    X_pack = pack_input(X, mask)
    y_tr = np.asarray(y[tr], dtype=np.int64)
    Y_tr = np.asarray(Y[tr], dtype=np.float32)
    idx_bal = oversample_indices(y_tr)
    X_bal = X_pack[np.where(tr)[0][idx_bal]]
    Y_bal = Y_tr[idx_bal]
    X_te = X_pack[te]

    print(
        f"[reg] {te_region}: soft-frac ROS {int(tr.sum())}→{len(Y_bal)} · CNN fit [{device}]",
        flush=True,
    )
    pred, model = fit_reg_cnn(X_bal, Y_bal, X_te, device)
    y_te = np.asarray(y[te], dtype=np.int64)
    y_te_frac = np.asarray(Y[te], dtype=np.float64)
    ckpt = cache / f"reg_ckpt_{te_region}.pt"
    torch.save(
        {"state_dict": model.state_dict(), "region": te_region, "task": "reg", "k": payload.get("k")},
        ckpt,
    )
    np.save(cache / f"reg_pred_{te_region}.npy", pred)
    np.save(cache / f"reg_idx_{te_region}.npy", np.where(te)[0].astype(np.int64))
    decoded = decode_fractions(
        pred[:, 0], pred[:, 1], pred[:, 2], building_share_max=BUILDING_SHARE_MAX
    )
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
        "device": str(device),
        "ckpt": str(ckpt),
        "building_share_max": BUILDING_SHARE_MAX,
    }
