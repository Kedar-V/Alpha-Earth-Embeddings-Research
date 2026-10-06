#!/usr/bin/env python3
"""Week 4 LORO: Vision Transformer on 100 m tiles (10×10 AE cells = patches).

Each 100 m tile is a 10×10 grid of AlphaEarth 10 m cells. Missing cells are
masked. The ViT treats each occupied cell's 64-D embedding as a patch token
(no image-style patchify — AE already is the patch content).

Protocol: leave-one-metro-out, all GPUs in parallel.
Baselines in-run: mean-pool 64-D LR; flattened grid LR (reuses cached features).

Writes outputs/week4_vit_100m/ (checkpoints, OOF preds + CLS embeddings, plots).
Build notebook: scripts/build_week4_vit_100m_notebook.py

Usage:
  .venv/bin/python -u scripts/train_week4_vit_100m_loro.py
  .venv/bin/python -u scripts/train_week4_vit_100m_loro.py --epochs 40 --d-model 128
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
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Quiet noisy libs in workers + parent (nested-tensor, sklearn, etc.)
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TQDM_MININTERVAL", "1")

ROOT = Path(__file__).resolve().parents[1]
TILES = ROOT / "datasets" / "week4_100m_tiles_balanced" / "tiles_balanced_min5.parquet"
CACHE_SRC = ROOT / "outputs" / "week4_100m_tile_train" / "_cache"
PLS_CKPT = ROOT / "outputs" / "week4_pls_loro" / "checkpoints"
PLS_FINE = ROOT / "outputs" / "week4_pls_loro" / "fine_k_sweep.json"
OUT = ROOT / "outputs" / "week4_vit_100m"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"
LOG_DIR = OUT / "logs"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
GRID = 10
PATCH_DIM = 64
N_PATCHES = GRID * GRID  # 100


def min_pls_k() -> tuple[int, str]:
    """Smallest PLS-DA width that stays within 0.02 LORO F1 of 64-D LR (10 m sweep)."""
    if PLS_FINE.exists():
        spec = json.loads(PLS_FINE.read_text())
        crit = spec.get("min_k_criteria", {}).get("within_0.02_F1", {})
        if crit.get("k"):
            k = int(crit["k"])
            gap = crit.get("gap")
            why = f"within 0.02 F1 of 64-D LR (gap={gap})"
            return k, why
    return 9, "fallback k=9 (fine sweep missing)"


def _project_pls(
    patches: np.ndarray,
    mask: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    R: np.ndarray,
    device: torch.device,
    log_path: Path | None = None,
    chunk: int = 8000,
) -> np.ndarray:
    """patches (N,100,64) → scores (N,100,k) on the fold GPU. Empty cells stay 0."""
    n, n_pos, _ = patches.shape
    k = R.shape[1]
    out = np.empty((n, n_pos, k), dtype=np.float32)
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32).view(1, 1, -1)
    sd_t = torch.as_tensor(sd, device=device, dtype=torch.float32).view(1, 1, -1)
    R_t = torch.as_tensor(np.ascontiguousarray(R), device=device, dtype=torch.float32)
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        x = torch.as_tensor(np.asarray(patches[start:end]), device=device, dtype=torch.float32)
        m = torch.as_tensor(np.asarray(mask[start:end]), device=device, dtype=torch.float32)
        scores = ((x - mu_t) / sd_t) @ R_t
        scores = scores * m.unsqueeze(-1)
        out[start:end] = scores.detach().cpu().numpy()
        if log_path is not None:
            _fold_log(
                log_path,
                f"project {end:,}/{n:,} tiles ({100.0 * end / n:.0f}%) on {device} {time.time() - t0:.0f}s",
            )
    del mu_t, sd_t, R_t
    torch.cuda.empty_cache()
    return out


def _fold_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    with log_path.open("a") as f:
        f.write(f"{stamp} {msg}\n")
        f.flush()


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
# Data
# ---------------------------------------------------------------------------


def load_arrays(min_cells: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return patches [N,100,64], mask [N,100], y, regions, tile_ids."""
    tag = f"min{min_cells}"
    x_path = CACHE_SRC / f"X_grid_{tag}_n52492_seed42.npy"
    if min_cells == 20:
        x_path = CACHE_SRC / "X_grid_min20_n34494_seed42.npy"
    # resolve by glob if exact name differs
    if not x_path.exists():
        cands = sorted(CACHE_SRC.glob(f"X_grid_{tag}_n*_seed42.npy"))
        if not cands:
            raise FileNotFoundError(
                f"Missing grid cache under {CACHE_SRC}. "
                "Run scripts/train_week4_100m_tiles_loro.py --min-cells 5 first."
            )
        x_path = cands[0]

    y_path = CACHE_SRC / f"y_bin_{tag}.npy"
    r_path = CACHE_SRC / f"regions_{tag}.npy"
    id_cands = sorted(CACHE_SRC.glob(f"tile_ids_{tag}_n*_seed42.npy"))
    if not id_cands:
        raise FileNotFoundError(f"Missing tile_ids for {tag}")
    id_path = id_cands[0]

    print(f"Loading grid features {x_path} …", flush=True)
    X_flat = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    regions = np.load(r_path, allow_pickle=True)
    tile_ids = np.load(id_path, allow_pickle=True)
    assert len(X_flat) == len(y) == len(regions) == len(tile_ids)

    emb = np.asarray(X_flat[:, : N_PATCHES * PATCH_DIM], dtype=np.float32).reshape(
        -1, N_PATCHES, PATCH_DIM
    )
    mask = np.asarray(X_flat[:, N_PATCHES * PATCH_DIM :], dtype=np.float32)
    assert mask.shape[1] == N_PATCHES
    print(
        f"  n={len(y):,} patches={emb.shape} mask_mean={mask.mean():.3f} "
        f"cells/tile≈{mask.sum(1).mean():.1f}",
        flush=True,
    )
    return emb, mask, y.astype(np.int64), regions, tile_ids.astype(str)


class TilePatchDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        patches: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray | None = None,
        sd: np.ndarray | None = None,
    ):
        self.indices = indices.astype(np.int64)
        self.patches = patches
        self.mask = mask
        self.y = y
        self.mu = mu
        self.sd = sd

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.patches[idx].astype(np.float32, copy=True)  # (100, 64)
        m = self.mask[idx].astype(np.float32, copy=False)  # (100,)
        if self.mu is not None:
            x = (x - self.mu) / self.sd
            x = x * m[:, None]  # keep empties at 0 after norm
        return (
            torch.from_numpy(x),
            torch.from_numpy(m),
            torch.tensor(self.y[idx], dtype=torch.float32),
            idx,
        )


def _train_stats(patches: np.ndarray, mask: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over occupied train patches."""
    p = patches[tr]  # (n_tr, 100, 64)
    m = mask[tr] > 0.5
    # flatten occupied
    occ = p[m]  # (n_occ, 64)
    if len(occ) == 0:
        return np.zeros(64, np.float32), np.ones(64, np.float32)
    mu = occ.mean(axis=0).astype(np.float32)
    sd = occ.std(axis=0).astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    return mu, sd


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AEPatchViT(nn.Module):
    """ViT over a fixed 10×10 grid of AE patch tokens (+ CLS)."""

    def __init__(
        self,
        patch_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        grid: int = 10,
    ):
        super().__init__()
        self.grid = grid
        self.d_model = d_model
        n_patches = grid * grid
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.zeros_(self.head.bias)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, 100, 64), mask: (B, 100) with 1=present. Returns CLS (B, d)."""
        B = x.shape[0]
        tokens = self.patch_proj(x)  # (B, 100, d)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 101, d)
        tokens = tokens + self.pos_embed
        # key_padding_mask: True = ignore. CLS never ignored.
        pad = torch.zeros(B, 1, device=x.device, dtype=torch.bool)
        key_pad = torch.cat([pad, mask < 0.5], dim=1)  # (B, 101)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        return self.norm(h[:, 0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, mask)
        return self.head(z).squeeze(-1)  # (B,) logits


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------


def _fit_fold(payload: tuple) -> dict:
    (
        te_region,
        patches_path,
        mask_path,
        y_path,
        reg_path,
        cache_dir,
        ckpt_dir,
        device_id,
        epochs,
        batch_size,
        d_model,
        n_heads,
        n_layers,
        lr,
        weight_decay,
        dropout,
        val_frac,
        patience,
        num_workers,
        pls_k,
        log_dir,
    ) = payload

    warnings.filterwarnings("ignore")
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + abs(hash(te_region)) % 10_000)
    np.random.seed(SEED + abs(hash(te_region)) % 10_000)

    log_path = Path(log_dir) / f"{te_region}.log"
    if log_path.exists():
        log_path.unlink()
    _fold_log(log_path, f"start cuda:{device_id} held_out={te_region} pls_k={pls_k}")

    patches64 = np.load(patches_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    y = np.load(y_path)
    regions = np.load(reg_path, allow_pickle=True)

    ckpt_pls = joblib.load(PLS_CKPT / f"fold_{te_region}.joblib")
    mu_ae = np.asarray(ckpt_pls["mu"], dtype=np.float32).reshape(-1)
    sd_ae = np.asarray(ckpt_pls["sd"], dtype=np.float32).reshape(-1)
    R = np.asarray(ckpt_pls["pls_da"]["x_rotations"], dtype=np.float32)[:, : int(pls_k)]
    _fold_log(log_path, f"projecting patches with LORO PLS rotations {tuple(R.shape)}")
    patches = _project_pls(patches64, mask, mu_ae, sd_ae, R, device, log_path=log_path)
    mask = np.asarray(mask)

    te = np.where(regions == te_region)[0]
    tr_all = np.where(regions != te_region)[0]
    # split train metros into train/val for early stopping (by tile, stratified-ish)
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_all)
    n_val = max(1, int(len(tr_all) * val_frac))
    va = tr_all[:n_val]
    tr = tr_all[n_val:]

    mu, sd = _train_stats(np.asarray(patches), np.asarray(mask), tr)

    train_loader = DataLoader(
        TilePatchDataset(tr, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TilePatchDataset(va, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TilePatchDataset(te, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = AEPatchViT(
        patch_dim=int(pls_k),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        grid=GRID,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # cosine over epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_state = None
    best_val = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(epochs):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xb, mb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            n_seen += len(yb)
        sched.step()

        # val F1
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, mb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                logits = model(xb, mb)
                p = torch.sigmoid(logits).cpu().numpy()
                preds.append((p >= 0.5).astype(np.int64))
                gts.append(yb.numpy().astype(np.int64))
        yhat = np.concatenate(preds)
        ytrue = np.concatenate(gts)
        val_f1 = float(f1_score(ytrue, yhat, average="macro", zero_division=0))
        tr_loss = loss_sum / max(n_seen, 1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_f1": val_f1})
        improved = val_f1 > best_val + 1e-4
        _fold_log(
            log_path,
            f"epoch {epoch + 1}/{epochs} loss={tr_loss:.4f} val_f1={val_f1:.4f} "
            f"best={max(best_val, val_f1):.4f} bad={0 if improved else bad + 1}/{patience}",
        )

        if val_f1 > best_val + 1e-4:
            best_val = val_f1
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # OOF preds + CLS embeddings on held-out metro
    n_te = len(te)
    y_pred = np.zeros(n_te, dtype=np.int64)
    p_res = np.zeros(n_te, dtype=np.float64)
    cls_emb = np.zeros((n_te, d_model), dtype=np.float32)
    # map global idx → position in te array
    te_pos = {int(g): i for i, g in enumerate(te.tolist())}

    with torch.no_grad():
        for xb, mb, yb, idx in test_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            z = model.encode(xb, mb)
            logits = model.head(z).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            z_np = z.cpu().numpy()
            idx_np = idx.numpy()
            for j, gidx in enumerate(idx_np):
                pos = te_pos[int(gidx)]
                p_res[pos] = p[j]
                y_pred[pos] = int(p[j] >= 0.5)
                cls_emb[pos] = z_np[j]

    yb_te = y[te]
    macro_f1 = float(f1_score(yb_te, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(yb_te, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(yb_te, y_pred, average="macro", zero_division=0))
    _fold_log(
        log_path,
        f"done test_f1={macro_f1:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"best_epoch={best_epoch} epochs_ran={len(history)} n_test={len(te)}",
    )

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    np.save(cache / f"oof_idx_{te_region}.npy", te.astype(np.int64))
    np.save(cache / f"oof_ypred_{te_region}.npy", y_pred)
    np.save(cache / f"oof_pres_{te_region}.npy", p_res)
    np.save(cache / f"oof_cls_{te_region}.npy", cls_emb)

    torch.save(
        {
            "region_held_out": te_region,
            "state_dict": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "arch": {
                "name": "AEPatchViT",
                "patch_dim": int(pls_k),
                "patch_features": "pls_da_scores",
                "pls_k": int(pls_k),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "grid": GRID,
                "dropout": dropout,
            },
            "hyperparams": {
                "epochs_ran": len(history),
                "best_epoch": best_epoch,
                "best_val_f1": best_val,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "patience": patience,
            },
            "seed": SEED,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "history": history,
        },
        ckpt / f"fold_{te_region}.pt",
    )

    del model
    torch.cuda.empty_cache()

    return {
        "region": te_region,
        "device": f"cuda:{device_id}",
        "macro_f1": macro_f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "n_test": int(len(te)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
    }


def _mean_pool_lr_baseline(
    patches: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    regions: np.ndarray,
    device_ids: list[int],
) -> pd.DataFrame:
    """Quick LORO mean-pool 64-D LR on GPU (closed-form-ish LBFGS)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed as ac

    def _one(payload):
        te_region, p_path, m_path, y_path, r_path, device_id = payload
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        P = np.load(p_path, mmap_mode="r")
        M = np.load(m_path, mmap_mode="r")
        yb = np.load(y_path)
        regs = np.load(r_path, allow_pickle=True)
        te = regs == te_region
        tr = ~te
        # mean pool
        m_tr = M[tr][:, :, None]
        m_te = M[te][:, :, None]
        sum_tr = (P[tr] * m_tr).sum(axis=1)
        cnt_tr = m_tr.sum(axis=1).clip(min=1.0)
        X_tr = sum_tr / cnt_tr
        sum_te = (P[te] * m_te).sum(axis=1)
        cnt_te = m_te.sum(axis=1).clip(min=1.0)
        X_te = sum_te / cnt_te
        X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
        X_te_t = torch.as_tensor(X_te, dtype=torch.float32, device=device)
        y_tr = torch.as_tensor(yb[tr], dtype=torch.float32, device=device)
        mu = X_tr_t.mean(0, keepdim=True)
        sd = X_tr_t.std(0, keepdim=True).clamp_min(1e-6)
        X_tr_s = (X_tr_t - mu) / sd
        X_te_s = (X_te_t - mu) / sd
        w = torch.zeros(64, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(X_tr_s @ w + b, y_tr)
            loss = loss + 0.5 * 1e-2 * (w * w).sum() / X_tr_s.shape[0]
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            p = torch.sigmoid(X_te_s @ w + b).cpu().numpy()
        yhat = (p >= 0.5).astype(np.int64)
        f1 = float(f1_score(yb[te], yhat, average="macro", zero_division=0))
        return {"region": te_region, "model": "mean_pool_lr", "macro_f1": f1, "n_test": int(te.sum())}

    # persist arrays for workers
    CACHE.mkdir(parents=True, exist_ok=True)
    p_path = str(CACHE / "patches.npy")
    m_path = str(CACHE / "mask.npy")
    y_path = str(CACHE / "y.npy")
    r_path = str(CACHE / "regions.npy")
    # already saved in main; reuse

    region_list = sorted(set(regions.tolist()))
    jobs = []
    for i, r in enumerate(region_list):
        jobs.append((r, p_path, m_path, y_path, r_path, device_ids[i % len(device_ids)]))
    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(device_ids), mp_context=ctx) as pool:
        futs = [pool.submit(_one, j) for j in jobs]
        for fut in tqdm(ac(futs), total=len(futs), desc="mean-pool LR"):
            rows.append(fut.result())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _claim_header(fig, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(0.01, 0.98, title, fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure)
    fig.text(0.01, 0.925, subtitle, fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure)


def make_plots(metrics: pd.DataFrame, oof: pd.DataFrame, summary: dict, baselines: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # per-metro F1
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    m = metrics.sort_values("macro_f1")
    y = np.arange(len(m))
    ax.barh(y, m["macro_f1"], color="#0072B2")
    for yi, v in zip(y, m["macro_f1"]):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("_", " ") for r in m["region"]])
    ax.set_xlabel("macro-F1")
    ax.set_xlim(0, 1.05)
    ax.axvline(float(summary["loro_macro_f1_mean"]), color="#333", ls="--", lw=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    _claim_header(
        fig,
        f"ViT LORO mean F1 = {summary['loro_macro_f1_mean']:.3f} (±{summary['loro_macro_f1_std']:.3f})",
        "Held-out metro. Dashed = mean across folds. 10×10 AE patches, empty cells masked.",
    )
    fig.savefig(FIG / "vit_f1_by_metro.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # comparison bars
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    labels, vals, colors = [], [], []
    order = [
        ("vit", "ViT (10×10 patches)", "#0072B2"),
        ("mean_pool_lr", "Mean-pool 64-D LR", "#009E73"),
        ("flat_grid_lr", "Flat 6500-D LR", "#D55E00"),
        ("pls_mean_pool", "PLS-DA on mean-pool", "#CC79A7"),
    ]
    for key, lab, c in order:
        if key == "vit":
            v = summary["loro_macro_f1_mean"]
        else:
            v = baselines.get(key)
        if v is None:
            continue
        labels.append(lab)
        vals.append(float(v))
        colors.append(c)
    order_idx = np.argsort(vals)[::-1]
    labels = [labels[i] for i in order_idx]
    vals = [vals[i] for i in order_idx]
    colors = [colors[i] for i in order_idx]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LORO mean macro-F1")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    best_lab = labels[0]
    _claim_header(
        fig,
        f"{best_lab} leads among 100 m tile heads",
        "Same balanced min5 tiles + LORO metros. Flat LR = prior week-4 grid logistic.",
    )
    fig.savefig(FIG / "vit_vs_baselines.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # confusion
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        oof["y_true"],
        oof["y_pred"],
        display_labels=["NRES", "RES"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("")
    _claim_header(
        fig,
        f"OOF macro-F1 = {summary['oof_macro_f1']:.3f}",
        "Pooled leave-one-metro-out predictions from ViT.",
    )
    fig.savefig(FIG / "vit_oof_confusion.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cells", type=int, default=5, choices=[5, 20])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None, help="Max parallel fold GPUs")
    ap.add_argument("--loader-workers", type=int, default=2)
    ap.add_argument(
        "--pls-k",
        type=int,
        default=None,
        help="PLS scores per patch (default: min k within 0.02 F1 of 64-D LR)",
    )
    ap.add_argument("--skip-mean-pool-baseline", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pls_k, pls_why = (args.pls_k, "cli") if args.pls_k else min_pls_k()
    print(f"PLS patch width k={pls_k} ({pls_why})", flush=True)

    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count() or 1
    print(f"CPUs={n_cpu}  GPUs={n_gpu}", flush=True)
    if n_gpu == 0:
        raise SystemExit("Need CUDA GPUs for ViT LORO.")
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)

    patches, mask, y, regions, tile_ids = load_arrays(args.min_cells)
    # materialize for workers (mmap path)
    p_path = CACHE / "patches.npy"
    m_path = CACHE / "mask.npy"
    y_path = CACHE / "y.npy"
    r_path = CACHE / "regions.npy"
    id_path = CACHE / "tile_ids.npy"
    np.save(p_path, np.asarray(patches, dtype=np.float32))
    np.save(m_path, np.asarray(mask, dtype=np.float32))
    np.save(y_path, y)
    np.save(r_path, regions)
    np.save(id_path, tile_ids)

    region_list = sorted(set(regions.tolist()))
    max_workers = args.workers or min(len(region_list), n_gpu)
    print(
        f"ViT LORO · n={len(y):,} folds={len(region_list)} workers={max_workers} "
        f"d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
        f"pls_k={pls_k} epochs≤{args.epochs}",
        flush=True,
    )

    jobs = []
    for i, r in enumerate(region_list):
        jobs.append(
            (
                r,
                str(p_path),
                str(m_path),
                str(y_path),
                str(r_path),
                str(CACHE),
                str(CKPT),
                i % n_gpu,
                args.epochs,
                args.batch_size,
                args.d_model,
                args.n_heads,
                args.n_layers,
                args.lr,
                args.weight_decay,
                args.dropout,
                args.val_frac,
                args.patience,
                args.loader_workers,
                int(pls_k),
                str(LOG_DIR),
            )
        )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit_fold, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="ViT folds"):
            res = fut.result()
            rows.append(res)
            print(
                f"  {res['region']} F1={res['macro_f1']:.4f} "
                f"best_ep={res['best_epoch']} on {res['device']}",
                flush=True,
            )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)

    # assemble OOF
    n = len(y)
    oof = pd.DataFrame(
        {
            "tile_id": tile_ids,
            "region": regions,
            "y_true": y,
            "y_pred": np.full(n, -1, dtype=np.int64),
            "p_res": np.full(n, np.nan),
        }
    )
    cls_oof = np.full((n, args.d_model), np.nan, dtype=np.float32)
    for r in region_list:
        idx = np.load(CACHE / f"oof_idx_{r}.npy")
        oof.loc[idx, "y_pred"] = np.load(CACHE / f"oof_ypred_{r}.npy")
        oof.loc[idx, "p_res"] = np.load(CACHE / f"oof_pres_{r}.npy")
        cls_oof[idx] = np.load(CACHE / f"oof_cls_{r}.npy")
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    np.save(OUT / "oof_cls_embeddings.npy", cls_oof)

    oof_f1 = float(f1_score(oof["y_true"], oof["y_pred"], average="macro", zero_division=0))
    loro_mean = float(metrics["macro_f1"].mean())
    loro_std = float(metrics["macro_f1"].std(ddof=0))

    # baselines
    baselines = {}
    flat_sum = ROOT / "outputs" / "week4_100m_tile_train" / "summary_min5.json"
    if flat_sum.exists():
        baselines["flat_grid_lr"] = json.loads(flat_sum.read_text()).get("loro_macro_f1_mean")
    pls_sum = ROOT / "outputs" / "week4_pls_10m_vs_100m" / "100m" / "summary.json"
    if pls_sum.exists():
        baselines["pls_mean_pool"] = (
            json.loads(pls_sum.read_text()).get("pls_da_best") or {}
        ).get("loro_macro_f1_mean")
        baselines["mean_pool_lr"] = (
            json.loads(pls_sum.read_text()).get("baseline_lr") or {}
        ).get("loro_macro_f1_mean")

    if not args.skip_mean_pool_baseline and baselines.get("mean_pool_lr") is None:
        print("Running mean-pool LR baseline …", flush=True)
        mp_df = _mean_pool_lr_baseline(
            patches, mask, y, regions, list(range(n_gpu))
        )
        mp_df.to_csv(OUT / "baseline_mean_pool_lr.csv", index=False)
        baselines["mean_pool_lr"] = float(mp_df["macro_f1"].mean())

    summary = {
        "name": "week4_vit_100m_loro_v1",
        "input": f"10x10 AE cells projected to {pls_k} LORO PLS-DA scores; empty masked",
        "pls_k": int(pls_k),
        "pls_k_reason": pls_why,
        "min_cells": args.min_cells,
        "n_tiles": int(len(y)),
        "n_res": int((y == 1).sum()),
        "n_nres": int((y == 0).sum()),
        "arch": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "patch_dim": int(pls_k),
            "grid": GRID,
            "dropout": args.dropout,
        },
        "train": {
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "patience": args.patience,
            "optimizer": "AdamW+cosine",
        },
        "protocol": "leave-one-region-out",
        "seed": SEED,
        "n_gpus": n_gpu,
        "workers": max_workers,
        "loro_macro_f1_mean": loro_mean,
        "loro_macro_f1_std": loro_std,
        "oof_macro_f1": oof_f1,
        "baselines": baselines,
        "delta_vs_mean_pool_lr": (
            None
            if baselines.get("mean_pool_lr") is None
            else float(loro_mean - baselines["mean_pool_lr"])
        ),
        "delta_vs_flat_grid_lr": (
            None
            if baselines.get("flat_grid_lr") is None
            else float(loro_mean - baselines["flat_grid_lr"])
        ),
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "oof": "outputs/week4_vit_100m/oof_predictions.parquet",
            "cls_embeddings": "outputs/week4_vit_100m/oof_cls_embeddings.npy",
            "metrics": "outputs/week4_vit_100m/loro_metrics.csv",
            "checkpoints": "outputs/week4_vit_100m/checkpoints",
            "figures": "outputs/week4_vit_100m/figures",
            "logs": "outputs/week4_vit_100m/logs",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    make_plots(metrics, oof, summary, baselines)

    print(json.dumps(_jsonable({
        "loro_macro_f1_mean": loro_mean,
        "oof_macro_f1": oof_f1,
        "baselines": baselines,
        "elapsed_s": summary["elapsed_s"],
    }), indent=2), flush=True)
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
