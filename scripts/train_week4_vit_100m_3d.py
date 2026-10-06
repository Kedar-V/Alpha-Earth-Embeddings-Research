#!/usr/bin/env python3
"""Week 4 LORO: ViT on 100 m tiles with a 3D class-token bottleneck.

The transformer stays wide (d_model=128) so attention can mix the 9 PLS scores.
The class token is then mapped to 3D. That 3D vector is the encoder output:

  loss = BCE(tile label) + λ * supervised contrastive(L2-normalized 3D)

Same-class tiles are pulled together; RES and NRES are pushed apart.
Empty cells stay masked. PLS width is the minimum k within 0.02 F1 of 64-D LR.

Writes outputs/week4_vit_100m_3d/ (per-fold checkpoints, OOF 3D embeddings,
epoch logs, figures). Safe to relaunch: finished folds are skipped if the
checkpoint and OOF arrays exist, unless --force.

Usage:
  .venv/bin/python -u scripts/train_week4_vit_100m_3d.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import json
import multiprocessing as mp
import sys
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
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_week4_vit_100m_loro import (  # noqa: E402
    GRID,
    PLS_CKPT,
    SEED,
    TilePatchDataset,
    _fold_log,
    _project_pls,
    _train_stats,
    load_arrays,
    min_pls_k,
)

OUT = ROOT / "outputs" / "week4_vit_100m_3d"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"
LOG_DIR = OUT / "logs"

EMB_DIM = 3


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


def supervised_contrastive(z: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Pull same-class 3D vectors together, push the other class away."""
    u = F.normalize(z, dim=1)
    logits = u @ u.T / tau
    b = z.shape[0]
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    labels = y.view(-1).long()
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    mean_log_prob = (log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    return -mean_log_prob[valid].mean()


class AEPatchViT3D(nn.Module):
    """Wide ViT over 10×10 PLS patches; class token compressed to 3D."""

    def __init__(
        self,
        patch_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        emb_dim: int = EMB_DIM,
    ):
        super().__init__()
        n_patches = GRID * GRID
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.to_3d = nn.Linear(d_model, emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.xavier_uniform_(self.to_3d.weight)
        nn.init.zeros_(self.to_3d.bias)
        nn.init.zeros_(self.head.bias)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.patch_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        pad = torch.zeros(b, 1, device=x.device, dtype=torch.bool)
        key_pad = torch.cat([pad, mask < 0.5], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        return self.to_3d(self.norm(h[:, 0]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        return self.head(z).squeeze(-1), z


def _cluster_stats(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Centroid separation and mean within-class radius in the 3D space."""
    if z.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    cents = []
    within = []
    for c in (0, 1):
        pts = z[y == c]
        if len(pts) == 0:
            return float("nan"), float("nan")
        mu = pts.mean(axis=0)
        cents.append(mu)
        within.append(float(np.linalg.norm(pts - mu, axis=1).mean()))
    sep = float(np.linalg.norm(cents[1] - cents[0]))
    return sep, float(np.mean(within))


def _fit_fold(payload: tuple) -> dict:
    (
        te_region,
        patches_path,
        mask_path,
        y_path,
        reg_path,
        cache_dir,
        ckpt_dir,
        log_dir,
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
        supcon_lambda,
        tau,
        force,
    ) = payload

    warnings.filterwarnings("ignore")
    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    log_path = Path(log_dir) / f"{te_region}.log"
    ckpt_path = ckpt / f"fold_{te_region}.pt"
    done_marker = cache / f"done_{te_region}.json"
    if done_marker.exists() and ckpt_path.exists() and not force:
        rec = json.loads(done_marker.read_text())
        rec["skipped"] = True
        _fold_log(log_path, f"skip existing checkpoint test_f1={rec.get('macro_f1')}")
        return rec

    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + abs(hash(te_region)) % 10_000)
    np.random.seed(SEED + abs(hash(te_region)) % 10_000)
    if log_path.exists():
        log_path.unlink()
    _fold_log(
        log_path,
        f"start cuda:{device_id} held_out={te_region} pls_k={pls_k} "
        f"emb={EMB_DIM} lambda={supcon_lambda} tau={tau}",
    )

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
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_all)
    n_val = max(1, int(len(tr_all) * val_frac))
    va = tr_all[:n_val]
    tr = tr_all[n_val:]
    mu, sd = _train_stats(patches, mask, tr)

    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        TilePatchDataset(tr, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        TilePatchDataset(va, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        TilePatchDataset(te, patches, mask, y, mu, sd),
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )

    model = AEPatchViT3D(
        patch_dim=int(pls_k),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_state = None
    best_val = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(epochs):
        model.train()
        bce_sum = sup_sum = n_seen = 0.0
        n_sup = 0
        for xb, mb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, z = model(xb, mb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            sup = supervised_contrastive(z, yb, tau)
            loss = bce + supcon_lambda * sup
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bce_sum += float(bce.item()) * len(yb)
            sup_sum += float(sup.item()) * len(yb)
            n_seen += len(yb)
            n_sup += len(yb)
        sched.step()

        model.eval()
        preds, gts, zs = [], [], []
        with torch.no_grad():
            for xb, mb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                logits, z = model(xb, mb)
                p = torch.sigmoid(logits).cpu().numpy()
                preds.append((p >= 0.5).astype(np.int64))
                gts.append(yb.numpy().astype(np.int64))
                zs.append(z.cpu().numpy())
        yhat = np.concatenate(preds)
        ytrue = np.concatenate(gts)
        zval = np.concatenate(zs)
        val_f1 = float(f1_score(ytrue, yhat, average="macro", zero_division=0))
        sep, within = _cluster_stats(zval, ytrue)
        tr_bce = bce_sum / max(n_seen, 1)
        tr_sup = sup_sum / max(n_sup, 1)
        history.append(
            {
                "epoch": epoch,
                "train_bce": tr_bce,
                "train_supcon": tr_sup,
                "val_f1": val_f1,
                "val_centroid_sep": sep,
                "val_within": within,
            }
        )
        improved = val_f1 > best_val + 1e-4
        _fold_log(
            log_path,
            f"epoch {epoch + 1}/{epochs} bce={tr_bce:.4f} supcon={tr_sup:.4f} "
            f"val_f1={val_f1:.4f} sep={sep:.3f} within={within:.3f} "
            f"best={max(best_val, val_f1):.4f} bad={0 if improved else bad + 1}/{patience}",
        )
        if improved:
            best_val = val_f1
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                _fold_log(log_path, f"early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    n_te = len(te)
    y_pred = np.zeros(n_te, dtype=np.int64)
    p_res = np.zeros(n_te, dtype=np.float64)
    z_te = np.zeros((n_te, EMB_DIM), dtype=np.float32)
    te_pos = {int(g): i for i, g in enumerate(te.tolist())}
    with torch.no_grad():
        for xb, mb, yb, idx in test_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            logits, z = model(xb, mb)
            p = torch.sigmoid(logits).cpu().numpy()
            z_np = z.cpu().numpy()
            for j, gidx in enumerate(idx.numpy()):
                pos = te_pos[int(gidx)]
                p_res[pos] = p[j]
                y_pred[pos] = int(p[j] >= 0.5)
                z_te[pos] = z_np[j]

    yb_te = y[te]
    macro_f1 = float(f1_score(yb_te, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(yb_te, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(yb_te, y_pred, average="macro", zero_division=0))
    sep_te, within_te = _cluster_stats(z_te, yb_te)
    head_w = model.head.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)

    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"oof_idx_{te_region}.npy", te.astype(np.int64))
    np.save(cache / f"oof_ypred_{te_region}.npy", y_pred)
    np.save(cache / f"oof_pres_{te_region}.npy", p_res)
    np.save(cache / f"oof_z3_{te_region}.npy", z_te)
    np.save(cache / f"head_w_{te_region}.npy", head_w)

    torch.save(
        {
            "region_held_out": te_region,
            "state_dict": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "mu_ae": mu_ae,
            "sd_ae": sd_ae,
            "pls_rotation": R,
            "head_w": head_w,
            "arch": {
                "name": "AEPatchViT3D",
                "patch_dim": int(pls_k),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "emb_dim": EMB_DIM,
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
                "supcon_lambda": supcon_lambda,
                "tau": tau,
                "pls_k": int(pls_k),
            },
            "seed": SEED,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "history": history,
        },
        ckpt_path,
    )

    result = {
        "region": te_region,
        "device": f"cuda:{device_id}",
        "macro_f1": macro_f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "test_centroid_sep": sep_te,
        "test_within": within_te,
        "n_test": int(len(te)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "skipped": False,
    }
    done_marker.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    _fold_log(
        log_path,
        f"done test_f1={macro_f1:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"sep={sep_te:.3f} within={within_te:.3f} "
        f"best_epoch={best_epoch} epochs_ran={len(history)} n_test={len(te)}",
    )
    del model, patches
    torch.cuda.empty_cache()
    return result


def _plot_3d_space(oof_z: np.ndarray, y: np.ndarray, regions: np.ndarray, head_ws: dict[str, np.ndarray]) -> None:
    """Gap bars, classifier densities, and one 3D scatter per metro."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_week4_vit_100m_3d_notebook import write_figures

    metrics = pd.read_csv(OUT / "loro_metrics.csv")
    metrics["ratio"] = metrics["test_centroid_sep"] / metrics["test_within"]
    write_figures(oof_z, y.astype(int), regions, head_ws, metrics)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument("--supcon-lambda", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--pls-k", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--loader-workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    pls_k, pls_why = (args.pls_k, "cli") if args.pls_k else min_pls_k()
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count() or 1
    if n_gpu == 0:
        raise SystemExit("Need CUDA GPUs.")
    print(f"CPUs={n_cpu}  GPUs={n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)
    print(f"PLS patch width k={pls_k} ({pls_why})  emb={EMB_DIM}  λ={args.supcon_lambda}", flush=True)

    patches, mask, y, regions, tile_ids = load_arrays(5)
    np.save(CACHE / "patches.npy", np.asarray(patches, dtype=np.float32))
    np.save(CACHE / "mask.npy", np.asarray(mask, dtype=np.float32))
    np.save(CACHE / "y.npy", y)
    np.save(CACHE / "regions.npy", regions)
    np.save(CACHE / "tile_ids.npy", tile_ids.astype(str))

    region_list = sorted(set(regions.tolist()))
    max_workers = min(len(region_list), n_gpu)
    loader_workers = args.loader_workers
    if loader_workers is None:
        loader_workers = max(2, (n_cpu // max_workers) - 1)
    print(
        f"ViT-3D LORO · n={len(y):,} folds={len(region_list)} gpus={max_workers} "
        f"loader_workers/fold={loader_workers} epochs≤{args.epochs}",
        flush=True,
    )

    jobs = []
    for i, region in enumerate(region_list):
        jobs.append(
            (
                region,
                str(CACHE / "patches.npy"),
                str(CACHE / "mask.npy"),
                str(CACHE / "y.npy"),
                str(CACHE / "regions.npy"),
                str(CACHE),
                str(CKPT),
                str(LOG_DIR),
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
                loader_workers,
                int(pls_k),
                args.supcon_lambda,
                args.tau,
                args.force,
            )
        )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit_fold, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="ViT-3D folds"):
            res = fut.result()
            rows.append(res)
            print(
                f"  {res['region']} F1={res['macro_f1']:.4f} sep={res['test_centroid_sep']:.3f} "
                f"ep={res['best_epoch']} skip={res.get('skipped')}",
                flush=True,
            )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)

    n = len(y)
    oof = pd.DataFrame(
        {
            "tile_id": tile_ids.astype(str),
            "region": regions,
            "y_true": y,
            "y_pred": np.full(n, -1, dtype=np.int64),
            "p_res": np.full(n, np.nan),
        }
    )
    z_oof = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    head_ws = {}
    for region in region_list:
        idx = np.load(CACHE / f"oof_idx_{region}.npy")
        oof.loc[idx, "y_pred"] = np.load(CACHE / f"oof_ypred_{region}.npy")
        oof.loc[idx, "p_res"] = np.load(CACHE / f"oof_pres_{region}.npy")
        z_oof[idx] = np.load(CACHE / f"oof_z3_{region}.npy")
        head_ws[region] = np.load(CACHE / f"head_w_{region}.npy")
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    np.save(OUT / "oof_z3.npy", z_oof)
    np.savez(OUT / "head_w_by_region.npz", **head_ws)

    oof_f1 = float(f1_score(oof["y_true"], oof["y_pred"], average="macro", zero_division=0))
    loro_mean = float(metrics["macro_f1"].mean())
    loro_std = float(metrics["macro_f1"].std(ddof=0))
    sep_mean = float(metrics["test_centroid_sep"].mean())
    within_mean = float(metrics["test_within"].mean())

    _plot_3d_space(z_oof, y, regions, head_ws)

    summary = {
        "name": "week4_vit_100m_3d_v1",
        "input": f"10x10 AE cells → {pls_k} LORO PLS scores; empty masked",
        "encoder_output": "3D class token",
        "loss": "BCE + supcon_lambda * supervised contrastive on L2-normalized 3D",
        "pls_k": int(pls_k),
        "pls_k_reason": pls_why,
        "supcon_lambda": args.supcon_lambda,
        "tau": args.tau,
        "n_tiles": int(len(y)),
        "arch": {
            "d_model": args.d_model,
            "emb_dim": EMB_DIM,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        },
        "protocol": "leave-one-region-out",
        "seed": SEED,
        "n_gpus": n_gpu,
        "n_cpus": n_cpu,
        "loader_workers_per_fold": loader_workers,
        "loro_macro_f1_mean": loro_mean,
        "loro_macro_f1_std": loro_std,
        "oof_macro_f1": oof_f1,
        "mean_test_centroid_sep": sep_mean,
        "mean_test_within": within_mean,
        "elapsed_s": round(time.time() - t0, 1),
        "reload": {
            "checkpoint": "outputs/week4_vit_100m_3d/checkpoints/fold_<metro>.pt",
            "class": "scripts.train_week4_vit_100m_3d.AEPatchViT3D",
            "oof_z3": "outputs/week4_vit_100m_3d/oof_z3.npy",
            "note": "Each fold has its own 3D coordinates. Do not pool raw z3 across metros.",
        },
        "paths": {
            "oof": "outputs/week4_vit_100m_3d/oof_predictions.parquet",
            "z3": "outputs/week4_vit_100m_3d/oof_z3.npy",
            "metrics": "outputs/week4_vit_100m_3d/loro_metrics.csv",
            "checkpoints": "outputs/week4_vit_100m_3d/checkpoints",
            "logs": "outputs/week4_vit_100m_3d/logs",
            "figures": "outputs/week4_vit_100m_3d/figures",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(
        json.dumps(
            _jsonable(
                {
                    "loro_macro_f1_mean": loro_mean,
                    "oof_macro_f1": oof_f1,
                    "mean_sep": sep_mean,
                    "mean_within": within_mean,
                    "elapsed_s": summary["elapsed_s"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
