#!/usr/bin/env python3
"""Week 4 LORO v3: same contrastive head as v2, longer floor, stop on validation loss.

v1 (outputs/week4_vit_100m_3d) added SupCon at λ=1, τ=0.1 on top of BCE.
The classifier gradient reached the 3D vector, so the fit drew one curve
instead of two class poles. This run changes four things:

  1. Stop the classifier gradient at the 3D vector. SupCon trains the encoder.
     BCE trains only the head.
  2. Cosine head. L2-normalize the 3D vector and the weight. No bias.
     A learnable scale (init 10) sets the logit range.
  3. Weight the contrastive term by λ * τ, with λ=10. Khosla rescales by τ
     because the SupCon gradient shrinks as 1/τ.
  4. Anneal τ linearly from 0.5 to 0.1 across the epoch budget.

  loss = BCE(scale * cos(z_hat.detach(), w_hat), y) + λ * τ * SupCon(z, y)

v2 stopped on validation F1 with patience 6, so a fold whose probabilities
had not yet crossed 0.5 was halted while BCE was still falling. This run
always trains 35 epochs. After that it stops when validation loss
(BCE + λ τ SupCon) has not improved for 6 epochs. The cap is 60.

Writes outputs/week4_vit_100m_3d_v3/. Finished folds are skipped unless --force.

Usage:
  PYTHONWARNINGS=ignore setsid nohup .venv/bin/python -u scripts/train_week4_vit_100m_3d_v3.py \\
    > outputs/week4_vit_100m_3d_v3/train.log 2>&1 < /dev/null &
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

from scripts.train_week4_vit_100m_3d import (  # noqa: E402
    EMB_DIM,
    _cluster_stats,
    _jsonable,
    supervised_contrastive,
)
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

OUT = ROOT / "outputs" / "week4_vit_100m_3d_v3"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"
LOG_DIR = OUT / "logs"

TAU_START = 0.5
TAU_END = 0.1
SUPCON_LAMBDA = 10.0
LOGIT_SCALE_INIT = 10.0
MIN_EPOCHS = 35
MAX_EPOCHS = 60


def _tau_at(epoch: int, epochs: int, tau_start: float, tau_end: float) -> float:
    if epochs <= 1:
        return float(tau_end)
    t = epoch / (epochs - 1)
    return float(tau_start + t * (tau_end - tau_start))


class AEPatchViT3DCosine(nn.Module):
    """Wide ViT. Class token is 3D. The head reads the unit vector and does not train the encoder."""

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
        self.head = nn.Linear(emb_dim, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(LOGIT_SCALE_INIT)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.xavier_uniform_(self.to_3d.weight)
        nn.init.zeros_(self.to_3d.bias)
        nn.init.xavier_uniform_(self.head.weight)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.patch_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        pad = torch.zeros(b, 1, device=x.device, dtype=torch.bool)
        key_pad = torch.cat([pad, mask < 0.5], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        return self.to_3d(self.norm(h[:, 0]))

    def cosine_logits(self, u: torch.Tensor) -> torch.Tensor:
        """u is L2-normalized. Gradient stops at u when the caller detaches it."""
        w = F.normalize(self.head.weight, dim=-1)
        return (self.scale * (u @ w.T)).squeeze(-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        u = F.normalize(z, dim=-1)
        return self.cosine_logits(u.detach()), z


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
        min_epochs,
        num_workers,
        pls_k,
        supcon_lambda,
        tau_start,
        tau_end,
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
    _fold_log(log_path, f"v3 cuda:{device_id} held_out={te_region} pls_k={pls_k}")
    _fold_log(log_path, "change 1: BCE does not backprop into the 3D vector")
    _fold_log(log_path, "change 2: cosine head, no bias, learnable scale init 10")
    _fold_log(
        log_path,
        f"change 3: loss = BCE + λ*τ*SupCon  λ={supcon_lambda}",
    )
    _fold_log(log_path, f"change 4: τ anneals {tau_start} → {tau_end} over {epochs} epochs")
    _fold_log(
        log_path,
        f"stop: train at least {min_epochs} epochs, then stop if val loss "
        f"(BCE + λτ SupCon) does not improve for {patience} epochs",
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

    model = AEPatchViT3DCosine(
        patch_dim=int(pls_k),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_state = None
    best_loss = float("inf")
    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(epochs):
        tau = _tau_at(epoch, epochs, tau_start, tau_end)
        model.train()
        bce_sum = sup_sum = wsup_sum = n_seen = 0.0
        for xb, mb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, z = model(xb, mb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            sup = supervised_contrastive(z, yb, tau)
            weighted = supcon_lambda * tau * sup
            loss = bce + weighted
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = len(yb)
            bce_sum += float(bce.item()) * bs
            sup_sum += float(sup.item()) * bs
            wsup_sum += float(weighted.item()) * bs
            n_seen += bs
        sched.step()

        model.eval()
        preds, gts, zs = [], [], []
        vb = vs = vn = 0.0
        with torch.no_grad():
            for xb, mb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                yb_d = yb.to(device, non_blocking=True)
                logits, z = model(xb, mb)
                bce_v = F.binary_cross_entropy_with_logits(logits, yb_d)
                sup_v = supervised_contrastive(z, yb_d, tau)
                vb += float(bce_v.item()) * len(yb)
                vs += float(sup_v.item()) * len(yb)
                vn += len(yb)
                p = torch.sigmoid(logits).cpu().numpy()
                preds.append((p >= 0.5).astype(np.int64))
                gts.append(yb.numpy().astype(np.int64))
                zs.append(z.cpu().numpy())
        yhat = np.concatenate(preds)
        ytrue = np.concatenate(gts)
        zval = np.concatenate(zs)
        val_f1 = float(f1_score(ytrue, yhat, average="macro", zero_division=0))
        val_bce = vb / max(vn, 1)
        val_sup = vs / max(vn, 1)
        val_loss = val_bce + supcon_lambda * tau * val_sup
        sep, within = _cluster_stats(zval, ytrue)
        z_unit = zval / np.clip(np.linalg.norm(zval, axis=1, keepdims=True), 1e-8, None)
        nsep, nwithin = _cluster_stats(z_unit, ytrue)
        tr_bce = bce_sum / max(n_seen, 1)
        tr_sup = sup_sum / max(n_seen, 1)
        tr_w = wsup_sum / max(n_seen, 1)
        scale = float(model.scale.detach().cpu())
        history.append(
            {
                "epoch": epoch,
                "tau": tau,
                "train_bce": tr_bce,
                "train_supcon": tr_sup,
                "train_weighted_supcon": tr_w,
                "val_bce": val_bce,
                "val_supcon": val_sup,
                "val_loss": val_loss,
                "logit_scale": scale,
                "val_f1": val_f1,
                "val_centroid_sep": sep,
                "val_within": within,
                "val_unit_sep": nsep,
                "val_unit_within": nwithin,
            }
        )
        improved = val_loss < best_loss - 1e-4
        _fold_log(
            log_path,
            f"epoch {epoch + 1}/{epochs} tau={tau:.3f} bce={tr_bce:.4f} "
            f"supcon={tr_sup:.4f} wsup={tr_w:.3f} scale={scale:.2f} "
            f"val_bce={val_bce:.4f} val_supcon={val_sup:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f} sep={sep:.3f} within={within:.3f} "
            f"unit_sep={nsep:.3f} unit_within={nwithin:.3f} "
            f"best_loss={min(best_loss, val_loss):.4f} bad={0 if improved else bad + 1}/{patience}",
        )
        if improved:
            best_loss = val_loss
            best_f1 = val_f1
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if (epoch + 1) >= min_epochs and bad >= patience:
                _fold_log(
                    log_path,
                    f"early stop at epoch {epoch + 1} on val loss "
                    f"(floor was {min_epochs} epochs)",
                )
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
    z_unit = z_te / np.clip(np.linalg.norm(z_te, axis=1, keepdims=True), 1e-8, None)
    nsep_te, nwithin_te = _cluster_stats(z_unit, yb_te)
    head_w = F.normalize(model.head.weight.detach(), dim=-1).cpu().numpy().reshape(-1).astype(np.float32)

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
            "logit_scale": float(model.scale.detach().cpu()),
            "arch": {
                "name": "AEPatchViT3DCosine",
                "patch_dim": int(pls_k),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "emb_dim": EMB_DIM,
                "grid": GRID,
                "dropout": dropout,
                "head": "cosine, bias-free, detached from encoder",
            },
            "hyperparams": {
                "epochs_ran": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "best_val_f1": best_f1,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "patience": patience,
                "min_epochs": min_epochs,
                "stop_metric": "val BCE + λ τ SupCon",
                "supcon_lambda": supcon_lambda,
                "tau_start": tau_start,
                "tau_end": tau_end,
                "tau_at_best_epoch": _tau_at(best_epoch, epochs, tau_start, tau_end) if best_epoch >= 0 else None,
                "pls_k": int(pls_k),
                "logit_scale_init": LOGIT_SCALE_INIT,
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
        "best_val_loss": best_loss,
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "tau_at_best_epoch": _tau_at(best_epoch, epochs, tau_start, tau_end) if best_epoch >= 0 else None,
        "final_logit_scale": float(model.scale.detach().cpu()),
        "test_centroid_sep": sep_te,
        "test_within": within_te,
        "test_unit_sep": nsep_te,
        "test_unit_within": nwithin_te,
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
        f"unit_sep={nsep_te:.3f} unit_within={nwithin_te:.3f} "
        f"best_epoch={best_epoch} epochs_ran={len(history)} n_test={len(te)}",
    )
    del model, patches
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--supcon-lambda", type=float, default=SUPCON_LAMBDA)
    ap.add_argument("--tau-start", type=float, default=TAU_START)
    ap.add_argument("--tau-end", type=float, default=TAU_END)
    ap.add_argument("--pls-k", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--loader-workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    (OUT / "train.pid").write_text(str(os.getpid()) + "\n")

    pls_k, pls_why = (args.pls_k, "cli") if args.pls_k else min_pls_k()
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count() or 1
    if n_gpu == 0:
        raise SystemExit("Need CUDA GPUs.")
    print(f"pid={os.getpid()} CPUs={n_cpu} GPUs={n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)
    print(
        f"v3 PLS k={pls_k} ({pls_why}) λ={args.supcon_lambda} "
        f"τ {args.tau_start}→{args.tau_end} cosine-head detach",
        flush=True,
    )

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
        f"ViT-3D v3 LORO · n={len(y):,} folds={len(region_list)} gpus={max_workers} "
        f"loader_workers/fold={loader_workers} epochs {args.min_epochs}–{args.epochs} stop=val_loss",
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
                args.min_epochs,
                loader_workers,
                int(pls_k),
                args.supcon_lambda,
                args.tau_start,
                args.tau_end,
                args.force,
            )
        )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit_fold, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="ViT-3D v3 folds"):
            res = fut.result()
            rows.append(res)
            print(
                f"  {res['region']} F1={res['macro_f1']:.4f} "
                f"sep={res['test_centroid_sep']:.3f} unit_sep={res['test_unit_sep']:.3f} "
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
    unit_sep_mean = float(metrics["test_unit_sep"].mean())
    unit_within_mean = float(metrics["test_unit_within"].mean())

    summary = {
        "name": "week4_vit_100m_3d_v3",
        "changes_from_v1": [
            "BCE gradient stopped at the 3D vector",
            "cosine head: normalize z and weight, no bias, learnable scale init 10",
            "contrastive term is λ * τ * SupCon with λ=10",
            "τ anneals linearly from 0.5 to 0.1",
            "train at least 35 epochs, then early-stop on validation loss (BCE + λ τ SupCon), patience 6, cap 60",
        ],
        "min_epochs": args.min_epochs,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "stop_metric": "val BCE + λ τ SupCon",
        "loss": "BCE(scale * cos(detach(z_hat), w_hat), y) + λ * τ * SupCon(z)",
        "pls_k": int(pls_k),
        "pls_k_reason": pls_why,
        "supcon_lambda": args.supcon_lambda,
        "tau_start": args.tau_start,
        "tau_end": args.tau_end,
        "n_tiles": int(len(y)),
        "arch": {
            "class": "scripts.train_week4_vit_100m_3d_v3.AEPatchViT3DCosine",
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
        "mean_test_unit_sep": unit_sep_mean,
        "mean_test_unit_within": unit_within_mean,
        "v1_loro_macro_f1_mean": 0.732,
        "elapsed_s": round(time.time() - t0, 1),
        "pid": os.getpid(),
        "paths": {
            "oof": "outputs/week4_vit_100m_3d_v3/oof_predictions.parquet",
            "z3": "outputs/week4_vit_100m_3d_v3/oof_z3.npy",
            "metrics": "outputs/week4_vit_100m_3d_v3/loro_metrics.csv",
            "checkpoints": "outputs/week4_vit_100m_3d_v3/checkpoints",
            "logs": "outputs/week4_vit_100m_3d_v3/logs",
            "figures": "outputs/week4_vit_100m_3d_v3/figures",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_week4_vit_100m_3d_v3_notebook import main as build_nb

        build_nb()
    except Exception as exc:
        print(f"notebook build failed: {exc}", flush=True)

    print(
        json.dumps(
            _jsonable(
                {
                    "loro_macro_f1_mean": loro_mean,
                    "oof_macro_f1": oof_f1,
                    "mean_sep": sep_mean,
                    "mean_unit_sep": unit_sep_mean,
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
