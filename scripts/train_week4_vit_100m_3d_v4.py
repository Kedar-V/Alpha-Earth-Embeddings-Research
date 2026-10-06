#!/usr/bin/env python3
"""Week 4 LORO v4: within-region clusters in the raw 3D encoder output.

v3 stopped the classifier gradient at z and trained the encoder with SupCon.
BCE fell while validation SupCon rose back to chance, because the two losses
did not train the same weights. This run drops SupCon, the detach, and the
temperature schedule.

Each training batch is one metro, balanced between residential and
non-residential. On the raw 3D vectors z:

  centers c_k = mean of class k in the batch
  pull     = mean squared distance of each tile to its own center
  short    = relu(3 - sep / within)^2
  logits   = scale * (z · unit w)          # gradient reaches z and w
  loss     = BCE(logits, y) + pull + short

sep is the distance between the two centers. within is the mean radius.
The plotted gap is sep/within. The target is 3, the clear-gap line.
A fold trains at least 35 epochs, then stops if the mean validation gap
has not improved for 6 epochs. The cap is 60. Higher gap is better.
F1 is logged. It does not choose the checkpoint.

Writes outputs/week4_vit_100m_3d_v4/. A finished fold (DONE + best.pt) is
skipped unless --force. A matching last.pt resumes.

Usage:
  PYTHONWARNINGS=ignore setsid nohup .venv/bin/python -u scripts/train_week4_vit_100m_3d_v4.py \\
    > outputs/week4_vit_100m_3d_v4/train.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
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
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_week4_vit_100m_3d import (  # noqa: E402
    EMB_DIM,
    _cluster_stats,
    _jsonable,
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

OUT = ROOT / "outputs" / "week4_vit_100m_3d_v4"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
FOLDS = OUT / "folds"

TARGET_RATIO = 3.0
LOGIT_SCALE_INIT = 10.0
MIN_EPOCHS = 35
MAX_EPOCHS = 60
CKPT_EVERY = 5

METRIC_FIELDS = [
    "epoch",
    "global_step",
    "lr",
    "train_bce",
    "train_pull",
    "train_short",
    "train_loss",
    "train_sep",
    "train_within",
    "train_ratio",
    "val_bce",
    "val_loss",
    "val_sep",
    "val_within",
    "val_ratio",
    "val_f1",
    "val_precision",
    "val_recall",
    "grad_norm",
    "head_norm",
    "to_3d_norm",
    "gpu_alloc_mb",
    "gpu_reserved_mb",
    "load_s",
    "step_s",
    "eval_s",
    "epoch_s",
    "elapsed_s",
    "samples_per_sec",
    "best_ratio",
    "bad",
    "checkpoint",
]


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_metrics(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})
        handle.flush()


class AEPatchViT3DCluster(nn.Module):
    """Wide ViT. The class token is 3D. The linear head trains that vector."""

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        w = F.normalize(self.head.weight, dim=-1)
        logits = (self.scale * (z @ w.T)).squeeze(-1)
        return logits, z


def _cluster_terms(z: torch.Tensor, y: torch.Tensor, target_ratio: float):
    """Pull each class to its mean and penalize a gap below target_ratio.

    The batch is one metro. Centers stay in the graph so the gap term can
    push the two means apart. Returns None when a class has fewer than 2 tiles.
    """
    labels = y.view(-1).long()
    centers = []
    pulls = []
    radii = []
    for k in (0, 1):
        pts = z[labels == k]
        if pts.shape[0] < 2:
            return None
        center = pts.mean(dim=0)
        centers.append(center)
        pulls.append((pts - center).pow(2).sum(dim=-1).mean())
        radii.append((pts - center).norm(dim=-1).mean())
    pull = torch.stack(pulls).mean()
    within = torch.stack(radii).mean()
    sep = (centers[1] - centers[0]).norm()
    ratio = sep / within.clamp_min(1e-6)
    short = F.relu(float(target_ratio) - ratio).pow(2)
    return pull, short, ratio, sep, within


class RegionBalancedSampler(Sampler[list[int]]):
    """Each batch is one metro, with an equal count of the two classes."""

    def __init__(self, indices, y, regions, batch_size: int, seed: int):
        groups: dict[str, dict[int, list[int]]] = {}
        for local_i, gidx in enumerate(np.asarray(indices).tolist()):
            region = str(regions[int(gidx)])
            label = int(y[int(gidx)])
            groups.setdefault(region, {0: [], 1: []})[label].append(local_i)
        self.groups = {
            region: buckets
            for region, buckets in groups.items()
            if len(buckets[0]) >= 2 and len(buckets[1]) >= 2
        }
        self.names = sorted(self.groups)
        sizes = np.array(
            [len(self.groups[r][0]) + len(self.groups[r][1]) for r in self.names],
            dtype=np.float64,
        )
        self.probs = sizes / sizes.sum()
        self.batch_size = int(batch_size)
        self.half = int(batch_size) // 2
        self.seed = int(seed)
        self.epoch = 0
        self._len = max(1, int(sizes.sum()) // int(batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        for _ in range(self._len):
            region = self.names[int(rng.choice(len(self.names), p=self.probs))]
            buckets = self.groups[region]
            left = _draw(buckets[0], self.half, rng)
            right = _draw(buckets[1], self.half, rng)
            batch = np.concatenate([left, right])
            rng.shuffle(batch)
            yield batch.tolist()


def _draw(pool: list[int], n: int, rng: np.random.RandomState) -> np.ndarray:
    values = np.asarray(pool, dtype=np.int64)
    return rng.choice(values, size=n, replace=len(values) < n)


def _region_gap(z: np.ndarray, y: np.ndarray, regions: np.ndarray) -> tuple[float, float, float]:
    ratios, seps, withins = [], [], []
    for region in pd.unique(regions):
        mask = regions == region
        sep, within = _cluster_stats(z[mask], y[mask])
        if not np.isfinite(sep) or not np.isfinite(within) or within <= 0:
            continue
        seps.append(sep)
        withins.append(within)
        ratios.append(sep / within)
    if not ratios:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(ratios)), float(np.mean(seps)), float(np.mean(withins))


def _assert_encoder_is_trained() -> None:
    """BCE and the cluster term must both reach the 3D layer."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    model = AEPatchViT3DCluster(patch_dim=9, d_model=32, n_heads=4, n_layers=1, dropout=0.0).to(device)
    x = torch.randn(32, 100, 9, device=device)
    mask = torch.ones(32, 100, device=device)
    y = torch.zeros(32, device=device)
    y[16:] = 1.0
    logits, z = model(x, mask)
    terms = _cluster_terms(z, y, TARGET_RATIO)
    if terms is None:
        raise RuntimeError("cluster terms were empty on a balanced batch")
    pull, short, ratio, sep, within = terms
    loss = F.binary_cross_entropy_with_logits(logits, y) + pull + short
    loss.backward()
    enc = model.to_3d.weight.grad
    head = model.head.weight.grad
    if enc is None or float(enc.abs().sum()) == 0.0:
        raise RuntimeError("cluster loss did not train the encoder")
    if head is None or float(head.abs().sum()) == 0.0:
        raise RuntimeError("BCE did not train the head")
    model.zero_grad(set_to_none=True)
    logits, z = model(x, mask)
    F.binary_cross_entropy_with_logits(logits, y).backward()
    if model.to_3d.weight.grad is None or float(model.to_3d.weight.grad.abs().sum()) == 0.0:
        raise RuntimeError("BCE did not reach the encoder")
    del model
    torch.cuda.empty_cache()
    print(
        f"grad check ok  ratio={float(ratio):.3f} sep={float(sep):.3f} within={float(within):.3f}",
        flush=True,
    )


def _ckpt_payload(
    model,
    opt,
    sched,
    epoch,
    best_ratio,
    best_epoch,
    bad,
    config,
    arch,
    history,
    extras,
) -> dict:
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": int(epoch),
        "best_metric": "val_sep_over_within",
        "best_metric_higher_is_better": True,
        "best_metric_value": float(best_ratio) if np.isfinite(best_ratio) else None,
        "best_epoch": int(best_epoch),
        "bad": int(bad),
        "seed": SEED,
        "config": config,
        "arch": arch,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "history": history,
        **extras,
    }


def _fit_fold(payload: dict) -> dict:
    te_region = payload["te_region"]
    fold_dir = Path(payload["fold_dir"])
    log_path = fold_dir / "train.log"
    metrics_path = fold_dir / "metrics.csv"
    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"
    done_path = fold_dir / "DONE"
    warnings.filterwarnings("ignore")
    try:
        return _fit_fold_body(payload, fold_dir, log_path, metrics_path, best_path, last_path, done_path)
    except Exception:
        _fold_log(log_path, "FAILED")
        (fold_dir / "FAILED").write_text(traceback.format_exc())
        raise


def _fit_fold_body(payload, fold_dir, log_path, metrics_path, best_path, last_path, done_path) -> dict:
    te_region = payload["te_region"]
    epochs = int(payload["epochs"])
    min_epochs = int(payload["min_epochs"])
    patience = int(payload["patience"])
    batch_size = int(payload["batch_size"])
    force = bool(payload["force"])
    device_id = int(payload["device_id"])
    pls_k = int(payload["pls_k"])
    target_ratio = float(payload["target_ratio"])
    config = payload["config"]
    config_id = json.dumps(config, sort_keys=True)

    fold_dir.mkdir(parents=True, exist_ok=True)
    if done_path.exists() and best_path.exists() and not force:
        rec = json.loads(done_path.read_text())
        rec["skipped"] = True
        _fold_log(log_path, f"skip existing DONE test_f1={rec.get('macro_f1')}")
        return rec
    if force:
        for stale in (done_path, fold_dir / "FAILED"):
            if stale.exists():
                stale.unlink()

    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED + abs(hash(te_region)) % 10_000)
    np.random.seed(SEED + abs(hash(te_region)) % 10_000)

    arch = {
        "name": "AEPatchViT3DCluster",
        "patch_dim": pls_k,
        "d_model": int(payload["d_model"]),
        "n_heads": int(payload["n_heads"]),
        "n_layers": int(payload["n_layers"]),
        "emb_dim": EMB_DIM,
        "grid": GRID,
        "dropout": float(payload["dropout"]),
        "head": "linear on raw z, unit weight, learnable scale, gradient into the encoder",
    }
    if log_path.exists() and force:
        log_path.unlink()
    _fold_log(log_path, f"v4 cuda:{device_id} held_out={te_region} pls_k={pls_k}")
    _fold_log(log_path, "batches are one metro, balanced res / non-res")
    _fold_log(
        log_path,
        f"loss = BCE(scale * z · unit w, y) + pull + relu({target_ratio:g} - sep/within)^2",
    )
    _fold_log(log_path, "BCE and the cluster term both train the 3D vector")
    _fold_log(
        log_path,
        f"stop: at least {min_epochs} epochs, then stop if val sep/within "
        f"does not improve for {patience} epochs, cap {epochs}. higher is better",
    )

    patches64 = np.load(payload["patches_path"], mmap_mode="r")
    mask = np.load(payload["mask_path"], mmap_mode="r")
    y = np.load(payload["y_path"])
    regions = np.load(payload["reg_path"], allow_pickle=True)
    ckpt_pls = joblib.load(PLS_CKPT / f"fold_{te_region}.joblib")
    mu_ae = np.asarray(ckpt_pls["mu"], dtype=np.float32).reshape(-1)
    sd_ae = np.asarray(ckpt_pls["sd"], dtype=np.float32).reshape(-1)
    rotation = np.asarray(ckpt_pls["pls_da"]["x_rotations"], dtype=np.float32)[:, :pls_k]
    _fold_log(log_path, f"projecting patches with LORO PLS rotations {tuple(rotation.shape)}")
    patches = _project_pls(patches64, mask, mu_ae, sd_ae, rotation, device, log_path=log_path)
    mask = np.asarray(mask)

    te = np.where(regions == te_region)[0]
    tr_all = np.where(regions != te_region)[0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_all)
    n_val = max(1, int(len(tr_all) * float(payload["val_frac"])))
    va = tr_all[:n_val]
    tr = tr_all[n_val:]
    mu, sd = _train_stats(patches, mask, tr)

    num_workers = int(payload["num_workers"])
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 4
    train_ds = TilePatchDataset(tr, patches, mask, y, mu, sd)
    sampler = RegionBalancedSampler(tr, y, regions, batch_size, SEED + abs(hash(te_region)) % 10_000)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kw)
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
    _fold_log(
        log_path,
        f"train batches/epoch={len(sampler)} regions={len(sampler.names)} "
        f"batch={batch_size} workers={num_workers}",
    )

    model = AEPatchViT3DCluster(
        patch_dim=pls_k,
        d_model=int(payload["d_model"]),
        n_heads=int(payload["n_heads"]),
        n_layers=int(payload["n_layers"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(payload["lr"]),
        weight_decay=float(payload["weight_decay"]),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_ratio = -float("inf")
    best_f1 = -1.0
    best_epoch = -1
    bad = 0
    history: list[dict] = []
    start_epoch = 0
    if last_path.exists() and not force:
        prior = torch.load(last_path, map_location=device, weights_only=False)
        if prior.get("config_id") != config_id:
            _fold_log(log_path, "last.pt config does not match this run; starting a new fold state")
        else:
            model.load_state_dict(prior["model"])
            opt.load_state_dict(prior["optimizer"])
            sched.load_state_dict(prior["scheduler"])
            start_epoch = int(prior["epoch"]) + 1
            best_ratio = float(prior.get("best_metric_value") or -float("inf"))
            best_epoch = int(prior.get("best_epoch", -1))
            bad = int(prior.get("bad", 0))
            history = list(prior.get("history") or [])
            torch.set_rng_state(prior["rng_torch"])
            torch.cuda.set_rng_state(prior["rng_cuda"].cpu() if hasattr(prior["rng_cuda"], "cpu") else prior["rng_cuda"])
            np.random.set_state(prior["rng_numpy"])
            _fold_log(log_path, f"resume at epoch {start_epoch + 1} best_ratio={best_ratio:.4f} bad={bad}/{patience}")

    extras = {
        "config_id": config_id,
        "region_held_out": te_region,
        "mu": mu,
        "sd": sd,
        "mu_ae": mu_ae,
        "sd_ae": sd_ae,
        "pls_rotation": rotation,
    }
    fold_t0 = time.time()
    steps_per_epoch = len(sampler)

    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.perf_counter()
        sampler.set_epoch(epoch)
        tau_lr = float(sched.get_last_lr()[0])
        model.train()
        sums = {k: 0.0 for k in ("bce", "pull", "short", "loss", "sep", "within", "ratio", "grad", "n")}
        load_s = step_s = 0.0
        cursor = time.perf_counter()
        for xb, mb, yb, _idx in train_loader:
            load_s += time.perf_counter() - cursor
            step_t0 = time.perf_counter()
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, z = model(xb, mb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            terms = _cluster_terms(z, yb, target_ratio)
            if terms is None:
                loss = bce
                pull = short = ratio = sep = within = z.new_zeros(())
            else:
                pull, short, ratio, sep, within = terms
                loss = bce + pull + short
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = int(yb.shape[0])
            sums["bce"] += float(bce.item()) * bs
            sums["pull"] += float(pull.item()) * bs
            sums["short"] += float(short.item()) * bs
            sums["loss"] += float(loss.item()) * bs
            sums["sep"] += float(sep.item()) * bs
            sums["within"] += float(within.item()) * bs
            sums["ratio"] += float(ratio.item()) * bs
            sums["grad"] += float(grad_norm) * bs
            sums["n"] += bs
            step_s += time.perf_counter() - step_t0
            cursor = time.perf_counter()
        sched.step()

        eval_t0 = time.perf_counter()
        model.eval()
        preds, gts, zs = [], [], []
        vb = vn = 0.0
        with torch.no_grad():
            for xb, mb, yb, _idx in val_loader:
                xb = xb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                yb_d = yb.to(device, non_blocking=True)
                logits, z = model(xb, mb)
                vb += float(F.binary_cross_entropy_with_logits(logits, yb_d).item()) * len(yb)
                vn += len(yb)
                p = torch.sigmoid(logits).cpu().numpy()
                preds.append((p >= 0.5).astype(np.int64))
                gts.append(yb.numpy().astype(np.int64))
                zs.append(z.cpu().numpy())
        yhat = np.concatenate(preds)
        ytrue = np.concatenate(gts)
        zval = np.concatenate(zs)
        val_f1 = float(f1_score(ytrue, yhat, average="macro", zero_division=0))
        val_prec = float(precision_score(ytrue, yhat, average="macro", zero_division=0))
        val_rec = float(recall_score(ytrue, yhat, average="macro", zero_division=0))
        val_bce = vb / max(vn, 1)
        val_ratio, val_sep, val_within = _region_gap(zval, ytrue, regions[va])
        eval_s = time.perf_counter() - eval_t0

        n_seen = max(sums["n"], 1.0)
        tr_bce = sums["bce"] / n_seen
        tr_pull = sums["pull"] / n_seen
        tr_short = sums["short"] / n_seen
        tr_loss = sums["loss"] / n_seen
        tr_sep = sums["sep"] / n_seen
        tr_within = sums["within"] / n_seen
        tr_ratio = sums["ratio"] / n_seen
        grad_mean = sums["grad"] / n_seen
        val_loss = val_bce + (0.0 if not np.isfinite(val_ratio) else float(F.relu(torch.tensor(target_ratio - val_ratio)).pow(2)))
        improved = np.isfinite(val_ratio) and val_ratio > best_ratio + 1e-3
        epoch_s = time.perf_counter() - epoch_t0
        elapsed_s = time.time() - fold_t0
        remaining = max(epochs - (epoch + 1), 0)
        eta_s = epoch_s * remaining
        ckpt_event = "last"
        if improved:
            best_ratio = float(val_ratio)
            best_f1 = val_f1
            best_epoch = epoch
            bad = 0
            ckpt_event = "best+last"
        else:
            bad += 1
        if (epoch + 1) % CKPT_EVERY == 0:
            ckpt_event = ckpt_event + f"+epoch_{epoch + 1:03d}"

        payload_ckpt = _ckpt_payload(
            model, opt, sched, epoch, best_ratio, best_epoch, bad, config, arch, history, extras
        )
        # history appended after the payload snapshot of prior epochs; store this epoch inside both files
        row = {
            "epoch": epoch + 1,
            "global_step": (epoch + 1) * steps_per_epoch,
            "lr": tau_lr,
            "train_bce": tr_bce,
            "train_pull": tr_pull,
            "train_short": tr_short,
            "train_loss": tr_loss,
            "train_sep": tr_sep,
            "train_within": tr_within,
            "train_ratio": tr_ratio,
            "val_bce": val_bce,
            "val_loss": val_loss,
            "val_sep": val_sep,
            "val_within": val_within,
            "val_ratio": val_ratio,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "grad_norm": grad_mean,
            "head_norm": float(model.head.weight.detach().norm()),
            "to_3d_norm": float(model.to_3d.weight.detach().norm()),
            "gpu_alloc_mb": torch.cuda.memory_allocated(device) / 1e6,
            "gpu_reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
            "load_s": load_s,
            "step_s": step_s,
            "eval_s": eval_s,
            "epoch_s": epoch_s,
            "elapsed_s": elapsed_s,
            "samples_per_sec": n_seen / max(epoch_s, 1e-6),
            "best_ratio": best_ratio if np.isfinite(best_ratio) else "",
            "bad": f"{bad}/{patience}",
            "checkpoint": ckpt_event,
        }
        history.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()})
        payload_ckpt["history"] = history
        payload_ckpt["bad"] = bad
        payload_ckpt["best_metric_value"] = float(best_ratio) if np.isfinite(best_ratio) else None
        payload_ckpt["best_epoch"] = int(best_epoch)
        _atomic_torch_save(payload_ckpt, last_path)
        if improved:
            _atomic_torch_save(payload_ckpt, best_path)
        if (epoch + 1) % CKPT_EVERY == 0:
            periodic = fold_dir / f"epoch_{epoch + 1:03d}.pt"
            if not periodic.exists():
                _atomic_torch_save(payload_ckpt, periodic)
        _append_metrics(metrics_path, row)
        _fold_log(
            log_path,
            f"epoch {epoch + 1}/{epochs} lr={tau_lr:.6f} bce={tr_bce:.4f} pull={tr_pull:.4f} "
            f"short={tr_short:.4f} ratio={tr_ratio:.3f} sep={tr_sep:.3f} within={tr_within:.3f} "
            f"val_bce={val_bce:.4f} val_ratio={val_ratio:.3f} val_sep={val_sep:.3f} val_within={val_within:.3f} "
            f"val_f1={val_f1:.4f} prec={val_prec:.4f} rec={val_rec:.4f} "
            f"grad={grad_mean:.3f} best_ratio={best_ratio:.3f} bad={bad}/{patience} "
            f"epoch_s={epoch_s:.1f} samples/s={row['samples_per_sec']:.0f} eta_min={eta_s / 60:.1f} ckpt={ckpt_event}",
        )
        if (not improved) and (epoch + 1) >= min_epochs and bad >= patience:
            _fold_log(
                log_path,
                f"early stop at epoch {epoch + 1} on val sep/within (floor was {min_epochs} epochs)",
            )
            break

    if best_path.exists():
        best_blob = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_blob["model"])
    model.eval()
    n_te = len(te)
    y_pred = np.zeros(n_te, dtype=np.int64)
    p_res = np.zeros(n_te, dtype=np.float64)
    z_te = np.zeros((n_te, EMB_DIM), dtype=np.float32)
    te_pos = {int(g): i for i, g in enumerate(te.tolist())}
    with torch.no_grad():
        for xb, mb, _yb, idx in test_loader:
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
    ratio_te = float(sep_te / within_te) if within_te and np.isfinite(within_te) and within_te > 0 else float("nan")
    head_w = F.normalize(model.head.weight.detach(), dim=-1).cpu().numpy().reshape(-1).astype(np.float32)

    np.save(fold_dir / "oof_idx.npy", te.astype(np.int64))
    np.save(fold_dir / "oof_ypred.npy", y_pred)
    np.save(fold_dir / "oof_pres.npy", p_res)
    np.save(fold_dir / "oof_z3.npy", z_te)
    np.save(fold_dir / "head_w.npy", head_w)
    result = {
        "region": te_region,
        "device": f"cuda:{device_id}",
        "status": "completed",
        "macro_f1": macro_f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "best_metric": "val_sep_over_within",
        "best_metric_value": None if not np.isfinite(best_ratio) else float(best_ratio),
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "final_epoch": len(history),
        "epochs_ran": len(history),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "test_centroid_sep": sep_te,
        "test_within": within_te,
        "test_ratio": ratio_te,
        "n_test": int(len(te)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "runtime_seconds": round(time.time() - fold_t0, 1),
        "skipped": False,
    }
    done_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    _fold_log(
        log_path,
        f"done test_f1={macro_f1:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"sep={sep_te:.3f} within={within_te:.3f} ratio={ratio_te:.3f} "
        f"best_epoch={best_epoch} best_ratio={best_ratio:.3f} n_test={len(te)}",
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
    ap.add_argument("--target-ratio", type=float, default=TARGET_RATIO)
    ap.add_argument("--pls-k", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--loader-workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    for directory in (OUT, FIG, CACHE, FOLDS):
        directory.mkdir(parents=True, exist_ok=True)
    (OUT / "train.pid").write_text(str(os.getpid()) + "\n")
    _assert_encoder_is_trained()

    pls_k, pls_why = (args.pls_k, "cli") if args.pls_k else min_pls_k()
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count() or 1
    if n_gpu == 0:
        raise SystemExit("Need CUDA GPUs.")
    print(f"pid={os.getpid()} CPUs={n_cpu} GPUs={n_gpu}", flush=True)
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)
    print(
        f"v4 PLS k={pls_k} ({pls_why}) target sep/within={args.target_ratio} "
        "within-region pull + gap, BCE trains the encoder",
        flush=True,
    )

    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_week4_vit_100m_3d_v4_notebook import main as build_nb

        build_nb()
    except Exception as exc:
        print(f"notebook build failed: {exc}", flush=True)

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
        f"ViT-3D v4 LORO · n={len(y):,} folds={len(region_list)} gpus={max_workers} "
        f"loader_workers/fold={loader_workers} epochs {args.min_epochs}–{args.epochs} "
        f"stop=val_sep/within",
        flush=True,
    )
    config = {
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "val_frac": args.val_frac,
        "patience": args.patience,
        "target_ratio": args.target_ratio,
        "pls_k": int(pls_k),
        "seed": SEED,
    }
    jobs = []
    for i, region in enumerate(region_list):
        jobs.append(
            {
                "te_region": region,
                "patches_path": str(CACHE / "patches.npy"),
                "mask_path": str(CACHE / "mask.npy"),
                "y_path": str(CACHE / "y.npy"),
                "reg_path": str(CACHE / "regions.npy"),
                "fold_dir": str(FOLDS / region),
                "device_id": i % n_gpu,
                "num_workers": loader_workers,
                "force": args.force,
                "config": config,
                **config,
            }
        )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_fit_fold, job): job["te_region"] for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ViT-3D v4 folds"):
            res = fut.result()
            rows.append(res)
            print(
                f"  {res['region']} F1={res['macro_f1']:.4f} "
                f"ratio={res['test_ratio']:.3f} sep={res['test_centroid_sep']:.3f} "
                f"ep={res['best_epoch']} skip={res.get('skipped')}",
                flush=True,
            )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)
    parts = []
    for region in region_list:
        path = FOLDS / region / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "region", region)
            parts.append(frame)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(OUT / "metrics.csv", index=False)

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
        fold_dir = FOLDS / region
        idx = np.load(fold_dir / "oof_idx.npy")
        oof.loc[idx, "y_pred"] = np.load(fold_dir / "oof_ypred.npy")
        oof.loc[idx, "p_res"] = np.load(fold_dir / "oof_pres.npy")
        z_oof[idx] = np.load(fold_dir / "oof_z3.npy")
        head_ws[region] = np.load(fold_dir / "head_w.npy")
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    np.save(OUT / "oof_z3.npy", z_oof)
    np.savez(OUT / "head_w_by_region.npz", **head_ws)

    oof_f1 = float(f1_score(oof["y_true"], oof["y_pred"], average="macro", zero_division=0))
    loro_mean = float(metrics["macro_f1"].mean())
    loro_std = float(metrics["macro_f1"].std(ddof=0))
    sep_mean = float(metrics["test_centroid_sep"].mean())
    within_mean = float(metrics["test_within"].mean())
    ratio_mean = float(metrics["test_ratio"].mean())
    summary = {
        "name": "week4_vit_100m_3d_v4",
        "status": "completed",
        "best_metric": "val_sep_over_within",
        "best_metric_higher_is_better": True,
        "loss": "BCE(scale * (z · unit w), y) + mean ||z - c_y||^2 + relu(3 - sep/within)^2",
        "target_ratio": args.target_ratio,
        "min_epochs": args.min_epochs,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "stop_metric": "mean within-region validation sep/within",
        "pls_k": int(pls_k),
        "pls_k_reason": pls_why,
        "n_tiles": int(len(y)),
        "arch": {
            "class": "scripts.train_week4_vit_100m_3d_v4.AEPatchViT3DCluster",
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
        "mean_test_ratio": ratio_mean,
        "v1_loro_macro_f1_mean": 0.732,
        "v3_loro_macro_f1_mean": 0.7347746243802467,
        "elapsed_s": round(time.time() - t0, 1),
        "runtime_seconds": round(time.time() - t0, 1),
        "pid": os.getpid(),
        "paths": {
            "oof": "outputs/week4_vit_100m_3d_v4/oof_predictions.parquet",
            "z3": "outputs/week4_vit_100m_3d_v4/oof_z3.npy",
            "metrics": "outputs/week4_vit_100m_3d_v4/metrics.csv",
            "loro_metrics": "outputs/week4_vit_100m_3d_v4/loro_metrics.csv",
            "folds": "outputs/week4_vit_100m_3d_v4/folds",
            "figures": "outputs/week4_vit_100m_3d_v4/figures",
            "log": "outputs/week4_vit_100m_3d_v4/train.log",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    try:
        from build_week4_vit_100m_3d_v4_notebook import main as build_nb

        build_nb()
    except Exception as exc:
        print(f"notebook build failed: {exc}", flush=True)
    print(
        json.dumps(
            _jsonable(
                {
                    "status": "completed",
                    "loro_macro_f1_mean": loro_mean,
                    "oof_macro_f1": oof_f1,
                    "mean_test_ratio": ratio_mean,
                    "runtime_seconds": summary["runtime_seconds"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
