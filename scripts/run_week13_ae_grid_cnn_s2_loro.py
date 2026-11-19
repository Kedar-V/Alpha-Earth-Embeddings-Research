#!/usr/bin/env python3
"""Week 13: small AE-grid CNN + late-fusion AE mean + S2 → soft-fraction LORO.

Apples-to-apples (docs/eval_protocol_apples_to_apples.md / balanced_eval):
  TRAIN: soft_gt_covered buildings (~86k)
  EVAL:  locked Section 3/4 building IDs (71,497)
  METRIC: 0.5*(MAE_res + MAE_nres)
  SPLIT:  leave-one-region-out

Architecture:
  64-band AE cells rasterized to 16×16 bbox grid (+ footprint mask)
  → tiny CNN → masked GAP (128-d)
  → concat AE mean 64-d + S2 NRES feats → MLP → (frac_res, frac_nres)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
W6_BUILD = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
XGB_BUILD = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "buildings_section3_with_xgb.parquet"
)
S2_TRAIN = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "train_buildings_ae_s2.parquet"
S2_OOF = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "buildings_oof_ae_s2.parquet"
RING_OOF = ROOT / "outputs" / "week8_ae_rings_loro" / "latest" / "artifacts" / "buildings_oof_ring.parquet"
SHARED_GRID = ROOT / "outputs" / "week13_ae_grid_cnn_s2_loro" / "_shared" / "grids_pad16"

BANDS = [f"A{i:02d}" for i in range(64)]
S2_FEATS = [
    "s2_blue",
    "s2_green",
    "s2_red",
    "s2_nir",
    "s2_brightness",
    "s2_blue_dom",
    "s2_ndvi",
    "s2_contrast_small",
    "s2_contrast_large",
    "s2_contrast_ratio",
    "fp_area_m2",
    "fp_equiv_diam_m",
    "fp_compactness",
    "s2_nres_score",
    "s2_n_pix",
    "s2_ok",
]
PAD = 16
EMB_DIM = 128
SEED = 42
BATCH_SIZE = 128
LR = 1e-3
EPOCHS_DEFAULT = 12

METRO_META = [
    ("Boston_NE", "Boston NE", "#ECE8F4"),
    ("NYC_NE", "NYC NE", "#ECE8F4"),
    ("Atlanta_SE", "Atlanta SE", "#E2EFE6"),
    ("Houston_S", "Houston S", "#F3EDD5"),
    ("Chicago_MW", "Chicago MW", "#F3E3E7"),
    ("Phoenix_SW", "Phoenix SW", "#EFE4D9"),
    ("Denver_W", "Denver W", "#E1ECEE"),
    ("LA_W", "LA W", "#E1ECEE"),
    ("SF_W", "SF W", "#E1ECEE"),
    ("Seattle_NW", "Seattle NW", "#E9EDDC"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    if "_t0" not in prev and "_t0" not in kwargs:
        kwargs["_t0"] = time.time()
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0.0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def detect_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if int(m.sum()) == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    mae_r = float(mean_absolute_error(y_res[m], p_res[m]))
    mae_n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": mae_r, "nres": mae_n, "mean": 0.5 * (mae_r + mae_n)}


def decorate_metro(rows: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in rows}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        out.append(r)
    return out


class BuildingGridDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        grids: np.ndarray,
        masks: np.ndarray,
        ae_mean: np.ndarray,
        s2: np.ndarray,
        y: np.ndarray,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.grids = grids
        self.masks = masks
        self.ae_mean = ae_mean
        self.s2 = s2
        self.y = y

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # (65, H, W): AE bands + mask
        x = np.concatenate([self.grids[idx], self.masks[idx][None, ...]], axis=0)
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.ae_mean[idx]),
            torch.from_numpy(self.s2[idx]),
            torch.from_numpy(self.y[idx]),
            idx,
        )


class FootprintCNNFusion(nn.Module):
    """Tiny CNN on AE grid + late fusion of AE mean + S2 tabular."""

    def __init__(self, s2_dim: int = 16, out_dim: int = 2, emb_dim: int = EMB_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(65, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, emb_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(emb_dim),
            nn.ReLU(inplace=True),
        )
        fuse_in = emb_dim + 64 + s2_dim
        self.head = nn.Sequential(
            nn.Linear(fuse_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, -1:, :, :]
        feats = self.backbone(x)
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return (feats * mask).sum(dim=(2, 3)) / denom

    def forward(self, x: torch.Tensor, ae_mean: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.head(torch.cat([z, ae_mean, s2], dim=1))


def _build_metro_grids_from_cells(region: str, cells_path: str, id_list: list[str], out_dir: str) -> dict[str, Any]:
    """Worker: build PAD×PAD AE grids from a metro cell shard parquet."""
    t0 = time.time()
    out = Path(out_dir)
    grid_path = out / f"{region}_grids.npy"
    mask_path = out / f"{region}_masks.npy"
    ids_path = out / f"{region}_ids.parquet"
    if grid_path.exists() and mask_path.exists() and ids_path.exists():
        return {"region": region, "n": len(pd.read_parquet(ids_path)), "seconds": 0.0, "skipped": True}

    cells = pd.read_parquet(cells_path)
    cells["id"] = cells["id"].astype(str)
    id_list = [str(x) for x in id_list]
    id_to_idx = {bid: i for i, bid in enumerate(id_list)}
    n = len(id_list)
    grids = np.zeros((n, 64, PAD, PAD), dtype=np.float32)
    masks = np.zeros((n, PAD, PAD), dtype=np.float32)

    for bid, g in cells.groupby("id", sort=False):
        if bid not in id_to_idx:
            continue
        i0 = id_to_idx[bid]
        ii = g["i"].to_numpy(np.int32)
        jj = g["j"].to_numpy(np.int32)
        vals = g[BANDS].to_numpy(np.float32)
        inside = g["inside_polygon"].to_numpy(bool)
        keep = (ii >= 0) & (jj >= 0) & (ii < PAD) & (jj < PAD)
        ii, jj, vals, inside = ii[keep], jj[keep], vals[keep], inside[keep]
        if len(ii) == 0:
            continue
        # Two-step index: grids[i0, :, ii, jj] would transpose advanced/slice dims.
        g = grids[i0]
        g[:, ii, jj] = vals.T
        if inside.any():
            masks[i0, ii[inside], jj[inside]] = 1.0
        else:
            masks[i0, ii, jj] = 1.0

    np.save(grid_path, grids)
    np.save(mask_path, masks)
    pd.DataFrame({"id": id_list, "region": region}).to_parquet(ids_path, index=False)
    return {"region": region, "n": n, "seconds": round(time.time() - t0, 1), "skipped": False}


def stage_build_grids(run_dir: Path, panel: pd.DataFrame, workers: int) -> Path:
    SHARED_GRID.mkdir(parents=True, exist_ok=True)
    regions = sorted(panel["region"].unique())
    # Skip if all metros already cached
    if all((SHARED_GRID / f"{r}_grids.npy").exists() for r in regions):
        log(run_dir, f"Reusing cached grids in {SHARED_GRID}")
        return SHARED_GRID

    soft_ids = set(panel["id"].astype(str))
    id_to_region = dict(zip(panel["id"].astype(str), panel["region"].astype(str)))
    write_progress(run_dir, status="running", stage="load_cells", completed=0, total=len(regions) + 1, failed=0)
    log(run_dir, f"Loading cells for {len(soft_ids):,} soft buildings from {CELLS}")
    t0 = time.time()
    cells = pd.read_parquet(CELLS, columns=["id", "i", "j", "inside_polygon", *BANDS])
    cells["id"] = cells["id"].astype(str)
    cells = cells[cells["id"].isin(soft_ids)].copy()
    cells["region"] = cells["id"].map(id_to_region)
    cells = cells[cells["region"].notna()].copy()
    log(run_dir, f"cells kept n={len(cells):,} in {time.time()-t0:.1f}s")

    shard_dir = SHARED_GRID / "_cell_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for region in regions:
        if (SHARED_GRID / f"{region}_grids.npy").exists():
            continue
        ids = panel.loc[panel["region"] == region, "id"].astype(str).tolist()
        shard = shard_dir / f"{region}.parquet"
        if not shard.exists():
            cells[cells["region"] == region].to_parquet(shard, index=False)
        jobs.append((region, str(shard), ids, str(SHARED_GRID)))

    write_progress(run_dir, stage="build_grids", completed=1, total=len(regions) + 1, failed=0)
    if not jobs:
        return SHARED_GRID

    max_w = max(1, min(workers, len(jobs), max(1, (os.cpu_count() or 8) // 4)))
    log(run_dir, f"Building PAD={PAD} grids for {len(jobs)} metros · workers={max_w}")
    done = 1
    failed = 0
    with ProcessPoolExecutor(max_workers=max_w) as pool:
        futs = {pool.submit(_build_metro_grids_from_cells, *j): j[0] for j in jobs}
        for fut in as_completed(futs):
            region = futs[fut]
            try:
                info = fut.result()
                log(
                    run_dir,
                    f"grids {region}: n={info['n']:,} skip={info['skipped']} in {info['seconds']}s",
                )
                atomic_json(run_dir / "checkpoints" / f"grid_{region}.json", info)
            except Exception as e:
                failed += 1
                log(run_dir, f"GRID FAIL {region}: {e}")
                atomic_json(run_dir / "failures" / f"grid_{region}.json", {"region": region, "error": str(e)})
            done += 1
            write_progress(
                run_dir,
                status="running",
                stage="build_grids",
                completed=done,
                total=len(regions) + 1,
                failed=failed,
            )
    if failed:
        raise RuntimeError(f"{failed} metro grid builds failed")
    return SHARED_GRID


def write_aligned_cache(panel: pd.DataFrame, grid_dir: Path, cache_dir: Path) -> Path:
    """Write panel-aligned memmap grids/masks + tabular npz (no full RAM concat)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel = panel.copy()
    panel["id"] = panel["id"].astype(str)
    n = len(panel)
    g_mm = np.memmap(cache_dir / "grids.f32", dtype=np.float32, mode="w+", shape=(n, 64, PAD, PAD))
    m_mm = np.memmap(cache_dir / "masks.f32", dtype=np.float32, mode="w+", shape=(n, PAD, PAD))

    # map (region,id) -> panel row
    key = panel["region"].astype(str) + "|" + panel["id"]
    row_of = {k: i for i, k in enumerate(key.tolist())}

    for region in sorted(panel["region"].unique()):
        ids = pd.read_parquet(grid_dir / f"{region}_ids.parquet")
        ids["id"] = ids["id"].astype(str)
        g = np.load(grid_dir / f"{region}_grids.npy", mmap_mode="r")
        m = np.load(grid_dir / f"{region}_masks.npy", mmap_mode="r")
        for local_i, bid in enumerate(ids["id"].tolist()):
            k = f"{region}|{bid}"
            if k not in row_of:
                continue
            ri = row_of[k]
            g_mm[ri] = g[local_i]
            m_mm[ri] = m[local_i]
    g_mm.flush()
    m_mm.flush()
    # Unicode arrays — avoid object dtype (np.load defaults allow_pickle=False).
    regions_u = panel["region"].astype(str).to_numpy()
    ids_u = panel["id"].astype(str).to_numpy()
    np.savez(
        cache_dir / "tabular.npz",
        ae_mean=panel[BANDS].to_numpy(np.float32),
        s2=np.nan_to_num(panel[S2_FEATS].to_numpy(np.float32), nan=0.0),
        y=panel[["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32),
        regions=np.asarray(regions_u, dtype=f"U{max(1, int(max(len(x) for x in regions_u)))}"),
        ids=np.asarray(ids_u, dtype=f"U{max(1, int(max(len(x) for x in ids_u)))}"),
        n=np.int64(n),
    )
    del g_mm, m_mm
    return cache_dir


def _load_tabular(cache_dir: Path) -> dict[str, Any]:
    # Object-dtype keys fail on __getitem__ under allow_pickle=False (not on open).
    meta = np.load(cache_dir / "tabular.npz", allow_pickle=True)
    regions = np.asarray(meta["regions"]).astype(str)
    return {
        "regions": regions,
        "ae_mean": np.asarray(meta["ae_mean"]),
        "s2": np.asarray(meta["s2"]),
        "y": np.asarray(meta["y"]),
        "n": int(meta["n"]),
        "ids": np.asarray(meta["ids"]).astype(str),
    }


def _loro_fold(payload: dict[str, Any]) -> dict[str, Any]:
    hold = payload["hold"]
    gpu_id = payload["gpu_id"]
    epochs = payload["epochs"]
    cache_dir = Path(payload["cache_dir"])
    t0 = time.time()

    meta = _load_tabular(cache_dir)
    regions = meta["regions"]
    ae_mean = meta["ae_mean"]
    s2 = meta["s2"]
    y = meta["y"]
    n = meta["n"]
    grids = np.memmap(cache_dir / "grids.f32", dtype=np.float32, mode="r", shape=(n, 64, PAD, PAD))
    masks = np.memmap(cache_dir / "masks.f32", dtype=np.float32, mode="r", shape=(n, PAD, PAD))

    te = np.where(regions == hold)[0]
    tr = np.where(regions != hold)[0]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # standardize tabular feats on train fold
    ae_mu, ae_sd = ae_mean[tr].mean(0), ae_mean[tr].std(0) + 1e-6
    s2_mu, s2_sd = s2[tr].mean(0), s2[tr].std(0) + 1e-6
    ae_s = ((ae_mean - ae_mu) / ae_sd).astype(np.float32)
    s2_s = ((s2 - s2_mu) / s2_sd).astype(np.float32)

    # num_workers=0: safer with memmap under spawn/CUDA folds
    train_loader = DataLoader(
        BuildingGridDataset(tr, grids, masks, ae_s, s2_s, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        BuildingGridDataset(te, grids, masks, ae_s, s2_s, y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    torch.manual_seed(SEED + abs(hash(hold)) % 10_000)
    model = FootprintCNNFusion(s2_dim=s2.shape[1], out_dim=2, emb_dim=EMB_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()

    model.train()
    for _ep in range(epochs):
        for xb, ae, s2b, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            ae = ae.to(device, non_blocking=True)
            s2b = s2b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, ae, s2b)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    preds = np.zeros((len(te), 2), dtype=np.float32)
    embs = np.zeros((len(te), EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        cursor = 0
        for xb, ae, s2b, yb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            ae = ae.to(device, non_blocking=True)
            s2b = s2b.to(device, non_blocking=True)
            z = model.encode(xb)
            p = model.head(torch.cat([z, ae, s2b], dim=1)).clamp(0, 1)
            n = p.shape[0]
            preds[cursor : cursor + n] = p.cpu().numpy()
            embs[cursor : cursor + n] = z.cpu().numpy()
            cursor += n

    mae_r = float(mean_absolute_error(y[te, 0], preds[:, 0]))
    mae_n = float(mean_absolute_error(y[te, 1], preds[:, 1]))
    return {
        "hold": hold,
        "te_idx": te.astype(np.int64),
        "pred": preds,
        "emb": embs,
        "mae_res": mae_r,
        "mae_nres": mae_n,
        "mae_mean": 0.5 * (mae_r + mae_n),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "seconds": round(time.time() - t0, 1),
        "device": str(device),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "ae_mu": ae_mu.astype(np.float32),
        "ae_sd": ae_sd.astype(np.float32),
        "s2_mu": s2_mu.astype(np.float32),
        "s2_sd": s2_sd.astype(np.float32),
    }


def train_loro(run_dir: Path, panel: pd.DataFrame, cache_dir: Path, epochs: int, fold_workers: int) -> pd.DataFrame:
    regions = sorted(panel["region"].unique())
    n_gpus = len(detect_gpus())
    log(run_dir, f"LORO CNN folds={len(regions)} fold_workers={fold_workers} n_gpus={n_gpus} epochs={epochs}")

    payloads = []
    for i, hold in enumerate(regions):
        payloads.append(
            {
                "hold": hold,
                "gpu_id": i % max(1, n_gpus),
                "epochs": epochs,
                "cache_dir": str(cache_dir),
            }
        )

    oof = np.full((len(panel), 2), np.nan, dtype=np.float32)
    oof_emb = np.full((len(panel), EMB_DIM), np.nan, dtype=np.float32)
    fold_metrics = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)

    write_progress(run_dir, status="running", stage="train_loro", completed=0, total=len(regions), failed=0)
    max_w = min(fold_workers, n_gpus) if n_gpus > 0 else 1
    max_w = max(1, min(max_w, len(payloads)))
    done = 0
    failed = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
        futs = {pool.submit(_loro_fold, p): p["hold"] for p in payloads}
        for fut in as_completed(futs):
            hold = futs[fut]
            try:
                res = fut.result()
                oof[res["te_idx"]] = res["pred"]
                oof_emb[res["te_idx"]] = res["emb"]
                fold_metrics.append(
                    {
                        "holdout": res["hold"],
                        "n_train": res["n_train"],
                        "n_test": res["n_test"],
                        "mae_res": res["mae_res"],
                        "mae_nres": res["mae_nres"],
                        "mae_mean": res["mae_mean"],
                        "seconds": res["seconds"],
                        "device": res["device"],
                    }
                )
                torch.save(
                    {
                        "hold": hold,
                        "state_dict": res["state_dict"],
                        "ae_mu": res["ae_mu"],
                        "ae_sd": res["ae_sd"],
                        "s2_mu": res["s2_mu"],
                        "s2_sd": res["s2_sd"],
                        "epochs": epochs,
                        "pad": PAD,
                        "emb_dim": EMB_DIM,
                    },
                    ckpt / f"fold_{hold}.pt",
                )
                np.savez_compressed(
                    ckpt / f"fold_{hold}_oof.npz",
                    pred=res["pred"],
                    emb=res["emb"],
                    te_idx=res["te_idx"],
                )
                log(
                    run_dir,
                    f"LORO hold={hold}: mean={res['mae_mean']:.4f} res={res['mae_res']:.4f} "
                    f"nres={res['mae_nres']:.4f} n_te={res['n_test']:,} "
                    f"device={res['device']} in {res['seconds']}s",
                )
            except Exception as e:
                failed += 1
                log(run_dir, f"LORO FAIL {hold}: {e}")
                atomic_json(run_dir / "failures" / f"fold_{hold}.json", {"hold": hold, "error": str(e)})
            done += 1
            write_progress(
                run_dir,
                status="running",
                stage="train_loro",
                completed=done,
                total=len(regions),
                failed=failed,
            )
    if failed:
        raise RuntimeError(f"{failed} LORO folds failed")
    if not np.isfinite(oof).all():
        raise RuntimeError("incomplete OOF")

    out = panel[
        [
            "id",
            "region",
            "soft_gt_frac_res",
            "soft_gt_frac_nres",
            "ae_frac_res",
            "ae_frac_nres",
            "ghsl_frac_res",
            "ghsl_frac_nres",
            "pls_frac_res",
            "pls_frac_nres",
        ]
    ].copy()
    out["cnn_frac_res"] = oof[:, 0]
    out["cnn_frac_nres"] = oof[:, 1]
    emb_df = pd.DataFrame(oof_emb, columns=[f"E{i:03d}" for i in range(EMB_DIM)])
    emb_out = pd.concat([out[["id", "region"]].reset_index(drop=True), emb_df], axis=1)
    emb_out.to_parquet(run_dir / "artifacts" / "buildings_oof_cnn_embeddings.parquet", index=False)

    if S2_OOF.exists():
        s2p = pd.read_parquet(S2_OOF.resolve(), columns=["id", "s2_frac_res", "s2_frac_nres"])
        s2p["id"] = s2p["id"].astype(str)
        out = out.merge(s2p, on="id", how="left")
    if RING_OOF.exists():
        rk = pd.read_parquet(RING_OOF.resolve(), columns=["id", "ring_frac_res", "ring_frac_nres"])
        rk["id"] = rk["id"].astype(str)
        out = out.merge(rk, on="id", how="left")

    out.to_parquet(run_dir / "artifacts" / "buildings_oof_ae_grid_cnn_s2.parquet", index=False)
    pd.DataFrame(fold_metrics).sort_values("holdout").to_csv(run_dir / "artifacts" / "fold_metrics.csv", index=False)
    atomic_json(run_dir / "artifacts" / "fold_metrics.json", sorted(fold_metrics, key=lambda x: x["holdout"]))
    return out


def score_buildings(run_dir: Path, oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    oof = oof.copy()
    oof["id"] = oof["id"].astype(str)
    m = cohort.merge(oof, on=["id", "region"], how="left")
    assert len(m) == 71497, f"building cohort size {len(m)} != 71497"

    if XGB_BUILD.exists():
        xb = pd.read_parquet(XGB_BUILD, columns=["id", "xgb_frac_res", "xgb_frac_nres"])
        xb["id"] = xb["id"].astype(str)
        m = m.merge(xb, on="id", how="left")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ae_frac_res", "ae_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
        "ae_ring": ("ring_frac_res", "ring_frac_nres"),
        "ae_s2": ("s2_frac_res", "s2_frac_nres"),
        "ae_cnn_s2": ("cnn_frac_res", "cnn_frac_nres"),
    }
    soft_pooled = {}
    for name, (pr, pn) in models.items():
        if pr not in m.columns:
            soft_pooled[name] = {
                "n": 0,
                "res": float("nan"),
                "nres": float("nan"),
                "mean": float("nan"),
                "model": name,
                "gt": "soft",
            }
            continue
        soft_pooled[name] = {
            **mae_pair(
                m["soft_gt_frac_res"].to_numpy(float),
                m["soft_gt_frac_nres"].to_numpy(float),
                m[pr].to_numpy(float),
                m[pn].to_numpy(float),
            ),
            "model": name,
            "gt": "soft",
        }

    metro = []
    for region, sub in m.groupby("region", sort=False):
        row: dict[str, Any] = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            if pr not in sub.columns:
                continue
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)

    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_ae_grid_cnn_s2.parquet", index=False)
    for name in models:
        r = soft_pooled[name]
        log(run_dir, f"building {name:12s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")

    return {
        "n_shared": len(m),
        "coverage": float(np.mean(np.isfinite(m["cnn_frac_res"].to_numpy(float)))),
        "soft_pooled": soft_pooled,
        "metro_soft": decorate_metro(metro),
        "cnn_beats_s2": sum(
            1
            for r in decorate_metro(metro)
            if "ae_cnn_s2" in r
            and "ae_s2" in r
            and np.isfinite(r["ae_cnn_s2"]["mean"])
            and np.isfinite(r["ae_s2"]["mean"])
            and r["ae_cnn_s2"]["mean"] + 1e-12 < r["ae_s2"]["mean"]
        ),
        "cnn_beats_ghsl": sum(
            1
            for r in decorate_metro(metro)
            if "ae_cnn_s2" in r
            and "ghsl_built_s" in r
            and np.isfinite(r["ae_cnn_s2"]["mean"])
            and np.isfinite(r["ghsl_built_s"]["mean"])
            and r["ae_cnn_s2"]["mean"] + 1e-12 < r["ghsl_built_s"]["mean"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week13_ae_grid_cnn_s2_loro")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument("--s2-train", type=Path, default=S2_TRAIN)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--fold-workers", type=int, default=2)
    ap.add_argument("--grid-workers", type=int, default=max(2, min(10, (os.cpu_count() or 8) // 4)))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.resume_dir or (args.out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    for sub in ("artifacts", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    latest = args.out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    gpu_names = detect_gpus()
    n_gpus = len(gpu_names)
    (run_dir / "launch.txt").write_text(
        f"started={utc_now()}\npid={os.getpid()}\nepochs={args.epochs}\n"
        f"fold_workers={args.fold_workers}\ngrid_workers={args.grid_workers}\n"
        f"n_gpus={n_gpus}\ngpus={gpu_names}\npad={PAD}\nemb_dim={EMB_DIM}\n"
        f"protocol=docs/eval_protocol_apples_to_apples.md\n"
        f"arch=FootprintCNNFusion(cnn+ae_mean+s2)\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpus={os.cpu_count()}\nn_gpus={n_gpus}\ngpus={gpu_names}\n"
        f"torch={torch.__version__}\npython={os.sys.executable}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="init",
        completed=0,
        total=5,
        failed=0,
        _t0=time.time(),
        output=str(run_dir),
        n_cpus=os.cpu_count(),
        n_gpus=n_gpus,
    )
    log(run_dir, f"Week13 AE-grid CNN+S2 LORO start → {run_dir} · GPUs={n_gpus} {gpu_names}")

    s2_path = args.s2_train.resolve()
    panel = pd.read_parquet(s2_path)
    panel["id"] = panel["id"].astype(str)
    need = ["id", "region", "soft_gt_frac_res", "soft_gt_frac_nres", *BANDS, *S2_FEATS,
            "ae_frac_res", "ae_frac_nres", "ghsl_frac_res", "ghsl_frac_nres", "pls_frac_res", "pls_frac_nres"]
    missing = [c for c in need if c not in panel.columns]
    if missing:
        # join baselines from week6 if needed
        w6 = pd.read_parquet(W6_BUILD)
        w6["id"] = w6["id"].astype(str)
        cols = [c for c in w6.columns if c not in panel.columns or c in ("id", "region")]
        panel = panel.merge(w6[cols], on=["id", "region"], how="left", suffixes=("", "_w6"))
        missing = [c for c in need if c not in panel.columns]
        if missing:
            raise RuntimeError(f"panel missing {missing}")
    log(run_dir, f"panel n={len(panel):,} regions={panel['region'].nunique()}")

    stage_build_grids(run_dir, panel, workers=args.grid_workers)
    write_progress(run_dir, stage="grids_done", completed=1, total=5)

    cache_dir = run_dir / "artifacts" / "train_cache"
    grids_ok = (cache_dir / "grids.f32").exists() and (cache_dir / "masks.f32").exists()
    tab_path = cache_dir / "tabular.npz"
    if grids_ok and tab_path.exists():
        probe = np.load(tab_path, allow_pickle=True)
        if probe["regions"].dtype.kind == "O" or probe["ids"].dtype.kind == "O":
            log(run_dir, f"Rewriting tabular.npz (object→unicode) in {cache_dir}")
            regions_u = panel["region"].astype(str).to_numpy()
            ids_u = panel["id"].astype(str).to_numpy()
            np.savez(
                tab_path,
                ae_mean=panel[BANDS].to_numpy(np.float32),
                s2=np.nan_to_num(panel[S2_FEATS].to_numpy(np.float32), nan=0.0),
                y=panel[["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32),
                regions=np.asarray(regions_u, dtype=f"U{max(1, int(max(len(x) for x in regions_u)))}"),
                ids=np.asarray(ids_u, dtype=f"U{max(1, int(max(len(x) for x in ids_u)))}"),
                n=np.int64(len(panel)),
            )
        else:
            log(run_dir, f"Reusing memmap cache {cache_dir}")
        _ = _load_tabular(cache_dir)  # smoke-load
    else:
        log(run_dir, f"Aligning metro grids → memmap cache {cache_dir}")
        write_aligned_cache(panel, SHARED_GRID, cache_dir)
    write_progress(run_dir, stage="arrays_loaded", completed=2, total=5)

    oof = train_loro(run_dir, panel, cache_dir, epochs=args.epochs, fold_workers=args.fold_workers)
    write_progress(run_dir, stage="score", completed=4, total=5)

    b_metrics = score_buildings(run_dir, oof)
    metrics = {
        "protocol": "docs/eval_protocol_apples_to_apples.md",
        "train_n_buildings": int(len(panel)),
        "eval_n_buildings": 71497,
        "pad": PAD,
        "emb_dim": EMB_DIM,
        "epochs": args.epochs,
        "arch": "FootprintCNNFusion",
        "buildings": b_metrics,
        "checklist": {
            "eval_ids_locked": True,
            "soft_gt_from_week6": True,
            "spatial_split": "LORO",
            "metric": "0.5*(MAE_res+MAE_nres)",
            "baselines_joined": True,
            "s2_features_reused_from_week9": True,
            "late_fusion": "cnn_gap + ae_mean64 + s2",
        },
    }
    atomic_json(run_dir / "artifacts" / "section10_ae_grid_cnn_s2_metrics.json", metrics)

    cnn_mean = b_metrics["soft_pooled"]["ae_cnn_s2"]["mean"]
    s2_mean = b_metrics["soft_pooled"].get("ae_s2", {}).get("mean")
    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "finished_at": utc_now(),
            "output": str(run_dir),
            "building_mean_ae_cnn_s2": cnn_mean,
            "building_mean_ae_s2": s2_mean,
            "building_delta_cnn_minus_s2": None if s2_mean is None else float(cnn_mean - s2_mean),
            "building_mean_ghsl": b_metrics["soft_pooled"]["ghsl_built_s"]["mean"],
            "coverage": b_metrics["coverage"],
            "metrics": str(run_dir / "artifacts" / "section10_ae_grid_cnn_s2_metrics.json"),
            "cnn_embeddings": str(run_dir / "artifacts" / "buildings_oof_cnn_embeddings.parquet"),
        },
    )
    write_progress(run_dir, status="completed", stage="done", completed=5, total=5, failed=0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, "DONE")

    try:
        board = ROOT / "scripts" / "rebuild_section10_ae_grid_cnn_s2_board.py"
        if board.exists():
            subprocess.check_call(
                [
                    os.sys.executable,
                    str(board),
                    "--metrics",
                    str(run_dir / "artifacts" / "section10_ae_grid_cnn_s2_metrics.json"),
                ],
                cwd=str(ROOT),
            )
            log(run_dir, "Section 10 canvas board injected")
    except Exception as e:
        log(run_dir, f"board inject skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
