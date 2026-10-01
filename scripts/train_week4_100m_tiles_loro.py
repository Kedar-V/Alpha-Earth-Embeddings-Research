#!/usr/bin/env python3
"""LORO train on 100 m tiles with 10×10 AE cell grids as input.

Input per tile: up to 10×10 AlphaEarth 10 m cells (64-D each), missing slots
zero-filled, plus a 10×10 occupancy mask → 6500-D feature vector.

Targets (tile-level):
  - Ridge multi-output on (frac_res, frac_nres)
  - LogisticRegression on y ∈ {0=NRES, 1=RES}

Primary set: datasets/week4_100m_tiles_balanced/tiles_balanced_min5.parquet
Also supports --min-cells 20.

Writes outputs/week4_100m_tile_train/.

Usage:
  .venv/bin/python -u scripts/train_week4_100m_tiles_loro.py
  .venv/bin/python -u scripts/train_week4_100m_tiles_loro.py --min-cells 20
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
BAL_DIR = ROOT / "datasets" / "week4_100m_tiles_balanced"
OUT = ROOT / "outputs" / "week4_100m_tile_train"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"
POOL_BASELINE = ROOT / "outputs" / "week4_100m_tiles_balanced" / "f1_best_summary.csv"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
TILE_M = 100.0
CELL_M = 10.0
GRID = 10  # 10×10
FEAT_DIM = GRID * GRID * 64 + GRID * GRID  # emb + mask


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def _tile_origin(tile_id: str) -> tuple[int, int]:
    """Parse tile_id '{region}|{tile_i}|{tile_j}' → (tile_i, tile_j)."""
    parts = str(tile_id).split("|")
    return int(parts[1]), int(parts[2])


def _build_one_chunk(payload: dict) -> str:
    """Worker: build 10×10×64 (+ mask) features for a list of tile_ids → .npy path."""
    tile_ids = payload["tile_ids"]
    cells = pd.DataFrame(payload["cells"])
    bands = np.asarray(payload["bands"], dtype=np.float32)  # aligned to cells rows
    out_path = Path(payload["out_path"])
    n = len(tile_ids)
    X = np.zeros((n, FEAT_DIM), dtype=np.float32)
    id_to_pos = {tid: i for i, tid in enumerate(tile_ids)}

    # group cells by tile
    cells = cells.copy()
    cells["_row"] = np.arange(len(cells))
    for tid, g in cells.groupby("tile_id", sort=False):
        if tid not in id_to_pos:
            continue
        ti, tj = _tile_origin(tid)
        west = ti * TILE_M
        south = tj * TILE_M
        ux = g["utm_x"].to_numpy(dtype=np.float64)
        uy = g["utm_y"].to_numpy(dtype=np.float64)
        # j: west→east; i: north→south (i=0 at north edge of tile)
        j = np.floor((ux - west) / CELL_M).astype(np.int64)
        i = np.floor(((south + TILE_M) - uy) / CELL_M).astype(np.int64)
        ok = (i >= 0) & (i < GRID) & (j >= 0) & (j < GRID)
        if not ok.any():
            continue
        i, j = i[ok], j[ok]
        rows = g["_row"].to_numpy()[ok]
        emb = bands[rows]  # (k, 64)
        pos = id_to_pos[tid]
        grid = X[pos, : GRID * GRID * 64].reshape(GRID, GRID, 64)
        mask = X[pos, GRID * GRID * 64 :].reshape(GRID, GRID)
        # last write wins on rare double-hits
        grid[i, j, :] = emb
        mask[i, j] = 1.0
        X[pos, : GRID * GRID * 64] = grid.reshape(-1)
        X[pos, GRID * GRID * 64 :] = mask.reshape(-1)

    np.save(out_path, X)
    return str(out_path)


def build_tile_features(
    tiles: pd.DataFrame, workers: int, force: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X [n, FEAT_DIM], tile_ids) for the given tile table."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"min{int(tiles['min_cells'].iloc[0])}_n{len(tiles)}_seed{SEED}"
    x_path = CACHE / f"X_grid_{tag}.npy"
    id_path = CACHE / f"tile_ids_{tag}.npy"
    if x_path.exists() and id_path.exists() and not force:
        print(f"Loaded cached features {x_path}", flush=True)
        return np.load(x_path, mmap_mode="r"), np.load(id_path, allow_pickle=True)

    tile_ids = tiles["tile_id"].astype(str).to_numpy()
    need = set(tile_ids.tolist())
    print(f"Loading cells + AE bands for {len(need):,} tiles …", flush=True)
    t0 = time.time()
    cells = pd.read_parquet(CELLS, columns=["pixel_id", "tile_id", "utm_x", "utm_y"])
    cells["tile_id"] = cells["tile_id"].astype(str)
    cells = cells.loc[cells["tile_id"].isin(need)].reset_index(drop=True)
    pix = pd.read_parquet(PIXELS, columns=["pixel_id", *BANDS])
    pix["pixel_id"] = pix["pixel_id"].astype(str)
    cells["pixel_id"] = cells["pixel_id"].astype(str)
    m = cells.merge(pix, on="pixel_id", how="inner")
    print(
        f"  joined cells={len(m):,} in {time.time() - t0:.1f}s · building grids "
        f"with {workers} workers …",
        flush=True,
    )
    bands = m[BANDS].to_numpy(dtype=np.float32)
    meta = m[["tile_id", "utm_x", "utm_y"]].copy()

    # shard tiles across workers
    splits = np.array_split(tile_ids, workers)
    chunk_dir = CACHE / f"_chunks_{tag}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for k, split in enumerate(splits):
        if len(split) == 0:
            continue
        sset = set(split.tolist())
        mask = meta["tile_id"].isin(sset)
        payloads.append(
            {
                "tile_ids": split.tolist(),
                "cells": meta.loc[mask].to_dict(orient="list"),
                "bands": bands[mask.to_numpy()],
                "out_path": str(chunk_dir / f"chunk_{k:03d}.npy"),
            }
        )

    chunk_paths: list[str] = [""] * len(payloads)
    # spawn: parent may already have initialized CUDA before this call
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = {pool.submit(_build_one_chunk, p): i for i, p in enumerate(payloads)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="build 10x10 grids"):
            i = futs[fut]
            chunk_paths[i] = fut.result()

    parts = [np.load(p) for p in chunk_paths]
    X = np.concatenate(parts, axis=0)
    # chunks were split in order of tile_ids splits → concat matches tile_ids order
    np.save(x_path, X)
    np.save(id_path, tile_ids)
    for p in chunk_paths:
        Path(p).unlink(missing_ok=True)
    print(f"  wrote {x_path} shape={X.shape}", flush=True)
    return X, tile_ids


def _fit_fold_cpu(payload: tuple) -> dict:
    """Sklearn fold worker (CPU)."""
    te_region, x_path, y_path, ybin_path, reg_path, cache_dir, ckpt_dir = payload
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = np.asarray(X[tr], dtype=np.float64)
    Y_tr = np.asarray(Y[tr], dtype=np.float64)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float64)
    y_te = np.asarray(Y[te], dtype=np.float64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(Ridge(alpha=1.0, random_state=SEED), n_jobs=1)),
        ]
    )
    reg.fit(X_tr, Y_tr)
    pred = np.clip(reg.predict(X_te), 0.0, 1.0)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    p_res = clf.predict_proba(X_te)[:, 1]

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"oof_pred_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    ypred_path = cache / f"oof_ypred_{te_region}.npy"
    pres_path = cache / f"oof_pres_{te_region}.npy"
    np.save(pred_path, pred.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))
    np.save(ypred_path, y_pred.astype(np.int64))
    np.save(pres_path, p_res.astype(np.float64))
    joblib.dump(
        {
            "region_held_out": te_region,
            "regressor": reg,
            "classifier": clf,
            "seed": SEED,
            "feat": "10x10_ae_grid_flat+mask",
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(Y_tr.shape[1])]
    return {
        "region": te_region,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "y_pred_path": str(ypred_path),
        "p_res_path": str(pres_path),
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
        "n_train_res": int((y_tr == 1).sum()),
        "n_train_nres": int((y_tr == 0).sum()),
        "mae_frac_res": mae[0],
        "mae_frac_nres": mae[1],
        "mae_mean": float(np.mean(mae)),
        "macro_f1": float(f1_score(yb_te, y_pred, average="macro", zero_division=0)),
        "device": "cpu",
    }


def _fit_fold_gpu(payload: tuple) -> dict:
    """Torch fold worker bound to one CUDA device (Ridge closed-form + LR LBFGS)."""
    te_region, x_path, y_path, ybin_path, reg_path, cache_dir, ckpt_dir, device_id = payload
    device = torch.device(f"cuda:{device_id}")
    X_np = np.load(x_path, mmap_mode="r")
    Y_np = np.load(y_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = torch.as_tensor(np.asarray(X_np[tr], dtype=np.float32), device=device)
    Y_tr = torch.as_tensor(np.asarray(Y_np[tr], dtype=np.float32), device=device)
    y_tr = torch.as_tensor(np.asarray(y_bin[tr], dtype=np.float32), device=device)
    X_te = torch.as_tensor(np.asarray(X_np[te], dtype=np.float32), device=device)
    y_te = np.asarray(Y_np[te], dtype=np.float64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    # Standardize using train stats
    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    X_tr_s = (X_tr - mu) / sd
    X_te_s = (X_te - mu) / sd

    # Ridge closed form for each target: beta = (X'X + aI)^{-1} X'y
    # add bias column
    ones_tr = torch.ones((X_tr_s.shape[0], 1), device=device)
    ones_te = torch.ones((X_te_s.shape[0], 1), device=device)
    Phi_tr = torch.cat([X_tr_s, ones_tr], dim=1)
    Phi_te = torch.cat([X_te_s, ones_te], dim=1)
    alpha = 1.0
    xtx = Phi_tr.T @ Phi_tr
    xtx = xtx + alpha * torch.eye(xtx.shape[0], device=device)
    xty = Phi_tr.T @ Y_tr
    beta = torch.linalg.solve(xtx, xty)  # (D+1, 2)
    pred_t = torch.clamp(Phi_te @ beta, 0.0, 1.0)
    pred = pred_t.detach().cpu().numpy().astype(np.float64)

    # Logistic regression via LBFGS on GPU
    D = X_tr_s.shape[1]
    w = torch.zeros(D, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = X_tr_s @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tr)
        # L2 like sklearn C default ~1.0 → weight decay ~1/C / (2n) ; use small wd
        loss = loss + 0.5 * 1e-2 * (w * w).sum() / X_tr_s.shape[0]
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    opt.step(closure)
    with torch.no_grad():
        logits_te = X_te_s @ w + b
        p_res = torch.sigmoid(logits_te).detach().cpu().numpy().astype(np.float64)
    y_pred = (p_res >= 0.5).astype(np.int64)

    cache = Path(cache_dir)
    ckpt = Path(ckpt_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"oof_pred_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    ypred_path = cache / f"oof_ypred_{te_region}.npy"
    pres_path = cache / f"oof_pres_{te_region}.npy"
    np.save(pred_path, pred)
    np.save(idx_path, np.where(te)[0].astype(np.int64))
    np.save(ypred_path, y_pred)
    np.save(pres_path, p_res)
    # save tensors as numpy for reload
    joblib.dump(
        {
            "region_held_out": te_region,
            "mu": mu.detach().cpu().numpy(),
            "sd": sd.detach().cpu().numpy(),
            "ridge_beta": beta.detach().cpu().numpy(),
            "lr_w": w.detach().cpu().numpy(),
            "lr_b": float(b.detach().cpu().item()),
            "seed": SEED,
            "feat": "10x10_ae_grid_flat+mask",
            "backend": "torch_gpu",
            "device_id": device_id,
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(pred.shape[1])]
    return {
        "region": te_region,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "y_pred_path": str(ypred_path),
        "p_res_path": str(pres_path),
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
        "n_train_res": int(int((y_bin[tr] == 1).sum())),
        "n_train_nres": int(int((y_bin[tr] == 0).sum())),
        "mae_frac_res": mae[0],
        "mae_frac_nres": mae[1],
        "mae_mean": float(np.mean(mae)),
        "macro_f1": float(f1_score(yb_te, y_pred, average="macro", zero_division=0)),
        "device": f"cuda:{device_id}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-cells", type=int, default=5, choices=[5, 20])
    ap.add_argument("--workers", type=int, default=None, help="CPU workers for grid build")
    ap.add_argument(
        "--backend",
        choices=["gpu", "cpu", "auto"],
        default="auto",
        help="Fold training backend (default: gpu if CUDA else cpu)",
    )
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT):
        d.mkdir(parents=True, exist_ok=True)

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
        print("No CUDA devices; falling back to CPU sklearn", flush=True)
        backend = "cpu"

    bal_path = BAL_DIR / f"tiles_balanced_min{args.min_cells}.parquet"
    if not bal_path.exists():
        raise FileNotFoundError(bal_path)
    tiles = pd.read_parquet(bal_path).reset_index(drop=True)
    print(
        f"Tiles {bal_path.name}: n={len(tiles):,}  "
        f"RES={(tiles.y==1).sum():,} NRES={(tiles.y==0).sum():,}",
        flush=True,
    )

    workers = args.workers or n_cpu
    X, tile_ids = build_tile_features(tiles, workers=workers, force=args.force_features)
    # align tile table to feature order
    tiles = tiles.set_index("tile_id").loc[tile_ids].reset_index()
    assert list(tiles["tile_id"].astype(str)) == list(tile_ids.astype(str))

    Y = tiles[["frac_res", "frac_nres"]].to_numpy(dtype=np.float32)
    y_bin = tiles["y"].to_numpy(dtype=np.int64)
    regions = tiles["region"].astype(str).to_numpy()
    region_list = sorted(tiles["region"].astype(str).unique().tolist())

    tag = f"min{args.min_cells}"
    x_path = str(CACHE / f"X_grid_{tag}_aligned.npy")
    y_path = str(CACHE / f"Y_frac_{tag}.npy")
    ybin_path = str(CACHE / f"y_bin_{tag}.npy")
    reg_path = str(CACHE / f"regions_{tag}.npy")
    # materialize X if mmap
    X_arr = np.asarray(X, dtype=np.float32)
    np.save(x_path, X_arr)
    np.save(y_path, Y)
    np.save(ybin_path, y_bin)
    np.save(reg_path, regions)

    fold_cache = CACHE / f"folds_{tag}"
    fold_ckpt = CKPT / tag
    fold_cache.mkdir(parents=True, exist_ok=True)
    fold_ckpt.mkdir(parents=True, exist_ok=True)

    print(
        f"LORO · backend={backend} · folds={len(region_list)} · "
        f"feat_dim={X_arr.shape[1]} · n={len(tiles):,}",
        flush=True,
    )

    rows = []
    oof_pred = np.full((len(tiles), 2), np.nan, dtype=np.float64)
    oof_ypred = np.full(len(tiles), -1, dtype=np.int64)
    oof_pres = np.full(len(tiles), np.nan, dtype=np.float64)

    if backend == "gpu":
        # one process per GPU; queue folds across GPUs
        max_workers = n_gpu
        jobs = []
        for i, r in enumerate(region_list):
            jobs.append(
                (
                    r,
                    x_path,
                    y_path,
                    ybin_path,
                    reg_path,
                    str(fold_cache),
                    str(fold_ckpt),
                    i % n_gpu,
                )
            )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            futs = {pool.submit(_fit_fold_gpu, job): job[0] for job in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO GPU folds"):
                res = fut.result()
                rows.append(res)
                te = np.load(res["idx_path"])
                oof_pred[te] = np.load(res["pred_path"])
                oof_ypred[te] = np.load(res["y_pred_path"])
                oof_pres[te] = np.load(res["p_res_path"])
                print(
                    f"  {res['region']:12s}  [{res['device']}]  "
                    f"mae={res['mae_mean']:.3f}  F1={res['macro_f1']:.3f}",
                    flush=True,
                )
    else:
        jobs = [
            (r, x_path, y_path, ybin_path, reg_path, str(fold_cache), str(fold_ckpt))
            for r in region_list
        ]
        with ProcessPoolExecutor(max_workers=min(len(jobs), n_cpu)) as pool:
            futs = {pool.submit(_fit_fold_cpu, job): job[0] for job in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO CPU folds"):
                res = fut.result()
                rows.append(res)
                te = np.load(res["idx_path"])
                oof_pred[te] = np.load(res["pred_path"])
                oof_ypred[te] = np.load(res["y_pred_path"])
                oof_pres[te] = np.load(res["p_res_path"])
                print(
                    f"  {res['region']:12s}  mae={res['mae_mean']:.3f}  F1={res['macro_f1']:.3f}",
                    flush=True,
                )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / f"loro_metrics_min{args.min_cells}.csv", index=False)

    oof = tiles[
        ["tile_id", "region", "y", "frac_res", "frac_nres", "frac_empty", "n_cells"]
    ].copy()
    oof["pred_frac_res"] = oof_pred[:, 0]
    oof["pred_frac_nres"] = oof_pred[:, 1]
    oof["y_pred"] = oof_ypred
    oof["p_res"] = oof_pres
    oof_path = OUT / f"oof_predictions_min{args.min_cells}.parquet"
    oof.to_parquet(oof_path, index=False)

    mae_res = float(mean_absolute_error(Y[:, 0], oof_pred[:, 0]))
    mae_nres = float(mean_absolute_error(Y[:, 1], oof_pred[:, 1]))
    mae_mean = 0.5 * (mae_res + mae_nres)
    macro_f1 = float(f1_score(y_bin, oof_ypred, average="macro", zero_division=0))
    loro_f1_mean = float(metrics["macro_f1"].mean())
    loro_f1_std = float(metrics["macro_f1"].std(ddof=0))
    loro_mae_mean = float(metrics["mae_mean"].mean())
    loro_mae_std = float(metrics["mae_mean"].std(ddof=0))

    # --- figures ---
    ms = metrics.sort_values("macro_f1", ascending=True).copy()
    ms["lab"] = ms["region"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(ms["lab"], ms["macro_f1"], color="#0072B2", height=0.7)
    ax.axvline(loro_f1_mean, color="#333", ls="--", lw=1)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"Tile-direct LR · LORO F1={loro_f1_mean:.3f} ± {loro_f1_std:.3f} (min≥{args.min_cells})",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        "Input = 10×10 AE grid (flattened + occupancy). Held out one metro at a time.",
        fontsize=9, color="#555", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / f"loro_f1_by_region_min{args.min_cells}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    mm = metrics.sort_values("mae_mean", ascending=False).copy()
    mm["lab"] = mm["region"].str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(mm["lab"], mm["mae_mean"], color="#009E73", height=0.7)
    ax.axvline(loro_mae_mean, color="#333", ls="--", lw=1)
    ax.set_xlabel("Mean MAE (frac_res, frac_nres)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"Tile-direct Ridge · LORO MAE={loro_mae_mean:.3f} ± {loro_mae_std:.3f}",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / f"loro_mae_by_region_min{args.min_cells}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, i, name, c in zip(
        axes, range(2), ["Residential", "Non-residential"], ["#0072B2", "#E69F00"]
    ):
        ax.scatter(Y[:, i], oof_pred[:, i], s=4, alpha=0.12, c=c, rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel(f"True frac ({name.lower()})")
        ax.set_ylabel("Predicted")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"OOF fraction preds · mean MAE={mae_mean:.3f}",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / f"oof_true_vs_pred_min{args.min_cells}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_bin, oof_ypred, display_labels=["NRES", "RES"], ax=ax, colorbar=False, cmap="Blues"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"Tile-direct LR confusion · macro-F1={macro_f1:.3f}",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / f"oof_confusion_min{args.min_cells}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # vs cell-pool baseline
    pool_f1 = None
    if POOL_BASELINE.exists():
        base = pd.read_csv(POOL_BASELINE)
        hit = base.loc[
            (base["min_cells"] == args.min_cells)
            & (base["model"] == "balanced_train_full")
        ]
        if len(hit):
            pool_f1 = float(hit.iloc[0]["f1_at_0.5"])

    summary = {
        "name": "week4_100m_tile_direct_loro_v1",
        "input": "10x10 AE cell grid flattened + occupancy mask (6500-D)",
        "targets": {"regression": ["frac_res", "frac_nres"], "classification": "y RES vs NRES"},
        "min_cells": args.min_cells,
        "backend": backend,
        "n_cpus": n_cpu,
        "n_gpus": n_gpu,
        "n_tiles": int(len(tiles)),
        "n_res": int((y_bin == 1).sum()),
        "n_nres": int((y_bin == 0).sum()),
        "feat_dim": int(X_arr.shape[1]),
        "oof_macro_f1": macro_f1,
        "loro_macro_f1_mean": loro_f1_mean,
        "loro_macro_f1_std": loro_f1_std,
        "oof_mae_frac_res": mae_res,
        "oof_mae_frac_nres": mae_nres,
        "oof_mae_mean": mae_mean,
        "loro_mae_mean_mean": loro_mae_mean,
        "loro_mae_mean_std": loro_mae_std,
        "cell_pool_baseline_f1_at_0.5": pool_f1,
        "delta_f1_vs_cell_pool": (macro_f1 - pool_f1) if pool_f1 is not None else None,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "oof": str(oof_path.relative_to(ROOT)),
            "metrics": f"outputs/week4_100m_tile_train/loro_metrics_min{args.min_cells}.csv",
            "figures": str(FIG.relative_to(ROOT)),
            "checkpoints": str(fold_ckpt.relative_to(ROOT)),
        },
    }
    (OUT / f"summary_min{args.min_cells}.json").write_text(json.dumps(summary, indent=2) + "\n")
    # also write/update aggregate summary
    agg_path = OUT / "summary.json"
    agg = jload_safe(agg_path)
    agg[f"min{args.min_cells}"] = summary
    agg_path.write_text(json.dumps(agg, indent=2) + "\n")

    print(
        f"\nOOF F1={macro_f1:.4f}  MAE={mae_mean:.4f}  "
        f"vs cell-pool F1@0.50={pool_f1}  "
        f"elapsed={summary['elapsed_s']}s",
        flush=True,
    )


def jload_safe(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text())
    return {}


if __name__ == "__main__":
    main()
