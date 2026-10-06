#!/usr/bin/env python3
"""Week 4 LORO: PLS 10 m pixels vs 100 m mean-pooled tiles.

Same protocol at both scales (fair 64-D comparison):
  - baseline LR on scaled A00–A63
  - PCA → LR
  - PLS-DA → LR (k sweep including k=1)
  - PLS fraction head (X → frac_*)
  - L1 logistic on k=32 PLS scores (C=10, 30) — sparsity story

10 m: balanced pure cells (datasets/week4_pixel_gt_balanced).
100 m: balanced tiles min5; features = mean AE embedding over cells in tile.

Writes outputs/week4_pls_10m_vs_100m/ (checkpoints, OOF, sweeps, figures, summary).
Build notebook: scripts/build_week4_pls_10m_vs_100m_notebook.py

Usage:
  .venv/bin/python -u scripts/train_week4_pls_10m_vs_100m.py --backend gpu
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
from sklearn.metrics import f1_score, mean_absolute_error
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_week4_pls_loro import (  # noqa: E402
    _torch_logistic,
    _torch_pca_fit_transform,
    _torch_pls_nipals,
    _torch_pls_predict,
    _torch_ridge,
    _torch_standardize,
)
from scripts.train_week4_l1_pls_loro import _torch_logistic_l1  # noqa: E402

PIXELS_10M = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
PIXELS_ALL = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
TILES_100M = ROOT / "datasets" / "week4_100m_tiles_balanced" / "tiles_balanced_min5.parquet"
OUT = ROOT / "outputs" / "week4_pls_10m_vs_100m"
FIG = OUT / "figures"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
N_COMPS = (1, 2, 4, 8, 12, 16, 24, 32)
N_COMP_MAX = max(N_COMPS)
L1_CS = (10.0, 30.0)
NZ_EPS = 1e-6
FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")


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


def load_10m() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    print(f"Loading 10 m {PIXELS_10M} …", flush=True)
    df = pd.read_parquet(PIXELS_10M).reset_index(drop=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    # prefer full-precision fracs when present
    fr = []
    for c in FRAC_COLS:
        full = f"{c}_full"
        fr.append(df[full].to_numpy(dtype=np.float32) if full in df.columns else df[c].to_numpy(dtype=np.float32))
    Y = np.stack(fr, axis=1)
    y = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].astype(str).to_numpy()
    meta = df[["region"]].copy()
    meta["unit_id"] = df["pixel_id"].astype(str) if "pixel_id" in df.columns else np.arange(len(df)).astype(str)
    return X, Y, y, regions, meta


def build_100m_mean_pool(force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cache = OUT / "_cache" / "100m"
    cache.mkdir(parents=True, exist_ok=True)
    x_p = cache / "X_mean64.npy"
    y_p = cache / "Y_frac.npy"
    yb_p = cache / "y_bin.npy"
    r_p = cache / "regions.npy"
    m_p = cache / "meta.parquet"
    if all(p.exists() for p in (x_p, y_p, yb_p, r_p, m_p)) and not force:
        print(f"Loaded cached 100 m mean-pool from {cache}", flush=True)
        meta = pd.read_parquet(m_p)
        return (
            np.load(x_p),
            np.load(y_p),
            np.load(yb_p),
            np.load(r_p, allow_pickle=True),
            meta,
        )

    print(f"Building 100 m mean-pooled AE from {TILES_100M} …", flush=True)
    t0 = time.time()
    tiles = pd.read_parquet(TILES_100M).reset_index(drop=True)
    tiles["tile_id"] = tiles["tile_id"].astype(str)
    need = set(tiles["tile_id"].tolist())

    cells = pd.read_parquet(CELLS, columns=["pixel_id", "tile_id"])
    cells["tile_id"] = cells["tile_id"].astype(str)
    cells["pixel_id"] = cells["pixel_id"].astype(str)
    cells = cells.loc[cells["tile_id"].isin(need)].reset_index(drop=True)

    pix = pd.read_parquet(PIXELS_ALL, columns=["pixel_id", *BANDS])
    pix["pixel_id"] = pix["pixel_id"].astype(str)
    m = cells.merge(pix, on="pixel_id", how="inner")
    print(f"  cells joined: {len(m):,} in {time.time() - t0:.1f}s", flush=True)

    means = m.groupby("tile_id", sort=False)[BANDS].mean()
    tiles = tiles.set_index("tile_id")
    # align to tile order
    means = means.reindex(tiles.index)
    if means.isna().any().any():
        n_bad = int(means.isna().any(axis=1).sum())
        raise RuntimeError(f"{n_bad} tiles missing AE means")

    X = means[BANDS].to_numpy(dtype=np.float32)
    Y = tiles[list(FRAC_COLS)].to_numpy(dtype=np.float32)
    y = tiles["y"].to_numpy(dtype=np.int64)
    regions = tiles["region"].astype(str).to_numpy()
    meta = pd.DataFrame(
        {
            "unit_id": tiles.index.astype(str).to_numpy(),
            "region": regions,
            "n_cells": tiles["n_cells"].to_numpy() if "n_cells" in tiles.columns else -1,
        }
    )

    np.save(x_p, X)
    np.save(y_p, Y)
    np.save(yb_p, y)
    np.save(r_p, regions)
    meta.to_parquet(m_p, index=False)
    print(f"  100 m n={len(X):,} built in {time.time() - t0:.1f}s", flush=True)
    return X, Y, y, regions, meta


# ---------------------------------------------------------------------------
# Fold worker
# ---------------------------------------------------------------------------


def _fit_fold(payload: tuple) -> dict:
    (
        scale,
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

    # baselines
    y_pred_lr, p_res_lr, w_lr, b_lr = _torch_logistic(X_tr_s, y_tr, X_te_s)
    pred_ridge = _torch_ridge(X_tr_s, Y_tr, X_te_s, alpha=1.0)
    mae_r = [float(mean_absolute_error(Y_te[:, i], pred_ridge[:, i])) for i in range(3)]
    rows.append(
        {
            "scale": scale,
            "region": te_region,
            "model": "baseline_lr",
            "n_components": 64,
            "C": None,
            "macro_f1": float(f1_score(yb_te, y_pred_lr, average="macro", zero_division=0)),
            "mae_mean": float("nan"),
            "nnz": int((w_lr.abs() > NZ_EPS).sum().item()),
        }
    )
    rows.append(
        {
            "scale": scale,
            "region": te_region,
            "model": "baseline_ridge",
            "n_components": 64,
            "C": None,
            "macro_f1": float("nan"),
            "mae_mean": float(np.mean(mae_r)),
            "nnz": None,
        }
    )

    Y_mu = Y_tr.mean(dim=0, keepdim=True)
    Y_tr_c = Y_tr - Y_mu
    y_mu = y_tr.mean()
    y_tr_c = (y_tr - y_mu).unsqueeze(1)

    pls_da = _torch_pls_nipals(X_tr_s, y_tr_c, n_comp_max)
    pls_frac = _torch_pls_nipals(X_tr_s, Y_tr_c, n_comp_max)
    _, _, pca_comp_max = _torch_pca_fit_transform(X_tr_s, X_te_s, n_comp_max)

    oof_pack: dict = {
        "baseline_lr": {"y_pred": y_pred_lr, "p_res": p_res_lr},
        "baseline_ridge": {"frac": pred_ridge},
        "pls_da_lr": {},
        "pls_frac": {},
        "pca_lr": {},
        "l1_pls": {},
    }

    # score stats for k_max (used by L1 + equation)
    R_full = pls_da["x_rotations"]
    T_tr_full = X_tr_s @ R_full
    T_te_full = X_te_s @ R_full
    tmu_full = T_tr_full.mean(0, keepdim=True)
    tsd_full = T_tr_full.std(0, keepdim=True).clamp_min(1e-6)
    Z_tr = (T_tr_full - tmu_full) / tsd_full
    Z_te = (T_te_full - tmu_full) / tsd_full

    for k in list(n_comps):
        R_da = R_full[:, :k]
        T_tr_da = X_tr_s @ R_da
        T_te_da = X_te_s @ R_da
        tmu = T_tr_da.mean(0, keepdim=True)
        tsd = T_tr_da.std(0, keepdim=True).clamp_min(1e-6)
        yhat, p_res, _, _ = _torch_logistic((T_tr_da - tmu) / tsd, y_tr, (T_te_da - tmu) / tsd)
        oof_pack["pls_da_lr"][k] = {"y_pred": yhat, "p_res": p_res}
        rows.append(
            {
                "scale": scale,
                "region": te_region,
                "model": "pls_da_lr",
                "n_components": int(k),
                "C": None,
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "nnz": int(k),
            }
        )

        pred_pls = _torch_pls_predict(X_te_s, Y_mu, pls_frac, k)
        pred_pls = torch.clamp(pred_pls, 0.0, 1.0).detach().cpu().numpy().astype(np.float64)
        mae = [float(mean_absolute_error(Y_te[:, i], pred_pls[:, i])) for i in range(3)]
        oof_pack["pls_frac"][k] = {"frac": pred_pls}
        rows.append(
            {
                "scale": scale,
                "region": te_region,
                "model": "pls_frac",
                "n_components": int(k),
                "C": None,
                "macro_f1": float("nan"),
                "mae_mean": float(np.mean(mae)),
                "nnz": int(k),
            }
        )

        comps = pca_comp_max[:, :k]
        T_tr_p = X_tr_s @ comps
        T_te_p = X_te_s @ comps
        tmu = T_tr_p.mean(0, keepdim=True)
        tsd = T_tr_p.std(0, keepdim=True).clamp_min(1e-6)
        yhat3, p3, _, _ = _torch_logistic((T_tr_p - tmu) / tsd, y_tr, (T_te_p - tmu) / tsd)
        oof_pack["pca_lr"][k] = {"y_pred": yhat3, "p_res": p3}
        rows.append(
            {
                "scale": scale,
                "region": te_region,
                "model": "pca_lr",
                "n_components": int(k),
                "C": None,
                "macro_f1": float(f1_score(yb_te, yhat3, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "nnz": int(k),
            }
        )

    # L1 on full k_max PLS scores
    l1_coef_rows = []
    for C in L1_CS:
        yhat, p, coef_np, intercept, lam = _torch_logistic_l1(Z_tr, y_tr, Z_te, C=float(C))
        coef_np = np.asarray(coef_np, dtype=np.float64)
        nnz = int(np.sum(np.abs(coef_np) > NZ_EPS))
        oof_pack["l1_pls"][C] = {"y_pred": yhat, "p_res": p, "coef": coef_np, "intercept": float(intercept)}
        rows.append(
            {
                "scale": scale,
                "region": te_region,
                "model": "l1_pls",
                "n_components": n_comp_max,
                "C": float(C),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
                "mae_mean": float("nan"),
                "nnz": nnz,
            }
        )
        for j, w in enumerate(coef_np):
            if abs(w) > NZ_EPS:
                l1_coef_rows.append(
                    {
                        "scale": scale,
                        "region": te_region,
                        "C": float(C),
                        "component": j + 1,
                        "coef_z": float(w),
                    }
                )
        l1_coef_rows.append(
            {
                "scale": scale,
                "region": te_region,
                "C": float(C),
                "component": 0,
                "coef_z": float(intercept),
            }
        )

    # t1-only head (explicit)
    yhat1, p1, w1, b1 = _torch_logistic(Z_tr[:, :1], y_tr, Z_te[:, :1])
    oof_pack["t1_only"] = {"y_pred": yhat1, "p_res": p1}
    rows.append(
        {
            "scale": scale,
            "region": te_region,
            "model": "t1_only",
            "n_components": 1,
            "C": None,
            "macro_f1": float(f1_score(yb_te, yhat1, average="macro", zero_division=0)),
            "mae_mean": float("nan"),
            "nnz": 1,
        }
    )

    ckpt_obj = {
        "scale": scale,
        "region_held_out": te_region,
        "device_id": device_id,
        "mu": mu.detach().cpu().numpy(),
        "sd": sd.detach().cpu().numpy(),
        "pls_da_rotations": R_full.detach().cpu().numpy(),
        "pls_da_loadings": pls_da["x_loadings"].detach().cpu().numpy(),
        "score_mu": tmu_full.detach().cpu().numpy().ravel(),
        "score_sd": tsd_full.detach().cpu().numpy().ravel(),
        "baseline_lr_w": w_lr.cpu().numpy(),
        "baseline_lr_b": float(b_lr.cpu().item()),
        "t1_only_w": float(w1.detach().cpu().item()),
        "t1_only_b": float(b1.detach().cpu().item()),
        "n_comps": list(n_comps),
        "seed": SEED,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }
    joblib.dump(ckpt_obj, ckpt / f"fold_{te_region}.joblib")
    joblib.dump(oof_pack, cache / f"oof_pack_{te_region}.joblib")
    np.save(cache / f"oof_idx_{te_region}.npy", te_idx)
    if l1_coef_rows:
        pd.DataFrame(l1_coef_rows).to_csv(cache / f"l1_coefs_{te_region}.csv", index=False)

    del X_tr, X_te, Y_tr, X_tr_s, X_te_s, pls_da, pls_frac, Z_tr, Z_te
    torch.cuda.empty_cache()

    return {
        "scale": scale,
        "region": te_region,
        "device": f"cuda:{device_id}",
        "rows": rows,
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
    }


# ---------------------------------------------------------------------------
# Aggregate + plots
# ---------------------------------------------------------------------------


def _agg_sweep(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scale", "model", "n_components", "C"]
    for key, g in metrics.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        rec["n_folds"] = len(g)
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
        if g["nnz"].notna().any():
            rec["mean_nnz"] = float(g["nnz"].mean())
        else:
            rec["mean_nnz"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _claim_header(fig, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(0.01, 0.98, title, fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure)
    fig.text(0.01, 0.925, subtitle, fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure)


def _plot_all(sweep: pd.DataFrame, metrics: pd.DataFrame, fig_dir: Path, summary: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"10m": "#0072B2", "100m": "#D55E00"}

    # 1) F1 vs k both scales
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for scale, c in colors.items():
        sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "pls_da_lr")].sort_values("n_components")
        ax.plot(sub["n_components"], sub["loro_macro_f1_mean"], marker="o", color=c, lw=1.8, label=f"PLS-DA {scale}")
        ax.fill_between(
            sub["n_components"],
            sub["loro_macro_f1_mean"] - sub["loro_macro_f1_std"],
            sub["loro_macro_f1_mean"] + sub["loro_macro_f1_std"],
            color=c,
            alpha=0.12,
        )
        base = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "baseline_lr")]
        if not base.empty:
            ax.axhline(
                float(base["loro_macro_f1_mean"].iloc[0]),
                color=c,
                ls="--",
                lw=1.2,
                alpha=0.8,
            )
    # annotate baselines in subtitle; direct labels for end points
    for scale, c in colors.items():
        sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "pls_da_lr")].sort_values("n_components")
        if sub.empty:
            continue
        x = float(sub["n_components"].iloc[-1])
        y = float(sub["loro_macro_f1_mean"].iloc[-1])
        ax.text(x + 0.4, y, f"{scale} {y:.3f}", color=c, fontsize=9, va="center")
        k1 = sub.loc[sub["n_components"] == 1]
        if not k1.empty:
            ax.text(
                1.15,
                float(k1["loro_macro_f1_mean"].iloc[0]),
                f"k=1 {float(k1['loro_macro_f1_mean'].iloc[0]):.3f}",
                color=c,
                fontsize=8,
                va="bottom",
            )
    ax.set_xlabel("PLS components (k)")
    ax.set_ylabel("LORO mean macro-F1")
    ax.set_ylim(0.55, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    b10 = float(summary["10m"]["baseline_lr"]["loro_macro_f1_mean"])
    b100 = float(summary["100m"]["baseline_lr"]["loro_macro_f1_mean"])
    _claim_header(
        fig,
        "10 m PLS stays near full LR; 100 m mean-pool is lower across every k",
        f"Dashed = 64-D baseline LR (10 m {b10:.3f}, 100 m {b100:.3f}). Shaded = ± metro std. Balanced RES/NRES.",
    )
    fig.savefig(fig_dir / "f1_vs_k_by_scale.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) Best-family comparison bars (sorted by 10m)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    families = [
        ("baseline_lr", "64-D LR"),
        ("pls_da_best", "PLS-DA best-k"),
        ("t1_only", "t₁ only"),
        ("l1_c10", "L1 on PLS (C=10)"),
        ("l1_c30", "L1 on PLS (C=30)"),
        ("pca_best", "PCA best-k"),
    ]
    vals_10, vals_100, labels = [], [], []
    for key, lab in families:
        v10 = summary["10m"].get(key, {}).get("loro_macro_f1_mean")
        v100 = summary["100m"].get(key, {}).get("loro_macro_f1_mean")
        if v10 is None and v100 is None:
            continue
        labels.append(lab)
        vals_10.append(float(v10) if v10 is not None else np.nan)
        vals_100.append(float(v100) if v100 is not None else np.nan)
    # sort by 10m descending
    order = np.argsort(vals_10)[::-1]
    labels = [labels[i] for i in order]
    vals_10 = [vals_10[i] for i in order]
    vals_100 = [vals_100[i] for i in order]
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, vals_10, height=h, color=colors["10m"], label="10 m cells")
    ax.barh(y - h / 2, vals_100, height=h, color=colors["100m"], label="100 m tiles")
    for yi, a, b in zip(y, vals_10, vals_100):
        if a == a:
            ax.text(a + 0.008, yi + h / 2, f"{a:.3f}", va="center", fontsize=8, color=colors["10m"])
        if b == b:
            ax.text(b + 0.008, yi - h / 2, f"{b:.3f}", va="center", fontsize=8, color=colors["100m"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LORO mean macro-F1")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    # color words in title instead of legend box when possible
    _claim_header(
        fig,
        "Every head is weaker at 100 m; t₁ still carries most of the signal",
        "Blue = 10 m pure cells. Orange = 100 m mean-pooled AE. Same LORO metros.",
    )
    fig.savefig(fig_dir / "f1_family_10m_vs_100m.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) Per-metro delta (best PLS-DA 10m − 100m)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    m10 = metrics.loc[(metrics["scale"] == "10m") & (metrics["model"] == "pls_da_lr")]
    m100 = metrics.loc[(metrics["scale"] == "100m") & (metrics["model"] == "pls_da_lr")]
    k10 = int(summary["10m"]["pls_da_best"]["n_components"])
    k100 = int(summary["100m"]["pls_da_best"]["n_components"])
    a = m10.loc[m10["n_components"] == k10, ["region", "macro_f1"]].rename(columns={"macro_f1": "f1_10m"})
    b = m100.loc[m100["n_components"] == k100, ["region", "macro_f1"]].rename(columns={"macro_f1": "f1_100m"})
    d = a.merge(b, on="region")
    d["delta"] = d["f1_10m"] - d["f1_100m"]
    d = d.sort_values("delta", ascending=True)
    y = np.arange(len(d))
    cols = ["#0072B2" if v >= 0 else "#D55E00" for v in d["delta"]]
    ax.barh(y, d["delta"], color=cols)
    ax.axvline(0, color="#333", lw=1)
    for yi, (_, row) in enumerate(d.iterrows()):
        ax.text(
            row["delta"] + (0.008 if row["delta"] >= 0 else -0.008),
            yi,
            f"{row['delta']:+.3f}",
            va="center",
            ha="left" if row["delta"] >= 0 else "right",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("_", " ") for r in d["region"]])
    ax.set_xlabel("macro-F1 (10 m best-k − 100 m best-k)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    mean_d = float(d["delta"].mean())
    _claim_header(
        fig,
        f"10 m beats 100 m in every metro (mean Δ F1 = {mean_d:+.3f})",
        f"Best PLS-DA k: 10 m = {k10}, 100 m = {k100}. Positive = 10 m higher.",
    )
    fig.savefig(fig_dir / "delta_f1_by_metro.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    d.to_csv(OUT / "delta_f1_by_metro.csv", index=False)

    # 4) L1 nnz vs scale
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for scale, c in colors.items():
        sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "l1_pls")].sort_values("C")
        ax.plot(sub["C"], sub["mean_nnz"], marker="o", color=c, lw=1.8, label=scale)
        for _, row in sub.iterrows():
            ax.text(row["C"] * 1.05, row["mean_nnz"] + 0.15, f"{row['mean_nnz']:.1f}", color=c, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("L1 C (smaller = stronger)")
    ax.set_ylabel("Mean nonzero PLS coefs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    _claim_header(
        fig,
        "Strong L1 collapses to ~t₁ at both scales",
        "Mean nnz over LORO folds on k=32 PLS scores. C=10 ≈ one surviving component.",
    )
    fig.savefig(fig_dir / "l1_nnz_by_scale.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 5) MAE vs k
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for scale, c in colors.items():
        sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "pls_frac")].sort_values("n_components")
        ax.plot(sub["n_components"], sub["loro_mae_mean"], marker="o", color=c, lw=1.8, label=scale)
        base = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "baseline_ridge")]
        if not base.empty:
            ax.axhline(float(base["loro_mae_mean"].iloc[0]), color=c, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("PLS components (k)")
    ax.set_ylabel("LORO mean MAE (3 fractions)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    _claim_header(
        fig,
        "Fraction MAE is higher at 100 m; PLS still matches Ridge with few components",
        "Dashed = 64-D Ridge. Targets = tile/cell (frac_res, frac_nres, frac_empty).",
    )
    fig.savefig(fig_dir / "mae_vs_k_by_scale.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _assemble_oof(scale: str, scale_dir: Path, y_bin: np.ndarray, regions: np.ndarray, meta: pd.DataFrame, best_k: int) -> pd.DataFrame:
    cache = scale_dir / "_cache"
    n = len(y_bin)
    out = {
        "scale": np.full(n, scale, dtype=object),
        "region": regions,
        "unit_id": meta["unit_id"].to_numpy(),
        "y_true": y_bin,
        "y_pred_baseline": np.full(n, -1, dtype=np.int64),
        "p_res_baseline": np.full(n, np.nan),
        "y_pred_pls": np.full(n, -1, dtype=np.int64),
        "p_res_pls": np.full(n, np.nan),
        "y_pred_t1": np.full(n, -1, dtype=np.int64),
        "p_res_t1": np.full(n, np.nan),
        "y_pred_l1_c10": np.full(n, -1, dtype=np.int64),
        "p_res_l1_c10": np.full(n, np.nan),
    }
    for reg in sorted(set(regions.tolist())):
        idx = np.load(cache / f"oof_idx_{reg}.npy")
        pack = joblib.load(cache / f"oof_pack_{reg}.joblib")
        out["y_pred_baseline"][idx] = pack["baseline_lr"]["y_pred"]
        out["p_res_baseline"][idx] = pack["baseline_lr"]["p_res"]
        out["y_pred_pls"][idx] = pack["pls_da_lr"][best_k]["y_pred"]
        out["p_res_pls"][idx] = pack["pls_da_lr"][best_k]["p_res"]
        out["y_pred_t1"][idx] = pack["t1_only"]["y_pred"]
        out["p_res_t1"][idx] = pack["t1_only"]["p_res"]
        out["y_pred_l1_c10"][idx] = pack["l1_pls"][10.0]["y_pred"]
        out["p_res_l1_c10"][idx] = pack["l1_pls"][10.0]["p_res"]
    return pd.DataFrame(out)


def _pick(sweep: pd.DataFrame, scale: str, model: str, metric: str, higher: bool) -> dict:
    sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == model)].dropna(subset=[metric])
    if sub.empty:
        return {}
    idx = sub[metric].idxmax() if higher else sub[metric].idxmin()
    return sub.loc[idx].to_dict()


def run_scale(scale: str, X, Y, y, regions, meta, n_comps, max_workers: int) -> dict:
    scale_dir = OUT / scale
    cache = scale_dir / "_cache"
    ckpt = scale_dir / "checkpoints"
    for d in (scale_dir, cache, ckpt):
        d.mkdir(parents=True, exist_ok=True)

    x_path = str(cache / "X.npy")
    y_path = str(cache / "Y.npy")
    ybin_path = str(cache / "y_bin.npy")
    reg_path = str(cache / "regions.npy")
    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(ybin_path, y)
    np.save(reg_path, regions)

    region_list = sorted(set(regions.tolist()))
    n_gpu = torch.cuda.device_count()
    jobs = []
    for i, r in enumerate(region_list):
        jobs.append(
            (
                scale,
                r,
                x_path,
                y_path,
                ybin_path,
                reg_path,
                str(cache),
                str(ckpt),
                i % n_gpu,
                n_comps,
                max(n_comps),
            )
        )

    print(
        f"[{scale}] LORO n={len(y):,} folds={len(region_list)} workers={max_workers} n_comps={list(n_comps)}",
        flush=True,
    )
    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = {pool.submit(_fit_fold, j): j[1] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{scale} folds"):
            res = fut.result()
            rows.extend(res["rows"])
            print(
                f"  [{scale}] {res['region']} done n_te={res['n_test']:,} on {res['device']}",
                flush=True,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(scale_dir / "loro_metrics.csv", index=False)
    sweep = _agg_sweep(metrics)
    sweep.to_csv(scale_dir / "sweep_summary.csv", index=False)

    # L1 coefs
    coef_frames = []
    for r in region_list:
        p = cache / f"l1_coefs_{r}.csv"
        if p.exists():
            coef_frames.append(pd.read_csv(p))
    if coef_frames:
        pd.concat(coef_frames, ignore_index=True).to_csv(scale_dir / "l1_coefs.csv", index=False)

    baseline = _pick(sweep, scale, "baseline_lr", "loro_macro_f1_mean", True)
    pls_best = _pick(sweep, scale, "pls_da_lr", "loro_macro_f1_mean", True)
    pca_best = _pick(sweep, scale, "pca_lr", "loro_macro_f1_mean", True)
    t1 = _pick(sweep, scale, "t1_only", "loro_macro_f1_mean", True)
    ridge = _pick(sweep, scale, "baseline_ridge", "loro_mae_mean", False)
    frac_best = _pick(sweep, scale, "pls_frac", "loro_mae_mean", False)

    def _l1(C):
        sub = sweep.loc[(sweep["scale"] == scale) & (sweep["model"] == "l1_pls") & (sweep["C"] == C)]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    best_k = int(pls_best.get("n_components", 32))
    oof = _assemble_oof(scale, scale_dir, y, regions, meta, best_k)
    oof.to_parquet(scale_dir / "oof_predictions.parquet", index=False)

    def _oof_f1(col):
        m = oof["y_pred_baseline"] >= 0
        return float(f1_score(oof.loc[m, "y_true"], oof.loc[m, col], average="macro", zero_division=0))

    scale_sum = {
        "scale": scale,
        "n": int(len(y)),
        "n_res": int((y == 1).sum()),
        "n_nres": int((y == 0).sum()),
        "regions": region_list,
        "baseline_lr": baseline,
        "baseline_ridge": ridge,
        "pls_da_best": pls_best,
        "pca_best": pca_best,
        "t1_only": t1,
        "l1_c10": _l1(10.0),
        "l1_c30": _l1(30.0),
        "pls_frac_best": frac_best,
        "oof_macro_f1_baseline": _oof_f1("y_pred_baseline"),
        "oof_macro_f1_pls": _oof_f1("y_pred_pls"),
        "oof_macro_f1_t1": _oof_f1("y_pred_t1"),
        "oof_macro_f1_l1_c10": _oof_f1("y_pred_l1_c10"),
        "paths": {
            "metrics": str(scale_dir / "loro_metrics.csv"),
            "sweep": str(scale_dir / "sweep_summary.csv"),
            "oof": str(scale_dir / "oof_predictions.parquet"),
            "checkpoints": str(ckpt),
        },
    }
    (scale_dir / "summary.json").write_text(json.dumps(_jsonable(scale_sum), indent=2) + "\n")
    return {"metrics": metrics, "sweep": sweep, "summary": scale_sum, "oof": oof}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["gpu"], default="gpu")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force-100m", action="store_true")
    ap.add_argument(
        "--n-comps",
        type=str,
        default=",".join(str(x) for x in N_COMPS),
    )
    ap.add_argument("--scales", type=str, default="10m,100m", help="Comma list: 10m,100m")
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG):
        d.mkdir(parents=True, exist_ok=True)

    n_comps = tuple(int(x) for x in args.n_comps.split(",") if x.strip())
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count() or 1
    print(f"CPUs={n_cpu}  GPUs={n_gpu}", flush=True)
    if n_gpu == 0:
        raise SystemExit("This comparison expects GPUs (8 TITAN RTX on this machine).")
    for i in range(n_gpu):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}", flush=True)

    max_workers = args.workers or min(10, n_gpu)
    want = [s.strip() for s in args.scales.split(",") if s.strip()]

    all_metrics = []
    all_sweep = []
    summaries = {}

    if "10m" in want:
        X, Y, y, regions, meta = load_10m()
        r10 = run_scale("10m", X, Y, y, regions, meta, n_comps, max_workers)
        all_metrics.append(r10["metrics"])
        all_sweep.append(r10["sweep"])
        summaries["10m"] = r10["summary"]

    if "100m" in want:
        X, Y, y, regions, meta = build_100m_mean_pool(force=args.force_100m)
        r100 = run_scale("100m", X, Y, y, regions, meta, n_comps, max_workers)
        all_metrics.append(r100["metrics"])
        all_sweep.append(r100["sweep"])
        summaries["100m"] = r100["summary"]

    metrics = pd.concat(all_metrics, ignore_index=True)
    sweep = pd.concat(all_sweep, ignore_index=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)
    sweep.to_csv(OUT / "sweep_summary.csv", index=False)

    # cross-scale deltas
    cross = {}
    if "10m" in summaries and "100m" in summaries:
        for key in ("baseline_lr", "pls_da_best", "t1_only", "l1_c10", "l1_c30", "pca_best"):
            a = summaries["10m"].get(key, {}).get("loro_macro_f1_mean")
            b = summaries["100m"].get(key, {}).get("loro_macro_f1_mean")
            if a is not None and b is not None:
                cross[f"delta_f1_{key}_10m_minus_100m"] = float(a) - float(b)

    summary = {
        "name": "week4_pls_10m_vs_100m_v1",
        "protocol": "leave-one-region-out",
        "features": "64-D AlphaEarth (10m cells; 100m = mean pool over cells in tile)",
        "n_comps_sweep": list(n_comps),
        "l1_C": list(L1_CS),
        "backend": "gpu",
        "n_gpus": n_gpu,
        "n_cpus": n_cpu,
        "workers": max_workers,
        "seed": SEED,
        "10m": summaries.get("10m"),
        "100m": summaries.get("100m"),
        "cross_scale": cross,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "metrics": "outputs/week4_pls_10m_vs_100m/loro_metrics.csv",
            "sweep": "outputs/week4_pls_10m_vs_100m/sweep_summary.csv",
            "figures": "outputs/week4_pls_10m_vs_100m/figures",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    if "10m" in summaries and "100m" in summaries:
        _plot_all(sweep, metrics, FIG, summaries)

    print(json.dumps(_jsonable({"elapsed_s": summary["elapsed_s"], "cross_scale": cross}), indent=2), flush=True)
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
