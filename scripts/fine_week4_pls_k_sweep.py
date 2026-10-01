#!/usr/bin/env python3
"""Fine PLS-DA LORO sweep k=1..20 to find the minimum orthogonal dimension count.

Writes outputs/week4_pls_loro/fine_k_sweep.{csv,json} and figures/fine_k_minimum.png.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from scripts.train_week4_pls_loro import (
    BANDS,
    _torch_logistic,
    _torch_pls_nipals,
    _torch_standardize,
)

PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_pls_loro"
CACHE = OUT / "_cache"
FIG = OUT / "figures"
N_COMPS = tuple(range(1, 21))
N_MAX = max(N_COMPS)


def worker(payload: tuple) -> dict:
    te_region, x_path, ybin_path, reg_path, device_id = payload
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    X = np.load(x_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    X_tr = torch.as_tensor(np.asarray(X[tr], dtype=np.float32), device=device)
    X_te = torch.as_tensor(np.asarray(X[te], dtype=np.float32), device=device)
    y_tr = torch.as_tensor(np.asarray(y_bin[tr], dtype=np.float32), device=device)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    X_tr_s, X_te_s, _, _ = _torch_standardize(X_tr, X_te)
    y_c = (y_tr - y_tr.mean()).unsqueeze(1)
    pls = _torch_pls_nipals(X_tr_s, y_c, N_MAX)
    R_full = pls["x_rotations"]
    T_tr_full = X_tr_s @ R_full
    yv = y_c.squeeze()
    cov = []
    for k in range(N_MAX):
        t = T_tr_full[:, k]
        c = float(((t - t.mean()) * (yv - yv.mean())).mean().item() ** 2)
        cov.append(c)

    rows = []
    for k in N_COMPS:
        R = R_full[:, :k]
        T_tr = X_tr_s @ R
        T_te = X_te_s @ R
        tmu = T_tr.mean(0, keepdim=True)
        tsd = T_tr.std(0, keepdim=True).clamp_min(1e-6)
        yhat, _, _, _ = _torch_logistic((T_tr - tmu) / tsd, y_tr, (T_te - tmu) / tsd)
        rows.append(
            {
                "region": te_region,
                "n_components": int(k),
                "macro_f1": float(f1_score(yb_te, yhat, average="macro", zero_division=0)),
            }
        )
    yhat64, _, _, _ = _torch_logistic(X_tr_s, y_tr, X_te_s)
    rows.append(
        {
            "region": te_region,
            "n_components": 64,
            "macro_f1": float(f1_score(yb_te, yhat64, average="macro", zero_division=0)),
        }
    )
    torch.cuda.empty_cache()
    return {"region": te_region, "rows": rows, "cov": cov}


def main() -> None:
    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    print(f"Fine PLS-DA sweep k=1..20 · GPUs={n_gpu}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    x_path = CACHE / "X.npy"
    ybin_path = CACHE / "y_bin.npy"
    reg_path = CACHE / "regions.npy"
    if not x_path.exists():
        df = pd.read_parquet(PIXELS)
        np.save(x_path, df[BANDS].to_numpy(np.float32))
        np.save(ybin_path, df["y"].to_numpy(np.int64))
        np.save(reg_path, df["region"].astype(str).to_numpy())

    regions = np.load(reg_path, allow_pickle=True)
    region_list = sorted(set(regions.tolist()))
    jobs = [
        (r, str(x_path), str(ybin_path), str(reg_path), i % n_gpu)
        for i, r in enumerate(region_list)
    ]

    ctx = torch.multiprocessing.get_context("spawn")
    all_rows, covs = [], []
    with ProcessPoolExecutor(max_workers=min(len(jobs), n_gpu), mp_context=ctx) as pool:
        futs = {pool.submit(worker, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="folds"):
            res = fut.result()
            all_rows.extend(res["rows"])
            covs.append(res["cov"])
            base = next(r for r in res["rows"] if r["n_components"] == 64)
            best = max(
                (r for r in res["rows"] if r["n_components"] <= 20),
                key=lambda r: r["macro_f1"],
            )
            print(
                f"  {res['region']:12s} base={base['macro_f1']:.3f} "
                f"best_k={best['n_components']} F1={best['macro_f1']:.3f}",
                flush=True,
            )

    m = pd.DataFrame(all_rows)
    sweep = (
        m.groupby("n_components")["macro_f1"]
        .agg(loro_mean="mean", loro_std=lambda s: float(s.std(ddof=0)))
        .reset_index()
    )
    base = float(sweep.loc[sweep.n_components == 64, "loro_mean"].iloc[0])
    sweep["gap"] = base - sweep["loro_mean"]
    sweep["rel_gap"] = sweep["gap"] / base

    mean_cov = np.asarray(covs, dtype=np.float64).mean(0)
    cum = np.cumsum(mean_cov) / max(mean_cov.sum(), 1e-12)

    def min_k(tol_abs=None, tol_rel=None):
        sub = sweep.loc[sweep.n_components < 64].sort_values("n_components")
        for _, row in sub.iterrows():
            if tol_abs is not None and row["gap"] > tol_abs:
                continue
            if tol_rel is not None and row["rel_gap"] > tol_rel:
                continue
            return int(row["n_components"]), float(row["loro_mean"]), float(row["gap"])
        return None, None, None

    picks = {
        "within_0.01_F1": min_k(tol_abs=0.01),
        "within_0.02_F1": min_k(tol_abs=0.02),
        "within_1pct_rel": min_k(tol_rel=0.01),
        "within_2pct_rel": min_k(tol_rel=0.02),
    }
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)

    out = {
        "baseline_loro_f1": base,
        "min_k_criteria": {
            k: {"k": v[0], "f1": v[1], "gap": v[2]} for k, v in picks.items()
        },
        "cov_cum_k90": k90,
        "cov_cum_k95": k95,
        "cov_cum_k99": k99,
        "mean_cov": mean_cov.tolist(),
        "cum_cov_frac": cum.tolist(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    sweep.to_csv(OUT / "fine_k_sweep.csv", index=False)
    (OUT / "fine_k_sweep.json").write_text(json.dumps(out, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sub = sweep.loc[sweep.n_components < 64]
    ax = axes[0]
    ax.plot(sub.n_components, sub.loro_mean, "o-", color="#0072B2", lw=1.8)
    ax.axhline(base, color="#333", ls="--", lw=1.2, label=f"64-D LR {base:.3f}")
    for name, color in (("within_0.02_F1", "#009E73"), ("within_0.01_F1", "#E69F00")):
        k, _, _ = picks[name]
        if k is not None:
            ax.axvline(k, color=color, ls=":", lw=1.3, label=f"{name} → k={k}")
    ax.set_xlabel("n_components (orthogonal PLS scores)")
    ax.set_ylabel("LORO mean macro-F1")
    ax.set_xticks(range(1, 21))
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E5E5E5")

    ax = axes[1]
    ax.bar(np.arange(1, N_MAX + 1), mean_cov / mean_cov.sum(), color="#0072B2", alpha=0.85)
    ax.plot(np.arange(1, N_MAX + 1), cum, color="#D55E00", lw=1.8, label="cumulative")
    ax.axhline(0.95, color="#333", ls="--", lw=1)
    ax.axvline(k95, color="#D55E00", ls=":", label=f"95% cov @ k={k95}")
    ax.set_xlabel("component")
    ax.set_ylabel("fraction of Σ cov²(t, y)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Minimum orthogonal PLS-DA dimensions vs 64-D LR",
        fontsize=12,
        fontweight="bold",
        transform=fig.transFigure,
        va="top",
    )
    fig.savefig(FIG / "fine_k_minimum.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n=== Absolute minimum k ===", flush=True)
    print(f"baseline 64-D LORO F1 = {base:.4f}", flush=True)
    for name, d in out["min_k_criteria"].items():
        print(
            f"  {name:20s}  k={d['k']}  F1={d['f1']:.4f}  gap={d['gap']:.4f}",
            flush=True,
        )
    print(f"cov spectrum: 90%@k={k90}  95%@k={k95}  99%@k={k99}", flush=True)
    print(sweep.to_string(index=False), flush=True)
    print(f"elapsed {out['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
