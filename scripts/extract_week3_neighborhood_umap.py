#!/usr/bin/env python3
"""Extract OOF NbCNN pooled embeddings + UMAP/PCA plots for notebook 06.

Uses leave-one-region-out clf checkpoints so embeddings are out-of-fold.
Also encodes with reg checkpoints for a second embedding view.

Usage:
  .venv/bin/python -u scripts/extract_week3_neighborhood_umap.py --k 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week3_pixel_neighborhood_trainlib import (  # noqa: E402
    BATCH_SIZE,
    EMB_DIM,
    NbCNN,
    pack_input,
)

SEED = 42
LABEL_ORDER = ["empty", "non_residential", "residential"]
LABEL_COLORS = {"empty": "#2a6f4e", "non_residential": "#2b6a8f", "residential": "#c45c26"}


def _encode_fold(payload: dict) -> dict:
    device = torch.device(
        f"cuda:{payload['gpu_id']}" if torch.cuda.is_available() and payload.get("gpu_id") is not None else "cpu"
    )
    X = np.load(payload["x_path"], mmap_mode="r")
    mask = np.load(payload["mask_path"], mmap_mode="r")
    te = np.load(payload["idx_path"])
    X_te = pack_input(X[te], mask[te])

    model = NbCNN(n_out=3, emb=EMB_DIM).to(device)
    ckpt = torch.load(payload["ckpt_path"], map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    chunks = []
    with torch.no_grad():
        for s in range(0, len(X_te), BATCH_SIZE):
            xb = torch.from_numpy(X_te[s : s + BATCH_SIZE]).to(device, non_blocking=True)
            chunks.append(model.encode(xb).cpu().numpy())
    emb = np.concatenate(chunks, axis=0).astype(np.float32)
    return {"te": te, "emb": emb, "region": payload["region"], "task": payload["task"]}


def extract_oof(task: str, cache: Path, out: Path, n_gpu: int) -> np.ndarray:
    x_path = str(cache / "X_nb.npy")
    mask_path = str(cache / "mask.npy")
    regions = sorted(
        p.name.replace(f"{task}_ckpt_", "").replace(".pt", "")
        for p in cache.glob(f"{task}_ckpt_*.pt")
    )
    if not regions:
        raise SystemExit(f"No {task} checkpoints in {cache}")

    n = len(np.load(cache / "y.npy", mmap_mode="r"))
    emb_oof = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    jobs = []
    for i, r in enumerate(regions):
        jobs.append(
            {
                "region": r,
                "task": task,
                "ckpt_path": str(cache / f"{task}_ckpt_{r}.pt"),
                "idx_path": str(cache / f"{task}_idx_{r}.npy"),
                "x_path": x_path,
                "mask_path": mask_path,
                "gpu_id": (i % n_gpu) if n_gpu else None,
            }
        )

    print(f"encode OOF {task} · {len(jobs)} folds · gpus={max(1, n_gpu)}", flush=True)
    ctx = mp.get_context("spawn")
    max_workers = max(1, n_gpu) if n_gpu else 1
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futs = [pool.submit(_encode_fold, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"encode-{task}"):
            r = fut.result()
            emb_oof[r["te"]] = r["emb"]
            print(f"  {r['region']:12s} emb[{r['task']}] n={len(r['te'])}", flush=True)

    if np.isnan(emb_oof).any():
        miss = int(np.isnan(emb_oof).any(axis=1).sum())
        raise RuntimeError(f"missing OOF embeddings for {miss} rows ({task})")
    path = out / f"oof_{task}_embeddings.npy"
    np.save(path, emb_oof)
    print(f"Wrote {path} shape={emb_oof.shape}", flush=True)
    return emb_oof


def run_umap(emb: np.ndarray, meta: pd.DataFrame, fig_dir: Path, tag: str) -> pd.DataFrame:
    import umap

    n = len(emb)
    rng = np.random.default_rng(SEED)
    # Fit one global UMAP on a balanced subsample, then plot per-region facets
    # (shared embedding space so regions are comparable).
    per_reg = 8_000
    idx_parts = []
    for reg, g in meta.groupby("region"):
        ii = g.index.to_numpy()
        take = min(per_reg, len(ii))
        idx_parts.append(rng.choice(ii, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    print(f"UMAP {tag} fit on {len(idx):,}/{n:,} rows (≤{per_reg}/region) …", flush=True)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    xy_sub = reducer.fit_transform(emb[idx])

    coords = meta.iloc[idx][["pixel_id", "region", "label"]].copy().reset_index(drop=True)
    coords["umap_1"] = xy_sub[:, 0]
    coords["umap_2"] = xy_sub[:, 1]
    coords.to_parquet(fig_dir.parent / f"umap_{tag}_coords.parquet", index=False)

    regions = sorted(coords["region"].unique())
    n_reg = len(regions)
    ncols = 5
    nrows = int(np.ceil(n_reg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax, reg in zip(axes.ravel(), regions):
        sub = coords.loc[coords["region"] == reg]
        for lab in LABEL_ORDER:
            m = sub["label"].astype(str) == lab
            if not m.any():
                continue
            ax.scatter(
                sub.loc[m, "umap_1"],
                sub.loc[m, "umap_2"],
                s=6,
                alpha=0.55,
                c=LABEL_COLORS.get(lab, "#444"),
                label=lab,
                rasterized=True,
            )
        ax.set_title(reg, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n_reg:]:
        ax.axis("off")
    # shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[l], markersize=8, label=l)
        for l in LABEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"NbCNN {EMB_DIM}-D OOF · UMAP by label per region ({tag})", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fig_dir / f"umap_{tag}_by_label_per_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also write individual PNGs under figures/umap_{tag}_by_label/
    per_dir = fig_dir / f"umap_{tag}_by_label"
    per_dir.mkdir(parents=True, exist_ok=True)
    for reg in regions:
        sub = coords.loc[coords["region"] == reg]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for lab in LABEL_ORDER:
            m = sub["label"].astype(str) == lab
            if not m.any():
                continue
            ax.scatter(
                sub.loc[m, "umap_1"],
                sub.loc[m, "umap_2"],
                s=8,
                alpha=0.6,
                c=LABEL_COLORS.get(lab, "#444"),
                label=lab,
                rasterized=True,
            )
        ax.set_title(f"{tag} · {reg} (n={len(sub):,})")
        ax.legend(markerscale=2, fontsize=9, frameon=False, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(per_dir / f"{reg}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # Keep a compact "by region" overview colored by region (global coords)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for reg in regions:
        m = coords["region"] == reg
        ax.scatter(
            coords.loc[m, "umap_1"],
            coords.loc[m, "umap_2"],
            s=4,
            alpha=0.45,
            label=reg,
            rasterized=True,
        )
    ax.set_title(f"NbCNN {EMB_DIM}-D OOF embedding · UMAP by region ({tag})")
    ax.legend(markerscale=3, fontsize=8, frameon=False, loc="best", ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / f"umap_{tag}_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote umap_{tag}_by_label_per_region.png + {per_dir}/", flush=True)
    return coords


def run_pca(emb: np.ndarray, meta: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    pca = PCA(n_components=min(32, emb.shape[1]), random_state=SEED)
    z = pca.fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
    axes[0].axhline(0.8, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("components")
    axes[0].set_ylabel("cumulative variance")
    axes[0].set_title(f"PCA on NbCNN OOF emb ({tag})")

    lab_to_i = {l: i for i, l in enumerate(LABEL_ORDER)}
    y = meta["label"].map(lab_to_i).to_numpy()
    # subsample scatter
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(z), size=min(60_000, len(z)), replace=False)
    sc = axes[1].scatter(z[idx, 0], z[idx, 1], c=y[idx], s=3, cmap="tab10", alpha=0.5, rasterized=True)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PC1–PC2 by label")
    cbar = fig.colorbar(sc, ax=axes[1], ticks=range(len(LABEL_ORDER)))
    cbar.ax.set_yticklabels(LABEL_ORDER, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"pca_{tag}_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote pca_{tag}_overview.png", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=["clf", "reg"])
    ap.add_argument(
        "--umap-only",
        action="store_true",
        help="Reuse existing oof_*_embeddings.npy (skip GPU encode)",
    )
    args = ap.parse_args()
    k = int(args.k)

    out = ROOT / "outputs" / f"week3_pixel_neighborhood_k{k}"
    cache = out / "_cache"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(ROOT / "datasets" / f"week3_pixel_neighborhood_k{k}" / "meta.parquet")
    meta = meta.loc[meta["label"] != "mixed"].reset_index(drop=True)
    assert len(meta) == len(np.load(cache / "y.npy", mmap_mode="r"))

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"n_gpu={n_gpu} k={k} tasks={args.tasks}", flush=True)

    man = {"k": k, "emb_dim": EMB_DIM, "tasks": {}}
    for task in args.tasks:
        emb_path = out / f"oof_{task}_embeddings.npy"
        if args.umap_only and emb_path.exists():
            emb = np.load(emb_path)
            print(f"loaded {emb_path} {emb.shape}", flush=True)
        else:
            emb = extract_oof(task, cache, out, n_gpu)
        run_umap(emb, meta, fig, tag=task)
        run_pca(emb, meta, fig, tag=task)
        man["tasks"][task] = {
            "embeddings": str(emb_path),
            "umap_coords": str(out / f"umap_{task}_coords.parquet"),
            "figures": [
                f"umap_{task}_by_label_per_region.png",
                f"umap_{task}_by_region.png",
                f"pca_{task}_overview.png",
                f"umap_{task}_by_label/*.png",
            ],
        }
    (out / "umap_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2), flush=True)


if __name__ == "__main__":
    main()
