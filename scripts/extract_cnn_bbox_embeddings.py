#!/usr/bin/env python3
"""Extract + cache uniform CNN pooled embeddings for UMAP / EDA.

Trains the same bbox-grid CNN as notebooks/week2/02 under leave-one-region-out,
and for each held-out building stores the 256-D masked-GAP vector (backbone
output before the classification head).

Writes under outputs/week2_cnn_bbox/:
  cnn_embeddings_oof.parquet   id, region, label, y_res, E000..E255
  cnn_embeddings_oof.npy       (N, 256) float32
  cnn_embedding_meta.json
  umap_cnn_by_label.png
  umap_cnn_by_region.png
  pca_cnn_overview.png

Usage:
  python scripts/extract_cnn_bbox_embeddings.py
  python scripts/extract_cnn_bbox_embeddings.py --epochs 25 --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
BBOX = ROOT / "datasets" / "building_bbox_grid"
CENTROID = ROOT / "outputs" / "building_type_dataset" / "dataset.parquet"
OUT = ROOT / "outputs" / "week2_cnn_bbox"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB_DIM = 256
PAD_H, PAD_W = 32, 32
BATCH_SIZE = 128
LR = 1e-3
SEED = 42
LABEL_ORDER = [
    "single_family",
    "multi_family",
    "education",
    "medical",
    "retail",
    "office",
    "hotel",
    "industrial",
    "religious",
    "civic",
]


class BuildingPatchDataset(Dataset):
    def __init__(self, indices: np.ndarray, y: np.ndarray, grids: np.ndarray, masks: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.grids = grids
        self.masks = masks

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.concatenate([self.grids[idx], self.masks[idx][None, ...]], axis=0)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long), idx


class BBoxCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 65):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, EMB_DIM, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(EMB_DIM),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(EMB_DIM, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, -1:, :, :]
        feats = self.backbone(x)
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        return (feats * mask).sum(dim=(2, 3)) / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def build_grids(meta: pd.DataFrame, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    id_to_idx = {bid: i for i, bid in enumerate(meta["id"].tolist())}
    grids = np.zeros((len(meta), 64, PAD_H, PAD_W), dtype=np.float32)
    masks = np.zeros((len(meta), PAD_H, PAD_W), dtype=np.float32)
    grouped = cells.sort_values(["id", "i", "j"]).groupby("id", sort=False)
    for bid, g in tqdm(grouped, total=meta["id"].nunique(), desc="build grids"):
        if bid not in id_to_idx:
            continue
        i = id_to_idx[bid]
        vals = g[BANDS].to_numpy(dtype=np.float32)
        ii = g["i"].to_numpy(dtype=int)
        jj = g["j"].to_numpy(dtype=int)
        keep = (ii >= 0) & (jj >= 0) & (ii < PAD_H) & (jj < PAD_W)
        ii, jj, vals = ii[keep], jj[keep], vals[keep]
        g = grids[i]
        g[:, ii, jj] = vals.T
        masks[i, ii, jj] = 1.0
    return grids, masks


def train_fold_and_embed(
    te_region: str,
    gpu_id: int,
    y: np.ndarray,
    regions: np.ndarray,
    grids: np.ndarray,
    masks: np.ndarray,
    n_classes: int,
    epochs: int,
) -> dict:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    te = np.where(regions == te_region)[0]
    tr = np.where(regions != te_region)[0]

    train_loader = DataLoader(
        BuildingPatchDataset(tr, y, grids, masks),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        BuildingPatchDataset(te, y, grids, masks),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(SEED + abs(hash(te_region)) % 10_000)
    model = BBoxCNN(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = np.bincount(y[tr], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weight = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    model.train()
    for _ in tqdm(range(epochs), desc=f"gpu{gpu_id}:{te_region}", leave=False, position=gpu_id):
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    emb = np.zeros((len(te), EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        cursor = 0
        for xb, _, idx in test_loader:
            xb = xb.to(device, non_blocking=True)
            z = model.encode(xb).cpu().numpy()
            emb[cursor : cursor + len(z)] = z
            cursor += len(z)

    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"bbox_cnn_loro_{te_region}.pt"
    torch.save(
        {
            "region_held_out": te_region,
            "state_dict": model.state_dict(),
            "n_classes": n_classes,
            "emb_dim": EMB_DIM,
            "epochs": epochs,
            "seed": SEED,
            "arch": "BBoxCNN",
        },
        ckpt_path,
    )

    return {
        "region": te_region,
        "te_idx": te,
        "emb": emb,
        "gpu": gpu_id,
        "ckpt": str(ckpt_path),
    }


def run_umap_plots(emb: np.ndarray, meta: pd.DataFrame) -> None:
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP plots", flush=True)
        return

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    xy = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    labels = meta["label"].astype(str)
    for lab in LABEL_ORDER:
        m = labels == lab
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.65, label=lab)
    ax.set_title("CNN 256-D pooled embedding · UMAP by building type")
    ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "umap_cnn_by_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    regions = meta["region"].astype(str)
    for reg in sorted(regions.unique()):
        m = regions == reg
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.65, label=reg)
    ax.set_title("CNN 256-D pooled embedding · UMAP by region")
    ax.legend(markerscale=2, fontsize=8, frameon=False, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "umap_cnn_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also stash coords for notebook reuse
    coords = meta[["id", "region", "label"]].copy()
    coords["umap_1"] = xy[:, 0]
    coords["umap_2"] = xy[:, 1]
    coords.to_parquet(OUT / "cnn_embeddings_umap_coords.parquet", index=False)
    print("Wrote UMAP plots + coords", flush=True)


def run_pca_plot(emb: np.ndarray, meta: pd.DataFrame) -> None:
    pca = PCA(n_components=min(32, emb.shape[1]), random_state=SEED)
    z = pca.fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
    axes[0].axhline(0.8, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("components")
    axes[0].set_ylabel("cumulative variance")
    axes[0].set_title("PCA on CNN 256-D embeddings")

    le = LabelEncoder().fit(LABEL_ORDER)
    y = le.transform(meta["label"].astype(str))
    sc = axes[1].scatter(z[:, 0], z[:, 1], c=y, s=6, cmap="tab10", alpha=0.7)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PC1–PC2 colored by label")
    cbar = fig.colorbar(sc, ax=axes[1], ticks=range(len(LABEL_ORDER)))
    cbar.ax.set_yticklabels([l.replace("_", " ") for l in LABEL_ORDER], fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "pca_cnn_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote PCA overview", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing parquet if present")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    emb_path = OUT / "cnn_embeddings_oof.parquet"
    npy_path = OUT / "cnn_embeddings_oof.npy"

    if args.skip_train and emb_path.exists() and npy_path.exists():
        print("Loading cached embeddings", emb_path, flush=True)
        meta = pd.read_parquet(emb_path)
        emb = np.load(npy_path)
    else:
        t0 = time.time()
        buildings = pd.read_parquet(BBOX / "buildings.parquet")
        cells = pd.read_parquet(BBOX / "building_cells.parquet")
        buildings["id"] = buildings["id"].astype(str)
        cells["id"] = cells["id"].astype(str)
        meta = buildings.copy()
        meta["y"] = meta["label"].map({lab: i for i, lab in enumerate(LABEL_ORDER)}).astype(int)
        grids, masks = build_grids(meta, cells)
        y = meta["y"].to_numpy(dtype=np.int64)
        regions = meta["region"].to_numpy()
        region_list = sorted(meta["region"].unique().tolist())

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        workers = args.workers or max(1, n_gpus) or 1
        workers = min(workers, len(region_list), max(1, n_gpus) if n_gpus else 1)
        print(f"Extracting OOF CNN embeddings · folds={len(region_list)} · workers={workers} · gpus={n_gpus}", flush=True)

        emb = np.zeros((len(meta), EMB_DIM), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for i, reg in enumerate(region_list):
                gpu_id = i % workers if n_gpus else 0
                fut = pool.submit(
                    train_fold_and_embed,
                    reg,
                    gpu_id,
                    y,
                    regions,
                    grids,
                    masks,
                    len(LABEL_ORDER),
                    args.epochs,
                )
                futs[fut] = reg
            for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO embed folds"):
                result = fut.result()
                emb[result["te_idx"]] = result["emb"]
                print(
                    f"  cached · {result['region']:12s} gpu={result['gpu']} "
                    f"n={len(result['te_idx'])} ckpt={Path(result['ckpt']).name}",
                    flush=True,
                )

        cols = {f"E{i:03d}": emb[:, i] for i in range(EMB_DIM)}
        out_df = meta[["id", "region", "label", "y_res", "lon", "lat"]].copy()
        for k, v in cols.items():
            out_df[k] = v
        out_df.to_parquet(emb_path, index=False)
        np.save(npy_path, emb)
        meta_out = {
            "name": "cnn_bbox_pooled_oof_v1",
            "description": (
                "256-D masked global-average-pool vector from BBoxCNN backbone, "
                "extracted leave-one-region-out (held-out buildings only)."
            ),
            "dim": EMB_DIM,
            "n_buildings": int(len(out_df)),
            "epochs": args.epochs,
            "pad": [PAD_H, PAD_W],
            "paths": {
                "parquet": str(emb_path.relative_to(ROOT)),
                "npy": str(npy_path.relative_to(ROOT)),
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(OUT / "cnn_embedding_meta.json", "w") as f:
            json.dump(meta_out, f, indent=2)
        print(f"Wrote {emb_path} and {npy_path} in {meta_out['elapsed_s']}s", flush=True)
        meta = out_df

    run_pca_plot(emb, meta)
    run_umap_plots(emb, meta)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
