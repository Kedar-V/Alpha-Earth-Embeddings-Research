#!/usr/bin/env python3
"""Generate notebooks/week2/02_cnn_bbox_grid_training.ipynb.

Repeats the supervised training from notebooks 08 (multi-class building type)
and 09 (residential binary + DoU stratification), but replaces the flat 64-D
centroid vector with a small CNN over the bbox-grid AlphaEarth patches in
datasets/building_bbox_grid/.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "week2" / "02_cnn_bbox_grid_training.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# Week 2 — CNN on bbox-grid AlphaEarth embeddings

**Goal.** Repeat the supervised setups from notebooks **08** (10-class building type)
and **09** (residential vs non-residential, sliced by Degree of Urbanization), but
feed each building a **variable-size 10 m embedding grid** instead of a single
centroid vector.

**Data.** `datasets/building_bbox_grid/` — enclosing-rectangle patches of AlphaEarth
2024 (`A00`–`A63`), no averaging. Built from the same labeled Overture set as
`outputs/building_type_dataset/dataset.parquet`.

**Models.**
1. **Baseline** — logistic regression on the centroid 64-D vector (notebooks 02 / 05 / 09).
2. **CNN** — small 2-D CNN over the padded `(64, H, W)` patch + mask channel, then a linear head.

**Protocol.** Leave-one-region-out (LORO) macro-F1. Binary residential also reported by
GHSL Degree of Urbanization where a DoU label is available.
"""
        ),
        md("## Setup"),
        code(
            r"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent
if not (ROOT / "requirements.txt").exists() and (ROOT.parent.parent / "requirements.txt").exists():
    ROOT = ROOT.parent.parent

BBOX = ROOT / "datasets" / "building_bbox_grid"
CENTROID = ROOT / "outputs" / "building_type_dataset" / "dataset.parquet"
DOU_CORE = ROOT / "outputs" / "week2_ghsl_dou" / "trial_core_with_dou.parquet"
OUT = ROOT / "outputs" / "week2_cnn_bbox"
OUT.mkdir(parents=True, exist_ok=True)

BANDS = [f"A{i:02d}" for i in range(64)]
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
DOU_ORDER = ["urban", "peri_urban", "rural"]
DOU_LABELS = {"urban": "Urban", "peri_urban": "Peri-urban", "rural": "Rural"}
DOU_COLORS = {"urban": "#d95f0e", "peri_urban": "#fec44f", "rural": "#2c7fb8"}

PAD_H, PAD_W = 32, 32
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
SEED = 42
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda:0" if N_GPUS else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
if N_GPUS:
    torch.cuda.manual_seed_all(SEED)

print("ROOT:", ROOT, flush=True)
print(f"GPUs: {N_GPUS}" + (f" · {[torch.cuda.get_device_name(i) for i in range(N_GPUS)]}" if N_GPUS else " · cpu"), flush=True)
print("BBOX:", BBOX, "exists=", BBOX.exists(), flush=True)
print("OUT:", OUT, flush=True)
"""
        ),
        md(
            """
## 1. Load buildings, centroid vectors, and bbox grids

Each building becomes a `(64, 32, 32)` tensor (channels-first) plus a binary
presence mask. Grids larger than 32×32 do not occur in this dataset (max 28×22).
Empty padded cells are zeroed and masked out of the CNN via an extra mask channel.
"""
        ),
        code(
            r"""
buildings = pd.read_parquet(BBOX / "buildings.parquet")
cells = pd.read_parquet(BBOX / "building_cells.parquet")
centroid = pd.read_parquet(CENTROID)
buildings["id"] = buildings["id"].astype(str)
cells["id"] = cells["id"].astype(str)
centroid["id"] = centroid["id"].astype(str)

meta = buildings.merge(
    centroid[["id"] + BANDS],
    on="id",
    how="inner",
    validate="one_to_one",
)
assert meta[BANDS].notna().all().all()
meta["y_mult"] = meta["label"].map({lab: i for i, lab in enumerate(LABEL_ORDER)}).astype(int)
meta["y_bin"] = meta["y_res"].astype(int)

# Optional DoU from notebook 09 core sample (partial coverage)
if DOU_CORE.exists():
    dou = pd.read_parquet(DOU_CORE)[["id", "dou", "smod_code"]].copy()
    dou["id"] = dou["id"].astype(str)
    meta = meta.merge(dou, on="id", how="left")
else:
    meta["dou"] = np.nan
    meta["smod_code"] = np.nan

print(
    f"buildings={len(meta)}  cells={len(cells)}  "
    f"with_dou={meta['dou'].notna().sum()}  "
    f"regions={meta['region'].nunique()}"
)
print(meta["label"].value_counts().reindex(LABEL_ORDER))
print("DoU coverage:\n", meta["dou"].value_counts(dropna=False))
"""
        ),
        code(
            r"""
def build_grid_tensor(group: pd.DataFrame, n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    # (C,H,W) float32 grid + (H,W) mask, padded to PAD_H x PAD_W
    grid = np.zeros((64, PAD_H, PAD_W), dtype=np.float32)
    mask = np.zeros((PAD_H, PAD_W), dtype=np.float32)
    if group.empty or n_rows <= 0 or n_cols <= 0:
        return grid, mask
    vals = group[BANDS].to_numpy(dtype=np.float32)
    ii = group["i"].to_numpy(dtype=int)
    jj = group["j"].to_numpy(dtype=int)
    # clip rare overflow (should not happen)
    keep = (ii >= 0) & (jj >= 0) & (ii < PAD_H) & (jj < PAD_W)
    ii, jj, vals = ii[keep], jj[keep], vals[keep]
    grid[:, ii, jj] = vals.T
    mask[ii, jj] = 1.0
    return grid, mask


# Precompute all patches once (fast enough at n≈11k, 32×32×64)
id_to_idx = {bid: i for i, bid in enumerate(meta["id"].tolist())}
grids = np.zeros((len(meta), 64, PAD_H, PAD_W), dtype=np.float32)
masks = np.zeros((len(meta), PAD_H, PAD_W), dtype=np.float32)

cells_sorted = cells.sort_values(["id", "i", "j"])
grouped = cells_sorted.groupby("id", sort=False)
n_built = 0
for bid, g in tqdm(grouped, total=meta["id"].nunique(), desc="build grids"):
    if bid not in id_to_idx:
        continue
    i = id_to_idx[bid]
    row = meta.iloc[i]
    grids[i], masks[i] = build_grid_tensor(g, int(row["n_rows"]), int(row["n_cols"]))
    n_built += 1

print(f"precomputed grids for {n_built}/{len(meta)} buildings · shape {grids.shape}", flush=True)
print(f"mask occupancy mean={masks.mean():.4f}  max cells in pad={(masks.sum(axis=(1,2))).max():.0f}", flush=True)

X_cent = meta[BANDS].to_numpy(dtype=np.float64)
y_mult = meta["y_mult"].to_numpy(dtype=np.int64)
y_bin = meta["y_bin"].to_numpy(dtype=np.int64)
regions = meta["region"].to_numpy()
region_list = sorted(meta["region"].unique().tolist())
"""
        ),
        md(
            """
## 2. Dataset / CNN

Input channels = **64 embedding bands + 1 presence mask**. Masked average pooling
ignores padded zeros so small houses and large hospitals share the same head.
"""
        ),
        code(
            r"""
class BuildingPatchDataset(Dataset):
    def __init__(self, indices: np.ndarray, y: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = np.concatenate([grids[idx], masks[idx][None, ...]], axis=0)  # (65,H,W)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


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
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 65, H, W) — channel -1 is presence mask
        mask = x[:, -1:, :, :]
        feats = self.backbone(x)
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (feats * mask).sum(dim=(2, 3)) / denom
        return self.head(pooled)


def _train_one_fold(
    te_region: str,
    gpu_id: int,
    y: np.ndarray,
    n_classes: int,
    epochs: int,
    class_weight: bool,
) -> dict:
    device = torch.device(f"cuda:{gpu_id}" if N_GPUS else "cpu")
    te = np.where(regions == te_region)[0]
    tr = np.where(regions != te_region)[0]
    train_loader = DataLoader(
        BuildingPatchDataset(tr, y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        BuildingPatchDataset(te, y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(SEED + abs(hash(te_region)) % 10_000)
    model = BBoxCNN(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    if class_weight:
        counts = np.bincount(y[tr], minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (n_classes * counts)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        weight = None
    crit = nn.CrossEntropyLoss(weight=weight)

    model.train()
    epoch_bar = tqdm(
        range(epochs),
        desc=f"gpu{gpu_id}:{te_region}",
        leave=False,
        position=gpu_id if N_GPUS else 0,
    )
    for _epoch in epoch_bar:
        total_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(yb)
            n_seen += len(yb)
        epoch_bar.set_postfix(loss=f"{total_loss / max(1, n_seen):.3f}")

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pr = torch.softmax(logits, dim=1)
            preds.append(pr.argmax(dim=1).cpu().numpy())
            probs.append(pr.max(dim=1).values.cpu().numpy())
    pred = np.concatenate(preds)
    prob = np.concatenate(probs)
    f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y[te], pred))
    return {
        "region": te_region,
        "te_idx": te,
        "pred": pred,
        "prob": prob,
        "n_test": int(len(te)),
        "accuracy": acc,
        "f1_macro": f1,
        "gpu": gpu_id,
    }


def train_cnn_loro(
    y: np.ndarray,
    n_classes: int,
    epochs: int = EPOCHS,
    class_weight: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Leave-one-region-out CNN; folds run in parallel across GPUs.
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    oof_prob = np.full(len(y), np.nan, dtype=np.float64)
    workers = max(1, N_GPUS) if N_GPUS else 1
    print(f"LORO CNN · {len(region_list)} folds · {workers} parallel workers", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for i, te_region in enumerate(region_list):
            gpu_id = i % workers if N_GPUS else 0
            fut = pool.submit(
                _train_one_fold, te_region, gpu_id, y, n_classes, epochs, class_weight
            )
            futs[fut] = te_region
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            result = fut.result()
            te = result["te_idx"]
            oof_pred[te] = result["pred"]
            oof_prob[te] = result["prob"]
            rows.append(
                {
                    "region": result["region"],
                    "n_test": result["n_test"],
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                    "gpu": result["gpu"],
                }
            )
            print(
                f"  done · {result['region']:12s}  gpu={result['gpu']}  "
                f"n={result['n_test']:4d}  macro-F1={result['f1_macro']:.3f}  "
                f"acc={result['accuracy']:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows), oof_pred, oof_prob


def train_lr_loro(X: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    oof = np.full(len(y), -1, dtype=np.int64)
    for te_region in tqdm(region_list, desc="LORO LR"):
        te = regions == te_region
        tr = ~te
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        oof[te] = pred
        rows.append(
            {
                "region": te_region,
                "n_test": int(te.sum()),
                "accuracy": float(accuracy_score(y[te], pred)),
                "f1_macro": float(f1_score(y[te], pred, average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows), oof


print("CNN + LR helpers ready · multi-GPU=", N_GPUS, flush=True)
"""
        ),
        md(
            """
## 3. Multi-class building type (notebook 08 / 05 setting)

10 Overture-derived labels · LORO macro-F1 · centroid LR vs bbox CNN.
"""
        ),
        code(
            r"""
print("=== Centroid logistic regression (LORO, 10-class) ===")
lr_mult, oof_lr_mult = train_lr_loro(X_cent, y_mult)
print(lr_mult.sort_values("f1_macro", ascending=False).to_string(index=False))
print(
    f"mean macro-F1={lr_mult['f1_macro'].mean():.3f} ± {lr_mult['f1_macro'].std(ddof=1):.3f}"
)

print("\n=== BBox CNN (LORO, 10-class) ===")
cnn_mult, oof_cnn_mult, _ = train_cnn_loro(y_mult, n_classes=len(LABEL_ORDER), epochs=EPOCHS)
print(cnn_mult.sort_values("f1_macro", ascending=False).to_string(index=False))
print(
    f"mean macro-F1={cnn_mult['f1_macro'].mean():.3f} ± {cnn_mult['f1_macro'].std(ddof=1):.3f}"
)

mult_cmp = lr_mult[["region", "f1_macro"]].rename(columns={"f1_macro": "lr_centroid"}).merge(
    cnn_mult[["region", "f1_macro"]].rename(columns={"f1_macro": "cnn_bbox"}),
    on="region",
)
mult_cmp["delta"] = mult_cmp["cnn_bbox"] - mult_cmp["lr_centroid"]
mult_cmp = mult_cmp.sort_values("delta", ascending=False)
mult_cmp.to_csv(OUT / "multiclass_loro_comparison.csv", index=False)
mult_cmp
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

# Per-region bars
order = mult_cmp.sort_values("lr_centroid")["region"]
x = np.arange(len(order))
w = 0.38
a = mult_cmp.set_index("region").loc[order, "lr_centroid"]
b = mult_cmp.set_index("region").loc[order, "cnn_bbox"]
axes[0].barh(x - w / 2, a, height=w, label="LR centroid", color="#4C78A8")
axes[0].barh(x + w / 2, b, height=w, label="CNN bbox-grid", color="#F58518")
axes[0].set_yticks(x)
axes[0].set_yticklabels(order)
axes[0].set_xlabel("LORO macro-F1")
axes[0].set_title("10-class building type · by held-out region")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].set_xlim(0, 1)

# Confusion — CNN
cm_ax = axes[1]
ConfusionMatrixDisplay.from_predictions(
    y_mult,
    oof_cnn_mult,
    display_labels=[lab.replace("_", "\n") for lab in LABEL_ORDER],
    cmap="Blues",
    colorbar=False,
    xticks_rotation=0,
    ax=cm_ax,
    values_format="d",
)
cm_ax.set_title("CNN LORO OOF confusion")
for lab in cm_ax.get_xticklabels() + cm_ax.get_yticklabels():
    lab.set_fontsize(7)

fig.tight_layout()
fig.savefig(OUT / "multiclass_loro_cnn_vs_lr.png", dpi=150, bbox_inches="tight")
plt.show()

print(
    classification_report(
        y_mult,
        oof_cnn_mult,
        target_names=LABEL_ORDER,
        digits=3,
        zero_division=0,
    )
)
"""
        ),
        md(
            """
## 4. Residential vs non-residential (notebooks 02 / 09 setting)

Same LORO protocol as the trial residential classifier, now with the CNN patch encoder.
"""
        ),
        code(
            r"""
print("=== Centroid LR (LORO, binary residential) ===")
lr_bin, oof_lr_bin = train_lr_loro(X_cent, y_bin)
print(lr_bin.sort_values("f1_macro", ascending=False).to_string(index=False))
print(f"mean macro-F1={lr_bin['f1_macro'].mean():.3f} ± {lr_bin['f1_macro'].std(ddof=1):.3f}")

print("\n=== BBox CNN (LORO, binary residential) ===")
cnn_bin, oof_cnn_bin, _ = train_cnn_loro(y_bin, n_classes=2, epochs=EPOCHS)
print(cnn_bin.sort_values("f1_macro", ascending=False).to_string(index=False))
print(f"mean macro-F1={cnn_bin['f1_macro'].mean():.3f} ± {cnn_bin['f1_macro'].std(ddof=1):.3f}")

bin_cmp = lr_bin[["region", "f1_macro"]].rename(columns={"f1_macro": "lr_centroid"}).merge(
    cnn_bin[["region", "f1_macro"]].rename(columns={"f1_macro": "cnn_bbox"}),
    on="region",
)
bin_cmp["delta"] = bin_cmp["cnn_bbox"] - bin_cmp["lr_centroid"]
bin_cmp = bin_cmp.sort_values("delta", ascending=False)
bin_cmp.to_csv(OUT / "binary_loro_comparison.csv", index=False)
bin_cmp
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
order = bin_cmp.sort_values("lr_centroid")["region"]
x = np.arange(len(order))
w = 0.38
a = bin_cmp.set_index("region").loc[order, "lr_centroid"]
b = bin_cmp.set_index("region").loc[order, "cnn_bbox"]
axes[0].barh(x - w / 2, a, height=w, label="LR centroid", color="#4C78A8")
axes[0].barh(x + w / 2, b, height=w, label="CNN bbox-grid", color="#F58518")
axes[0].set_yticks(x)
axes[0].set_yticklabels(order)
axes[0].set_xlabel("LORO macro-F1")
axes[0].set_title("Residential vs non-residential")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].set_xlim(0, 1)

ConfusionMatrixDisplay.from_predictions(
    y_bin,
    oof_cnn_bin,
    display_labels=["non-res", "residential"],
    cmap="Oranges",
    colorbar=False,
    ax=axes[1],
    values_format="d",
)
axes[1].set_title("CNN LORO OOF confusion")
fig.tight_layout()
fig.savefig(OUT / "binary_loro_cnn_vs_lr.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        md(
            """
## 5. Degree of Urbanization slice (notebook 09)

DoU labels come from the notebook 09 GHSL core join (partial overlap with this
multi-class set). We report CNN vs LR **out-of-fold** metrics inside urban /
peri-urban / rural for buildings that have a DoU tag.
"""
        ),
        code(
            r"""
has_dou = meta["dou"].isin(DOU_ORDER)
print(f"buildings with DoU={has_dou.sum()} / {len(meta)}")

dou_rows = []
for model_name, pred in [("lr_centroid", oof_lr_bin), ("cnn_bbox", oof_cnn_bin)]:
    for dou in DOU_ORDER:
        m = has_dou & (meta["dou"].to_numpy() == dou)
        if m.sum() == 0:
            continue
        dou_rows.append(
            {
                "model": model_name,
                "dou": dou,
                "n": int(m.sum()),
                "accuracy": float(accuracy_score(y_bin[m], pred[m])),
                "f1_macro": float(f1_score(y_bin[m], pred[m], average="macro", zero_division=0)),
                "recall_residential": float(
                    ((pred[m] == 1) & (y_bin[m] == 1)).sum() / max(1, (y_bin[m] == 1).sum())
                ),
            }
        )

dou_metrics = pd.DataFrame(dou_rows)
dou_metrics.to_csv(OUT / "binary_f1_by_dou.csv", index=False)
print(dou_metrics.to_string(index=False))

if len(dou_metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(DOU_ORDER))
    w = 0.36
    for i, (model_name, color) in enumerate(
        [("lr_centroid", "#4C78A8"), ("cnn_bbox", "#F58518")]
    ):
        sub = dou_metrics[dou_metrics["model"] == model_name].set_index("dou").reindex(DOU_ORDER)
        ax.bar(
            x + (i - 0.5) * w,
            sub["f1_macro"].to_numpy(),
            width=w,
            label=model_name,
            color=color,
            edgecolor="k",
            linewidth=0.4,
        )
        for xi, (f1, n) in enumerate(zip(sub["f1_macro"], sub["n"])):
            if pd.notna(f1):
                ax.text(xi + (i - 0.5) * w, f1 + 0.02, f"{f1:.2f}\nn={int(n)}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DOU_LABELS[d] for d in DOU_ORDER])
    ax.set_ylabel("OOF macro-F1 (binary residential)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Notebook 09 slice · F1 by Degree of Urbanization")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "binary_f1_by_dou.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No DoU overlap — skip plot.")
"""
        ),
        md("## 6. Snapshot"),
        code(
            r"""
summary = {
    "device": str(DEVICE),
    "n_buildings": int(len(meta)),
    "pad": [PAD_H, PAD_W],
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "multiclass": {
        "lr_centroid_macro_f1_mean": float(lr_mult["f1_macro"].mean()),
        "cnn_bbox_macro_f1_mean": float(cnn_mult["f1_macro"].mean()),
        "delta_mean": float((cnn_mult["f1_macro"] - lr_mult["f1_macro"]).mean()),
    },
    "binary_residential": {
        "lr_centroid_macro_f1_mean": float(lr_bin["f1_macro"].mean()),
        "cnn_bbox_macro_f1_mean": float(cnn_bin["f1_macro"].mean()),
        "delta_mean": float((cnn_bin["f1_macro"] - lr_bin["f1_macro"]).mean()),
    },
    "dou_coverage": int(has_dou.sum()),
    "dou_metrics": dou_metrics.to_dict(orient="records") if len(dou_metrics) else [],
    "outputs": sorted(p.name for p in OUT.iterdir()),
}
with open(OUT / "cnn_bbox_snapshot.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nWrote", OUT)
"""
        ),
        md(
            """
### How to read this

| Piece | What it mirrors | What changed |
|---|---|---|
| §3 multi-class LORO | Notebook 08 labels / notebook 05 protocol | CNN on bbox grid vs centroid LR |
| §4 binary LORO | Notebooks 02 / 09 residential split | Same, with CNN patch encoder |
| §5 DoU slice | Notebook 09 urban / peri-urban / rural | OOF metrics on the GHSL-joined subset |

**Claim language.** This is still AlphaEarth-only on high-precision Overture tags.
The CNN is an architecture ablation on **how** we pool the 10 m cells under each
footprint bbox — not a new modality.
"""
        ),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbf.write(nb, OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
