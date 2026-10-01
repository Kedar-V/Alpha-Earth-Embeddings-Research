#!/usr/bin/env python3
"""Export leave-one-region-out CNN predictions from saved checkpoints.

Reads outputs/week2_cnn_bbox/checkpoints/bbox_cnn_loro_<region>.pt (10-class)
and writes outputs/week2_cnn_bbox/cnn_oof_predictions.parquet for map QA.

Also derives a residential binary view by collapsing single_family + multi_family.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
BBOX = ROOT / "datasets" / "building_bbox_grid"
OUT = ROOT / "outputs" / "week2_cnn_bbox"
CKPT_DIR = OUT / "checkpoints"

BANDS = [f"A{i:02d}" for i in range(64)]
EMB_DIM = 256
PAD_H, PAD_W = 32, 32
BATCH_SIZE = 128
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
RESIDENTIAL = {"single_family", "multi_family"}


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
        grids[i][:, ii, jj] = vals.T
        masks[i, ii, jj] = 1.0
    return grids, masks


def predict_region(
    te_region: str,
    gpu_id: int,
    meta: pd.DataFrame,
    y: np.ndarray,
    regions: np.ndarray,
    grids: np.ndarray,
    masks: np.ndarray,
) -> pd.DataFrame:
    ckpt_path = CKPT_DIR / f"bbox_cnn_loro_{te_region}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_classes = int(ckpt["n_classes"])
    model = BBoxCNN(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    te = np.where(regions == te_region)[0]
    loader = DataLoader(
        BuildingPatchDataset(te, y, grids, masks),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    preds, probs, idxs = [], [], []
    with torch.no_grad():
        for xb, _, idx in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pr = torch.softmax(logits, dim=1)
            pred = pr.argmax(dim=1)
            conf, _ = pr.max(dim=1)
            preds.append(pred.cpu().numpy())
            probs.append(conf.cpu().numpy())
            idxs.append(idx.numpy())

    pred = np.concatenate(preds)
    conf = np.concatenate(probs)
    te_idx = np.concatenate(idxs)
    rows = meta.iloc[te_idx][["id", "region", "label", "y_res", "lon", "lat"]].copy()
    rows["y_true"] = y[te_idx]
    rows["pred"] = pred
    rows["pred_label"] = [LABEL_ORDER[int(p)] for p in pred]
    rows["pred_prob"] = conf
    rows["true_label"] = rows["label"].astype(str)
    rows["misclassified_multi"] = rows["true_label"] != rows["pred_label"]
    rows["y_res"] = rows["y_res"].astype(int)
    rows["pred_res"] = rows["pred_label"].isin(RESIDENTIAL).astype(int)
    rows["misclassified_res"] = rows["pred_res"] != rows["y_res"]
    # FP: predicted residential but not; FN: residential missed
    err = np.full(len(rows), "correct", dtype=object)
    err[(rows["pred_res"] == 1) & (rows["y_res"] == 0)] = "fp"
    err[(rows["pred_res"] == 0) & (rows["y_res"] == 1)] = "fn"
    err[rows["misclassified_multi"] & ~rows["misclassified_res"]] = "multi_only"
    # keep residential error type when both wrong; multi_only when res correct but type wrong
    err_res = np.full(len(rows), "correct", dtype=object)
    err_res[(rows["pred_res"] == 1) & (rows["y_res"] == 0)] = "fp"
    err_res[(rows["pred_res"] == 0) & (rows["y_res"] == 1)] = "fn"
    rows["error_res"] = err_res
    rows["error_multi"] = np.where(rows["misclassified_multi"], "wrong_type", "correct")
    return rows.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0, help="GPU workers (0=all)")
    args = ap.parse_args()

    buildings = pd.read_parquet(BBOX / "buildings.parquet")
    cells = pd.read_parquet(BBOX / "building_cells.parquet")
    buildings["id"] = buildings["id"].astype(str)
    cells["id"] = cells["id"].astype(str)
    buildings = buildings.sort_values("id").reset_index(drop=True)
    label_to_y = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    buildings["y"] = buildings["label"].map(label_to_y).astype(int)

    grids, masks = build_grids(buildings, cells)
    regions = buildings["region"].astype(str).to_numpy()
    y = buildings["y"].to_numpy()
    region_list = sorted(buildings["region"].unique())
    missing = [r for r in region_list if not (CKPT_DIR / f"bbox_cnn_loro_{r}.pt").exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints for: {missing}")

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    workers = args.workers or max(1, n_gpus)
    print(f"regions={len(region_list)} workers={workers} gpus={n_gpus}", flush=True)

    parts: list[pd.DataFrame] = []
    if n_gpus and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    predict_region,
                    reg,
                    i % n_gpus,
                    buildings,
                    y,
                    regions,
                    grids,
                    masks,
                ): reg
                for i, reg in enumerate(region_list)
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc="OOF predict"):
                parts.append(fut.result())
                print(f"  done {futs[fut]}", flush=True)
    else:
        for i, reg in enumerate(tqdm(region_list, desc="OOF predict")):
            parts.append(predict_region(reg, 0 if n_gpus else -1, buildings, y, regions, grids, masks))

    out = pd.concat(parts, ignore_index=True)
    # align to buildings order
    out = buildings[["id"]].merge(out, on="id", how="left")
    path = OUT / "cnn_oof_predictions.parquet"
    out.to_parquet(path, index=False)

    n = len(out)
    multi_err = int(out["misclassified_multi"].sum())
    res_err = int(out["misclassified_res"].sum())
    meta = {
        "n": n,
        "multiclass_error_rate": multi_err / n,
        "residential_error_rate": res_err / n,
        "n_misclassified_multi": multi_err,
        "n_misclassified_res": res_err,
        "source": "LORO BBoxCNN checkpoints (10-class); residential = collapse sf+mf",
        "checkpoints": str(CKPT_DIR),
        "path": str(path),
    }
    (OUT / "cnn_oof_predictions_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
