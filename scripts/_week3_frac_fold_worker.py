#!/usr/bin/env python3
"""Single LORO fold worker. Expects CUDA_VISIBLE_DEVICES already set by parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 40
BATCH = 4096
LR = 1e-3
WD = 1e-4
DROPOUT = 0.1
BANDS = [f"A{i:02d}" for i in range(64)]


class FractionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--regions", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[fold {args.region}] device={device} physical_gpu={args.physical_gpu}", flush=True)

    X = np.load(args.x, mmap_mode="r")
    Y = np.load(args.y, mmap_mode="r")
    regions = np.load(args.regions, allow_pickle=True)
    te = regions == args.region
    tr = ~te

    Xtr = torch.from_numpy(np.asarray(X[tr], dtype=np.float32))
    Ytr = torch.from_numpy(np.asarray(Y[tr], dtype=np.float32))
    Xte = torch.from_numpy(np.asarray(X[te], dtype=np.float32))

    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    torch.manual_seed(SEED + abs(hash(args.region)) % 10_000)
    model = FractionMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)
        if (epoch + 1) % 10 == 0:
            print(f"  [{args.region}] epoch {epoch+1}/{EPOCHS} loss={total/max(n,1):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(Xte_n.to(device)).cpu().numpy()

    y_te = np.asarray(Y[te], dtype=np.float32)
    mae_res = float(mean_absolute_error(y_te[:, 0], pred[:, 0]))
    mae_nres = float(mean_absolute_error(y_te[:, 1], pred[:, 1]))
    rmse_res = float(root_mean_squared_error(y_te[:, 0], pred[:, 0]))
    rmse_nres = float(root_mean_squared_error(y_te[:, 1], pred[:, 1]))

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = ckpt_dir.parent / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"oof_pred_{args.region}.npy"
    idx_path = cache / f"oof_idx_{args.region}.npy"
    np.save(pred_path, pred.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu": mu.numpy(),
            "sd": sd.numpy(),
            "region": args.region,
            "arch": "FractionMLP",
            "epochs": EPOCHS,
            "bands": BANDS,
            "physical_gpu": args.physical_gpu,
        },
        ckpt_dir / f"frac_mlp_loro_{args.region}.pt",
    )

    result = {
        "region": args.region,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "n_test": int(te.sum()),
        "mae_res": mae_res,
        "mae_nres": mae_nres,
        "rmse_res": rmse_res,
        "rmse_nres": rmse_nres,
        "gpu": args.physical_gpu,
    }
    Path(args.out_json).write_text(json.dumps(result) + "\n")
    print(
        f"  done · {args.region:12s} gpu={args.physical_gpu}  "
        f"MAE_res={mae_res:.3f} MAE_nres={mae_nres:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
