#!/usr/bin/env python3
"""Encode 100 m tiles with each frozen v4 fold and compare class directions.

No training. Each checkpoint was fit without its held-out metro. Class directions
of the source metros and of the held-out metro are measured in that same 3D space.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_week4_vit_100m_3d_v4 import AEPatchViT3DCluster  # noqa: E402
from train_week4_vit_100m_loro import TilePatchDataset, _project_pls  # noqa: E402

import lib  # noqa: E402

V4 = ROOT / "outputs" / "week4_vit_100m_3d_v4"


def _direction(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = z[y == 1].mean(0) - z[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def _encode_fold(region: str, device: torch.device, log_path: Path) -> dict:
    fold = V4 / "folds" / region
    ckpt = torch.load(fold / "best.pt", map_location=device, weights_only=False)
    regions = np.load(V4 / "_cache" / "regions.npy", allow_pickle=True).astype(str)
    y = np.load(V4 / "_cache" / "y.npy").astype(np.int64)
    mask = np.load(V4 / "_cache" / "mask.npy", mmap_mode="r")
    patches64 = np.load(V4 / "_cache" / "patches.npy", mmap_mode="r")
    lib.log(log_path, f"project {region} on {device}")
    patches = _project_pls(patches64, mask, ckpt["mu_ae"], ckpt["sd_ae"], ckpt["pls_rotation"], device, log_path=log_path)
    arch = ckpt["arch"]
    model = AEPatchViT3DCluster(
        patch_dim=int(arch["patch_dim"]),
        d_model=int(arch["d_model"]),
        n_heads=int(arch["n_heads"]),
        n_layers=int(arch["n_layers"]),
        dropout=float(arch["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(
        TilePatchDataset(np.arange(len(y)), patches, np.asarray(mask), y, ckpt["mu"], ckpt["sd"]),
        batch_size=512,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    zs = []
    with torch.no_grad():
        for xb, mb, _yb, _idx in loader:
            _logits, z = model(xb.to(device, non_blocking=True), mb.to(device, non_blocking=True))
            zs.append(z.cpu().numpy())
    z = np.concatenate(zs)
    dirs = {}
    for metro in sorted(set(regions.tolist())):
        m = regions == metro
        dirs[metro] = _direction(z[m], y[m])
    held = dirs[region]
    rows = []
    for metro, vec in dirs.items():
        rows.append(
            {
                "fold_held_out": region,
                "metro": metro,
                "cosine_to_heldout_direction": float(np.dot(held, vec)),
                "is_held_out": metro == region,
            }
        )
    # linear probe: train LR on source metros' z, test on held-out z
    src = regions != region
    pred, proba, _clf = lib.fit_lr(z[src], y[src], z[~src])
    metrics = lib.binary_metrics(y[~src], pred, proba)
    metrics.update({"fold_held_out": region, "model": "lr_on_v4_z"})
    np.savez_compressed(log_path.parent / f"z_{region}.npz", z=z.astype(np.float32), y=y, regions=regions)
    return {"rows": rows, "probe": metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "v4_frozen_directions",
            "command": " ".join(sys.argv),
            "gpus": 1,
            "gpu": args.gpu,
            "note": "Inference only. No v4 weights are updated.",
        },
    )
    log_path = run_dir / "train.log"
    try:
        device = torch.device(f"cuda:{args.gpu}")
        folds = sorted(p.parent.name for p in (V4 / "folds").glob("*/DONE"))
        if len(folds) != 10:
            raise RuntimeError(f"expected 10 v4 folds, found {folds}")
        t0 = time.time()
        all_rows = []
        probes = []
        for i, region in enumerate(folds, start=1):
            lib.log(log_path, f"fold {i}/10 {region}")
            out = _encode_fold(region, device, log_path)
            all_rows.extend(out["rows"])
            probes.append(out["probe"])
            lib.atomic_df(run_dir / "direction_cosine.csv", __import__("pandas").DataFrame(all_rows))
            lib.atomic_df(run_dir / "probe_metrics.csv", __import__("pandas").DataFrame(probes))
        lib.atomic_json(run_dir / "summary.json", {"status": "completed", "runtime_seconds": time.time() - t0, "folds": folds})
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
