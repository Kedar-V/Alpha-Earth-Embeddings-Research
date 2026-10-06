#!/usr/bin/env python3
"""Plot a few validation-split clusters from the NYC fold's best checkpoint.

The logged val gap is the mean sep/within on a 10% split of the training
metros. Those coordinates were not saved. This rebuilds that split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_week4_vit_100m_3d import _cluster_stats  # noqa: E402
from scripts.train_week4_vit_100m_3d_v4 import AEPatchViT3DCluster  # noqa: E402
from scripts.train_week4_vit_100m_loro import (  # noqa: E402
    SEED,
    TilePatchDataset,
    _project_pls,
)
import build_week4_vit_100m_3d_notebook as viz  # noqa: E402

HELD = "NYC_NE"
FOLD = ROOT / "outputs" / "week4_vit_100m_3d_v4" / "folds" / HELD
CACHE = ROOT / "outputs" / "week4_vit_100m_3d_v4" / "_cache"
FIG = ROOT / "outputs" / "week4_vit_100m_3d_v4" / "figures"
N_SHOW = 99  # all validation metros in the NYC fold


def _plot(region: str, z: np.ndarray, y: np.ndarray, w: np.ndarray, gap: float, sep: float, within: float, n0: int, n1: int) -> Path:
    return viz.plot_metro(
        region,
        z,
        y,
        w,
        float("nan"),
        sep,
        within,
        subtitle=(
            f"Validation tiles while New York is held out. n={n0 + n1} "
            f"({n0} non-residential, {n1} residential). Gap {gap:.2f}. "
            "Mauve is where both centers still reach. Influence falls with distance from a center. The plane and the dashed line are the classifier cut."
        ),
        path=FIG / f"val_nycfold_{region}.png",
    )


def main() -> None:
    device = torch.device("cuda:0")
    ckpt = torch.load(FOLD / "best.pt", map_location=device, weights_only=False)
    arch = ckpt["arch"]
    regions = np.load(CACHE / "regions.npy", allow_pickle=True)
    y = np.load(CACHE / "y.npy")
    mask = np.load(CACHE / "mask.npy", mmap_mode="r")
    patches64 = np.load(CACHE / "patches.npy", mmap_mode="r")
    tr_all = np.where(regions != HELD)[0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_all)
    n_val = max(1, int(len(tr_all) * 0.1))
    va = tr_all[:n_val]

    patches = _project_pls(
        patches64,
        mask,
        ckpt["mu_ae"],
        ckpt["sd_ae"],
        ckpt["pls_rotation"],
        device,
    )
    loader = DataLoader(
        TilePatchDataset(va, patches, np.asarray(mask), y, ckpt["mu"], ckpt["sd"]),
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model = AEPatchViT3DCluster(
        patch_dim=int(arch["patch_dim"]),
        d_model=int(arch["d_model"]),
        n_heads=int(arch["n_heads"]),
        n_layers=int(arch["n_layers"]),
        dropout=float(arch["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    import torch.nn.functional as F

    head = F.normalize(model.head.weight.detach(), dim=-1).cpu().numpy().reshape(-1)
    head = head * (1.0 if float(model.scale.detach().cpu()) >= 0 else -1.0)
    zs = []
    with torch.no_grad():
        for xb, mb, _yb, _idx in loader:
            _logits, z = model(xb.to(device, non_blocking=True), mb.to(device, non_blocking=True))
            zs.append(z.cpu().numpy())
    z = np.concatenate(zs)
    yv = y[va]
    rv = regions[va]
    rows = []
    for region in sorted(set(rv.tolist())):
        m = rv == region
        sep, within = _cluster_stats(z[m], yv[m])
        if not np.isfinite(sep) or not np.isfinite(within) or within <= 0:
            continue
        rows.append(
            {
                "region": region,
                "gap": sep / within,
                "sep": sep,
                "within": within,
                "n0": int((yv[m] == 0).sum()),
                "n1": int((yv[m] == 1).sum()),
                "z": z[m],
                "y": yv[m].astype(int),
            }
        )
    rows.sort(key=lambda r: r["gap"], reverse=True)
    mean_gap = float(np.mean([r["gap"] for r in rows]))
    print(f"val regions={len(rows)} mean gap={mean_gap:.3f} logged best={ckpt.get('best_metric_value')}")
    for r in rows:
        print(f"  {r['region']:14} gap={r['gap']:.2f} sep={r['sep']:.3f} within={r['within']:.3f} n0={r['n0']} n1={r['n1']}")
    FIG.mkdir(parents=True, exist_ok=True)
    # Widest gaps, and the narrowest, so the picture is not only the best panels.
    show = rows[:N_SHOW]
    if rows[-1]["region"] not in {r["region"] for r in show}:
        show.append(rows[-1])
    for r in show:
        path = _plot(r["region"], r["z"], r["y"], head, r["gap"], r["sep"], r["within"], r["n0"], r["n1"])
        print("wrote", path.name)


if __name__ == "__main__":
    main()
