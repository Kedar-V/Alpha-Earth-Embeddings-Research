#!/usr/bin/env python3
"""Held-out v4 3D cluster figures for the PLS comparison notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_week4_vit_100m_3d_notebook as viz

V4 = ROOT / "outputs" / "week4_vit_100m_3d_v4"
V3 = ROOT / "outputs" / "week4_vit_100m_3d_v3" / "loro_metrics.csv"
FIG = V4 / "figures"
Y_PATH = V4 / "_cache" / "y.npy"

INK = "#222222"
MUTED = "#555555"
BLUE = "#0072B2"
GREY = "#8A8A8A"


def _done_rows() -> list[dict]:
    y_all = np.load(Y_PATH)
    v3 = pd.read_csv(V3).set_index("region")
    v3_ratio = v3["test_centroid_sep"] / v3["test_within"]
    rows = []
    for done in sorted((V4 / "folds").glob("*/DONE")):
        rec = json.loads(done.read_text())
        region = rec["region"]
        fold = done.parent
        idx = np.load(fold / "oof_idx.npy")
        z = np.load(fold / "oof_z3.npy")
        rows.append(
            {
                "region": region,
                "name": viz._name(region),
                "z": z,
                "y": y_all[idx].astype(int),
                "f1": float(rec["macro_f1"]),
                "sep": float(rec["test_centroid_sep"]),
                "within": float(rec["test_within"]),
                "ratio": float(rec["test_ratio"]),
                "v3_f1": float(v3.loc[region, "macro_f1"]),
                "v3_ratio": float(v3_ratio.loc[region]),
            }
        )
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return rows


def _gap_bars(rows: list[dict]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r["ratio"])
    y = np.arange(len(ordered))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh(y - h / 2, [r["v3_ratio"] for r in ordered], height=h, color=GREY, zorder=2, label="v3")
    ax.barh(y + h / 2, [r["ratio"] for r in ordered], height=h, color=BLUE, zorder=2, label="v4")
    for i, r in enumerate(ordered):
        ax.text(r["ratio"] + 0.08, i + h / 2, f"{r['ratio']:.2f}", va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(3, color="#333333", lw=0.9, ls="--", zorder=1)
    ax.axvline(1, color="#333333", lw=0.9, ls=":", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in ordered])
    ax.set_xlabel("Gap  (distance between class centers ÷ size of each class)")
    ax.set_xlim(0, max(4.0, max(r["ratio"] for r in ordered) + 0.7))
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.14)
    clear = [r["name"] for r in rows if r["ratio"] >= 3]
    if len(clear) == 2:
        title = f"{clear[0]} and {clear[1]} clear a gap of 3"
    elif len(clear) > 2:
        title = f"{', '.join(clear[:-1])}, and {clear[-1]} clear a gap of 3"
    elif len(clear) == 1:
        title = f"{clear[0]} clears a gap of 3"
    else:
        title = "No held-out metro clears a gap of 3"
    viz._header(
        fig,
        title,
        "Held-out 100 m tiles. Blue is v4. Grey is v3. Dotted line = 1 (one cloud). Dashed line = 3 (clear gap).",
        title_y=0.96,
        sub_y=0.88,
    )
    path = FIG / "v4_vs_v3_gap.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _oriented_head(region: str) -> np.ndarray:
    import torch

    blob = torch.load(V4 / "folds" / region / "best.pt", map_location="cpu", weights_only=False)
    scale = float(np.asarray(blob["model"]["scale"]).reshape(-1)[0])
    w = np.load(V4 / "folds" / region / "head_w.npy").astype(np.float64).reshape(-1)
    return w if scale >= 0 else -w


def write_figures() -> list[dict]:
    rows = _done_rows()
    if not rows:
        return []
    viz.FIG = FIG
    FIG.mkdir(parents=True, exist_ok=True)
    _gap_bars(rows)
    for r in rows:
        viz.plot_metro(
            r["region"],
            r["z"],
            r["y"],
            _oriented_head(r["region"]),
            r["f1"],
            r["sep"],
            r["within"],
        )
    return rows


def markdown() -> str:
    rows = _done_rows()
    if not rows:
        return ""
    # Figures are written by write_figures(); this only describes files already on disk.
    table = [
        "| Metro | v4 F1 | v3 F1 | v4 gap | v3 gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r['name']} | {r['f1']:.3f} | {r['v3_f1']:.3f} | {r['ratio']:.2f} | {r['v3_ratio']:.2f} |"
        )
    pending = sorted(
        p.name for p in (V4 / "folds").iterdir() if p.is_dir() and not (p / "DONE").exists()
    )
    pending_line = ""
    if pending:
        names = ", ".join(viz._name(p) for p in pending)
        pending_line = f"\n{names} {'is' if len(pending) == 1 else 'are'} still training and not in the figure.\n"
    gallery = []
    for r in rows:
        gallery.append(
            f"### {r['name']}\n\n"
            f"![3D {r['name']}](../../outputs/week4_vit_100m_3d_v4/figures/vit3d_{r['region']}.png)\n"
        )
    return f"""## Held-out 3D clusters after the within-region loss

Each panel is that metro's raw encoder output on the tiles the fold did not train on. The axes are encoder 1, 2, and 3. Blue is residential. Orange is non-residential. The gap is the distance between the two class centers divided by the mean radius. A gap of 3 is a clear separation. A gap below 1 is one cloud. v3 is the previous contrastive run.
{pending_line}
{chr(10).join(table)}

![v4 versus v3 gap](../../outputs/week4_vit_100m_3d_v4/figures/v4_vs_v3_gap.png)

{chr(10).join(gallery)}
Each fold has its own axes. The coordinates are not pooled across metros.
"""


def main() -> None:
    rows = write_figures()
    print(f"Wrote {len(rows)} metros → {FIG}")


if __name__ == "__main__":
    main()
