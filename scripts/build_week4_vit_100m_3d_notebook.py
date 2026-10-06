#!/usr/bin/env python3
"""Plot per-metro 3D ViT spaces and write notebooks/week4/12_vit_100m_3d.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_vit_100m_3d"
FIG = OUT / "figures"
NB = ROOT / "notebooks" / "week4" / "12_vit_100m_3d.ipynb"

RES = "#0072B2"
NRES = "#D55E00"
INK = "#222222"
MUTED = "#555555"

NAMES = {
    "Atlanta_SE": "Atlanta",
    "Boston_NE": "Boston",
    "Chicago_MW": "Chicago",
    "Denver_W": "Denver",
    "Houston_S": "Houston",
    "LA_W": "Los Angeles",
    "NYC_NE": "New York",
    "Phoenix_SW": "Phoenix",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}


def _name(region: str) -> str:
    return NAMES.get(region, region.replace("_", " "))


def _ratio(sep: float, within: float) -> float:
    if within is None or within != within or within <= 0:
        return float("nan")
    return float(sep) / float(within)


def _frame(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Columns: classifier (residential to the right), widest leftover, thinnest leftover."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)
    resid = z - np.outer(z @ w, w)
    _, _, vt = np.linalg.svd(resid - resid.mean(axis=0), full_matrices=False)
    a = vt[0] - w * (w @ vt[0])
    a = a / (np.linalg.norm(a) + 1e-12)
    b = np.cross(w, a)
    return z @ np.stack([w, a, b], axis=1)


def _orient_residential_right(coords: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = coords.copy()
    if y.size and out[y == 1, 0].mean() < out[y == 0, 0].mean():
        out[:, 0] *= -1
    return out


def _header(fig, title: str, subtitle: str, title_y: float = 0.98, sub_y: float = 0.92) -> None:
    fig.text(0.01, title_y, title, fontsize=13, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.01, sub_y, subtitle, fontsize=9, ha="left", va="top", color=MUTED)


def _claim(name: str, ratio: float) -> str:
    if ratio >= 2.5:
        return f"{name}: residential and non-residential are apart"
    if ratio >= 1.5:
        return f"{name}: the two classes touch"
    return f"{name}: residential and non-residential are one cloud"


def plot_gap_bars(metrics: pd.DataFrame) -> Path:
    """One claim: which metros have a real gap."""
    d = metrics.sort_values("ratio", ascending=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    color = ["#0072B2" if r >= 2.5 else "#8A8A8A" for r in d["ratio"]]
    y = np.arange(len(d))
    ax.barh(y, d["ratio"], color=color, height=0.72, zorder=2)
    for i, val in enumerate(d["ratio"]):
        ax.text(val + 0.12, i, f"{val:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.axvline(3, color="#333333", lw=0.9, ls="--", zorder=1)
    ax.axvline(1, color="#333333", lw=0.9, ls=":", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_name(r) for r in d["region"]])
    ax.set_xlabel("Gap  (distance between class centers ÷ size of each class)")
    ax.set_xlim(0, max(4.0, float(d["ratio"].max()) + 0.85))
    ax.set_ylim(-0.55, len(d) - 0.45)
    ax.tick_params(length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.16)
    _header(
        fig,
        "Phoenix is the only held-out metro with a clear gap",
        "Grey bars overlap or only touch. Blue is a gap of 3 or more. Dotted line = 1 (one cloud). Dashed line = 3 (clear gap).",
        title_y=0.96,
        sub_y=0.88,
    )
    path = FIG / "vit3d_sep_ratio.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _project2(z: np.ndarray) -> np.ndarray:
    """Drop the encoder output onto the plane where it spreads the most."""
    xc = z - z.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


def plot_class_axis(z: np.ndarray, y: np.ndarray, regions: np.ndarray, heads: dict, metrics: pd.DataFrame) -> Path:
    """2D projection of the 3 encoder outputs, largest gap to smallest."""
    del heads
    order = metrics.sort_values("ratio", ascending=False)["region"].tolist()
    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(2, 5, figsize=(12.4, 6.0))
    for ax, region in zip(axes.ravel(), order):
        m = regions == region
        xy = _project2(z[m])
        yy = y[m]
        keep = []
        for lab in (0, 1):
            idx = np.where(yy == lab)[0]
            if len(idx) > 900:
                idx = rng.choice(idx, 900, replace=False)
            keep.append(idx)
        sel = np.concatenate(keep)
        pts = xy[sel]
        for lab, color in ((0, NRES), (1, RES)):
            hit = yy[sel] == lab
            ax.scatter(pts[hit, 0], pts[hit, 1], s=6, c=color, alpha=0.4, linewidths=0)
        span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
        mid = pts.mean(axis=0)
        half = 0.58 * span
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ratio = float(metrics.loc[metrics["region"] == region, "ratio"].iloc[0])
        ax.set_title(f"{_name(region)}   {ratio:.2f}", loc="left", fontsize=9, color=INK, pad=3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.tick_params(labelsize=7, length=2, colors="#666666")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1:].ravel():
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.12, wspace=0.22, hspace=0.32)
    fig.text(0.52, 0.02, "projection 1 of the encoder output", fontsize=8, color=MUTED, ha="center")
    fig.text(0.012, 0.46, "projection 2", fontsize=8, color=MUTED, ha="left", va="center", rotation=90)
    _header(
        fig,
        "The classes pull apart in Phoenix and sit together in New York",
        "Each panel projects that metro's three encoder outputs onto the plane they spread across. Own scale. The number is the gap. Blue is residential. Orange is non-residential.",
        title_y=0.97,
        sub_y=0.90,
    )
    path = FIG / "vit3d_projection.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _cut_basis(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit classifier axis, widest leftover axis, and the third axis of the cut plane.

    The displayed classifier axis points toward residential, so the cut stays at score 0.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)
    if np.any(y == 1) and np.any(y == 0) and (z[y == 1] @ w).mean() < (z[y == 0] @ w).mean():
        w = -w
    resid = z - np.outer(z @ w, w)
    _, _, vt = np.linalg.svd(resid - resid.mean(axis=0), full_matrices=False)
    across = vt[0] - w * np.dot(w, vt[0])
    if np.linalg.norm(across) < 1e-8:
        seed = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        across = seed - w * np.dot(w, seed)
    across = across / np.linalg.norm(across)
    normal = np.cross(w, across)
    return w, across, normal


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)


_RES_LIN = _srgb_to_linear(mcolors.to_rgb(RES))
_NRES_LIN = _srgb_to_linear(mcolors.to_rgb(NRES))
OVERLAP = mcolors.to_hex(_linear_to_srgb(0.5 * _RES_LIN + 0.5 * _NRES_LIN))


def _centers_radii(coords: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class center and the mean radius of that class. Influence falls as radius grows."""
    centers = np.zeros((2, coords.shape[1]), dtype=np.float64)
    radii = np.zeros(2, dtype=np.float64)
    for lab in (0, 1):
        pts = coords[labels == lab]
        center = np.median(pts, axis=0)
        radius = float(np.linalg.norm(pts - center, axis=1).mean())
        centers[lab] = center
        radii[lab] = max(radius, 1e-6)
    # Reach at least halfway to the other center, so the overlap is a mix instead of a gap.
    gap = float(np.linalg.norm(centers[0] - centers[1]))
    radii = np.maximum(radii, 0.5 * gap)
    return centers, radii


def _influences(query: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for lab in (0, 1):
        dist = np.linalg.norm(query - centers[lab], axis=-1)
        out.append(np.exp(-0.5 * (dist / radii[lab]) ** 2))
    return out[0], out[1]


def _blend_rgb(inf_nres: np.ndarray, inf_res: np.ndarray) -> np.ndarray:
    total = inf_nres + inf_res + 1e-12
    t = (inf_res / total)[..., None]
    linear = (1.0 - t) * _NRES_LIN + t * _RES_LIN
    return np.clip(_linear_to_srgb(linear), 0, 1)


def _robust_lim(values: np.ndarray, pad: float = 0.14) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.01, 0.99])
    span = max(float(hi - lo), 1e-6)
    return float(lo - pad * span), float(hi + pad * span)


def _draw_cut_plane(ax, z: np.ndarray, across: np.ndarray, normal: np.ndarray) -> None:
    """The classifier cut is the plane through the origin perpendicular to the class axis."""
    ua = z @ across
    ub = z @ normal
    lo_a, hi_a = np.quantile(ua, [0.01, 0.99])
    lo_b, hi_b = np.quantile(ub, [0.01, 0.99])
    pad_a = 0.06 * (hi_a - lo_a + 1e-6)
    pad_b = 0.06 * (hi_b - lo_b + 1e-6)
    aa, bb = np.meshgrid(
        np.linspace(lo_a - pad_a, hi_a + pad_a, 2),
        np.linspace(lo_b - pad_b, hi_b + pad_b, 2),
    )
    plane = aa[..., None] * across + bb[..., None] * normal
    ax.plot_surface(
        plane[..., 0],
        plane[..., 1],
        plane[..., 2],
        color="#4D4D4D",
        alpha=0.18,
        linewidth=0,
        shade=False,
        zorder=0,
    )
    corners = np.array(
        [
            across * aa[0, 0] + normal * bb[0, 0],
            across * aa[0, 1] + normal * bb[0, 1],
            across * aa[1, 1] + normal * bb[1, 1],
            across * aa[1, 0] + normal * bb[1, 0],
            across * aa[0, 0] + normal * bb[0, 0],
        ]
    )
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color="#333333", lw=1.0, zorder=1)


def plot_metro(
    region: str,
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    f1: float,
    sep: float,
    within: float,
    subtitle: str | None = None,
    path: Path | None = None,
) -> Path:
    rng = np.random.RandomState(42)
    keep = []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        if len(idx) > 800:
            idx = rng.choice(idx, 800, replace=False)
        keep.append(idx)
    sel = np.concatenate(keep)
    pts = z[sel]
    yy = y[sel]
    ratio = _ratio(sep, within)
    name = _name(region)
    direction, across, normal = _cut_basis(pts, yy, w)
    score = pts @ direction
    side = pts @ across

    centers3, radii3 = _centers_radii(pts, yy)
    inf_nres, inf_res = _influences(pts, centers3, radii3)
    strength = inf_nres + inf_res
    reach = 1.0 - np.exp(-2.2 * strength)
    face = np.concatenate([_blend_rgb(inf_nres, inf_res), (0.22 + 0.68 * reach)[:, None]], axis=1)
    rank = np.argsort(strength, kind="mergesort")

    fig = plt.figure(figsize=(12.4, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    _draw_cut_plane(ax, pts, across, normal)
    ax.scatter(
        pts[rank, 0],
        pts[rank, 1],
        pts[rank, 2],
        s=8 + 16 * reach[rank],
        c=face[rank],
        linewidths=0,
        depthshade=False,
        zorder=2,
    )
    ax.scatter(
        centers3[:, 0],
        centers3[:, 1],
        centers3[:, 2],
        s=36,
        c=[NRES, RES],
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
        zorder=3,
    )
    spans = np.array([_robust_lim(pts[:, i])[1] - _robust_lim(pts[:, i])[0] for i in range(3)])
    ax.set_box_aspect(tuple(np.clip(spans / spans.max(), 0.32, 1.0)))
    for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(*_robust_lim(pts[:, i], pad=0.08))
    ax.set_xlabel("encoder 1", labelpad=8, fontsize=8)
    ax.set_ylabel("encoder 2", labelpad=8, fontsize=8)
    ax.set_zlabel("encoder 3", labelpad=8, fontsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7, pad=1)
    ax.view_init(elev=16, azim=-62)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#E0E0E0")

    ax2 = fig.add_subplot(122)
    flat = np.column_stack([score, side])
    inf0, inf1 = _influences(flat, *_centers_radii(flat, yy))
    reach2 = 1.0 - np.exp(-2.2 * (inf0 + inf1))
    face2 = np.concatenate([_blend_rgb(inf0, inf1), (0.2 + 0.72 * reach2)[:, None]], axis=1)
    rank2 = np.argsort(inf0 + inf1, kind="mergesort")
    ax2.scatter(
        score[rank2],
        side[rank2],
        s=4 + 22 * reach2[rank2],
        c=face2[rank2],
        linewidths=0,
        zorder=2,
    )
    ax2.axvline(0, color="#333333", lw=1.1, ls="--", zorder=3)
    ax2.text(0, 1.03, "cut", transform=ax2.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK)
    centers2, _ = _centers_radii(flat, yy)
    ax2.scatter(centers2[:, 0], centers2[:, 1], s=28, c=[NRES, RES], edgecolors="white", linewidths=0.6, zorder=4)
    x0, x1 = _robust_lim(score)
    y0, y1 = _robust_lim(side)
    xspan = x1 - x0
    if (y1 - y0) < 0.12 * xspan:
        mid = 0.5 * (y0 + y1)
        half = 0.06 * xspan
        y0, y1 = mid - half, mid + half
    ax2.set_xlabel("classifier score", fontsize=8)
    ax2.set_ylabel("across the cut", fontsize=8)
    ax2.tick_params(labelsize=7, length=2, colors="#666666")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    ax2.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=NRES, markeredgecolor="white", markersize=7, label="non-residential"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=OVERLAP, markeredgecolor="white", markersize=7, label="overlap"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RES, markeredgecolor="white", markersize=7, label="residential"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.72, 0.0))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.16, wspace=0.18)
    if subtitle is None:
        subtitle = (
            f"Held-out tiles. F1 {f1:.3f}. Gap {ratio:.2f}. "
            "Mauve is where both centers still reach. Influence falls with distance from a center. The plane and the dashed line are the classifier cut."
        )
    _header(fig, _claim(name, ratio), subtitle, title_y=0.97, sub_y=0.90)
    out = Path(path) if path is not None else FIG / f"vit3d_{region}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def write_figures(z, y, regions, heads, metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_gap_bars(metrics)
    plot_class_axis(z, y, regions, heads, metrics)
    for _, row in metrics.iterrows():
        region = row["region"]
        m = regions == region
        plot_metro(
            region,
            z[m],
            y[m],
            heads[region],
            float(row["macro_f1"]),
            float(row["test_centroid_sep"]),
            float(row["test_within"]),
        )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "summary.json").read_text())
    metrics = pd.read_csv(OUT / "loro_metrics.csv")
    oof = pd.read_parquet(OUT / "oof_predictions.parquet")
    z = np.load(OUT / "oof_z3.npy")
    heads = np.load(OUT / "head_w_by_region.npz")
    regions = oof["region"].to_numpy()
    y = oof["y_true"].to_numpy().astype(int)

    metrics["ratio"] = metrics["test_centroid_sep"] / metrics["test_within"]
    metrics = metrics.sort_values("macro_f1", ascending=False)

    write_figures(z, y, regions, heads, metrics)

    by_gap = metrics.sort_values("ratio", ascending=False)
    rows = []
    for _, r in metrics.iterrows():
        rows.append(
            f"| {r['region']} | {r['macro_f1']:.3f} | {r['test_centroid_sep']:.2f} | {r['test_within']:.2f} | {r['ratio']:.2f} |"
        )
    gallery = []
    for _, r in by_gap.iterrows():
        region = r["region"]
        gallery.append(
            f"### {_name(region)}\n\n"
            f"![3D {region}](../../outputs/week4_vit_100m_3d/figures/vit3d_{region}.png)\n"
        )

    f1 = summary["loro_macro_f1_mean"]
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# 3D ViT on 100 m tiles

Each 100 m tile is a 10×10 grid of AlphaEarth cells. Empty slots are masked. Each occupied cell is 9 PLS-DA scores (the smallest k within 0.02 LORO F1 of the 64-D logistic). A transformer (width 128) mixes those cells. The class token is then mapped to **3 dimensions**. That 3D vector is the encoder output.

Loss on the 3D vector:

$$
\\mathcal{{L}} = \\mathrm{{BCE}} + \\lambda \\, \\mathcal{{L}}_{{\\mathrm{{SupCon}}}}, \\quad \\lambda = 1,\\ \\tau = 0.1
$$

Same-class tiles are pulled together. RES and NRES are pushed apart. Each metro is its own 3D coordinate system. The scatters below plot those three encoder outputs.

**Result.** LORO macro-F1 = **{f1:.3f}** ± {summary['loro_macro_f1_std']:.3f}. Pooled OOF F1 = {summary['oof_macro_f1']:.3f}.

That is below the mean-pool logistic on the same tiles (0.761) and below the 128-D ViT with no contrastive term (0.756). Phoenix is the only metro with a clear gap in the 3D space (sep/within = 3.0). NYC and SF are one cloud.

```bash
.venv/bin/python -u scripts/train_week4_vit_100m_3d.py
.venv/bin/python -u scripts/build_week4_vit_100m_3d_notebook.py
```
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Held-out scores

sep is the distance between class centroids. within is the mean distance to that centroid. sep/within above 3 is a visible gap. Near 1 the clouds overlap. Below 1 the centroids sit inside one blob.

| Metro | F1 | sep | within | sep/within |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Chance for the contrastive term, at batch size 256, is about 5.54. A perfect cosine split floors near 4.84. The runs ended around 5.3, so the contrastive term never packed each class onto a point. sep grew, and within grew with it.
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Where the classes separate

Gap is the distance between the two class centers divided by the average distance from a tile to its own center. Above 3 the groups are apart. Near 1 they are one cloud. The 3D figures below use the same order, largest gap first.

![Gap by metro](../../outputs/week4_vit_100m_3d/figures/vit3d_sep_ratio.png)

![2D projection of the encoder output](../../outputs/week4_vit_100m_3d/figures/vit3d_projection.png)
"""
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3D views\n\n"
            "Each figure is that metro's own encoder output, largest gap first. "
            "The three axes are the three coordinates the model wrote.\n\n"
            + "\n".join(gallery)
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            """## Code

The training driver is `scripts/train_week4_vit_100m_3d.py`. The pieces that define the 3D space are the encoder bottleneck and the contrastive loss. The next cell reloads the saved coordinates and redraws one metro.
"""
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            '''"""3D bottleneck ViT and supervised contrastive loss.

Matches scripts/train_week4_vit_100m_3d.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 3
GRID = 10


def supervised_contrastive(z, y, tau: float = 0.1):
    """Pull same-class 3D vectors together, push the other class away."""
    u = F.normalize(z, dim=1)
    logits = u @ u.T / tau
    b = z.shape[0]
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    labels = y.view(-1).long()
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    mean_log_prob = (log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    return -mean_log_prob[valid].mean()


class AEPatchViT3D(nn.Module):
    """Wide ViT over 10x10 PLS patches. Class token compressed to 3D."""

    def __init__(self, patch_dim, d_model=128, n_heads=4, n_layers=4, dropout=0.1, emb_dim=EMB_DIM):
        super().__init__()
        n_patches = GRID * GRID
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.to_3d = nn.Linear(d_model, emb_dim)
        self.head = nn.Linear(emb_dim, 1)

    def encode(self, x, mask):
        b = x.shape[0]
        tokens = self.patch_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        pad = torch.zeros(b, 1, device=x.device, dtype=torch.bool)
        key_pad = torch.cat([pad, mask < 0.5], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=key_pad)
        return self.to_3d(self.norm(h[:, 0]))

    def forward(self, x, mask):
        z = self.encode(x, mask)
        return self.head(z).squeeze(-1), z


# Training step (inside each LORO fold, on that fold's GPU):
#   logits, z = model(x, mask)
#   loss = BCE(logits, y) + lambda * supervised_contrastive(z, y, tau=0.1)
'''
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            '''"""Reload one metro and scatter its three encoder outputs.

Blue is residential, orange is non-residential.
Axes are the three encoder outputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("../../outputs/week4_vit_100m_3d")
REGION = "Phoenix_SW"  # try NYC_NE for the overlapping cloud

oof = pd.read_parquet(OUT / "oof_predictions.parquet")
z = np.load(OUT / "oof_z3.npy")
m = oof["region"].to_numpy() == REGION
zz = z[m]
yy = oof.loc[m, "y_true"].to_numpy().astype(int)

fig = plt.figure(figsize=(7.4, 5.5))
ax = fig.add_subplot(111, projection="3d")
for lab, color in ((0, "#D55E00"), (1, "#0072B2")):
    sel = yy == lab
    ax.scatter(zz[sel, 0], zz[sel, 1], zz[sel, 2], s=9, c=color, alpha=0.45, linewidths=0, depthshade=False)
spans = np.ptp(zz, axis=0)
spans = np.clip(np.maximum(spans, 1e-6) / spans.max(), 0.32, 1.0)
ax.set_box_aspect(tuple(spans))
ax.set_xlabel("encoder 1")
ax.set_ylabel("encoder 2")
ax.set_zlabel("encoder 3")
ax.view_init(elev=16, azim=-62)
ax.set_title(REGION.replace("_", " ") + "  ·  blue residential, orange non-residential", loc="left")
plt.show()
'''
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Reload

| Path | Contents |
|---|---|
| `outputs/week4_vit_100m_3d/checkpoints/fold_<metro>.pt` | Weights, PLS rotation, score mean/std, history |
| `outputs/week4_vit_100m_3d/oof_z3.npy` | Held-out 3D vectors, shape (52492, 3). Axes differ by metro. |
| `outputs/week4_vit_100m_3d/oof_predictions.parquet` | Labels and probabilities |
| `outputs/week4_vit_100m_3d/logs/<metro>.log` | Epoch bce, supcon, val F1, sep, within |

Elapsed {summary['elapsed_s']} s on {summary['n_gpus']} GPUs. Class: `scripts.train_week4_vit_100m_3d.AEPatchViT3D`.
"""
        )
    )
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB}")
    print(f"Figures in {FIG}")


if __name__ == "__main__":
    main()
