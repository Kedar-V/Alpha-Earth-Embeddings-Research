#!/usr/bin/env python3
"""Pipeline diagrams for Week 3 Kyle briefing: data prep → model → output.

Writes outputs/week3_model_comparison/figures/pipeline_*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week3_model_comparison" / "figures"

COLORS = {
    "data": ("#e8f5e9", "#2e7d32"),
    "prep": ("#fff8e1", "#f9a825"),
    "model": ("#e3f2fd", "#1565c0"),
    "out": ("#fce4ec", "#c2185b"),
    "eval": ("#f3e5f5", "#7b1fa2"),
}

PIPELINES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "building_cnn": (
        "A. Building · AlphaEarth bbox CNN",
        [
            ("data", "Overture Track-A footprints\n+ enclosing 10 m AE grid cells\n(datasets/building_bbox_grid)"),
            ("prep", "Pack variable H×W×64 AE patch\nper footprint (inside-polygon mask)"),
            ("model", "Small CNN over AE grid\n→ pooled embedding → binary head\nLORO leave-one-metro-out"),
            ("out", "Per-building RES / NRES\n(+ OOF embeddings for UMAP)"),
            ("eval", "Macro-F1 vs Overture GT (~0.75)\nalso vs GHSL Built-C (~0.67)"),
        ],
    ),
    "ghsl": (
        "G. GHSL GHS-BUILT-C FUN (baseline)",
        [
            (
                "data",
                "GHS-BUILT-C FUN R2023A\n10 m raster (ESRI:54009)\ncodes: 1=RES, 2=NRES, 0=nonbuilt",
            ),
            (
                "prep",
                "No training — frozen map\nReproject Overture GT to Mollweide\nAlign to exact GHSL grid",
            ),
            (
                "model",
                "Two scorings of the same raster:\n• Pixel: read cell value as-is\n• Building: majority under footprint\n  (centroid fallback if empty mask)",
            ),
            (
                "out",
                "Pixel: RES/NRES/NONBUILT per cell\nBuilding: one label per footprint",
            ),
            (
                "eval",
                "Same Track-A GT as AE models\nBuilding macro-F1 ~0.67\nPixel macro-F1 ~0.51\n(functional subset)",
            ),
        ],
    ),
    "pixel_binary_lr": (
        "B. Pixel binary · logistic regression",
        [
            ("data", "10 m AE cells overlapping\nTrack-A footprints\n(datasets/week3_pixel_gt)"),
            ("prep", "Drop mixed · keep pure RES/NRES\nX = A00–A63 (64-D)\nSMOTE on each LORO train fold"),
            ("model", "StandardScaler\n→ LogisticRegression\n(class_weight balanced)"),
            ("out", "Per-pixel y_pred / P_res"),
            ("eval", "LORO macro-F1 ~0.56"),
        ],
    ),
    "pixel_binary_reg": (
        "C. Pixel binary · fraction regression",
        [
            ("data", "Same pure pixels as binary LR\n(+ coverage fracs for QA)"),
            ("prep", "Drop mixed · SMOTE on train fold\nTargets: RES/NRES fractions"),
            ("model", "StandardScaler\n→ multi-output Ridge / soft head"),
            ("out", "Predicted fractions\n→ argmax → RES / NRES"),
            ("eval", "LORO macro-F1 ~0.57\n(nearly tied with LR)"),
        ],
    ),
    "pixel_3class_clf": (
        "D. Pixel 3-class · hard classifier",
        [
            ("data", "week3_pixel_gt_v2:\nbuilding-overlapped cells\n+ AOI-sampled empties\n(n_empty ≈ n_res per metro)"),
            ("prep", "Drop mixed · labels ∈ {RES, NRES, empty}\nX = 64-D AE · SMOTE 3-way\non each LORO train fold"),
            ("model", "CE classifier on hard labels\n(GPU LORO workers)"),
            ("out", "Per-pixel class ∈\n{RES, NRES, empty}"),
            ("eval", "OOF macro-F1 ~0.46\n(empty recall relatively strong)"),
        ],
    ),
    "pixel_3class_reg": (
        "E. Pixel 3-class · soft-fraction + decode",
        [
            ("data", "Same v2 pixels\nsoft GT: frac_res / frac_nres / frac_empty"),
            ("prep", "Drop mixed · RandomOverSampler\ncarrying soft Y with each row\nX = 64-D AE"),
            ("model", "Softmax MSE head\npredicts 3 fractions"),
            ("out", "Decode τ=0.15:\nif frac_res+frac_nres < 0.15 → empty\nelse argmax(RES, NRES)"),
            ("eval", "OOF macro-F1 ~0.52\nwins 9/10 metros vs hard clf"),
        ],
    ),
    "pixel_nb_cnn": (
        "F. Neighborhood 3×3 · Conv (clf or reg)",
        [
            ("data", "Center pixel from v2 labels\n+ 3×3 AE window around it\n(X_nb N×64×3×3 + validity mask)"),
            ("prep", "Stack AE + mask → 65×3×3\nROS on hard labels (clf)\nor soft fracs (reg)"),
            ("model", "NbCNN: Conv→BN→ReLU ×2\n→ masked GAP → Linear→3"),
            ("out", "Clf: class logits\nReg: soft fracs → τ=0.15 decode\n(+ 128-D OOF embeddings)"),
            ("eval", "OOF macro-F1 ~0.35 / 0.33\n(below single-pixel soft-reg)"),
        ],
    ),
}


def draw_flow(ax, title: str, steps: list[tuple[str, str]]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)

    heights = []
    for _, text in steps:
        nlines = text.count("\n") + 1
        heights.append(max(0.9, 0.34 * nlines + 0.38))
    gap = 0.22
    total = sum(heights) + gap * (len(steps) - 1)
    y = 5 + total / 2
    x, w = 0.7, 8.6
    boxes = []
    for (kind, text), h in zip(steps, heights):
        y -= h
        fc, ec = COLORS[kind]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.35,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.8, linespacing=1.2)
        boxes.append((x + w / 2, y, y + h))
        y -= gap

    for i in range(len(boxes) - 1):
        cx = boxes[i][0]
        ax.annotate(
            "",
            xy=(cx, boxes[i + 1][2] + 0.02),
            xytext=(cx, boxes[i][1] - 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.25),
        )


def legend_bar(fig) -> None:
    name = {"data": "Data", "prep": "Prep", "model": "Model", "out": "Output", "eval": "Eval"}
    handles = [Patch(facecolor=COLORS[k][0], edgecolor=COLORS[k][1], label=name[k]) for k in name]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (title, steps) in PIPELINES.items():
        fig, ax = plt.subplots(figsize=(5.4, 7.8))
        draw_flow(ax, title, steps)
        legend_bar(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = OUT / f"pipeline_{key}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", path)

    order = [
        "building_cnn",
        "ghsl",
        "pixel_binary_lr",
        "pixel_binary_reg",
        "pixel_3class_clf",
        "pixel_3class_reg",
        "pixel_nb_cnn",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 14.5))
    for ax, key in zip(axes.ravel(), order):
        title, steps = PIPELINES[key]
        draw_flow(ax, title, steps)
    for ax in axes.ravel()[len(order) :]:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "GHSL is the frozen\nbaseline scored against\nthe same Track-A GT",
            ha="center",
            va="center",
            fontsize=11,
            color="#555",
            transform=ax.transAxes,
        )
    legend_bar(fig)
    fig.suptitle(
        "Week 3 model pipelines (incl. GHSL baseline): data → prep → model → output → eval",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    for name in ("pipeline_overview_all.png", "pipeline_overview_six.png"):
        path = OUT / name
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
