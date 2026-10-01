#!/usr/bin/env python3
"""Compare week3 pixel AE LORO vs building-level (polygon) binary LORO.

Primary polygon baseline: week2 centroid LogReg on the same labeled Overture set
(outputs/week2_cnn_bbox/binary_loro_comparison.csv). Also plots week2 bbox CNN
and the notebook-02 trial_residential LORO (balanced subsample) for context.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week3_pixel_ae"
FIG = OUT / "figures"

PIXEL = OUT / "loro_metrics.csv"
BUILDING = ROOT / "outputs" / "week2_cnn_bbox" / "binary_loro_comparison.csv"
TRIAL = ROOT / "outputs" / "trial_residential" / "loro_metrics.csv"
SUMMARY = OUT / "summary.json"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    pixel = pd.read_csv(PIXEL)[["region", "f1_macro", "accuracy", "n_test"]].rename(
        columns={
            "f1_macro": "pixel_ae_lr",
            "accuracy": "pixel_acc",
            "n_test": "n_pixels",
        }
    )
    building = pd.read_csv(BUILDING).rename(
        columns={"lr_centroid": "building_centroid_lr", "cnn_bbox": "building_bbox_cnn"}
    )
    cmp = building.merge(pixel, on="region", how="inner")
    cmp["delta_centroid_minus_pixel"] = cmp["building_centroid_lr"] - cmp["pixel_ae_lr"]
    cmp["delta_cnn_minus_pixel"] = cmp["building_bbox_cnn"] - cmp["pixel_ae_lr"]
    cmp = cmp.sort_values("region").reset_index(drop=True)
    cmp.to_csv(OUT / "polygon_vs_pixel_loro_comparison.csv", index=False)

    means = {
        "building_centroid_lr": float(cmp["building_centroid_lr"].mean()),
        "building_bbox_cnn": float(cmp["building_bbox_cnn"].mean()),
        "pixel_ae_lr": float(cmp["pixel_ae_lr"].mean()),
        "building_centroid_lr_std": float(cmp["building_centroid_lr"].std(ddof=1)),
        "building_bbox_cnn_std": float(cmp["building_bbox_cnn"].std(ddof=1)),
        "pixel_ae_lr_std": float(cmp["pixel_ae_lr"].std(ddof=1)),
    }

    trial_mean = None
    if TRIAL.exists():
        trial = pd.read_csv(TRIAL)
        trial_mean = float(trial["f1_macro"].mean())
        means["trial_building_lr_balanced_subsample"] = trial_mean

    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text())
    else:
        summary = {}
    summary["polygon_vs_pixel"] = {
        **means,
        "note": (
            "Building metrics from week2 binary LORO on ~11k Overture buildings "
            "(centroid LR / bbox CNN). Pixel metrics from week3 pure-cell AE LR. "
            "Units differ (building vs 10 m cell); compare directionally."
        ),
        "comparison_csv": "outputs/week3_pixel_ae/polygon_vs_pixel_loro_comparison.csv",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    regions = cmp["region"].tolist()
    x = np.arange(len(regions))
    w = 0.27

    # --- grouped per-region bars ---
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, cmp["building_centroid_lr"], w, label="Building centroid LR", color="#2c7fb8")
    ax.bar(x, cmp["building_bbox_cnn"], w, label="Building bbox CNN", color="#7fcdbb")
    ax.bar(x + w, cmp["pixel_ae_lr"], w, label="Pixel AE LR", color="#d95f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=35, ha="right")
    ax.set_ylabel("LORO macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Polygon (building) vs pixel — LORO macro-F1 by region")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(means["building_centroid_lr"], color="#2c7fb8", ls=":", lw=1, alpha=0.7)
    ax.axhline(means["pixel_ae_lr"], color="#d95f0e", ls=":", lw=1, alpha=0.7)
    fig.tight_layout()
    fig.savefig(FIG / "polygon_vs_pixel_f1_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- mean summary ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["Building\ncentroid LR", "Building\nbbox CNN", "Pixel\nAE LR"]
    vals = [
        means["building_centroid_lr"],
        means["building_bbox_cnn"],
        means["pixel_ae_lr"],
    ]
    errs = [
        means["building_centroid_lr_std"],
        means["building_bbox_cnn_std"],
        means["pixel_ae_lr_std"],
    ]
    colors = ["#2c7fb8", "#7fcdbb", "#d95f0e"]
    bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, ecolor="#333")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    if trial_mean is not None:
        ax.axhline(
            trial_mean,
            color="#636363",
            ls="--",
            label=f"Trial building LR (balanced subsample) mean={trial_mean:.3f}",
        )
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylabel("Mean LORO macro-F1 ± std")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean LORO macro-F1 — polygon vs pixel")
    fig.tight_layout()
    fig.savefig(FIG / "polygon_vs_pixel_mean_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- scatter building centroid vs pixel ---
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.scatter(cmp["building_centroid_lr"], cmp["pixel_ae_lr"], s=55, c="#d95f0e", zorder=3)
    for _, r in cmp.iterrows():
        ax.annotate(
            r["region"].replace("_", "\n"),
            (r["building_centroid_lr"], r["pixel_ae_lr"]),
            fontsize=7,
            ha="center",
            va="bottom",
        )
    lims = [0.35, 0.95]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Building centroid LR macro-F1")
    ax.set_ylabel("Pixel AE LR macro-F1")
    ax.set_title("Per-region: building vs pixel")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "polygon_vs_pixel_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- delta bars ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = cmp.sort_values("delta_centroid_minus_pixel")
    colors_d = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in d["delta_centroid_minus_pixel"]]
    ax.barh(d["region"], d["delta_centroid_minus_pixel"], color=colors_d)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ macro-F1 (building centroid LR − pixel AE LR)")
    ax.set_title("Where polygon beats pixel (positive) vs reverse")
    fig.tight_layout()
    fig.savefig(FIG / "polygon_vs_pixel_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(cmp.to_string(index=False))
    print(
        f"\nmeans · building_centroid={means['building_centroid_lr']:.3f}  "
        f"building_cnn={means['building_bbox_cnn']:.3f}  "
        f"pixel={means['pixel_ae_lr']:.3f}",
        flush=True,
    )
    print(f"Wrote figures under {FIG}", flush=True)


if __name__ == "__main__":
    main()
