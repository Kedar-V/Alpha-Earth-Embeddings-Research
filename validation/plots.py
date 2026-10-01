"""Figures for the validation report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Binary framing for class metrics: RES = positive.
# Also show macro-F1 (mean of RES and NRES one-vs-rest F1).
_COMPARE_METRICS = [
    ("macro_f1", "macro-F1"),
    ("precision", "precision (RES+)"),
    ("recall", "recall (RES+)"),
    ("residential_f1", "F1 (RES+)"),
    ("residential_iou", "IoU (RES+)"),
]


def plot_comparison_bars(comparison: pd.DataFrame, out_dir: Path) -> Path:
    """comparison columns: level, system, metric, value."""
    metrics = [m for m, _ in _COMPARE_METRICS]
    labels = [lab for _, lab in _COMPARE_METRICS]
    levels = comparison["level"].unique().tolist()
    fig, axes = plt.subplots(1, len(levels), figsize=(5.5 * len(levels), 4.2), squeeze=False)
    for ax, level in zip(axes[0], levels):
        sub = comparison[comparison["level"] == level]
        x = np.arange(len(metrics))
        w = 0.35
        model_vals = [float(sub[(sub["system"] == "model") & (sub["metric"] == m)]["value"].iloc[0])
                      if len(sub[(sub["system"] == "model") & (sub["metric"] == m)]) else np.nan for m in metrics]
        ghsl_vals = [float(sub[(sub["system"] == "ghsl") & (sub["metric"] == m)]["value"].iloc[0])
                     if len(sub[(sub["system"] == "ghsl") & (sub["metric"] == m)]) else np.nan for m in metrics]
        ax.bar(x - w / 2, model_vals, w, label="Our model (LORO CNN)", color="#1f77b4")
        ax.bar(x + w / 2, ghsl_vals, w, label="GHSL GHS-BUILT-C FUN", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{level}-level · macro-F1 + RES+")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "figures" / "functional_comparison_bars.png"
    _savefig(fig, path)
    return path


def plot_per_aoi_macro_f1(per_aoi: pd.DataFrame, out_dir: Path, level: str) -> Path:
    """Per-AOI macro-F1 (functional subset)."""
    sub = per_aoi[per_aoi["level"] == level].copy() if "level" in per_aoi.columns else per_aoi.copy()
    metric = "macro_f1"
    aois = sorted(sub["aoi_id"].unique())
    model = [float(sub[(sub["aoi_id"] == a) & (sub["system"] == "model")][metric].iloc[0]) for a in aois]
    ghsl = [float(sub[(sub["aoi_id"] == a) & (sub["system"] == "ghsl")][metric].iloc[0]) for a in aois]
    y = np.arange(len(aois))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(aois))))
    ax.barh(y - 0.2, model, 0.4, label="Model", color="#1f77b4")
    ax.barh(y + 0.2, ghsl, 0.4, label="GHSL", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(aois, fontsize=8)
    ax.set_xlabel("Macro-F1 (functional subset)")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Per-AOI macro-F1 · {level}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "figures" / f"per_aoi_macro_f1_{level}.png"
    _savefig(fig, path)
    return path


def plot_coverage(end_to_end: dict[str, Any], out_dir: Path, level: str) -> Path:
    systems = ["model", "ghsl"]
    cov = [end_to_end[s]["coverage"] for s in systems]
    sel = [end_to_end[s]["selective_macro_f1"] for s in systems]
    eff = [end_to_end[s]["effective_accuracy"] for s in systems]
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.25, cov, 0.25, label="Coverage", color="#2ca02c")
    ax.bar(x, sel, 0.25, label="Selective macro-F1", color="#1f77b4")
    ax.bar(x + 0.25, eff, 0.25, label="Effective accuracy", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(["Our model", "GHSL"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"End-to-end coverage vs accuracy · {level}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "figures" / f"end_to_end_{level}.png"
    _savefig(fig, path)
    return path
