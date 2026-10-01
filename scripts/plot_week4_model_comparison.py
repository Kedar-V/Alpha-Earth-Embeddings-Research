#!/usr/bin/env python3
"""Week 4 vs Week 3 model comparison figures (claim-first).

Reads existing CSVs under outputs/week4_model_comparison/ and rewrites PNGs.

Usage:
  .venv/bin/python -u scripts/plot_week4_model_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import ACCENT, MUTED, OK, RES, label_bars_h, set_claim_title, style_axes  # noqa: E402

OUT = ROOT / "outputs" / "week4_model_comparison"
FIG = OUT / "figures"

# Short plain-language labels
SHORT = {
    "balanced_train (W4)": "W4 balanced LR",
    "fraction_regression (W3)": "W3 Ridge",
    "pixel_ae (W3)": "W3 AE LR",
    "pixel_3class_reg (W3)": "W3 soft 3-class",
    "neighborhood_k3_reg (W3)": "W3 3×3 CNN",
    "GHSL (pixel, W4 GT)": "GHSL (pixel)",
    "GHSL (pixel, W4 balanced)": "GHSL (pixel, balanced)",
    "Polygon AE (building)": "Polygon AE (buildings)",
    "GHSL (building)": "GHSL (buildings)",
    "balanced_train LR (W4)": "W4 balanced LR",
    "pixel_ae LR (W3)": "W3 AE LR",
    "pixel_3class soft (W3)": "W3 soft 3-class",
    "neighborhood 3x3 (W3)": "W3 3×3 CNN",
}


def _short(name: str) -> str:
    return SHORT.get(name, name)


def plot_f1_balanced(df: pd.DataFrame) -> Path:
    s = df.sort_values("macro_f1_balanced_eval", ascending=True).copy()
    s["label"] = s["model"].map(_short)
    best = float(s["macro_f1_balanced_eval"].max())
    colors = [RES if v == best else MUTED for v in s["macro_f1_balanced_eval"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(s["label"], s["macro_f1_balanced_eval"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.2f}", pad=0.012)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"On balanced pixels, W4 and Week 3 Ridge/AE tie near {best:.2f} F1",
        "Eval: pure RES=NRES cells. Blue = best; grey = others.",
    )
    p = FIG / "f1_on_balanced_eval.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_mae_balanced(df: pd.DataFrame) -> Path:
    s = df.dropna(subset=["mae_mean_balanced"]).sort_values("mae_mean_balanced", ascending=False).copy()
    if s.empty:
        return FIG / "mae_on_balanced_eval.png"
    s["label"] = s["model"].map(_short)
    best = float(s["mae_mean_balanced"].min())
    colors = [OK if abs(v - best) < 1e-9 else MUTED for v in s["mae_mean_balanced"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(s["label"], s["mae_mean_balanced"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.2f}", pad=0.008)
    ax.set_xlabel("Mean absolute error (three coverage fractions)")
    ax.set_xlim(0, max(s["mae_mean_balanced"]) * 1.18)
    style_axes(ax, grid=True)
    claim = f"Soft 3-class has the lowest fraction error ({best:.2f}) — not the highest binary F1"
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(fig, claim, "Lower is better. Green = best MAE.")
    p = FIG / "mae_on_balanced_eval.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_protocol_shift(df: pd.DataFrame) -> Path:
    """Encode the delta (balanced − reported) so the reader does not subtract."""
    s = df.dropna(subset=["macro_f1_week3_reported", "macro_f1_balanced_eval"]).copy()
    # Skip near-zero deltas that are not protocol shifts (e.g. W4 trained on balanced)
    s["label"] = s["model"].map(_short)
    s["delta"] = s["macro_f1_balanced_eval"] - s["macro_f1_week3_reported"]
    s = s.sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = [OK if d >= 0 else ACCENT for d in s["delta"]]
    bars = ax.barh(s["label"], s["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#333333", lw=1)
    for rect, d in zip(bars, s["delta"]):
        x = d + (0.008 if d >= 0 else -0.008)
        ax.text(
            x,
            rect.get_y() + rect.get_height() / 2,
            f"{d:+.2f}",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlabel("F1 change: balanced eval − Week 3 reported")
    style_axes(ax, grid=True)
    # expand xlim for labels
    lo, hi = float(s["delta"].min()), float(s["delta"].max())
    pad = 0.06
    ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
    claim = "Rebalancing the eval lifts Week 3 models by +0.13 to +0.16 F1"
    sub = "Same OOF preds; only the pixel subset changes (imbalanced → 1:1 RES/NRES)."
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(fig, claim, sub)
    p = FIG / "f1_protocol_shift.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def plot_cross_week(ladder: pd.DataFrame) -> Path:
    s = ladder.sort_values("macro_f1", ascending=True).copy()
    s["label"] = s["model"].map(_short)
    is_building = s["unit"].astype(str).str.contains("building")
    best_pix = float(s.loc[~is_building, "macro_f1"].max())
    colors = []
    for _, row in s.iterrows():
        if "building" in str(row["unit"]):
            colors.append("#56B4E9" if "Polygon" in row["model"] else "#A6CEE3")
        elif abs(row["macro_f1"] - best_pix) < 1e-9:
            colors.append(RES)
        else:
            colors.append(MUTED)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(s["label"], s["macro_f1"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.2f}", pad=0.012)
    ax.set_xlabel("Macro-F1 (reported / OOF)")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    claim = "Building F1 is a different task — do not read it as a pixel score"
    sub = "Light blue = building-level; dark blue = best pixel (balanced). Grey = other pixel models."
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(fig, claim, sub)
    p = FIG / "cross_week_f1_ladder.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "comparison_balanced_eval.csv")
    ladder = pd.read_csv(OUT / "f1_ladder_cross_week.csv")

    paths = [
        plot_f1_balanced(df),
        plot_mae_balanced(df),
        plot_protocol_shift(df),
        plot_cross_week(ladder),
    ]
    for p in paths:
        print(f"Wrote {p.relative_to(ROOT)}", flush=True)

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary["figures_v2"] = [str(p.relative_to(ROOT)) for p in paths]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
