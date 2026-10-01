#!/usr/bin/env python3
"""Shared Week 4 figure style (data-viz principles)."""

from __future__ import annotations

# Colorblind-safe RES / NRES (Okabe–Ito-ish blue / orange)
RES = "#0072B2"
NRES = "#E69F00"
MUTED = "#B0B0B0"
ACCENT = "#D55E00"  # errors / focal
OK = "#009E73"
CONTEXT = "#7F7F7F"

RES_COLORS = {1: RES, 0: NRES}
RES_LABELS = {1: "residential", 0: "non-residential"}

MODEL_COLORS = {
    "balanced_train": "#0072B2",
    "fraction_regression": "#E69F00",
    "pixel_ae": "#009E73",
    "pixel_3class_reg": "#CC79A7",
    "neighborhood_k3_reg": "#56B4E9",
}

MODEL_SHORT = {
    "balanced_train": "W4 balanced LR",
    "fraction_regression": "W3 Ridge",
    "pixel_ae": "W3 AE LR",
    "pixel_3class_reg": "W3 soft 3-class",
    "neighborhood_k3_reg": "W3 3×3 CNN",
}


def style_axes(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not grid:
        ax.grid(False)
    else:
        ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
        ax.set_axisbelow(True)


def set_claim_title(fig, claim: str, subtitle: str | None = None) -> None:
    """Left-aligned claim title above the axes. Call after tight_layout(rect=...)."""
    fig.text(
        0.01,
        0.995,
        claim,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
        wrap=True,
    )
    if subtitle:
        fig.text(
            0.01,
            0.955,
            subtitle,
            fontsize=9,
            color="#555555",
            ha="left",
            va="top",
            transform=fig.transFigure,
        )


def label_bars_h(ax, bars, fmt: str = "{:.2f}", pad: float = 0.01) -> None:
    for rect in bars:
        w = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        ax.text(w + pad, y, fmt.format(w), va="center", ha="left", fontsize=9, color="#222222")
