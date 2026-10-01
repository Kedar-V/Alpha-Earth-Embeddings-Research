#!/usr/bin/env python3
"""Week 3 — four-model comparison figures (data-viz principles).

Each figure has one claim, an active title, sorted bars, direct labels,
and minimal legend/decoding. Writes outputs/week3_model_comparison/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week3_model_comparison"
FIG = OUT / "figures"

W2_BIN = ROOT / "outputs" / "week2_cnn_bbox" / "binary_loro_comparison.csv"
W3_02 = ROOT / "outputs" / "week3_pixel_ae" / "loro_metrics.csv"
W3_03 = ROOT / "outputs" / "week3_fraction_regression" / "loro_majority_metrics.csv"
VAL_AOI = ROOT / "outputs" / "validation" / "per_aoi_metrics.csv"
W3_02_SUM = ROOT / "outputs" / "week3_pixel_ae" / "summary.json"
W3_03_SUM = ROOT / "outputs" / "week3_fraction_regression" / "summary.json"
W2_SNAP = ROOT / "outputs" / "week2_cnn_bbox" / "cnn_bbox_snapshot.json"

# Accent + muted neutrals (colorblind-safer; color has a job)
C_FOCUS = "#c45c26"  # warm accent for AE focal
C_FOCUS2 = "#2b6a8f"  # cool accent for second AE
C_MUTE = "#b0b0b0"
C_GHSL = "#5a5a5a"
C_POS = "#2a6f4e"
C_NEG = "#8b3a3a"

REGION_SHORT = {
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


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _claim_header(fig, ax, title: str, subtitle: str) -> None:
    """Active title + subtitle in figure coords — never overlapping.

    Avoid ax.set_title + ax.text(y>1) together (they share one band), and avoid
    relying on bbox_inches='tight' to place headers (it can crush them together).
    """
    ax.set_title("")  # clear default
    # Reserved band above axes (see _save top≈0.80): title then subtitle
    fig.text(
        0.02,
        0.97,
        title,
        transform=fig.transFigure,
        fontsize=13,
        fontweight="bold",
        color="#111",
        ha="left",
        va="top",
    )
    fig.text(
        0.02,
        0.915,
        subtitle,
        transform=fig.transFigure,
        fontsize=10,
        color="#555",
        ha="left",
        va="top",
    )


def _save(fig, path: Path, *, top: float = 0.80, left: float = 0.20, bottom: float = 0.14) -> None:
    """Fixed margins (no bbox_inches='tight') so headers/labels keep their gaps."""
    fig.subplots_adjust(top=top, left=left, right=0.96, bottom=bottom)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _label_bars_h(ax, bars, vals, fmt="{:.2f}", dx: float | None = None) -> None:
    vals = np.asarray(vals, dtype=float)
    span = max(float(np.nanmax(vals)) - float(np.nanmin(np.minimum(vals, 0))), 1e-6)
    pad = dx if dx is not None else max(0.015, 0.03 * span)
    for bar, v in zip(bars, vals):
        ax.text(
            float(v) + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(float(v)),
            va="center",
            ha="left",
            fontsize=10,
            color="#222",
            clip_on=True,
        )
    hi = float(np.nanmax(vals)) + pad * 4
    ax.set_xlim(0, max(hi, ax.get_xlim()[1]))


def _label_deltas_h(ax, bars, vals, fmt="{:+.2f}") -> None:
    """Label delta bars without colliding with y-tick city names.

    Positive: label to the right of the tip (outward).
    Negative: label to the right of the tip (inward toward zero) so text never
    crosses the left spine into the y-tick gutter.
    Near-zero: park the label just right of the zero line (clear of the spine).
    """
    vals = np.asarray(vals, dtype=float)
    vmax = max(float(np.nanmax(np.abs(vals))), 0.05)
    pad = max(0.012, 0.045 * vmax)
    spine_clear = max(0.028, pad * 1.8)
    for bar, v in zip(bars, vals):
        v = float(v)
        y = bar.get_y() + bar.get_height() / 2
        x = v + pad
        # Keep labels off the zero spine (tiny bars like −0.01)
        if abs(x) < spine_clear:
            x = spine_clear
        ax.text(
            x,
            y,
            fmt.format(v),
            va="center",
            ha="left",
            fontsize=9,
            color="#222",
            clip_on=True,
        )
    lo = min(0.0, float(np.nanmin(vals))) - pad * 2
    hi = max(0.0, float(np.nanmax(vals)), spine_clear) + pad * 5
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="y", pad=10, length=0)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    poly = pd.read_csv(W2_BIN).rename(
        columns={"lr_centroid": "polygon_centroid_lr", "cnn_bbox": "polygon_bbox_cnn"}
    )
    p02 = pd.read_csv(W3_02)[["region", "f1_macro"]].rename(columns={"f1_macro": "pixel_ae_lr"})
    p03 = pd.read_csv(W3_03)[["region", "f1_macro_pure_binary"]].rename(
        columns={"f1_macro_pure_binary": "pixel_frac_majority"}
    )
    aoi = pd.read_csv(VAL_AOI)
    ghsl_b = aoi[(aoi["system"] == "ghsl") & (aoi["level"] == "building")][
        ["aoi_id", "macro_f1"]
    ].rename(columns={"aoi_id": "region", "macro_f1": "ghsl_building"})
    ghsl_p = aoi[(aoi["system"] == "ghsl") & (aoi["level"] == "pixel")][
        ["aoi_id", "macro_f1"]
    ].rename(columns={"aoi_id": "region", "macro_f1": "ghsl_pixel"})

    per = (
        poly.merge(p02, on="region")
        .merge(p03, on="region")
        .merge(ghsl_b, on="region", how="left")
        .merge(ghsl_p, on="region", how="left")
    )
    per["city"] = per["region"].map(REGION_SHORT)
    per.to_csv(OUT / "per_region_macro_f1.csv", index=False)

    s02 = json.loads(W3_02_SUM.read_text())
    s03 = json.loads(W3_03_SUM.read_text())
    s2 = json.loads(W2_SNAP.read_text())

    m_poly_cnn = float(per["polygon_bbox_cnn"].mean())
    m_poly_lr = float(per["polygon_centroid_lr"].mean())
    m_ghsl_b = float(per["ghsl_building"].mean())
    m_pix_lr = float(s02["loro_macro_f1_mean"])
    m_pix_frac = float(s03["loro_f1_macro_pure_binary_mean"])
    m_ghsl_p = float(per["ghsl_pixel"].mean())

    primary_four = pd.DataFrame(
        [
            {"model": "Polygon AE (bbox CNN)", "short": "Polygon AE", "unit": "building", "macro_f1": m_poly_cnn, "std": float(per["polygon_bbox_cnn"].std(ddof=1))},
            {"model": "GHSL (building)", "short": "GHSL", "unit": "building", "macro_f1": m_ghsl_b, "std": float(per["ghsl_building"].std(ddof=1))},
            {"model": "Pixel AE LR + SMOTE (02)", "short": "Pixel LR", "unit": "pixel", "macro_f1": m_pix_lr, "std": float(s02["loro_macro_f1_std"])},
            {"model": "Pixel AE regression + SMOTE (03)", "short": "Regression", "unit": "pixel", "macro_f1": m_pix_frac, "std": float(per["pixel_frac_majority"].std(ddof=1))},
        ]
    )
    primary_four.to_csv(OUT / "primary_four_means.csv", index=False)

    headline = pd.DataFrame(
        [
            {"model": "Polygon AE (centroid LR)", "unit": "building", "macro_f1_mean": m_poly_lr, "macro_f1_std": float(per["polygon_centroid_lr"].std(ddof=1))},
            {"model": "Polygon AE (bbox CNN)", "unit": "building", "macro_f1_mean": m_poly_cnn, "macro_f1_std": float(per["polygon_bbox_cnn"].std(ddof=1))},
            {"model": "GHSL", "unit": "building", "macro_f1_mean": m_ghsl_b, "macro_f1_std": float(per["ghsl_building"].std(ddof=1))},
            {"model": "Pixel AE LR + SMOTE (02)", "unit": "pixel", "macro_f1_mean": m_pix_lr, "macro_f1_std": float(s02["loro_macro_f1_std"])},
            {"model": "Pixel AE regression + SMOTE (03)", "unit": "pixel", "macro_f1_mean": m_pix_frac, "macro_f1_std": float(per["pixel_frac_majority"].std(ddof=1))},
            {"model": "GHSL", "unit": "pixel", "macro_f1_mean": m_ghsl_p, "macro_f1_std": float(per["ghsl_pixel"].std(ddof=1))},
        ]
    )
    headline.to_csv(OUT / "headline_means.csv", index=False)

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "name": "week3_four_model_comparison_v1",
                "primary_four": primary_four.to_dict(orient="records"),
                "claims": [
                    "On buildings, Polygon AE CNN beats GHSL by ~0.07 macro-F1.",
                    f"On 10 m pixels (pure+SMOTE), regression ({m_pix_frac:.2f}) edges LR ({m_pix_lr:.2f}).",
                    f"Switching the unit from building to pixel costs ~{m_poly_cnn - m_pix_lr:.2f} F1 for AE.",
                ],
                "caveat": "Building and pixel F1 are different tasks; within-unit panels are the fair comparisons. Pixel models: pure only + SMOTE.",
                "protocol_pixel": "pure RES/NRES only · mixed dropped · SMOTE on each LORO train fold",
                "week2_binary_means": s2["binary_residential"],
            },
            indent=2,
        )
        + "\n"
    )

    # ------------------------------------------------------------------
    # Fig 1 — Building unit: AE CNN beats GHSL (sorted, highlight winner)
    # Claim: Polygon AE CNN leads on buildings
    # ------------------------------------------------------------------
    bldg = pd.DataFrame(
        [
            ("Polygon AE CNN", m_poly_cnn, C_FOCUS),
            ("Polygon AE centroid", m_poly_lr, C_MUTE),
            ("GHSL", m_ghsl_b, C_GHSL),
        ],
        columns=["label", "f1", "color"],
    ).sort_values("f1")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.barh(bldg["label"], bldg["f1"], color=bldg["color"], height=0.65)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean leave-one-region-out macro-F1")
    gap = m_poly_cnn - m_ghsl_b
    _claim_header(
        fig,
        ax,
        "On buildings, Polygon AE CNN beats GHSL",
        f"10 US metros · residential vs non-residential · building unit · +{gap:.2f} vs GHSL",
    )
    _label_bars_h(ax, bars, bldg["f1"].to_numpy())
    ax.tick_params(axis="y", pad=8, length=0)
    _save(fig, FIG / "01_building_ae_beats_ghsl.png", left=0.28)

    # ------------------------------------------------------------------
    # Fig 2 — Pixel unit: regression + SMOTE edges LR + SMOTE (fair pure-only)
    # ------------------------------------------------------------------
    pix = pd.DataFrame(
        [
            ("Pixel AE LR + SMOTE (02)", m_pix_lr, C_FOCUS),
            ("Pixel AE regression + SMOTE (03)", m_pix_frac, C_FOCUS2),
            ("GHSL (pixel eval)", m_ghsl_p, C_GHSL),
        ],
        columns=["label", "f1", "color"],
    ).sort_values("f1")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.barh(pix["label"], pix["f1"], color=pix["color"], height=0.65)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean macro-F1")
    gap_reg = m_pix_frac - m_pix_lr
    _claim_header(
        fig,
        ax,
        f"On 10 m pixels, regression + SMOTE edges LR by {gap_reg:+.2f} F1",
        "Pure RES/NRES only · mixed dropped · SMOTE on each LORO train fold",
    )
    _label_bars_h(ax, bars, pix["f1"].to_numpy())
    ax.tick_params(axis="y", pad=8, length=0)
    _save(fig, FIG / "02_pixel_lr_vs_frac_vs_ghsl.png", left=0.32)

    # ------------------------------------------------------------------
    # Fig 3 — Unit cost: encode the Δ directly (do not make reader subtract)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = pd.DataFrame(
        [
            ("Building · Polygon AE CNN", m_poly_cnn, C_FOCUS),
            ("Pixel · AE LR (02)", m_pix_lr, C_MUTE),
        ],
        columns=["label", "f1", "color"],
    )
    bars = ax.barh(rows["label"], rows["f1"], color=rows["color"], height=0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean macro-F1")
    drop = m_poly_cnn - m_pix_lr
    _claim_header(
        fig,
        ax,
        f"Moving from buildings to pixels costs {drop:.2f} macro-F1 for AE",
        "Not a fair horse-race — different prediction units — but the drop is the story",
    )
    _label_bars_h(ax, bars, rows["f1"].to_numpy())
    ax.annotate(
        "",
        xy=(m_pix_lr, 0),
        xytext=(m_poly_cnn, 0),
        arrowprops=dict(arrowstyle="<->", color=C_NEG, lw=1.4),
    )
    ax.text(
        (m_pix_lr + m_poly_cnn) / 2,
        -0.55,
        f"Δ = −{drop:.2f}",
        ha="center",
        color=C_NEG,
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )
    ax.set_ylim(-0.9, 1.6)
    ax.tick_params(axis="y", pad=8, length=0)
    _save(fig, FIG / "03_unit_drop_building_to_pixel.png")

    # ------------------------------------------------------------------
    # Fig 4 — Per-city: does Polygon AE beat GHSL? (encode delta)
    # ------------------------------------------------------------------
    d = per.copy()
    d["delta"] = d["polygon_bbox_cnn"] - d["ghsl_building"]
    d = d.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = [C_POS if v >= 0 else C_NEG for v in d["delta"]]
    bars = ax.barh(d["city"], d["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_xlabel("Macro-F1 advantage of Polygon AE CNN over GHSL")
    _claim_header(
        fig,
        ax,
        "Polygon AE beats GHSL in most metros (building unit)",
        "Positive = AE CNN higher · leave-one-region-out / validation AOIs",
    )
    _label_deltas_h(ax, bars, d["delta"].to_numpy())
    _save(fig, FIG / "04_poly_minus_ghsl_by_city.png", left=0.18, bottom=0.16)

    # ------------------------------------------------------------------
    # Fig 5 — Per-city: Pixel LR vs GHSL pixel (encode delta)
    # ------------------------------------------------------------------
    d = per.copy()
    d["delta"] = d["pixel_ae_lr"] - d["ghsl_pixel"]
    d = d.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = [C_POS if v >= 0 else C_NEG for v in d["delta"]]
    bars = ax.barh(d["city"], d["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_xlabel("Macro-F1 advantage of Pixel AE LR over GHSL")
    n_win = int((d["delta"] >= 0).sum())
    _claim_header(
        fig,
        ax,
        f"At pixel scale, AE LR beats GHSL in {n_win}/10 metros",
        "Positive = Pixel AE LR + SMOTE (02) higher than GHSL pixel eval",
    )
    _label_deltas_h(ax, bars, d["delta"].to_numpy())
    _save(fig, FIG / "05_pixel_lr_minus_ghsl_by_city.png", left=0.18, bottom=0.16)

    # ------------------------------------------------------------------
    # Fig 6 — Classifier (02) vs regression (03): encode delta
    # ------------------------------------------------------------------
    d = per.copy()
    d["delta"] = d["pixel_ae_lr"] - d["pixel_frac_majority"]
    d = d.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = [C_FOCUS if v >= 0 else C_FOCUS2 for v in d["delta"]]
    bars = ax.barh(d["city"], d["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_xlabel("Macro-F1 gap (Pixel LR − regression)")
    n_reg_win = int((d["delta"] < 0).sum())
    _claim_header(
        fig,
        ax,
        f"Regression + SMOTE wins {n_reg_win}/10 metros vs LR + SMOTE",
        "Orange = LR higher · blue = regression higher · pure only · SMOTE train",
    )
    _label_deltas_h(ax, bars, d["delta"].to_numpy())
    _save(fig, FIG / "06_lr_minus_frac_by_city.png", left=0.18, bottom=0.16)

    # ------------------------------------------------------------------
    # Fig 7 — Four-model strip sorted within unit (mute cross-unit temptation)
    # Two small panels, one claim each already covered; this is a compact scoreboard
    # with direct labels and unit captions — sorted by value within panel
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    left = pd.DataFrame(
        [("AE CNN", m_poly_cnn, C_FOCUS), ("Centroid LR", m_poly_lr, C_MUTE), ("GHSL", m_ghsl_b, C_GHSL)],
        columns=["label", "f1", "color"],
    ).sort_values("f1")
    right = pd.DataFrame(
        [("AE LR", m_pix_lr, C_FOCUS), ("Regression", m_pix_frac, C_FOCUS2), ("GHSL", m_ghsl_p, C_GHSL)],
        columns=["label", "f1", "color"],
    ).sort_values("f1")
    for ax, df_, panel in [
        (axes[0], left, "Building unit"),
        (axes[1], right, "Pixel unit"),
    ]:
        bars = ax.barh(df_["label"], df_["f1"], color=df_["color"], height=0.62)
        ax.set_xlim(0, 1.0)
        ax.set_title(panel, fontsize=12, pad=10)
        _label_bars_h(ax, bars, df_["f1"].to_numpy(), dx=0.015)
        ax.tick_params(axis="y", pad=8, length=0)
    axes[0].set_xlabel("Mean macro-F1")
    axes[1].set_xlabel("Mean macro-F1")
    # Figure-level claim above panels — keep clear of panel titles
    fig.suptitle(
        "Same-unit scoreboards — compare within each panel only",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.82, left=0.12, right=0.96, bottom=0.16, wspace=0.35)
    fig.savefig(FIG / "07_unit_scoreboards.png", dpi=150, facecolor="white")
    plt.close(fig)

    # Keep aliases expected by older notebook refs (point to refined figs)
    # Write a small manifest of figures + claims
    fig_manifest = [
        {"file": "01_building_ae_beats_ghsl.png", "claim": "On buildings, Polygon AE CNN beats GHSL"},
        {"file": "02_pixel_lr_vs_frac_vs_ghsl.png", "claim": f"On pixels, regression+SMOTE edges LR by {m_pix_frac - m_pix_lr:+.2f} F1"},
        {"file": "03_unit_drop_building_to_pixel.png", "claim": f"Building→pixel costs {m_poly_cnn - m_pix_lr:.2f} F1 for AE"},
        {"file": "04_poly_minus_ghsl_by_city.png", "claim": "AE CNN − GHSL by city (building)"},
        {"file": "05_pixel_lr_minus_ghsl_by_city.png", "claim": "Pixel LR − GHSL by city"},
        {"file": "06_lr_minus_frac_by_city.png", "claim": "Regression+SMOTE wins most metros vs LR+SMOTE"},
        {"file": "07_unit_scoreboards.png", "claim": "Within-unit scoreboards"},
    ]
    (OUT / "figure_manifest.json").write_text(json.dumps(fig_manifest, indent=2) + "\n")

    print(primary_four.to_string(index=False))
    print(f"\nWrote refined figures under {FIG}", flush=True)


if __name__ == "__main__":
    main()
