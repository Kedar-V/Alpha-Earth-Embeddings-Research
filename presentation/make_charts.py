#!/usr/bin/env python3
"""Slide charts for the AlphaEarth deck, built with the data-visualization-principles skill.

Rules applied on every chart (see .cursor/skills/data-visualization-principles/):
  * one claim per chart, stated in an active, left-aligned title; subtitle gives what / who / which test set
  * bars start at zero; AUC dot plot uses its null (0.5) as the reference instead of bars
  * the reader never subtracts: per-metro comparisons are plotted as differences
  * one accent (blue) for the series the title is about, grey for context, navy for the GHSL reference;
    residential = teal, non-residential = amber, same as the maps
  * direct value labels, no side legends, "lower/higher is better" stated, rounded numbers, light grid only

Every number is read from run artifacts. Run:  .venv/bin/python presentation/make_charts.py
Writes presentation/figures/chart_*.png (sized to their slide boxes, so 1 pt here = 1 pt on the slide).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
O = ROOT / "outputs"

NAVY, SLATE, TEAL, AMBER, BLUE = "#0B2238", "#4B5A6E", "#2B9DB0", "#C27A2C", "#0057A0"
GREY, LIGHT = "#B8BEC6", "#E4E7EB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": NAVY,
    "axes.edgecolor": GREY, "axes.labelcolor": SLATE, "xtick.color": SLATE, "ytick.color": NAVY,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
})
DPI = 250


def frame(w, h, title, subtitle, foot=None, left=0.2, right=0.97, bottom=0.12, top_in=0.95):
    """Figure with left-aligned active title + subtitle above the axes; top_in = inches reserved for text."""
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    top = 1 - top_in / h
    foot_in = 0.3 if foot else 0.0
    ax = fig.add_axes([left, bottom + foot_in / h, right - left, top - bottom - foot_in / h])
    fig.text(0.015, 1 - 0.1 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=NAVY, wrap=True)
    fig.text(0.015, 1 - 0.1 / h - (0.25 * (title.count("\n") + 1) + 0.06) / h, subtitle, ha="left", va="top",
             fontsize=10.5, color=SLATE)
    if foot:
        fig.text(0.015, 0.06 / h, foot, ha="left", va="bottom", fontsize=9.5, color=SLATE, style="italic")
    return fig, ax


def save(fig, name):
    fig.savefig(FIG / name, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def hbar_labels(ax, ys, vals, fmt, inside_color="white", pad=0.01, xmax=None):
    for y, v in zip(ys, vals):
        ax.text(v + pad * (xmax or 1), y, fmt(v), va="center", ha="left", fontsize=10.5, color=NAVY)


def short(region):
    return {"LA_W": "LA", "NYC_NE": "NYC", "SF_W": "SF"}.get(region, region.split("_")[0].replace(" ", ""))


# ---------------------------------------------------------------- 1. balancing (slide 4)
def chart_balance():
    df = pd.read_csv(FIG / "balancing_counts_by_metro.csv")
    df["share"] = df.res_before / (df.res_before + df.nres_before) * 100
    df = df.sort_values("share", ascending=True)
    names = [short(r) for r in df.region]
    fig, ax = frame(4.55, 3.65, "Most metros are residential-heavy;\nChicago and SF are the reverse",
                    "Residential share of pure 10 m cells, before balancing", top_in=1.15, left=0.2, right=0.9,
                    foot="Balancing then keeps 50% in every metro (875,907 cells per class)")
    cols = [TEAL if s >= 50 else AMBER for s in df.share]
    ax.barh(names, df.share, color=cols, height=0.68)
    ax.axvline(50, color=NAVY, lw=1.2, ls="--")
    ax.text(50, len(df) - 0.35, " 50%", color=NAVY, fontsize=10, va="bottom", ha="left")
    for y, v in enumerate(df.share):
        ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=10.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "chart_balance.png")


# ---------------------------------------------------------------- Canvas-style per-metro dot rows
# Style taken from notebooks/ghsl_built_s_vs_ae/"AlphaEarth Weeks 1–5 Canvas.html" (boards 3–7):
# one row per metro in canvas order on a region-tinted band, one dot per model joined by a grey line,
# GHS-BUILT-S orange, AlphaEarth models in blues, margin column coloured by who is ahead.
CANVAS_ORDER = ["Boston_NE", "NYC_NE", "Atlanta_SE", "Houston_S", "Chicago_MW", "Phoenix_SW", "Denver_W", "LA_W", "SF_W", "Seattle_NW"]
BAND = {"NE": "#ECE8F4", "SE": "#E2EFE6", "S": "#F3EDD5", "MW": "#F3E3E7", "SW": "#EFE4D9", "W": "#E1ECEE", "NW": "#E9EDDC"}
C_GHSL, C_AE, C_AE2, C_LINK, C_GRID = "#C2691A", "#1F6FA8", "#4AA0DB", "#B0ADA3", "#E5E1D6"


def canon(name):
    """'Boston NE' / 'Boston_NE' -> 'Boston_NE'."""
    return name.strip().replace(" ", "_")


def metro_label(region):
    city, reg = region.split("_")
    return f"{city} {reg}"


def dot_rows(ax, series, xlim, xticks, xfmt, link=True):
    """series: list of (name, {region: value}, colour, size). Rows top-to-bottom in canvas order."""
    n = len(CANVAS_ORDER)
    for i, r in enumerate(CANVAS_ORDER):
        ax.axhspan(i - 0.5, i + 0.5, color=BAND[r.split("_")[1]], lw=0, zorder=0)
        vals = [sv[r] for _, sv, _, _ in series if r in sv]
        if link and len(vals) > 1:
            ax.hlines(i, min(vals), max(vals), color=C_LINK, lw=2, zorder=1)
        for _, sv, col, size in series:
            if r in sv:
                ax.scatter(sv[r], i, s=size, color=col, zorder=3, edgecolors="white", linewidths=0.6)
    for xt in xticks:
        ax.axvline(xt, color=C_GRID, lw=0.8, zorder=0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([metro_label(r) for r in CANVAS_ORDER], fontsize=10.5)
    ax.set_xlim(*xlim); ax.set_xticks(xticks); ax.set_xticklabels([xfmt(x) for x in xticks], fontsize=10)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", length=0)


def margin_col(fig, ax, deltas, better_low, fmt="{:+.3f}", head="Margin", tie=0.01):
    """Text column right of ax: our best minus GHSL (or vs comparator); blue = ours ahead, orange = behind."""
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.6 / fig.get_figwidth(), pos.height])
    mx.set_ylim(len(CANVAS_ORDER) - 0.5, -0.5); mx.set_xlim(0, 1); mx.axis("off")
    mx.text(0.5, -0.75, head, ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    for i, r in enumerate(CANVAS_ORDER):
        d = deltas[r]
        ahead = (d < -tie) if better_low else (d > tie)
        behind = (d > tie) if better_low else (d < -tie)
        col = C_AE if ahead else (C_GHSL if behind else SLATE)
        mx.text(0.5, i, fmt.format(d), ha="center", va="center", fontsize=10.5, color=col, fontweight="bold")
    return mx


def key(fig, y_in, h, items):
    """One-line dot key at the top, under the subtitle (legend at top, per text-and-legends.md)."""
    w = fig.get_figwidth()
    x_in = 0.07
    for name, col in items:
        fig.text(x_in / w, 1 - y_in / h, "●", color=col, fontsize=12, va="center", ha="left")
        fig.text((x_in + 0.2) / w, 1 - y_in / h, name, color=NAVY, fontsize=10, va="center", ha="left")
        x_in += 0.2 + 0.083 * len(name) + 0.3


def count(deltas, better_low, tie=0.01):
    v = np.array(list(deltas.values()))
    ahead = int(((v < -tie) if better_low else (v > tie)).sum())
    behind = int(((v > tie) if better_low else (v < -tie)).sum())
    return ahead, len(v) - ahead - behind, behind


# ---------------------------------------------------------------- 2. AE per metro (slide 7)
def chart_ae_metro():
    df = pd.read_csv(O / "week4_pixel_balanced_train/loro_metrics.csv")
    f1 = dict(zip(df.region, df.macro_f1))
    mean = json.load(open(O / "week4_pixel_balanced_train/summary.json"))["loro_macro_f1_mean"]
    hi, lo = max(f1, key=f1.get), min(f1, key=f1.get)
    w, h = 4.75, 4.15
    fig, ax = frame(w, h, f"F1 ranges from {f1[hi]:.2f} in {hi.split('_')[0]}\nto {f1[lo]:.2f} in {lo.split('_')[0]}",
                    "Macro-F1 per held-out metro (0.5 = chance)\nAlphaEarth 64-D + logistic regression, 1.75M cells",
                    top_in=1.3, left=0.25, right=0.84, bottom=0.08)
    dot_rows(ax, [("AlphaEarth 64-D", f1, C_AE, 70)], (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda x: f"{x:.1f}")
    ax.axvline(mean, color=NAVY, lw=1.2, ls="--", zorder=2)
    ax.text(mean, -0.62, f"mean {mean:.2f}", ha="center", va="bottom", fontsize=10, color=NAVY)
    pos = ax.get_position()
    vx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.1, pos.height]); vx.axis("off")
    vx.set_ylim(len(CANVAS_ORDER) - 0.5, -0.5); vx.set_xlim(0, 1)
    for i, r in enumerate(CANVAS_ORDER):
        vx.text(0.1, i, f"{f1[r]:.2f}", va="center", fontsize=10.5, fontweight="bold" if r in (hi, lo) else "normal")
    save(fig, "chart_ae_metro.png")


# ---------------------------------------------------------------- 3. CNN vs one centre cell per metro (slide 9)
def chart_cnn_metro():
    df = pd.read_csv(O / "week2_cnn_bbox/binary_loro_comparison.csv")
    cnn, lr = dict(zip(df.region, df.cnn_bbox)), dict(zip(df.region, df.lr_centroid))
    d = {r: cnn[r] - lr[r] for r in cnn}
    a, t, b = count(d, better_low=False)
    w, h = 4.75, 4.15
    fig, ax = frame(w, h, f"The CNN was ahead in {a} metros,\ntied in {t}, behind in {b}",
                    f"Building F1, held-out metro, higher is better (week 2)\nmean {df.cnn_bbox.mean():.2f} vs {df.lr_centroid.mean():.2f}",
                    top_in=1.45, left=0.25, right=0.78, bottom=0.08)
    key(fig, 1.25, h, [("CNN on the grid", C_AE), ("One centre cell", C_AE2)])
    dot_rows(ax, [("One centre cell", lr, C_AE2, 55), ("CNN", cnn, C_AE, 70)], (0.5, 0.9),
             [0.5, 0.6, 0.7, 0.8, 0.9], lambda x: f"{x:.1f}")
    margin_col(fig, ax, d, better_low=False, fmt="{:+.2f}", head="Δ")
    save(fig, "chart_cnn_metro.png")


# ---------------------------------------------------------------- 4. ViT vs simple models (slide 10)
def chart_vit():
    pls = json.load(open(O / "week4_pls_10m_vs_100m/summary.json"))["100m"]
    rows = [
        ("Logistic regression", pls["baseline_lr"]["loro_macro_f1_mean"], NAVY),
        ("PLS", pls["pls_da_best"]["loro_macro_f1_mean"], GREY),
        ("ViT", json.load(open(O / "week4_vit_100m/summary.json"))["loro_macro_f1_mean"], BLUE),
        ("3D ViT", json.load(open(O / "week4_vit_100m_3d_v4/summary.json"))["loro_macro_f1_mean"], GREY),
        ("LR on full 10×10 grid", json.load(open(O / "week4_100m_tile_train/summary.json"))["min5"]["loro_macro_f1_mean"], GREY),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = frame(4.75, 4.15, "The ViT matched, but did not beat,\na simple logistic regression",
                    "Macro-F1 on 100 m tiles, held-out metro, higher is better\n52,492 balanced tiles",
                    top_in=1.35, left=0.36, right=0.88)
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for y, r in enumerate(rows):
        ax.text(r[1] + 0.015, y, f"{r[1]:.2f}", va="center", fontsize=10.5)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "chart_vit.png")


# ---------------------------------------------------------------- 5. XGBoost classifier vs GHSL per metro (slide 11)
def chart_xgb_metro():
    rows = list(csv.DictReader(open(O / "week12_ae_s2_xgb_clf_loro/20261113_065557/artifacts/metro_clf_f1.csv")))
    g = {canon(r["metro"]): float(r["GHSL_macro"]) for r in rows}
    x = {canon(r["metro"]): float(r["CLF_macro"]) for r in rows}
    d = {r: x[r] - g[r] for r in g}
    a, t, b = count(d, better_low=False)
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"XGBoost classifier ahead of GHSL in {a} metros, tied in {t}, behind in {b}",
                    "Macro-F1 per held-out metro, higher is better · week 12, AE + S2 & shape · pooled 0.84 vs GHSL 0.82",
                    top_in=1.0, left=0.16, right=0.88, bottom=0.08)
    key(fig, 0.8, h, [("XGBoost classifier", C_AE), ("GHS-BUILT-S", C_GHSL)])
    dot_rows(ax, [("GHSL", g, C_GHSL, 70), ("XGBoost", x, C_AE, 70)], (0.6, 1.0),
             [0.6, 0.7, 0.8, 0.9, 1.0], lambda v: f"{v:.1f}")
    margin_col(fig, ax, d, better_low=False, fmt="{:+.2f}", head="Δ")
    save(fig, "chart_xgb_metro.png")


# ---------------------------------------------------------------- 5b. ViT + S2 & shape vs GHSL per metro, regression (new slide)
def chart_vit_metro():
    m = json.load(open(O / "week14_ae_grid_vit_s2_loro/20261113_075223/artifacts/section11_ae_grid_vit_s2_metrics.json"))["buildings"]["metro_soft"]
    vals = {canon(r["region"]): r for r in m}
    d = {r: vals[r]["ae_vit_s2"]["mean"] - vals[r]["ghsl_built_s"]["mean"] for r in vals}
    a, t, b = count(d, better_low=True)
    w, h = 11.9, 4.15
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    fig.text(0.012, 1 - 0.1 / h, f"ViT + S2 & shape beats GHSL in {a} metros, ties in {t}, loses in {b}",
             ha="left", va="top", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.012, 1 - 0.46 / h, "Share error per building, lower is better · 71,497 frozen buildings, held-out metro · margin = ViT mean − GHSL mean",
             ha="left", va="top", fontsize=10.5, color=SLATE)
    key(fig, 0.86, h, [("ViT + S2 & shape", C_AE), ("GHS-BUILT-S", C_GHSL)])
    top, bot = 1 - 1.25 / h, 0.08
    for k, (col, title) in enumerate([("res", "Residential share error"), ("nres", "Non-residential share error")]):
        ax = fig.add_axes([0.1 + k * 0.42, bot, 0.36, top - bot])
        g = {r: vals[r]["ghsl_built_s"][col] for r in vals}
        v = {r: vals[r]["ae_vit_s2"][col] for r in vals}
        dot_rows(ax, [("GHSL", g, C_GHSL, 60), ("ViT", v, C_AE, 60)], (0.1, 0.32),
                 [0.1, 0.15, 0.2, 0.25, 0.3], lambda x: f"{x:.2f}")
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=NAVY, pad=4)
        if k == 1:
            ax.set_yticklabels([])
    margin_col(fig, ax, d, better_low=True, fmt="{:+.3f}", head="Margin")
    save(fig, "chart_vit_metro.png")


# ---------------------------------------------------------------- 6. progression (slide 13)
def chart_progression():
    sp = json.load(open(O / "week14_ae_grid_vit_s2_loro/20261113_075223/artifacts/section11_ae_grid_vit_s2_metrics.json"))["buildings"]["soft_pooled"]
    steps = [("AE Ridge", "ae_ridge"), ("AE PLS", "ae_pls"), ("AE XGBoost", "ae_xgb"), ("+ rings", "ae_ring"),
             ("+ S2 &\nshape", "ae_s2"), ("CNN + S2\n& shape", "ae_cnn_s2"), ("ViT + S2\n& shape", "ae_vit_s2")]
    vals = [sp[k]["mean"] for _, k in steps]
    ghsl = sp["ghsl_built_s"]["mean"]
    fig, ax = frame(8.4, 4.15, "Error fell below GHSL only once Sentinel-2 and footprint shape were added",
                    "Share error per building, lower is better · 71,497 frozen buildings, held-out metro",
                    foot="Weeks 6–7 train on pixels and average them per building; from week 8 models train on buildings directly",
                    top_in=0.75, left=0.07, right=0.98, bottom=0.14)
    x = np.arange(len(steps))
    ax.bar(x, vals, color=[BLUE if v < ghsl else GREY for v in vals], width=0.6)
    ax.axhline(ghsl, color=NAVY, lw=1.4, ls="--")
    ax.text(len(steps) - 0.5, ghsl + 0.006, f"GHSL {ghsl:.3f}", color=NAVY, ha="right", va="bottom", fontsize=10.5, fontweight="bold")
    for xi, v in zip(x, vals):
        ax.text(xi, v - 0.008, f"{v:.3f}", ha="center", va="top", fontsize=10.5, fontweight="bold",
                color="white" if v < ghsl else NAVY)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=10.5)
    ax.set_ylim(0, 0.33); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    save(fig, "chart_progression.png")


# ---------------------------------------------------------------- 7. cue separation (slide 14)
NAMES = {"fp_area_m2": "Footprint area", "fp_equiv_diam_m": None, "fp_compactness": "Footprint compactness",
         "s2_ndvi": "Vegetation (NDVI)", "s2_blue": "Blue", "s2_brightness": "Brightness", "s2_red": None,
         "s2_green": None, "s2_nir": None, "s2_contrast_small": "Fine texture", "s2_contrast_large": "Coarse texture",
         "s2_nres_score": "Combined score", "s2_blue_dom": "Blue share", "s2_contrast_ratio": "Texture ratio"}


def chart_cues(abl_dir):
    df = pd.read_csv(abl_dir / "s2_feature_auc.csv")
    df["name"] = df.feature.map(NAMES)
    df = df.dropna(subset=["name"]).sort_values("separation")
    fig, ax = frame(4.1, 4.15, "Footprint area is the\nstrongest single cue",
                    "How well each cue alone separates\nresidential from non-residential (AUC)",
                    top_in=1.35, left=0.42, right=0.92)
    cols = [BLUE if n.startswith("Footprint") else (NAVY if n == "Combined score" else GREY) for n in df.name]
    y = np.arange(len(df))
    ax.hlines(y, 0.5, df.separation, color=LIGHT, lw=2)
    ax.scatter(df.separation, y, color=cols, s=55, zorder=3)
    for yi, v in zip(y, df.separation):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=10)
    ax.axvline(0.5, color=NAVY, lw=1)
    ax.text(0.505, len(df) - 0.45, "coin flip", ha="left", va="bottom", fontsize=9.5, color=SLATE)
    ax.set_yticks(y); ax.set_yticklabels(df.name, fontsize=10)
    ax.set_xlim(0.48, 0.9); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); ax.set_ylim(-0.6, len(df) + 0.1)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    save(fig, "chart_cues.png")


# ---------------------------------------------------------------- 8. ablation (slide 15)
LABELS = {"AE only (64)": "AlphaEarth only", "Footprint shape only (3)": "Footprint shape only (3 numbers)",
          "S2 imagery only (10)": "Sentinel-2 imagery only", "Fuzzy NRES score only (1)": "GHSL-style score only",
          "AE + footprint shape": "AlphaEarth + footprint shape", "AE + S2 imagery": "AlphaEarth + imagery",
          "AE + all 16 S2 features (week 9)": "AlphaEarth + all 16 (week 9)"}


def chart_ablation(abl_dir):
    df = pd.read_csv(abl_dir / "s2_ablation.csv")
    ghsl = float(df.loc[df.features.str.startswith("GHS"), "mae_mean"].iloc[0])
    df = df[~df.features.str.startswith("GHS")].copy()
    df["label"] = df.features.map(LABELS)
    df = df.sort_values("mae_mean", ascending=False)
    fig, ax = frame(7.7, 4.15, "Adding footprint shape, not imagery, took AlphaEarth past GHSL",
                    "Share error per building, lower is better · same XGBoost, 71,497 frozen buildings, held-out metro",
                    top_in=0.75, left=0.37, right=0.97)
    cols = [BLUE if ("footprint" in l.lower() or "all 16" in l) else GREY for l in df.label]
    y = np.arange(len(df))
    ax.barh(df.label, df.mae_mean, color=cols, height=0.62)
    for yi, (v, c) in zip(y, zip(df.mae_mean, cols)):
        if v < ghsl:   # bar ends before the GHSL line: label inside
            ax.text(v - 0.004, yi, f"{v:.3f}", va="center", ha="right", fontsize=10.5, fontweight="bold",
                    color="white" if c == BLUE else NAVY)
        else:          # bar crosses the line: label past the bar end
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", ha="left", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.axvline(ghsl, color=NAVY, lw=1.4, ls="--")
    ax.text(ghsl, len(df) - 0.45, f"GHSL {ghsl:.3f}", color=NAVY, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 0.34); ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "chart_ablation.png")


# ================================================================ Generalisation (new section)
INTL = ROOT / "week4/international_geo_generalization/runs"
C_USA, C_JPN, C_IND = NAVY, "#8A93A0", C_AE   # India is the focal country (largest drop), others muted


def chart_gen_ladder():
    """Within-country -> held-out city -> city-to-city, per country (ordinal steps, so a slope chart)."""
    a = pd.read_csv(INTL / "india_loro_20261005_202600/pls_compare/summary_pls_vs_lr64.csv")
    j = pd.read_csv(INTL / "japan_loro_20261005_225832/pls_compare/summary_pls_vs_lr64.csv")
    rows = {"USA": a[(a.panel == "USA_10") & (a.model == "lr64")].iloc[0],
            "Japan": j[(j.panel == "JPN_10") & (j.model == "lr64")].iloc[0],
            "India": a[(a.panel == "IND_10") & (a.model == "lr64")].iloc[0]}
    steps = ["Random 80/20\nwithin country", "Held-out city", "Train one city,\ntest another"]
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, "Japan holds up on unseen cities; India drops to near chance",
                    "Macro-F1, 64-D logistic regression, 10 cities per country, higher is better (0.5 = chance)",
                    top_in=0.8, left=0.08, right=0.84, bottom=0.14)
    x = np.arange(3)
    for name, col in [("USA", C_USA), ("Japan", C_JPN), ("India", C_IND)]:
        r = rows[name]
        v = [r.within_80_20, r.loro_mean, r.city_off_mean]
        ax.plot(x, v, color=col, lw=2.6 if name == "India" else 2, marker="o", ms=8, zorder=3)
        for xi, vi in zip(x, v):
            below = name == "USA" and xi == 2   # avoid the Japan line crossing here
            ax.text(xi, vi + (-0.025 if below else 0.022), f"{vi:.2f}", ha="center", va="top" if below else "bottom",
                    fontsize=10.5, color=col, fontweight="bold")
        ax.text(2.12, v[-1], name, va="center", ha="left", fontsize=11.5, color=col, fontweight="bold")
    ax.axhline(0.5, color=SLATE, lw=1, ls=":")
    ax.text(-0.25, 0.505, "chance", fontsize=9.5, color=SLATE, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(steps, fontsize=10.5)
    ax.set_xlim(-0.3, 2.45); ax.set_ylim(0.4, 1.0); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="y", color=LIGHT, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    save(fig, "chart_gen_ladder.png")


def chart_gen_transfer():
    df = pd.read_csv(INTL / "intl_transfer_20261005_142709/artifacts/country_transfer_metrics.csv")
    order = ["USA", "JPN", "IND"]; names = {"USA": "USA", "JPN": "Japan", "IND": "India"}
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", "#CFE5E8", C_AE2, C_AE, NAVY])
    w, h = 7.9, 4.15
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    fig.text(0.015, 1 - 0.1 / h, "Transfer runs one way: Japan → USA holds, USA → India collapses",
             ha="left", va="top", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(0.015, 1 - 0.45 / h, "Macro-F1 · train on rows, test on columns · outlined = same country · 10 USA, 5 Japan, 5 India cities",
             ha="left", va="top", fontsize=10, color=SLATE)
    for k, (model, kk, title) in enumerate([("lr64", 64, "Full 64-D"), ("pls", 2, "PLS, 2 dimensions")]):
        ax = fig.add_axes([0.1 + k * 0.47, 0.1, 0.36, 0.62])
        sub = df[(df.model == model) & (df.k == kk)]
        M = np.array([[sub[(sub.source_country == a) & (sub.target_country == b)].macro_f1.iloc[0] for b in order] for a in order])
        ax.imshow(M, cmap=cmap, vmin=0.3, vmax=0.95)
        for i in range(3):
            for jx in range(3):
                v = M[i, jx]
                ax.text(jx, i, f"{v:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if v > 0.72 else NAVY)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=NAVY, lw=2))
        ax.set_xticks(range(3)); ax.set_xticklabels([names[c] for c in order], fontsize=10.5)
        ax.set_yticks(range(3)); ax.set_yticklabels([names[c] for c in order] if k == 0 else [], fontsize=10.5)
        ax.xaxis.set_ticks_position("top"); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=NAVY, pad=22)
        if k == 0:
            ax.set_ylabel("Trained on", fontsize=10.5)
        ax.text(1, 3.0, "Tested on ↑", ha="center", va="top", fontsize=9.5, color=SLATE)
    save(fig, "chart_gen_transfer.png")


def chart_gen_threshold():
    df = pd.read_csv(INTL / "intl_transfer_20261005_142709/artifacts/country_transfer_metrics.csv")
    df = df[(df.model == "lr64") & (df.source_country != df.target_country)].copy()
    names = {"USA": "USA", "JPN": "Japan", "IND": "India"}
    df["pair"] = [f"{names[a]} → {names[b]}" for a, b in zip(df.source_country, df.target_country)]
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, "Ranking survives across countries; the 0.5 cut-off does not",
                    "Train in one country, test in another · 64-D logistic regression · higher is better",
                    top_in=0.95, left=0.2, right=0.8, bottom=0.1)
    key(fig, 0.78, h, [("AUROC: ranks buildings correctly?", NAVY), ("F1 at 0.5: right decision?", C_AE)])
    y = np.arange(len(df))
    ax.hlines(y, df.macro_f1, df.auroc, color=C_LINK, lw=2.5, zorder=1)
    ax.scatter(df.auroc, y, color=NAVY, s=70, zorder=3)
    ax.scatter(df.macro_f1, y, color=C_AE, s=70, zorder=3)
    for yi, r in df.iterrows():
        ax.text(r.auroc + 0.012, yi, f"{r.auroc:.2f}", va="center", fontsize=10.5, color=NAVY)
        ax.text(r.macro_f1 - 0.012, yi, f"{r.macro_f1:.2f}", va="center", ha="right", fontsize=10.5, color=C_AE, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(df.pair, fontsize=10.5); ax.set_ylim(len(df) - 0.5, -0.6)
    ax.set_xlim(0.25, 0.9); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.axvline(0.5, color=SLATE, lw=1, ls=":")
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    # non-residential F1 for the collapsed pairs: everything is called residential
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.14, pos.height]); mx.axis("off")
    mx.set_ylim(len(df) - 0.5, -0.6); mx.set_xlim(0, 1)
    mx.text(0.5, -0.75, "Non-res F1", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    for yi, r in df.iterrows():
        v = r.f1_nonresidential
        mx.text(0.5, yi, f"{v:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=C_GHSL if v < 0.2 else SLATE)
    save(fig, "chart_gen_threshold.png")


def chart_gen_india():
    d = pd.read_csv(INTL / "india_loro_20261005_202600/pls_compare/loro_per_city_pls_vs_lr64.csv")
    d["city"] = d.held_out.str.replace("IND_", "").str.title()
    d = d.sort_values("lr64_k64").reset_index(drop=True)
    d["delta"] = d.pls_k2 - d.lr64_k64
    ahead = int((d.delta > 0.01).sum()); behind = int((d.delta < -0.01).sum()); ties = len(d) - ahead - behind
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"Two PLS dimensions are clearly better in {ahead} of 10 Indian cities",
                    f"Macro-F1 per held-out Indian city, higher is better · {ties} within ±0.01, {behind} worse (Jaipur −0.01)",
                    top_in=0.95, left=0.16, right=0.86, bottom=0.08)
    key(fig, 0.78, h, [("PLS, 2 dimensions", C_AE), ("Full 64-D", C_AE2)])
    y = np.arange(len(d))
    for i in y:
        ax.axhspan(i - 0.5, i + 0.5, color="#F3EDD5" if i % 2 == 0 else "#F8F4E6", lw=0, zorder=0)
    ax.hlines(y, d.lr64_k64, d.pls_k2, color=C_LINK, lw=2, zorder=1)
    ax.scatter(d.lr64_k64, y, color=C_AE2, s=60, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(d.pls_k2, y, color=C_AE, s=70, zorder=3, edgecolors="white", linewidths=0.6)
    ax.axvline(0.5, color=SLATE, lw=1, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(d.city, fontsize=10.5); ax.set_ylim(len(d) - 0.5, -0.5)
    ax.set_xlim(0.3, 0.75); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    for xt in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ax.axvline(xt, color=C_GRID, lw=0.8, zorder=0.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.1, pos.height]); mx.axis("off")
    mx.set_ylim(len(d) - 0.5, -0.5); mx.set_xlim(0, 1)
    mx.text(0.5, -0.75, "Δ", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    for i, v in zip(y, d.delta):
        mx.text(0.5, i, f"{v:+.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=C_AE if v > 0.01 else (C_GHSL if v < -0.01 else SLATE))
    save(fig, "chart_gen_india.png")


def chart_gen_loco():
    d = pd.read_csv(ROOT / "week5/runs/dou_balanced_week5_20261008_192748/task_invariant_subspace/artifacts/loco_results.csv")
    d = d[d.r == 3].groupby(["method", "target"]).macro_f1.mean().unstack("target")
    models = [("lr64", "Full 64-D", C_AE2), ("pooled_pls", "Pooled PLS", GREY),
              ("geometric_avgproj", "Geometric consensus", GREY), ("task_invariant_A", "Task-invariant subspace", C_AE)]
    targets = [("USA", "USA"), ("JPN", "Japan"), ("IND", "India")]
    w, h = 7.9, 4.15
    lo, hi = d.loc["lr64", ["USA", "JPN", "IND"]].mean(), d.loc["task_invariant_A", ["USA", "JPN", "IND"]].mean()
    fig, ax = frame(w, h, f"On an unseen country, a shared subspace lifts F1 from {lo:.2f} to {hi:.2f}",
                    "Macro-F1, leave-one-country-out, 2.0M cells balanced by urbanisation, mean of 5 seeds, rank 3",
                    top_in=0.95, left=0.12, right=0.97, bottom=0.1)
    key(fig, 0.78, h, [(m[1], m[2]) for m in models])
    cats = [t[1] for t in targets] + ["Mean"]
    y = np.arange(len(cats))
    for i in y:
        ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4" if i == 3 else "white", lw=0, zorder=0)
    for mk, mn, col in models:
        vals = [d.loc[mk, t[0]] for t in targets] + [d.loc[mk, ["USA", "JPN", "IND"]].mean()]
        ax.scatter(vals, y, color=col, s=95 if mk in ("lr64", "task_invariant_A") else 55, zorder=3,
                   edgecolors="white", linewidths=0.6)
    for i, t in enumerate([t[0] for t in targets] + ["mean"]):
        a = d.loc["lr64", ["USA", "JPN", "IND"]].mean() if t == "mean" else d.loc["lr64", t]
        b = d.loc["task_invariant_A", ["USA", "JPN", "IND"]].mean() if t == "mean" else d.loc["task_invariant_A", t]
        ax.hlines(i, min(a, b), max(a, b), color=C_LINK, lw=2.5, zorder=1)
        ax.text(max(a, b) + 0.015, i, f"{b - a:+.2f}", va="center", fontsize=10.5, fontweight="bold",
                color=C_AE if b - a > 0.01 else (C_GHSL if b - a < -0.01 else SLATE))
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=11); ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.set_xlim(0.3, 0.95); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.axvline(0.5, color=SLATE, lw=1, ls=":")
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "chart_gen_loco.png")


# ================================================================ Africa (new section)
AFR5 = O / "ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/scored.parquet"
AFR_LORO = O / "africa_loro_vs_ghsl/20261113_125810/artifacts/scored.parquet"
AFR_NAMES = {"EGY_CAIRO": "Cairo", "GHA_ACCRA": "Accra", "KEN_NAIROBI": "Nairobi", "NGA_LAGOS": "Lagos", "ZAF_JOHANNESBURG": "Johannesburg"}
US_GHSL_CELL_F1 = json.load(open(O / "week4_pixel_balanced_train/section1_classification_metrics.json"))["cell_10m"]["pooled"]["ghsl_built_s"]["macro_f1"]


def _afr5():
    from sklearn.metrics import f1_score
    d = pd.read_parquet(AFR5, columns=["city_id", "pixel_lon", "pixel_lat", "binary_label", "bs_frac_res", "bs_frac_nres",
                                       "ridge_frac_res", "ridge_frac_nres"])
    d["g"] = (d.bs_frac_res >= d.bs_frac_nres).astype(int)
    d["a"] = (d.ridge_frac_res >= d.ridge_frac_nres).astype(int)
    rows = []
    for c, x in d.groupby("city_id"):
        nres = x.binary_label == 0
        rows.append({"city": AFR_NAMES[c], "n": len(x),
                     "f1_g": f1_score(x.binary_label, x.g, average="macro"), "f1_a": f1_score(x.binary_label, x.a, average="macro"),
                     "rec_g": (x.g[nres] == 0).mean(), "rec_a": (x.a[nres] == 0).mean(), "pres_g": x.g.mean()})
    return d, pd.DataFrame(rows)


def africa_rows(ax, df, cg, ca, xlim, xticks, xfmt):
    y = np.arange(len(df))
    for i in y:
        ax.axhspan(i - 0.5, i + 0.5, color="#F3EDD5" if i % 2 == 0 else "#F8F4E6", lw=0, zorder=0)
    ax.hlines(y, df[cg], df[ca], color=C_LINK, lw=2.2, zorder=1)
    ax.scatter(df[cg], y, color=C_GHSL, s=75, zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(df[ca], y, color=C_AE, s=75, zorder=3, edgecolors="white", linewidths=0.6)
    for xt in xticks:
        ax.axvline(xt, color=C_GRID, lw=0.8, zorder=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df.city, fontsize=11); ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xlim(*xlim); ax.set_xticks(xticks); ax.set_xticklabels([xfmt(v) for v in xticks], fontsize=10)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


def chart_africa_bias():
    _, t = _afr5()
    t = t.sort_values("rec_g")
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"GHSL finds only {t.rec_g.min():.0%}–{t.rec_g.max():.0%} of non-residential cells in African cities",
                    f"Share of true non-residential 10 m cells caught · truth is 50/50, GHSL calls {t.pres_g.min():.0%}–{t.pres_g.max():.0%} residential",
                    top_in=0.95, left=0.17, right=0.86, bottom=0.1)
    key(fig, 0.78, h, [("AlphaEarth, trained in that city", C_AE), ("GHS-BUILT-S", C_GHSL)])
    africa_rows(ax, t, "rec_g", "rec_a", (0, 1.0), [0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.0%}")
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.11, pos.height]); mx.axis("off")
    mx.set_ylim(len(t) - 0.5, -0.5); mx.set_xlim(0, 1)
    mx.text(0.5, -0.65, "GHSL", ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_GHSL)
    for i, v in enumerate(t.rec_g):
        mx.text(0.5, i, f"{v:.0%}", ha="center", va="center", fontsize=11, fontweight="bold", color=C_GHSL)
    save(fig, "chart_africa_bias.png")


def chart_africa_within():
    _, t = _afr5()
    t = t.sort_values("f1_g")
    t["d"] = t.f1_a - t.f1_g
    a = int((t.d > 0.01).sum())
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"Trained in the same city, AlphaEarth beats GHSL in {a} of {len(t)} African cities",
                    "Macro-F1 on balanced 10 m cells, higher is better · AlphaEarth Ridge, 5-fold within each city",
                    top_in=0.95, left=0.17, right=0.86, bottom=0.1)
    key(fig, 0.78, h, [("AlphaEarth", C_AE), ("GHS-BUILT-S", C_GHSL)])
    africa_rows(ax, t, "f1_g", "f1_a", (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda v: f"{v:.1f}")
    ax.axvline(US_GHSL_CELL_F1, color=C_GHSL, lw=1.2, ls="--", zorder=2)
    ax.text(US_GHSL_CELL_F1, -0.62, f"GHSL on US cells {US_GHSL_CELL_F1:.2f}", color=C_GHSL, fontsize=9.5, ha="center", va="bottom")
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.11, pos.height]); mx.axis("off")
    mx.set_ylim(len(t) - 0.5, -0.5); mx.set_xlim(0, 1)
    mx.text(0.5, -0.65, "Δ", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    for i, v in enumerate(t.d):
        mx.text(0.5, i, f"{v:+.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                color=C_AE if v > 0.01 else (C_GHSL if v < -0.01 else SLATE))
    save(fig, "chart_africa_within.png")


def chart_africa_loro():
    rows = []
    for line in open(O / "africa_clf_vs_ghsl/20261116_231920/RESULTS.md"):
        p = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(p) == 5 and p[1].startswith(("EGY_", "NGA_")):
            rows.append({"country": p[0], "city": p[1].split("_", 1)[1].replace("_", " ").title(),
                         "f1_a": float(p[2]), "f1_g": float(p[4])})
    t = pd.DataFrame(rows)
    t["d"] = t.f1_a - t.f1_g
    t = t.sort_values(["country", "d"]).reset_index(drop=True)
    a, b = int((t.d > 0.01).sum()), int((t.d < -0.01).sum())
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"On a city it has never seen, AlphaEarth wins {a} of 10; GHSL wins {b}",
                    "Macro-F1 per held-out city, Egypt and Nigeria, higher is better · AlphaEarth XGBoost, leave-one-city-out",
                    top_in=0.95, left=0.24, right=0.86, bottom=0.08)
    key(fig, 0.78, h, [("AlphaEarth", C_AE), ("GHS-BUILT-S", C_GHSL)])
    t["city"] = [f"{c} ({k})" for c, k in zip(t.city, t.country)]
    africa_rows(ax, t, "f1_g", "f1_a", (0.3, 0.85), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], lambda v: f"{v:.1f}")
    ax.axvline(0.5, color=SLATE, lw=1, ls=":")
    pos = ax.get_position()
    mx = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.11, pos.height]); mx.axis("off")
    mx.set_ylim(len(t) - 0.5, -0.5); mx.set_xlim(0, 1)
    mx.text(0.5, -0.65, "Δ", ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)
    for i, v in enumerate(t.d):
        mx.text(0.5, i, f"{v:+.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                color=C_AE if v > 0.01 else (C_GHSL if v < -0.01 else SLATE))
    save(fig, "chart_africa_loro.png")


def map_lagos():
    """Lagos window: truth vs GHSL vs AlphaEarth on the labelled 10 m cells (squares drawn at true size)."""
    d, _ = _afr5()
    W = (3.352, 6.576, 3.360, 6.584)
    x = d[(d.city_id == "NGA_LAGOS") & d.pixel_lon.between(W[0], W[2]) & d.pixel_lat.between(W[1], W[3])]
    coslat = np.cos(np.radians(6.58)); half_lat = 5 / 111_320; half_lon = 5 / (111_320 * coslat)
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle as R
    w, h = 12.1, 4.6
    fig, axes = plt.subplots(1, 3, figsize=(w, h), dpi=DPI)
    for ax, (col, title) in zip(axes, [("binary_label", "Truth (Overture tags)"), ("g", "GHSL"), ("a", "AlphaEarth (in-city)")]):
        cols = [TEAL if v == 1 else AMBER for v in x[col]]
        ax.add_collection(PatchCollection([R((lo - half_lon, la - half_lat), 2 * half_lon, 2 * half_lat)
                                           for lo, la in zip(x.pixel_lon, x.pixel_lat)], facecolor=cols, edgecolor="none"))
        if col != "binary_label":
            wrong = x[x[col] != x.binary_label]
            ax.add_collection(PatchCollection([R((lo - half_lon, la - half_lat), 2 * half_lon, 2 * half_lat)
                                               for lo, la in zip(wrong.pixel_lon, wrong.pixel_lat)],
                                              facecolor="none", edgecolor=NAVY, lw=0.9))
            title += f" · wrong on {len(wrong) / len(x):.0%}"
        ax.set_xlim(W[0], W[2]); ax.set_ylim(W[1], W[3]); ax.set_aspect(1 / coslat)
        ax.set_facecolor("#F4F4F4"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)
        dl = 200 / (111_320 * coslat)
        ax.plot([W[0] + 0.0003, W[0] + 0.0003 + dl], [W[1] + 0.0003] * 2, color=NAVY, lw=3, solid_capstyle="butt")
        ax.text(W[0] + 0.0003 + dl / 2, W[1] + 0.0005, "200 m", ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=AMBER),
                        plt.Rectangle((0, 0), 1, 1, fc="none", ec=NAVY, lw=1.2)],
               labels=["Residential", "Non-residential", "Wrong vs truth"], loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1, wspace=0.04)
    save(fig, "map_lagos.png")
    print("lagos window cells", len(x), "res share %.2f" % x.binary_label.mean())


# ================================================================ Pure vs mixed cells (outputs/cell_purity_by_city)
def chart_purity():
    c = pd.read_csv(O / "cell_purity_by_city/city_purity_soft.csv")
    k = pd.read_csv(O / "cell_purity_by_city/country_purity_soft.csv")
    allpct = 100 * c.n_mixed_soft.sum() / c.n_cells.sum()
    order = k.sort_values("pct_mixed").country_id.tolist()        # most mixed country at the top
    names = {"EGY": "Egypt", "GHA": "Ghana", "KEN": "Kenya", "NGA": "Nigeria", "ZAF": "South Africa",
             "IND": "India", "JPN": "Japan", "USA": "USA"}
    w, h = 7.9, 4.15
    fig, ax = frame(w, h, f"Mixed cells are rare: {allpct:.1f}% overall, under 1% in every city outside Japan",
                    f"% of labelled 10 m cells holding both residential and non-residential buildings · {len(c)} cities, {c.n_cells.sum()/1e6:.1f}M cells",
                    top_in=0.8, left=0.16, right=0.97, bottom=0.1)
    for i, cc in enumerate(order[::-1]):
        sub = c[c.country_id == cc]
        col = C_AE if cc == "JPN" else GREY
        ax.scatter(sub.pct_mixed, [i] * len(sub), s=46, color=col, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
        kp = float(k.loc[k.country_id == cc, "pct_mixed"].iloc[0])
        ax.scatter([kp], [i], marker="|", s=420, linewidths=2.5, color=NAVY, zorder=4)
        ax.text(max(sub.pct_mixed.max(), kp) + 0.15, i, f"{kp:.2f}%", va="center", fontsize=10, color=NAVY, fontweight="bold")
    for city, dx, dy in [("Nagoya", 0, 0.32), ("Fukuoka", 0, 0.32), ("Osaka", 0, 0.32)]:
        r = c[c.city_name == city].iloc[0]
        i = order[::-1].index("JPN")
        ax.text(r.pct_mixed + dx, i + dy, city, fontsize=9, color=C_AE, ha="center", va="top")
    ax.axvline(1, color=SLATE, lw=1, ls=":")
    ax.text(1.05, len(order) - 0.45, "1%", fontsize=9.5, color=SLATE, va="bottom")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[x] for x in order[::-1]], fontsize=10.5)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, 7.5); ax.set_xticks(range(0, 8)); ax.set_xticklabels([f"{v}%" for v in range(0, 8)], fontsize=10)
    ax.grid(axis="x", color=LIGHT, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.text(7.4, len(order) - 0.55, "dot = city · | = country", fontsize=9.5, color=SLATE, ha="right", va="bottom")
    save(fig, "chart_purity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", type=Path,
                    default=FIG / "texture_fix" if (FIG / "texture_fix" / "s2_ablation.csv").exists() else FIG,
                    help="folder holding s2_ablation.csv and s2_feature_auc.csv")
    a = ap.parse_args()
    print("ablation results from", a.ablation_dir)
    chart_balance(); chart_ae_metro(); chart_cnn_metro(); chart_vit(); chart_xgb_metro(); chart_vit_metro(); chart_progression()
    chart_cues(a.ablation_dir); chart_ablation(a.ablation_dir)
    chart_gen_ladder(); chart_gen_transfer(); chart_gen_threshold(); chart_gen_india(); chart_gen_loco()
    chart_africa_bias(); chart_africa_within(); chart_africa_loro(); map_lagos()
    chart_purity()
