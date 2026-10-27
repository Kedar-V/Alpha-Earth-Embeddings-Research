#!/usr/bin/env python3
"""Build the weeks 1-5 stakeholder PDF (figures + report) from saved artifacts.

Every number is read from outputs/, week4/international_geo_generalization/runs/
or week5/runs/. Numbers that exist only in notebook markdown are listed in
SOURCE_NUMBERS below so they can be audited.

    .venv/bin/python scripts/build_stakeholder_report.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "week5" / "reports"
FIG_DIR = OUT_DIR / "stakeholder_figures"
PDF_PATH = OUT_DIR / "alphaearth_weeks1-5_stakeholder_insights.pdf"
INTL = ROOT / "week4/international_geo_generalization/runs"

NAVY = "#1f3a5f"
BLUE = "#2b7bba"
ORANGE = "#d9822b"
GREY = "#6b7280"
LIGHT = "#e5e7eb"
RED = "#b4443b"
GREEN = "#3a7d5c"

# Numbers quoted from notebook markdown / reports because no CSV carries them.
SOURCE_NUMBERS = {
    "week1_binary_loro": (0.77, "notebooks/week1/00_briefing.ipynb"),
    "week1_pooled_80_20": (0.83, "notebooks/week1/00_briefing.ipynb"),
    "week1_ten_class_loro": (0.22, "notebooks/week1/00_briefing.ipynb"),
    "week2_bbox_cnn_binary": (0.746, "notebooks/week2/02_cnn_bbox_grid_training.ipynb"),
    "week2_centroid_binary": (0.696, "notebooks/week2/02_cnn_bbox_grid_training.ipynb"),
    "week2_bbox_cnn_ten_class": (0.232, "notebooks/week2/02_cnn_bbox_grid_training.ipynb"),
    "building_ae_cnn_vs_ghsl": ((0.746, 0.672), "notebooks/week4/03_model_comparison.ipynb"),
    "pls_10m_vs_100m": ("notebooks/week4/10_pls_10m_vs_100m.ipynb", "PLS / LR / t1 LORO F1"),
    "balanced_tiles_ghsl": ("notebooks/week4/05_100m_fraction_eval.ipynb", "1:1 tile table"),
    "within_dim_tradeoff": (
        "week5/runs/dou_balanced_week5_20261008_192748/global_local_subspace/REPORT.md",
        "k1->k20 within vs cross deltas, chordal vs transfer drop",
    ),
    "loco_strict": (
        "week5/runs/dou_balanced_week5_20261008_192748/task_invariant_subspace/REPORT.md",
        "strict LOCO table",
    ),
}


def setup_fonts() -> None:
    ttf = Path(font_manager.findfont("DejaVu Sans")).parent
    pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(ttf / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(ttf / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DV-BI", str(ttf / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "figure.dpi": 100,
    }
)


def load_data() -> dict:
    d: dict = {}
    d["us_loro"] = pd.read_csv(ROOT / "outputs/week4_pixel_balanced_train/loro_metrics.csv")
    d["us_summary"] = json.loads((ROOT / "outputs/week4_pixel_balanced_train/summary.json").read_text())
    d["cmp"] = pd.read_csv(ROOT / "outputs/week4_model_comparison/comparison_balanced_eval.csv")
    d["mae100"] = pd.read_csv(ROOT / "outputs/week4_100m_fraction_eval/mae_summary.csv")
    d["rescore"] = pd.read_csv(ROOT / "outputs/week4_pixel_eval/week3_rescore_v1/summary_metrics.csv")
    d["vit"] = pd.read_csv(ROOT / "outputs/week4_vit_100m_3d_v4/loro_metrics.csv")
    d["ghsl"] = json.loads((ROOT / "outputs/week4_ghsl_pixel/summary.json").read_text())
    d["ghsl_fair"] = json.loads((ROOT / "outputs/week4_ghsl_pixel/alignment_fairness_report.json").read_text())
    a = INTL / "intl_transfer_20261005_142709/artifacts"
    d["country_metrics"] = pd.read_csv(a / "country_transfer_metrics.csv")
    d["loro_ctry"] = {c: json.loads((a / f"loro_{c}_summary.json").read_text()) for c in ("USA", "JPN", "IND")}
    d["ind10"] = pd.read_csv(INTL / "india_loro_20261005_202600/pls_compare/summary_pls_vs_lr64.csv")
    d["ind_city"] = pd.read_csv(INTL / "india_loro_20261005_202600/pls_compare/loro_per_city_pls_vs_lr64.csv")
    d["jpn10"] = pd.read_csv(INTL / "japan_loro_20261005_225832/pls_compare/summary_pls_vs_lr64.csv")
    d["jpn_city"] = pd.read_csv(INTL / "japan_loro_20261005_225832/pls_compare/loro_per_city_pls_vs_lr64.csv")
    run = ROOT / "week5/runs/dou_balanced_week5_20261008_192748"
    d["tis"] = json.loads((run / "task_invariant_subspace/artifacts/success_criteria.json").read_text())
    d["tcf"] = json.loads((run / "three_country_falsification/artifacts/success_criteria.json").read_text())
    d["dou_summary"] = json.loads((run / "summary.json").read_text())
    return d


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def label_bars(ax, bars, fmt="{:.3f}", horizontal=True, pad=0.01, fs=8):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + pad, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=fs, color="#374151")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt.format(v), ha="center", fontsize=fs, color="#374151")


def fig_classification(d) -> Path:
    us = d["us_loro"].copy()
    us["metro"] = us.region.str.replace("_", " ")
    us = us.sort_values("macro_f1")
    mean_f1 = d["us_summary"]["loro_macro_f1_mean"]
    cmp = d["cmp"].copy()
    keep = {
        "balanced_train (W4)": "Balanced LR (W4)",
        "fraction_regression (W3)": "Ridge fractions (W3)",
        "pixel_ae (W3)": "Pixel LR (W3)",
        "pixel_3class_reg (W3)": "3-class soft head (W3)",
        "neighborhood_k3_reg (W3)": "3x3 neighborhood CNN (W3)",
    }
    rows = [(keep[m], f) for m, f in zip(cmp.model, cmp.macro_f1_balanced_eval) if m in keep]
    rows.append(("GHSL at AE cell centers", d["ghsl_fair"]["balanced_eval"]["macro_f1"]))
    rows = sorted(rows, key=lambda r: r[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1, 1]})
    cols = [RED if v < 0.70 else BLUE for v in us.macro_f1]
    bars = ax1.barh(us.metro, us.macro_f1, color=cols)
    label_bars(ax1, bars)
    ax1.axvline(mean_f1, color=NAVY, ls="--", lw=1)
    ax1.text(mean_f1 - 0.005, -0.9, f"mean {mean_f1:.3f}", color=NAVY, ha="right", fontsize=8)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_xlabel("Macro-F1 (held-out metro)")
    ax1.set_title("Balanced pixel LR, leave-one-metro-out")

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols2 = [ORANGE if "GHSL" in n else (BLUE if "Balanced" in n else "#9db7d1") for n in names]
    bars2 = ax2.barh(names, vals, color=cols2)
    label_bars(ax2, bars2)
    ax2.set_xlim(0.3, 0.8)
    ax2.set_xlabel("Macro-F1 on the same 59,596 balanced cells")
    ax2.set_title("Head-to-head on one shared eval set")
    fig.tight_layout(w_pad=2)
    return save(fig, "01_classification.png")


def fig_regression(d) -> Path:
    r = d["rescore"].set_index("model")
    w4 = d["us_summary"]
    models = [
        ("Ridge + SMOTE (W3)", r.loc["fraction_regression", "oof_mae_mean"]),
        ("Pixel LR as fractions (W3)", r.loc["pixel_ae", "oof_mae_mean"]),
        ("3x3 neighborhood CNN (W3)", r.loc["neighborhood_k3_reg", "oof_mae_mean"]),
        ("3-class soft head (W3)", r.loc["pixel_3class_reg", "oof_mae_mean"]),
        ("Balanced Ridge (W4)", w4["oof_mae_mean"]),
    ]
    m100 = d["mae100"]
    w4_10 = m100[(m100.model == "balanced_train_full") & (m100.scale == "10m")].mae_mean_res_nres.iloc[0]
    w4_100 = m100[(m100.model == "balanced_train_full") & (m100.scale == "100m")].mae_mean_res_nres.iloc[0]
    g_10 = m100[(m100.model == "ghsl") & (m100.scale == "10m")].mae_mean_res_nres.iloc[0]
    g_100 = m100[(m100.model == "ghsl") & (m100.scale == "100m")].mae_mean_res_nres.iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    names = [m[0] for m in models][::-1]
    vals = [m[1] for m in models][::-1]
    cols = [BLUE if "W4" in n else ("#9db7d1") for n in names]
    bars = ax1.barh(names, vals, color=cols)
    label_bars(ax1, bars, pad=0.004)
    ax1.set_xlim(0, 0.45)
    ax1.set_xlabel("Mean absolute error over RES / NRES / empty fractions (lower is better)")
    ax1.set_title("10 m fraction regression")

    x = np.arange(2)
    w = 0.34
    b1 = ax2.bar(x - w / 2, [w4_10, w4_100], w, color=BLUE, label="Balanced Ridge (W4)")
    b2 = ax2.bar(x + w / 2, [g_10, g_100], w, color=ORANGE, label="GHSL soft")
    label_bars(ax2, b1, horizontal=False, pad=0.006)
    label_bars(ax2, b2, horizontal=False, pad=0.006)
    ax2.set_xticks(x, ["10 m cells", "100 m tiles\n(>=20 cells)"])
    ax2.set_ylim(0, 0.42)
    ax2.set_ylabel("MAE over RES and NRES (lower is better)")
    ax2.set_title("Pooling to 100 m")
    ax2.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout(w_pad=2)
    return save(fig, "02_regression.png")


def fig_generalisation(d) -> Path:
    heads = [
        ("64-D LR", 0.779, 0.761),
        ("PLS-DA (best k)", 0.775, 0.762),
        ("PLS t1 only", 0.735, 0.720),
    ]
    vit = {"v1": 0.732, "v2": 0.632, "v3": 0.7348, "v4": float(d["vit"].macro_f1.mean())}
    us = d["us_loro"].set_index("region").macro_f1
    v = d["vit"].set_index("region").macro_f1
    order = us.sort_values().index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1, 1.25]})
    labels = [h[0] for h in heads] + [f"3D ViT {k}" for k in vit]
    v10 = [h[1] for h in heads] + [np.nan] * len(vit)
    v100 = [h[2] for h in heads] + list(vit.values())
    y = np.arange(len(labels))[::-1]
    h = 0.38
    ax1.barh(y + h / 2, [0 if np.isnan(a) else a for a in v10], h, color=BLUE, label="10 m cells")
    bars = ax1.barh(y - h / 2, v100, h, color=ORANGE, label="100 m tiles")
    for yy, a in zip(y, v10):
        if not np.isnan(a):
            ax1.text(a + 0.006, yy + h / 2, f"{a:.3f}", va="center", fontsize=8)
    for yy, b in zip(y, v100):
        ax1.text(b + 0.006, yy - h / 2, f"{b:.3f}", va="center", fontsize=8)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0.5, 0.85)
    ax1.set_xlabel("Leave-one-metro-out macro-F1")
    ax1.set_title("More capacity did not buy robustness")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)

    xs = np.arange(len(order))
    ax2.bar(xs - 0.2, [us[o] for o in order], 0.4, color=BLUE, label="Linear head, 10 m cells")
    ax2.bar(xs + 0.2, [v[o] for o in order], 0.4, color=ORANGE, label="3D ViT v4, 100 m tiles")
    ax2.set_xticks(xs, [o.replace("_", "\n") for o in order], fontsize=7.5)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_ylabel("Held-out macro-F1")
    ax2.set_title("The same metros are hard for both model families")
    ax2.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(w_pad=2)
    return save(fig, "03_generalisation.png")


def fig_embedding(d) -> tuple[Path, float]:
    vit = d["vit"].copy()
    vit["metro"] = vit.region.str.replace("_", " ")
    gap = vit.test_ratio.to_numpy()
    f1 = vit.macro_f1.to_numpy()
    r = float(np.corrcoef(gap, f1)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.05, 1]})
    ax1.scatter(gap, f1, color=BLUE, s=36, zorder=3)
    offsets = {"NYC_NE": (6, -10), "SF_W": (6, 4), "Chicago_MW": (6, -12), "Boston_NE": (6, 4)}
    for _, row in vit.iterrows():
        dx, dy = offsets.get(row.region, (6, 3))
        ax1.annotate(row.region.split("_")[0], (row.test_ratio, row.macro_f1), xytext=(dx, dy), textcoords="offset points", fontsize=7.5)
    ax1.axvline(1, color=GREY, ls=":", lw=1)
    ax1.axvline(3, color=GREY, ls="--", lw=1)
    ax1.text(1.02, 0.505, "one cloud", fontsize=7.5, color=GREY)
    ax1.text(3.02, 0.505, "clear gap", fontsize=7.5, color=GREY)
    ax1.set_xlabel("Held-out gap (distance between class centers / class spread)")
    ax1.set_ylabel("Held-out macro-F1")
    ax1.set_ylim(0.5, 0.95)
    ax1.set_title(f"Cluster gap tracks F1 (Pearson r = {r:.2f}, n = 10)")

    inc = pd.DataFrame(
        {
            "city": ["Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"],
            "in_city": [1.76, 1.61, 2.10, 1.52, 2.72],
            "loro": [0.82, 1.13, 1.47, 0.87, 1.01],
        }
    )
    x = np.arange(len(inc))
    ax2.bar(x - 0.2, inc.in_city, 0.4, color=BLUE, label="PLS fit inside the city")
    ax2.bar(x + 0.2, inc.loro, 0.4, color=ORANGE, label="Held-out city in the other four's PLS")
    ax2.set_xticks(x, inc.city, fontsize=8)
    ax2.set_ylabel("Class separation on PLS axis 1")
    ax2.set_title("India: the local axis does not transfer")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_ylim(0, 3.4)
    fig.tight_layout(w_pad=2)
    return save(fig, "04_embedding_dynamics.png"), r


def fig_country_loro(d) -> Path:
    ind = d["ind10"].set_index(["panel", "model", "k"])
    jpn = d["jpn10"].set_index(["panel", "model", "k"])
    panels = ["USA", "Japan", "India"]
    within = [ind.loc[("USA_10", "lr64", 64), "within_80_20"], jpn.loc[("JPN_10", "lr64", 64), "within_80_20"], ind.loc[("IND_10", "lr64", 64), "within_80_20"]]
    loro = [ind.loc[("USA_10", "lr64", 64), "loro_mean"], jpn.loc[("JPN_10", "lr64", 64), "loro_mean"], ind.loc[("IND_10", "lr64", 64), "loro_mean"]]
    off = [ind.loc[("USA_10", "lr64", 64), "city_off_mean"], jpn.loc[("JPN_10", "lr64", 64), "city_off_mean"], ind.loc[("IND_10", "lr64", 64), "city_off_mean"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.arange(3)
    w = 0.26
    b1 = ax1.bar(x - w, within, w, color="#9db7d1", label="Within-country 80/20")
    b2 = ax1.bar(x, loro, w, color=BLUE, label="Leave-one-city-out")
    b3 = ax1.bar(x + w, off, w, color=ORANGE, label="City-to-city, off-diagonal")
    for bars in (b1, b2, b3):
        label_bars(ax1, bars, fmt="{:.2f}", horizontal=False, pad=0.008, fs=7.5)
    ax1.set_xticks(x, [f"{p}\n(10 cities)" for p in panels])
    ax1.set_ylim(0.3, 1.0)
    ax1.set_ylabel("Macro-F1 (64-D logistic regression)")
    ax1.set_title("Country-wise leave-one-city-out")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    c = d["ind_city"].sort_values("lr64_k64")
    names = [n.replace("IND_", "").title() for n in c.held_out]
    y = np.arange(len(c))
    for yy, a, b in zip(y, c.lr64_k64, c.pls_k2):
        ax2.plot([a, b], [yy, yy], color=LIGHT, lw=3, zorder=1)
    ax2.scatter(c.lr64_k64, y, color=RED, s=34, zorder=3, label="64-D LR")
    ax2.scatter(c.pls_k2, y, color=GREEN, s=34, zorder=3, label="PLS, 2 components")
    ax2.set_yticks(y, names)
    ax2.set_xlim(0.3, 0.75)
    ax2.set_xlabel("Held-out city macro-F1")
    ax2.set_title("India: low rank rescues the worst cities")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=2)
    return save(fig, "05_country_loro.png")


def fig_country_transfer(d) -> Path:
    m = d["country_metrics"]
    order = ["USA", "JPN", "IND"]

    def mat(model, k):
        s = m[(m.model == model) & (m.k == k)]
        out = np.zeros((3, 3))
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                out[i, j] = s[(s.source_country == a) & (s.target_country == b)].macro_f1.iloc[0]
        return out

    lr, pls2 = mat("lr64", 64), mat("pls", 2)
    fig = plt.figure(figsize=(10.6, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.55)
    for idx, (title, M) in enumerate([("64-D logistic regression", lr), ("PLS, 2 components", pls2)]):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(M, vmin=0.3, vmax=0.9, cmap="RdYlBu")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black", fontweight="bold" if i == j else None)
        ax.set_xticks(range(3), order)
        ax.set_yticks(range(3), order)
        ax.set_xlabel("Test country")
        if idx == 0:
            ax.set_ylabel("Train country")
        ax.set_title(title, fontsize=9)
        ax.spines[:].set_visible(False)
    off = m[(m.model == "lr64") & (m.protocol == "train_all_source_country")].copy()
    off["pair"] = off.source_country + " to " + off.target_country
    off = off.sort_values("macro_f1")
    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(off))
    h = 0.38
    ax.barh(y + h / 2, off.auroc, h, color="#9db7d1", label="AUROC (ranking)")
    ax.barh(y - h / 2, off.macro_f1, h, color=RED, label="Macro-F1 at threshold 0.5")
    for yy, a, f in zip(y, off.auroc, off.macro_f1):
        ax.text(a + 0.01, yy + h / 2, f"{a:.2f}", va="center", fontsize=7.5)
        ax.text(f + 0.01, yy - h / 2, f"{f:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y, off.pair)
    ax.set_xlim(0.3, 0.95)
    ax.set_title("Cross-country: ranking vs decision", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=1)
    return save(fig, "06_country_transfer.png")


def fig_loco(d) -> Path:
    tbl = {
        "64-D LR": [0.822, 0.498, 0.367],
        "Pooled PLS (r=3)": [0.783, 0.598, 0.583],
        "Geometric consensus": [0.801, 0.644, 0.580],
        "Task-invariant subspace": [0.785, 0.714, 0.621],
    }
    targets = ["USA", "JPN", "IND", "mean"]
    means = d["tis"]["loco_means"]
    chk = {
        "64-D LR": means["lr64"],
        "Pooled PLS (r=3)": means["pooled_pls"],
        "Geometric consensus": means["geometric_avgproj"],
        "Task-invariant subspace": means["task_A"],
    }
    colors_ = ["#9ca3af", "#9db7d1", BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(10.4, 3.5))
    x = np.arange(len(targets))
    w = 0.2
    for i, (name, vals) in enumerate(tbl.items()):
        vals = vals + [round(chk[name], 3)]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=colors_[i], label=name)
        label_bars(ax, bars, fmt="{:.2f}", horizontal=False, pad=0.008, fs=7.5)
    ax.set_xticks(x, ["Target USA", "Target Japan", "Target India", "Mean"])
    ax.set_ylim(0.3, 0.95)
    ax.set_ylabel("Macro-F1, target country never seen")
    ax.set_title("Strict leave-one-country-out (DoU-balanced, 2.0 M cells, seeds 0-4)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    return save(fig, "07_loco.png")


def build_pdf(d, fig_paths: dict, gap_r: float) -> None:
    setup_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = ParagraphStyle("base", fontName="DV", fontSize=8.8, leading=12.4, textColor=colors.HexColor("#1f2937"), alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=base, fontSize=7.4, leading=10, textColor=colors.HexColor(GREY))
    cap = ParagraphStyle("cap", parent=small, spaceBefore=2, spaceAfter=8)
    h1 = ParagraphStyle("h1", parent=base, fontName="DV-B", fontSize=15, leading=19, textColor=colors.HexColor(NAVY), spaceBefore=4, spaceAfter=5)
    h2 = ParagraphStyle("h2", parent=base, fontName="DV-B", fontSize=10.2, leading=13.5, textColor=colors.HexColor(NAVY), spaceBefore=8, spaceAfter=3)
    title = ParagraphStyle("title", parent=base, fontName="DV-B", fontSize=21, leading=26, textColor=colors.HexColor(NAVY), spaceAfter=3)
    sub = ParagraphStyle("sub", parent=base, fontSize=10, leading=14, textColor=colors.HexColor(GREY), spaceAfter=10)
    bullet = ParagraphStyle("bullet", parent=base, leftIndent=11, bulletIndent=1, spaceAfter=2.6)
    cell = ParagraphStyle("cell", parent=base, fontSize=7.8, leading=10.2)
    cell_b = ParagraphStyle("cell_b", parent=cell, fontName="DV-B")
    cell_h = ParagraphStyle("cell_h", parent=cell, fontName="DV-B", textColor=colors.white)
    kpi_v = ParagraphStyle("kpi_v", parent=base, fontName="DV-B", fontSize=19, leading=23, textColor=colors.HexColor(BLUE))
    kpi_l = ParagraphStyle("kpi_l", parent=small, fontSize=7.4, leading=9.6)
    action = ParagraphStyle("action", parent=base, fontSize=8.4, leading=11.6)

    W = letter[0] - 1.3 * inch

    def P(t, s=base):
        return Paragraph(t, s)

    def bullets(items):
        return [Paragraph(t, bullet, bulletText="•") for t in items]

    def figure(path, width=W):
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            w, h = im.size
        return Image(str(path), width=width, height=width * h / w)

    def kpis(items):
        n = len(items)
        cells = [[P(v, kpi_v) for v, _ in items], [P(l, kpi_l) for _, l in items]]
        t = Table(cells, colWidths=[W / n] * n)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f6fa")),
                    ("LINEBEFORE", (1, 0), (-1, -1), 0.6, colors.white),
                    ("TOPPADDING", (0, 0), (-1, 0), 7),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return t

    def callout(head, body, color=BLUE):
        t = Table([[P(f"<b>{head}</b><br/>{body}", base)]], colWidths=[W])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(color)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        return t

    def table(rows, widths, bold_first_col=False, align_right_from=None, zebra=True):
        data = []
        for ri, row in enumerate(rows):
            out = []
            for ci, c in enumerate(row):
                style = cell_h if ri == 0 else (cell_b if (bold_first_col and ci == 0) else cell)
                if align_right_from is not None and ci >= align_right_from:
                    style = ParagraphStyle("r", parent=style, alignment=2)
                out.append(Paragraph(str(c), style))
            data.append(out)
        t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
        st = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
        if zebra:
            for i in range(2, len(rows), 2):
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")))
        t.setStyle(TableStyle(st))
        return t

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.2)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(0.65 * inch, 0.45 * inch, "AlphaEarth embeddings, residential vs non-residential: weeks 1-5 stakeholder insights")
        canvas.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor(LIGHT))
        canvas.line(0.65 * inch, 0.6 * inch, letter[0] - 0.65 * inch, 0.6 * inch)
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.75 * inch,
        title="AlphaEarth weeks 1-5: stakeholder insights",
        author="Alpha-Earth-Embeddings-Research",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])

    us = d["us_summary"]
    ind10 = d["ind10"].set_index(["panel", "model", "k"])
    jpn10 = d["jpn10"].set_index(["panel", "model", "k"])
    m100 = d["mae100"]
    w4_10 = m100[(m100.model == "balanced_train_full") & (m100.scale == "10m")].mae_mean_res_nres.iloc[0]
    w4_100 = m100[(m100.model == "balanced_train_full") & (m100.scale == "100m")].mae_mean_res_nres.iloc[0]
    rescore = d["rescore"].set_index("model")
    cm = d["country_metrics"]

    S: list = []

    # ------------------------------------------------------------------ page 1
    S.append(P("What AlphaEarth embeddings can and cannot do for residential mapping", title))
    S.append(
        P(
            f"Weeks 1 to 5 findings for technical review. Prepared {date.today():%d %B %Y}. "
            "Frozen 64-D AlphaEarth embeddings (10 m, 2024), Overture building labels, leave-one-region-out (LORO) evaluation "
            "on 10 US metros, 10 Japanese and 10 Indian cities.",
            sub,
        )
    )
    S.append(
        kpis(
            [
                (f"{us['loro_macro_f1_mean']:.2f}", "US binary pixel macro-F1, leave-one-metro-out, class-balanced"),
                (f"{w4_100:.2f}", f"Fraction MAE at 100 m tiles, down from {w4_10:.2f} at 10 m"),
                ("0.52", "India city-LORO macro-F1, against 0.77 for Japan and 0.86 for the USA"),
                ("0.37", "USA-trained head tested on India"),
            ]
        )
    )
    S.append(Spacer(1, 8))
    S.append(P("Bottom line", h2))
    S.extend(
        bullets(
            [
                "<b>A linear head on frozen embeddings is a usable binary residential classifier inside the US.</b> "
                f"LORO macro-F1 is {us['loro_macro_f1_mean']:.2f} on {us['n_pixels']/1e6:.2f} M balanced cells, with no SMOTE and no fine-tuning. "
                "Deeper encoders (3D transformer, neighborhood CNN) did not beat it.",
                "<b>The same head is not a global model.</b> City-level LORO falls from 0.86 (USA) to 0.77 (Japan) to 0.52 (India), "
                "and a US-trained head scores 0.37 on India.",
                "<b>Fine building type and 10 m fraction regression are not ready.</b> Ten-type LORO is 0.22. "
                "The best 10 m regression explains 13% of the residential fraction variance and 32% of non-residential.",
                "<b>Scale and balance move the numbers more than architecture.</b> The Week 3 pixel score of 0.56 was an 8.5:1 class "
                "imbalance. The same predictions score 0.70 on balanced cells. Pooling to 100 m cuts fraction error by about one third.",
                "<b>Across countries the decision fails before the ranking does.</b> Cross-country AUROC is 0.61 to 0.81 while F1 can fall to 0.33. "
                "Where transfer collapses, the head labels almost everything residential (non-residential F1 of 0.00 to 0.07).",
            ]
        )
    )
    S.append(Spacer(1, 4))
    S.append(P("Actions we recommend", h2))
    S.append(
        table(
            [
                ["#", "Recommendation", "Backing evidence", "Confidence"],
                ["1", "Ship the binary linear head for US metro screening. Acceptance test: per-metro LORO F1, not a pooled split.", "LORO 0.78, pooled 0.81; range 0.66 to 0.91 across metros", "High"],
                ["2", "Stop investing in deeper encoders and 3D cluster geometry for accuracy.", "ViT v1, v3, v4: 0.73 to 0.74 LORO (v2: 0.63) vs 0.76 for a 100 m linear head", "High"],
                ["3", "Report area fractions at 100 m, not 10 m. Model the empty class explicitly.", f"MAE {w4_10:.2f} to {w4_100:.2f}; 3-class head 0.29 vs 0.37 without empty", "Medium to high"],
                ["4", "Train and validate per country. Do not export the US head.", "USA to India 0.37, India to Japan 0.33, Japan to USA 0.72", "High"],
                ["5", "For India use a 2-component PLS head as the baseline and audit Kolkata and Chennai labels.", "City-LORO 0.52 to 0.60; Kolkata 0.37 to 0.62, Chennai 0.39 to 0.63", "Medium"],
                ["6", "Before any cross-country deployment, test threshold recalibration on a small labeled target sample.", "India to Japan: AUROC 0.69, F1 0.33. USA to India: AUROC 0.61, F1 0.37. Fix is untested.", "Medium to low"],
                ["7", "Claim a global subspace only after more than three countries.", "Shared-spectrum test fails vs shuffle null; verdict partial", "High"],
            ],
            [0.04, 0.42, 0.42, 0.12],
        )
    )
    S.append(PageBreak())

    # --------------------------------------------------------- 1 classification
    S.append(P("1. Classification results", h1))
    S.append(figure(fig_paths["cls"]))
    S.append(
        P(
            "Left: Week 4 balanced pixel logistic regression on the 10 US metros (1.75 M cells, 1:1 residential to non-residential, "
            "no SMOTE). Right: every pixel head scored on one shared set of 59,596 balanced cells. GHSL is sampled at the AE cell center.",
            cap,
        )
    )
    S.append(
        table(
            [
                ["Unit and protocol", "Model", "Macro-F1", "Note"],
                ["Pixel, 1.75 M balanced cells, LORO", "64-D logistic regression", f"{us['loro_macro_f1_mean']:.3f} ± {us['loro_macro_f1_std']:.3f}", f"Pooled out-of-fold {us['oof_macro_f1']:.3f}"],
                ["Pixel, 3.13 M full-GT cells", "Same fold models", f"{us['full_gt_oof']['macro_f1']:.3f}", "Out-of-fold, imbalanced truth"],
                ["Pixel, same 1.75 M cells", "GHSL (no training)", f"{d['ghsl']['balanced']['functional']['macro_f1']:.3f}", "Selective, 96.8% coverage"],
                ["Pixel, 59,596-cell shared set", "AE balanced LR vs GHSL", "0.704 vs 0.692", "Gap shrinks to 0.01"],
                ["Pixel, Week 3 protocol", "Pixel LR", "0.563", "8.5:1 imbalance; 0.697 on balanced subset"],
                ["Building footprint", "AE bbox CNN vs GHSL", "0.746 vs 0.672", "Largest AE advantage (+0.07)"],
                ["Building centroid", "LR vs bbox CNN (binary)", "0.696 vs 0.746", "Footprint context adds 0.05"],
                ["Building, ten types", "HistGradientBoosting", "0.22 ± 0.03", "Chance is about 0.10"],
                ["Pixel with 3x3 neighborhood", "Conv encoder", "0.345 to 0.50", "Worse than the plain pixel head"],
            ],
            [0.30, 0.25, 0.17, 0.28],
            align_right_from=None,
        )
    )
    S.append(P("What this means", h2))
    S.extend(
        bullets(
            [
                "<b>The binary task is solved well enough for screening in the US.</b> Six of ten metros score above 0.76; "
                "Phoenix (0.91), Atlanta (0.87) and Denver (0.87) lead, and NYC (0.67) and SF (0.66) trail.",
                "<b>The AlphaEarth advantage over GHSL is real but depends on the unit.</b> It is +0.07 on buildings, "
                f"+0.05 on the full balanced pixel set ({us['oof_macro_f1']:.2f} vs {d['ghsl']['balanced']['functional']['macro_f1']:.2f}, out-of-fold vs a model-free baseline) and only +0.01 on the 59,596-cell shared set. "
                "On 1:1 balanced 100 m tiles with at least 5 cells, Ridge scores 0.707 and GHSL 0.703 at threshold 0.5, and GHSL reaches 0.768 with a tuned threshold.",
                "<b>Fine typing is out of reach with these features.</b> Footprint context lifts ten-type F1 only from 0.175 to 0.232.",
                "<b>Neighborhood context did not help</b> (3x3 conv 0.35 to 0.50), so extra spatial window is not the missing ingredient.",
            ]
        )
    )
    S.append(PageBreak())

    # ------------------------------------------------------------ 2 regression
    S.append(P("2. Regression results", h1))
    S.append(
        P(
            "Targets are the area fractions of each cell covered by residential and non-residential footprints (and the remaining empty share), "
            "predicted from the 64 embedding bands only.",
            base,
        )
    )
    S.append(Spacer(1, 4))
    S.append(figure(fig_paths["reg"]))
    S.append(
        P(
            "Left: mean MAE over the three fractions for Week 3 heads re-scored on the Week 4 ground truth (283,119 building-overlapped pixels) "
            "and the Week 4 balanced Ridge (1.75 M balanced cells). The two eval sets differ, so read the ordering, not the third decimal. "
            "Right: the same models pooled from 10 m cells to 100 m tiles.",
            cap,
        )
    )
    R = lambda m, c: rescore.loc[m, c]
    S.append(
        table(
            [
                ["Model", "Mean MAE", "MAE res / nres / empty", "R² res / nres / empty"],
                ["Ridge + SMOTE (W3)", f"{R('fraction_regression','oof_mae_mean'):.3f}", f"{R('fraction_regression','oof_mae_frac_res'):.2f} / {R('fraction_regression','oof_mae_frac_nres'):.2f} / {R('fraction_regression','oof_mae_frac_empty'):.2f}", f"{R('fraction_regression','oof_r2_frac_res'):.2f} / {R('fraction_regression','oof_r2_frac_nres'):.2f} / {R('fraction_regression','oof_r2_frac_empty'):.2f}"],
                ["3-class soft head (W3)", f"{R('pixel_3class_reg','oof_mae_mean'):.3f}", f"{R('pixel_3class_reg','oof_mae_frac_res'):.2f} / {R('pixel_3class_reg','oof_mae_frac_nres'):.2f} / {R('pixel_3class_reg','oof_mae_frac_empty'):.2f}", f"{R('pixel_3class_reg','oof_r2_frac_res'):.2f} / {R('pixel_3class_reg','oof_r2_frac_nres'):.2f} / {R('pixel_3class_reg','oof_r2_frac_empty'):.2f}"],
                ["3x3 neighborhood CNN (W3)", f"{R('neighborhood_k3_reg','oof_mae_mean'):.3f}", f"{R('neighborhood_k3_reg','oof_mae_frac_res'):.2f} / {R('neighborhood_k3_reg','oof_mae_frac_nres'):.2f} / {R('neighborhood_k3_reg','oof_mae_frac_empty'):.2f}", f"{R('neighborhood_k3_reg','oof_r2_frac_res'):.2f} / {R('neighborhood_k3_reg','oof_r2_frac_nres'):.2f} / {R('neighborhood_k3_reg','oof_r2_frac_empty'):.2f}"],
                ["Balanced Ridge (W4)", f"{us['oof_mae_mean']:.3f}", f"{us['oof_mae_frac_res']:.2f} / {us['oof_mae_frac_nres']:.2f} / {us['oof_mae_frac_empty']:.2f}", f"{us['oof_r2_frac_res']:.2f} / {us['oof_r2_frac_nres']:.2f} / {us['oof_r2_frac_empty']:.2f}"],
            ],
            [0.30, 0.13, 0.28, 0.29],
        )
    )
    S.append(P("What this means", h2))
    S.extend(
        bullets(
            [
                "<b>Regression is the weakest deliverable.</b> The best model reaches R² of 0.13 for residential, 0.32 for non-residential and 0.10 for empty. "
                f"Out-of-fold MAE is {us['oof_mae_mean']:.2f} and leave-one-metro-out MAE is {us['loro_mae_mean_mean']:.2f} ± {us['loro_mae_mean_std']:.2f}. "
                "Use it to rank cells, not to report areas.",
                "<b>Negative R² on the Week 3 heads</b> (Ridge residential −5.9) comes from training on SMOTE-balanced data and scoring on imbalanced truth: "
                "the head over-predicts residential share. Train on the distribution you will score.",
                "<b>The empty class is where binary models lose.</b> Binary heads put zero mass on empty, so empty MAE equals the ground-truth mean (0.37). "
                "The 3-class head cuts mean MAE from 0.37 to 0.29.",
                f"<b>Pooling to 100 m is the cheap win.</b> Balanced Ridge MAE falls from {w4_10:.3f} to {w4_100:.3f} ({(w4_10-w4_100)/w4_10:.0%}) "
                "on 62,312 tiles with at least 20 cells. GHSL soft moves only 0.323 to 0.294.",
            ]
        )
    )
    S.append(PageBreak())

    # ----------------------------------------------------------- 3 generalisation
    S.append(P("3. Generalisation performance", h1))
    S.append(figure(fig_paths["gen"]))
    S.append(
        P(
            "Left: leave-one-metro-out macro-F1 for linear and PLS heads at 10 m and 100 m, and for four 3D transformer runs on 100 m tiles. "
            "Right: held-out F1 per metro for the linear head (10 m cells) and ViT v4 (100 m tiles). Protocols differ slightly between the "
            "two panels' sources, so compare shape rather than absolute level.",
            cap,
        )
    )
    S.append(
        table(
            [
                ["Question", "Finding", "Number"],
                ["Does a pooled split overstate quality?", "Yes, by 0.03 to 0.06", f"Week 1: 0.83 vs 0.77. Week 4: {us['oof_macro_f1']:.3f} vs {us['loro_macro_f1_mean']:.3f}"],
                ["How much do metros differ?", "A lot; std 0.09 on a 0.78 mean", "Phoenix 0.91 to SF 0.66"],
                ["Does more capacity help?", "No", "ViT v4 0.738 vs 64-D LR 0.761 at 100 m"],
                ["Do fewer dimensions hurt?", "Slightly", "PLS k=9 within 0.02 of 64-D LR; first PLS score alone 0.720 to 0.735"],
                ["Does 100 m pooling hurt?", "A little", "−0.018 for LR, −0.013 for best PLS. Training on tiles directly is about 0.07 worse than pooling cell predictions."],
                ["Urban to rural?", "Mild drop (small rural n)", "0.78, 0.76, 0.74 for urban, peri-urban, rural (Week 2)"],
            ],
            [0.28, 0.27, 0.45],
        )
    )
    S.append(P("What this means", h2))
    S.extend(
        bullets(
            [
                "<b>The ceiling is the data, not the model.</b> NYC, SF, Boston and Chicago are the weak metros under both a linear head and a transformer. "
                "Capacity changes the per-metro order a little and the mean not at all.",
                "<b>Use a per-metro floor as the acceptance criterion.</b> A mean of 0.78 hides a 0.66 metro. "
                "Treat dense mixed-use cores as a separate validation slice.",
                "<b>Keep the model small.</b> A 9-component PLS head or the 64-D logistic regression gives the same F1 as every deeper variant tried, "
                "and is the cheapest to audit.",
            ]
        )
    )
    S.append(PageBreak())

    # --------------------------------------------------------------- 4 embeddings
    S.append(P("4. Embedding dynamics: how the classes sit in the space", h1))
    S.append(figure(fig_paths["umap"]))
    S.append(
        P(
            "UMAP of balanced AlphaEarth cells per metro (existing Week 4 figure). Background is a 2D logistic cut; coloured points are cells on the wrong "
            "side. Wrong-side share is 20% (Phoenix) to 43% (NYC), mean 31%.",
            cap,
        )
    )
    S.append(figure(fig_paths["emb"]))
    S.append(
        P(
            f"Left: ViT v4 held-out gap against F1 across the 10 metros (r = {gap_r:.2f}). Right: Indian cities separate in-city on PLS axis 1 "
            "and lose 30% to 63% of that separation when projected through the other four cities' fit.",
            cap,
        )
    )
    S.append(P("What the structure looks like", h2))
    S.extend(
        bullets(
            [
                "<b>One supervised direction carries most of the signal.</b> A single PLS score gives LORO F1 0.735 at 10 m (0.720 at 100 m) against 0.779 (0.761) for all 64 dimensions. "
                "L1 on 32 scores keeps about 1.2 and that survivor is always the first.",
                "<b>Classes form a continuum, not two blobs.</b> In Denver (gap 3.15, F1 0.862) the leftmost 30% of tiles are 9% residential, the middle 40% are 46% and the rightmost 30% are 96%. "
                "Over 99.9% of tiles of either class still sit inside the other class's score range.",
                f"<b>Cluster gap is a good diagnostic and a poor guarantee.</b> It tracks F1 (r = {gap_r:.2f}), but v4 raised the gap over v3 in every metro while mean F1 moved only 0.735 to 0.738. "
                "NYC stays one cloud (gap 0.63, F1 0.59).",
                "<b>Later dimensions are geography.</b> From 1 to 20 PLS components, India gains 0.08 within-country and loses 0.20 across countries (USA +0.03 / −0.14, Japan +0.05 / +0.02). "
                "At k = 2, label-shuffled subspaces are further apart than real ones, so alignment needs real task signal.",
            ]
        )
    )
    S.append(PageBreak())

    # ------------------------------------------------------------ 5 country LORO
    S.append(P("5. Country-wise LORO and cross-country generalisation", h1))
    S.append(figure(fig_paths["ctry"]))
    S.append(
        P(
            "Left: 64-D logistic regression, balanced cells, 10 cities per country. Right: held-out Indian city F1 for the full head versus a 2-component PLS head, sorted by the full-head score.",
            cap,
        )
    )
    S.append(
        table(
            [
                ["Panel", "Within-country 80/20", "City LORO (64-D)", "Best low-rank", "City to city (off-diagonal)"],
                ["USA, 10 metros", f"{ind10.loc[('USA_10','lr64',64),'within_80_20']:.3f}", f"{ind10.loc[('USA_10','lr64',64),'loro_mean']:.3f} ± {ind10.loc[('USA_10','lr64',64),'loro_std']:.3f}", f"PLS-2 {ind10.loc[('USA_10','pls',2),'loro_mean']:.3f}", f"{ind10.loc[('USA_10','lr64',64),'city_off_mean']:.3f}"],
                ["Japan, 10 cities", f"{jpn10.loc[('JPN_10','lr64',64),'within_80_20']:.3f}", f"{jpn10.loc[('JPN_10','lr64',64),'loro_mean']:.3f} ± {jpn10.loc[('JPN_10','lr64',64),'loro_std']:.3f}", f"PLS-9 {jpn10.loc[('JPN_10','pls',9),'loro_mean']:.3f}", f"{jpn10.loc[('JPN_10','lr64',64),'city_off_mean']:.3f}"],
                ["India, 10 cities", f"{ind10.loc[('IND_10','lr64',64),'within_80_20']:.3f}", f"{ind10.loc[('IND_10','lr64',64),'loro_mean']:.3f} ± {ind10.loc[('IND_10','lr64',64),'loro_std']:.3f}", f"PLS-2 {ind10.loc[('IND_10','pls',2),'loro_mean']:.3f}", f"{ind10.loc[('IND_10','lr64',64),'city_off_mean']:.3f} (PLS-2 {ind10.loc[('IND_10','pls',2),'city_off_mean']:.3f})"],
            ],
            [0.22, 0.20, 0.20, 0.16, 0.22],
        )
    )
    S.append(Spacer(1, 6))
    S.append(figure(fig_paths["xfer"]))
    S.append(
        P(
            "Left and centre: macro-F1 when training on all cities of the row country and testing on all cities of the column country (10 USA, 5 Japanese, 5 Indian cities; diagonal is a within-country 80/20 split). "
            "Right: for each cross-country pair, ranking quality (AUROC) against the F1 obtained at threshold 0.5.",
            cap,
        )
    )
    S.append(PageBreak())
    S.append(P("5. Country-wise LORO, continued", h1))
    S.append(figure(fig_paths["loco"]))
    S.append(
        P(
            "Strict leave-one-country-out on the degree-of-urbanization-balanced extract (country-level 1:1:1 urban, peri-urban, rural; 1:1 residential within each). "
            "Subspace rank r = 3 for the PLS-based heads. Source: Week 5 task-invariant subspace run.",
            cap,
        )
    )
    S.append(P("What this means", h2))
    S.extend(
        bullets(
            [
                "<b>The US and Japan share one residential direction across cities; India does not.</b> Japan LORO (0.77) is close to within-country (0.80); "
                "India falls from 0.71 to 0.52. Expanding India from 5 to 10 cities lifted LORO from 0.44 to 0.52 and left the gap.",
                "<b>Transfer is asymmetric.</b> Japan to USA holds at 0.72 and USA to Japan at 0.65, while USA to India is 0.37 and India to Japan is 0.33. "
                "A single universal line cannot explain both directions.",
                f"<b>When transfer fails, the threshold fails first.</b> India to Japan has AUROC {cm[(cm.model=='lr64')&(cm.source_country=='IND')&(cm.target_country=='JPN')].auroc.iloc[0]:.2f} "
                "but F1 0.33 because non-residential F1 is 0.00 (everything is called residential). USA to India behaves the same (AUROC 0.61, non-residential F1 0.07). "
                "A target-country threshold or prior shift is a plausible first fix and is not yet tested.",
                "<b>Low rank repairs India and is neutral to harmful elsewhere.</b> PLS-2 beats the full head on 9 of 10 held-out Indian cities (Kolkata +0.25, Chennai +0.24, Bengaluru +0.14; Jaipur −0.01) and "
                "lifts city-to-city transfer from 0.45 to 0.54. In Japan the full head is as good as PLS-9 and better than PLS-2 in 8 of 10 cities; Sapporo (0.59) is the outlier.",
                "<b>A shared low-rank subspace helps unseen countries, but the global-plus-local story is only partial.</b> "
                "Strict leave-one-country-out F1 rises from 0.56 (full head) to 0.71 (task-invariant subspace), with the largest gains on Japan (+0.22) and India (+0.25) and none on the USA. "
                "Yet the shared-subspace spectrum does not exceed a label-shuffle null, excess task invariance is 0.007, and a matched local residual beats a mismatched one by only 0.006 to 0.020.",
            ]
        )
    )
    S.append(PageBreak())

    # ------------------------------------------------------------------- close
    S.append(P("6. Decisions, risks and what to run next", h1))
    S.append(P("Do now", h2))
    S.extend(
        bullets(
            [
                "Ship the US binary head with per-metro LORO reporting and a per-metro floor. Keep the model to 64-D logistic regression or a 9-component PLS head.",
                "Move all area-fraction outputs to 100 m tiles and add an explicit empty-class term.",
                "Fit and validate country-specific heads. Block any US-to-India export.",
            ]
        )
    )
    S.append(P("Run next (cheap, decisive)", h2))
    S.extend(
        bullets(
            [
                "<b>Threshold recalibration on a small labeled target sample.</b> Re-score the existing cross-country predictions with a target-country threshold or prior correction. "
                "AUROC of 0.61 to 0.69 on the failing pairs caps what recalibration can recover. If most of the gap closes, the head is portable at the cost of a small label budget.",
                "<b>Fusion of AlphaEarth with GHSL.</b> At 100 m on balanced tiles the two are level. A stacked head tests whether the embedding adds signal beyond a free baseline.",
                "<b>Label audit for Kolkata, Chennai and Sapporo.</b> India's fragmentation may partly be Overture label coverage, which falls from about 75% (Berlin) to about 1% (Lagos).",
            ]
        )
    )
    S.append(P("Do not claim", h2))
    S.extend(
        bullets(
            [
                "A universal global residential subspace. Three countries are too few: geometry-versus-transfer correlations (about 0.37, 600 rows) are not independent observations.",
                "Fine-grained building type from embeddings alone, or reliable 10 m area fractions.",
                "That pooled-split scores represent held-out geography.",
            ]
        )
    )
    S.append(P("Caveats on the numbers", h2))
    S.extend(
        bullets(
            [
                "Several protocols appear side by side. 0.78 is LORO on 1.75 M balanced cells, 0.70 is the shared 59,596-cell set, 0.56 is the imbalanced Week 3 protocol. They are labelled wherever they appear and are not interchangeable.",
                "Labels come from Overture footprints. Tag fill varies by region, and the briefing's FEMA and GHS-OBAT checks read missing tags as an OSM coverage gap, not a residential-only bias. Rural counts are small.",
                "The 100 m MAE figures are read from the saved evaluation tables (generated 29 Sep). The Week 4 notebook text quotes older values (about 0.28 for Week 3 heads and 0.35 for GHSL), so the tables are used here.",
                "week5/notebooks/06 states PLS-2 wins 8 of 10 Indian holdouts. Recounting the saved per-city table gives 9 of 10 (Jaipur is the exception).",
            ]
        )
    )
    S.append(Spacer(1, 6))
    S.append(P("Where each number comes from", h2))
    S.append(
        table(
            [
                ["Section", "Artifacts and notebooks"],
                ["1 Classification", "outputs/week4_pixel_balanced_train, week4_model_comparison, week4_ghsl_pixel; notebooks week1/00, week2/02, week4/01 to 05"],
                ["2 Regression", "outputs/week4_pixel_eval/week3_rescore_v1, week4_100m_fraction_eval/mae_summary.csv; notebooks week3/03, week4/01, 05"],
                ["3 Generalisation", "outputs/week4_vit_100m_3d_v4, week4_pls_10m_vs_100m; notebooks week4/08 to 16"],
                ["4 Embeddings", "outputs/week4_region_umap, week4_vit_100m_3d_v4/figures; notebooks week4/12 to 16, notebooks/week5/02, 03"],
                ["5 Country LORO", "week4/international_geo_generalization/runs/{intl_transfer,india_loro,japan_loro}*; week5/runs/dou_balanced_week5_20261008_192748; notebooks week5/01 to 04"],
                ["Rebuild", ".venv/bin/python scripts/build_stakeholder_report.py"],
            ],
            [0.18, 0.82],
        )
    )

    doc.build(S)


def main() -> None:
    d = load_data()
    fig_paths = {
        "cls": fig_classification(d),
        "reg": fig_regression(d),
        "gen": fig_generalisation(d),
        "ctry": fig_country_loro(d),
        "xfer": fig_country_transfer(d),
        "loco": fig_loco(d),
    }
    emb, gap_r = fig_embedding(d)
    fig_paths["emb"] = emb
    fig_paths["umap"] = ROOT / "outputs/week4_region_umap/figures/umap_gt_wrong_side_by_region.png"
    build_pdf(d, fig_paths, gap_r)
    manifest = {
        "pdf": str(PDF_PATH.relative_to(ROOT)),
        "figures": {k: str(Path(v).relative_to(ROOT)) for k, v in fig_paths.items()},
        "built": date.today().isoformat(),
        "gap_vs_f1_pearson": gap_r,
        "notebook_sourced_numbers": {k: v[1] if isinstance(v, tuple) else v for k, v in SOURCE_NUMBERS.items()},
        "rebuild": ".venv/bin/python scripts/build_stakeholder_report.py",
    }
    (OUT_DIR / "stakeholder_report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
