#!/usr/bin/env python3
"""Build international generalization notebook + PNG figures from a finished run.

Usage:
  .venv/bin/python week4/international_geo_generalization/scripts/build_intl_generalization_notebook.py \
    --run-dir week4/international_geo_generalization/runs/intl_transfer_20261005_142709
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
INK = "#1a1a1a"
MUTED = "#555555"
LIGHT = "#F7F7F5"

COUNTRY_COLOR = {"USA": BLUE, "IND": ORANGE, "JPN": TEAL}
COUNTRY_LABEL = {"USA": "United States", "IND": "India", "JPN": "Japan"}

CITY_SHORT = {
    "USA_ATLANTA": "Atlanta",
    "USA_BOSTON": "Boston",
    "USA_CHICAGO": "Chicago",
    "USA_DENVER": "Denver",
    "USA_HOUSTON": "Houston",
    "USA_LOS_ANGELES": "Los Angeles",
    "USA_NEW_YORK": "New York",
    "USA_PHOENIX": "Phoenix",
    "USA_SAN_FRANCISCO": "San Francisco",
    "USA_SEATTLE": "Seattle",
    "IND_BENGALURU": "Bengaluru",
    "IND_CHENNAI": "Chennai",
    "IND_DELHI": "Delhi",
    "IND_KOLKATA": "Kolkata",
    "IND_MUMBAI": "Mumbai",
    "JPN_FUKUOKA": "Fukuoka",
    "JPN_NAGOYA": "Nagoya",
    "JPN_OSAKA": "Osaka",
    "JPN_SAPPORO": "Sapporo",
    "JPN_TOKYO": "Tokyo",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")


def _heat(ax, mat: pd.DataFrame, vmin, vmax, cmap, title: str, annotate=True, fmt=".2f"):
    order_r = list(mat.index)
    order_c = list(mat.columns)
    arr = mat.loc[order_r, order_c].to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(order_c)))
    ax.set_yticks(range(len(order_r)))
    ax.set_xticklabels([CITY_SHORT.get(x, x) for x in order_c], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([CITY_SHORT.get(x, x) for x in order_r], fontsize=7)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=6.5 if arr.shape[0] <= 5 else 5.2,
                    color="white" if v < (vmin + vmax) / 2 else INK,
                )
    return im


def _f1_cmap():
    # pale → blue (good F1)
    return LinearSegmentedColormap.from_list(
        "f1",
        ["#F7F7F5", "#D6EAF5", "#6BAED6", "#2171B5", "#08306B"],
    )


def load_run(run_dir: Path):
    art = run_dir / "artifacts"
    loro = pd.concat(
        [
            pd.read_csv(art / "loro_USA_metrics.csv"),
            pd.read_csv(art / "loro_IND_metrics.csv"),
            pd.read_csv(art / "loro_JPN_metrics.csv"),
        ],
        ignore_index=True,
    )
    country = pd.read_csv(art / "country_transfer_metrics.csv")
    city = pd.read_csv(art / "city_transfer_metrics.csv")
    counts = pd.read_csv(run_dir / "extract" / "city_counts.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    return loro, country, city, counts, summary


def make_figures(run_dir: Path) -> dict[str, Path]:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    loro, country, city, counts, summary = load_run(run_dir)
    paths: dict[str, Path] = {}
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
        }
    )

    # ---- 1. Headline cards: LORO means + key transfers ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    loro64 = loro[loro.model == "lr64"].copy()
    means = loro64.groupby("country")["macro_f1"].agg(["mean", "std"]).reindex(["USA", "JPN", "IND"])
    x = np.arange(len(means))
    bars = ax.bar(
        x,
        means["mean"],
        yerr=means["std"],
        color=[COUNTRY_COLOR[c] for c in means.index],
        edgecolor="none",
        width=0.62,
        capsize=4,
        error_kw={"ecolor": MUTED, "lw": 1},
    )
    ax.axhline(0.779, color="#999999", ls="--", lw=1, label="Week4 US downtown LORO ≈ 0.78")
    ax.set_xticks(x)
    ax.set_xticklabels([COUNTRY_LABEL[c] for c in means.index])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Leave-one-city-out macro-F1")
    ax.set_title("Within-country geographic sharing", loc="left", color=INK)
    for b, v in zip(bars, means["mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)

    ax = axes[1]
    ct = country[(country.model == "lr64")].copy()
    pairs = [
        ("USA", "USA", "USA within"),
        ("USA", "JPN", "USA → Japan"),
        ("USA", "IND", "USA → India"),
        ("JPN", "USA", "Japan → USA"),
        ("IND", "USA", "India → USA"),
        ("JPN", "IND", "Japan → India"),
        ("IND", "JPN", "India → Japan"),
    ]
    vals = []
    for s, t, _ in pairs:
        v = float(ct[(ct.source_country == s) & (ct.target_country == t)]["macro_f1"].iloc[0])
        vals.append(v)
    colors = []
    for s, t, _ in pairs:
        if s == t:
            colors.append(COUNTRY_COLOR[s])
        elif s == "USA":
            colors.append(BLUE)
        elif s == "JPN":
            colors.append(TEAL)
        else:
            colors.append(ORANGE)
    y = np.arange(len(pairs))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.68)
    ax.axvline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([p[2] for p in pairs])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("macro-F1 (lr64)")
    ax.set_title("Can a country-trained head transfer?", loc="left", color=INK)
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "01_headline_loro_and_transfer.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["headline"] = p

    # ---- 2. Country transfer heatmap ----
    mat = ct.pivot(index="source_country", columns="target_country", values="macro_f1").reindex(
        index=["USA", "IND", "JPN"], columns=["USA", "IND", "JPN"]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    display = mat.copy()
    display.index = [COUNTRY_LABEL[c] for c in display.index]
    display.columns = [COUNTRY_LABEL[c] for c in display.columns]
    im = _heat(ax, display, 0.3, 0.95, _f1_cmap(), "Country → country transfer (lr64)")
    ax.set_xticklabels(list(display.columns), rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(list(display.index), fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("macro-F1")
    ax.set_xlabel("Test country")
    ax.set_ylabel("Train country")
    fig.tight_layout()
    p = fig_dir / "02_country_transfer_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["country_heat"] = p

    # ---- 3. Per-city LORO strip ----
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    order_countries = ["USA", "JPN", "IND"]
    plot_df = loro64.copy()
    plot_df["city"] = plot_df["held_out"].map(CITY_SHORT)
    xpos, labels, colors, vals = [], [], [], []
    x = 0.0
    for c in order_countries:
        sub = plot_df[plot_df.country == c].sort_values("macro_f1", ascending=False)
        for _, r in sub.iterrows():
            xpos.append(x)
            vals.append(float(r["macro_f1"]))
            labels.append(r["city"])
            colors.append(COUNTRY_COLOR[c])
            x += 1.0
        x += 1.2
    ax.bar(xpos, vals, color=colors, edgecolor="none", width=0.8)
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("LORO macro-F1")
    ax.set_title("Leave-one-city-out performance by metro", loc="left", color=INK)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=COUNTRY_COLOR[c], label=COUNTRY_LABEL[c]) for c in order_countries],
        frameon=False,
        loc="upper right",
    )
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "03_loro_by_city.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["loro_city"] = p

    # ---- 4. Full 20×20 city transfer heatmap ----
    city64 = city[city.model == "lr64"].copy()
    city_order = (
        counts.sort_values(["country_id", "city_id"])["city_id"].tolist()
    )
    # Prefer USA, JPN, IND blocks
    city_order = (
        [c for c in city_order if c.startswith("USA_")]
        + [c for c in city_order if c.startswith("JPN_")]
        + [c for c in city_order if c.startswith("IND_")]
    )
    mat_c = city64.pivot(index="source", columns="target", values="macro_f1").reindex(
        index=city_order, columns=city_order
    )
    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    im = _heat(ax, mat_c, 0.30, 0.95, _f1_cmap(), "City → city transfer (lr64)", annotate=False)
    # country block lines
    cuts = [10, 15]  # after USA, after JPN
    for cut in cuts:
        ax.axhline(cut - 0.5, color="white", lw=2)
        ax.axvline(cut - 0.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("macro-F1")
    ax.set_xlabel("Test city")
    ax.set_ylabel("Train city")
    # block labels
    for mid, lab, col in ((4.5, "USA", BLUE), (12, "Japan", TEAL), (17, "India", ORANGE)):
        ax.text(mid, -1.6, lab, ha="center", color=col, fontsize=10, fontweight="bold")
        ax.text(-1.8, mid, lab, va="center", rotation=90, color=col, fontsize=10, fontweight="bold")
    fig.tight_layout()
    p = fig_dir / "04_city_transfer_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["city_heat"] = p

    # ---- 5. Cross-country block averages ----
    city64 = city64.copy()
    off = city64[city64.source != city64.target]
    rows = []
    for s in ["USA", "JPN", "IND"]:
        for t in ["USA", "JPN", "IND"]:
            g = off[(off.source_country == s) & (off.target_country == t)]
            rows.append(
                {
                    "source": s,
                    "target": t,
                    "mean": float(g.macro_f1.mean()) if len(g) else np.nan,
                    "median": float(g.macro_f1.median()) if len(g) else np.nan,
                    "n": int(len(g)),
                }
            )
    block = pd.DataFrame(rows)
    block_mat = block.pivot(index="source", columns="target", values="mean").reindex(
        index=["USA", "JPN", "IND"], columns=["USA", "JPN", "IND"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    display = block_mat.copy()
    display.index = [COUNTRY_LABEL[c] for c in display.index]
    display.columns = [COUNTRY_LABEL[c] for c in display.columns]
    im = _heat(ax, display, 0.30, 0.85, _f1_cmap(), "Mean city-pair F1 (off-diagonal)")
    ax.set_xticklabels(list(display.columns), rotation=0, ha="center")
    ax.set_yticklabels(list(display.index))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mean macro-F1")

    ax = axes[1]
    # retention vs source within-country split
    within = {
        r.source_country: float(r.macro_f1)
        for r in ct[ct.source_country == ct.target_country].itertuples()
    }
    transfer_pairs = [
        ("USA", "JPN"),
        ("USA", "IND"),
        ("JPN", "USA"),
        ("IND", "USA"),
        ("JPN", "IND"),
        ("IND", "JPN"),
    ]
    labels, retention, abs_f1 = [], [], []
    for s, t in transfer_pairs:
        f = float(ct[(ct.source_country == s) & (ct.target_country == t)].macro_f1.iloc[0])
        labels.append(f"{s}→{t}")
        abs_f1.append(f)
        retention.append(f / within[s])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, abs_f1, width=0.36, color=BLUE, label="Transfer F1")
    ax.bar(x + 0.18, retention, width=0.36, color=ORANGE, label="Retention (÷ source within)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Transfer strength vs source within-country", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "05_block_averages_and_retention.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["blocks"] = p

    # ---- 6. Model comparison (lr64 / pca / pls) for country transfer ----
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    focus = [
        ("USA", "USA"),
        ("USA", "JPN"),
        ("USA", "IND"),
        ("JPN", "USA"),
        ("IND", "USA"),
    ]
    model_specs = [("lr64", 64, BLUE), ("pls", 9, TEAL), ("pca", 9, ORANGE)]
    width = 0.25
    x = np.arange(len(focus))
    for i, (model, k, color) in enumerate(model_specs):
        vals = []
        for s, t in focus:
            r = country[(country.model == model) & (country.k == k) & (country.source_country == s) & (country.target_country == t)]
            vals.append(float(r.macro_f1.iloc[0]))
        ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=f"{model} k={k}", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}→{t}" for s, t in focus])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1")
    ax.set_title("Transfer is not rescued by PCA/PLS compression", loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "06_model_comparison_transfer.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["models"] = p

    # ---- 7. Sample-size context ----
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    counts2 = counts.copy()
    counts2["country_id"] = pd.Categorical(counts2["country_id"], ["USA", "JPN", "IND"])
    counts2 = counts2.sort_values(["country_id", "n_balanced_total"], ascending=[True, False])
    xpos = np.arange(len(counts2))
    ax.bar(
        xpos,
        counts2["n_balanced_total"] / 1000,
        color=[COUNTRY_COLOR[c] for c in counts2["country_id"]],
        edgecolor="none",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(counts2["city_name"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Balanced pure cells (thousands)")
    ax.set_title("Analysis set size per city (1:1 pure-exclusive)", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "07_sample_sizes.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["sizes"] = p

    # ---- 8. Asymmetry scatter: A→B vs B→A ----
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    directed = [("USA", "JPN"), ("USA", "IND"), ("JPN", "IND")]
    for s, t in directed:
        ab = float(ct[(ct.source_country == s) & (ct.target_country == t)].macro_f1.iloc[0])
        ba = float(ct[(ct.source_country == t) & (ct.target_country == s)].macro_f1.iloc[0])
        ax.scatter([ab], [ba], s=120, color=COUNTRY_COLOR[s], zorder=3, edgecolor="white", lw=1.2)
        ax.annotate(f"{s}↔{t}", (ab, ba), textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK)
    ax.plot([0.3, 0.9], [0.3, 0.9], ls="--", color="#AAAAAA", lw=1)
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(0.3, 0.9)
    ax.set_xlabel("F1 train A → test B")
    ax.set_ylabel("F1 train B → test A")
    ax.set_title("Transfer asymmetry", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "08_transfer_asymmetry.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["asym"] = p

    # save figure index
    (fig_dir / "manifest.json").write_text(
        json.dumps({k: str(v.relative_to(run_dir)) for k, v in paths.items()}, indent=2) + "\n"
    )
    return paths


def build_notebook(
    run_dir: Path,
    paths: dict[str, Path],
    out_nb: Path,
    *,
    fig_rel: dict[str, str] | None = None,
    run_rel: str | None = None,
    builder_hint: str | None = None,
) -> Path:
    if fig_rel is None:
        fig_rel = {k: f"../runs/{run_dir.name}/figures/{v.name}" for k, v in paths.items()}
    if run_rel is None:
        run_rel = f"../runs/{run_dir.name}"
    if builder_hint is None:
        builder_hint = (
            f"!../.venv/bin/python ../scripts/build_intl_generalization_notebook.py "
            f"--run-dir {run_rel}"
        )

    loro, country, city, counts, summary = load_run(run_dir)
    loro64 = loro[loro.model == "lr64"]
    ct = country[country.model == "lr64"]

    def f1(s, t):
        return float(ct[(ct.source_country == s) & (ct.target_country == t)].macro_f1.iloc[0])

    nb = nbf.v4.new_notebook()
    cells = []

    def md(text: str):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text: str):
        cells.append(nbf.v4.new_code_cell(text))

    md(
        f"""# International generalization — USA / India / Japan

Week4-parity analysis of whether frozen **64-D AlphaEarth** residential vs non-residential
heads transfer across countries.

**Run:** `{run_dir.name}`  
**n cells (balanced):** {summary['n_samples']:,} across {summary['n_cities']} cities  
**Protocol:** pure-exclusive clear cells · per-city 1:1 balance (seed 42) · `StandardScaler` on train only · threshold 0.5 · lr64 (+ PCA/PLS probes)

Figures below are pre-rendered from `artifacts/`; re-run the last cells to regenerate.
"""
    )

    md(
        f"""## 1. Headline

| Within-country LORO (mean F1) | | Key transfers (lr64) | |
|---|---:|---|---:|
| USA | **{summary['loro_lr64_mean']['USA']:.3f}** | USA → Japan | **{f1('USA','JPN'):.3f}** |
| Japan | **{summary['loro_lr64_mean']['JPN']:.3f}** | USA → India | **{f1('USA','IND'):.3f}** |
| India | **{summary['loro_lr64_mean']['IND']:.3f}** | Japan → USA | **{f1('JPN','USA'):.3f}** |

![headline]({fig_rel['headline']})
"""
    )

    md(
        f"""## 2. Country → country transfer matrix

Diagonal = within-country 80/20 stratified split. Off-diagonal = train on all source-country cities, test on all target-country cities.

![country heat]({fig_rel['country_heat']})

**Read:** USA and Japan have strong within-country ceilings ({f1('USA','USA'):.2f} / {f1('JPN','JPN'):.2f}).
USA→India collapses ({f1('USA','IND'):.2f}); Japan→USA remains competitive ({f1('JPN','USA'):.2f}).
"""
    )

    md(
        f"""## 3. Leave-one-city-out by metro

![loro by city]({fig_rel['loro_city']})

USA metros cluster tightly near week4 downtown LORO (~0.78). Japan is mostly strong with Sapporo weaker.
India LORO is near chance for several cities — a national Indian head does not emerge under leave-one-city-out.
"""
    )

    md(
        f"""## 4. Full city → city transfer heatmap

Blocks outlined for USA (10) / Japan (5) / India (5). Bright = high transfer F1.

![city heat]({fig_rel['city_heat']})
"""
    )

    md(
        f"""## 5. Block averages & retention

Retention = transfer F1 ÷ source within-country F1 (1.0 = no drop).

![blocks]({fig_rel['blocks']})
"""
    )

    md(
        f"""## 6. PCA / PLS do not fix cross-country transfer

![models]({fig_rel['models']})
"""
    )

    md(
        f"""## 7. Transfer asymmetry

Points above the diagonal mean B→A works better than A→B.

![asym]({fig_rel['asym']})
"""
    )

    md(
        f"""## 8. Sample sizes (analysis extract)

![sizes]({fig_rel['sizes']})
"""
    )

    md(
        """## Takeaways

1. **Embeddings are not globally interchangeable for this task.** Within-USA LORO is excellent; Japan is good; India is fragmented across cities.
2. **A USA-trained head is not safe for India** (F1 ≈ 0.37). Japan is a partial cold-start (≈ 0.65) but still well below USA within-country.
3. **Japan → USA transfers surprisingly well** (≈ 0.72) — asymmetry argues against one universal linear direction.
4. **Operational rule:** train or fine-tune in-country (or at least validate transfer before shipping a US-only head).
"""
    )

    md("## Reproduce / reload tables")
    code(
        f'''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path("{run_rel}")
ART = RUN / "artifacts"
FIG = RUN / "figures"

loro = pd.concat([pd.read_csv(ART / f) for f in ["loro_USA_metrics.csv", "loro_IND_metrics.csv", "loro_JPN_metrics.csv"]], ignore_index=True)
country = pd.read_csv(ART / "country_transfer_metrics.csv")
city = pd.read_csv(ART / "city_transfer_metrics.csv")
summary = json.loads((RUN / "summary.json").read_text())

print("LORO lr64 means:")
print(loro[loro.model=="lr64"].groupby("country")["macro_f1"].agg(["mean","std","min","max"]).round(3))
print("\\nCountry transfer lr64:")
print(country[country.model=="lr64"].pivot(index="source_country", columns="target_country", values="macro_f1").round(3))
print("\\nFigures:", sorted(p.name for p in FIG.glob("*.png")))
'''
    )

    md("### Optional: regenerate figures")
    code(
        f'''# Re-run the builder (writes PNGs + can rebuild this notebook)
# {builder_hint}
from IPython.display import Image, display
for name in sorted(FIG.glob("*.png")):
    print(name.name)
    display(Image(filename=str(name), width=900))
'''
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_nb)
    return out_nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("week4/international_geo_generalization/runs/intl_transfer_20261005_142709"),
    )
    ap.add_argument(
        "--notebook",
        type=Path,
        default=Path("week4/international_geo_generalization/notebooks/01_international_generalization.ipynb"),
    )
    ap.add_argument(
        "--week5-notebook",
        type=Path,
        default=Path("week5/notebooks/01_international_generalization.ipynb"),
    )
    ap.add_argument("--skip-week5", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"run not complete: {run_dir}")
    paths = make_figures(run_dir)
    nb_path = build_notebook(run_dir, paths, args.notebook.resolve())
    print(f"Wrote {len(paths)} figures → {run_dir / 'figures'}")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"Notebook: {nb_path}")

    if not args.skip_week5:
        week5_nb = args.week5_notebook.resolve()
        week5_fig = week5_nb.parent.parent / "figures"
        week5_fig.mkdir(parents=True, exist_ok=True)
        for v in paths.values():
            dest = week5_fig / v.name
            dest.write_bytes(v.read_bytes())
        # Also mirror under notebooks/week5 for parity with notebooks/week4
        notebooks_week5 = Path("notebooks/week5/01_international_generalization.ipynb").resolve()
        fig_rel_week5_pkg = {k: f"../figures/{v.name}" for k, v in paths.items()}
        run_rel_from_week5 = f"../../week4/international_geo_generalization/runs/{run_dir.name}"
        build_notebook(
            run_dir,
            paths,
            week5_nb,
            fig_rel=fig_rel_week5_pkg,
            run_rel=run_rel_from_week5,
            builder_hint=(
                "!../../.venv/bin/python "
                "../../week4/international_geo_generalization/scripts/build_intl_generalization_notebook.py "
                f"--run-dir {run_rel_from_week5}"
            ),
        )
        # notebooks/week5 uses paths up two levels into week4 run figures (no local copy required)
        fig_rel_nb = {
            k: f"../../week4/international_geo_generalization/runs/{run_dir.name}/figures/{v.name}"
            for k, v in paths.items()
        }
        build_notebook(
            run_dir,
            paths,
            notebooks_week5,
            fig_rel=fig_rel_nb,
            run_rel=f"../../week4/international_geo_generalization/runs/{run_dir.name}",
            builder_hint=(
                "!../../.venv/bin/python "
                "../../week4/international_geo_generalization/scripts/build_intl_generalization_notebook.py "
                f"--run-dir ../../week4/international_geo_generalization/runs/{run_dir.name}"
            ),
        )
        readme = week5_nb.parent.parent / "README.md"
        readme.write_text(
            f"""# Week 5 — International geographic generalization

Cross-country AlphaEarth residential / non-residential transfer (USA · India · Japan).

## Notebook

- [`notebooks/01_international_generalization.ipynb`](notebooks/01_international_generalization.ipynb) — figures + takeaways
- Also mirrored at [`../notebooks/week5/01_international_generalization.ipynb`](../notebooks/week5/01_international_generalization.ipynb)

## Figures

Local copies in [`figures/`](figures/). Source run:

`week4/international_geo_generalization/runs/{run_dir.name}/`

## Rebuild

```bash
.venv/bin/python week4/international_geo_generalization/scripts/build_intl_generalization_notebook.py \\
  --run-dir week4/international_geo_generalization/runs/{run_dir.name}
```
"""
        )
        print(f"Week5 notebook: {week5_nb}")
        print(f"Week5 figures: {week5_fig}")
        print(f"Also: {notebooks_week5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
