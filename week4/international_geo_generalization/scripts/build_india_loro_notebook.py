#!/usr/bin/env python3
"""Build India 10-city LORO notebook + PNG figures from a finished run.

Usage:
  .venv/bin/python week4/international_geo_generalization/scripts/build_india_loro_notebook.py \
    --run-dir week4/international_geo_generalization/runs/india_loro_20261005_202600
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

ROOT = Path(__file__).resolve().parents[3]
ORANGE = "#D55E00"
BLUE = "#0072B2"
TEAL = "#009E73"
PURPLE = "#6A3D9A"
INK = "#1a1a1a"
MUTED = "#555555"

CITY_SHORT = {
    "IND_AHMEDABAD": "Ahmedabad",
    "IND_BENGALURU": "Bengaluru",
    "IND_CHENNAI": "Chennai",
    "IND_DELHI": "Delhi",
    "IND_HYDERABAD": "Hyderabad",
    "IND_JAIPUR": "Jaipur",
    "IND_KOLKATA": "Kolkata",
    "IND_LUCKNOW": "Lucknow",
    "IND_MUMBAI": "Mumbai",
    "IND_PUNE": "Pune",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")


def _f1_cmap():
    return LinearSegmentedColormap.from_list(
        "f1",
        ["#F7F7F5", "#D6EAF5", "#6BAED6", "#2171B5", "#08306B"],
    )


def _short(city: str) -> str:
    return CITY_SHORT.get(city, city.replace("IND_", "").title())


def load_run(run_dir: Path):
    art = run_dir / "artifacts"
    loro = pd.read_csv(art / "loro_IND_metrics.csv")
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

    loro64 = loro[loro.model == "lr64"].copy()
    mean_loro = float(loro64["macro_f1"].mean())
    within = float(
        country[(country.model == "lr64") & (country.source_country == "IND") & (country.target_country == "IND")][
            "macro_f1"
        ].iloc[0]
    )
    city_lr = city[city.model == "lr64"].copy()
    off = city_lr[city_lr.source != city_lr.target]
    mean_city = float(off["macro_f1"].mean()) if len(off) else float("nan")

    # ---- 1. Headline ----
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    labels = [
        "India LORO\n(10-city)",
        "Within-India\n80/20",
        "City→city\n(off-diag mean)",
        "Prior 5-city\nIndia LORO",
        "USA LORO\n(week4 intl)",
    ]
    vals = [mean_loro, within, mean_city, 0.437, 0.860]
    colors = [ORANGE, TEAL, BLUE, "#BBBBBB", BLUE]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="none", width=0.62)
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1 (lr64)")
    ax.set_title("India 10-city panel — headline", loc="left", color=INK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "01_headline.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["headline"] = p

    # ---- 2. Per-city LORO ----
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    d = loro64.sort_values("macro_f1").copy()
    y = np.arange(len(d))
    ax.barh(y, d["macro_f1"], color=ORANGE, edgecolor="none", height=0.72)
    ax.axvline(mean_loro, color=INK, ls="--", lw=1, label=f"mean {mean_loro:.3f}")
    ax.axvline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(c) for c in d["held_out"]], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Leave-one-city-out macro-F1 (lr64)")
    ax.set_title("Per-city LORO — train on other 9 Indian cities", loc="left", color=INK)
    for yi, v, n in zip(y, d["macro_f1"], d["n"]):
        ax.text(v + 0.012, yi, f"{v:.3f}  (n={int(n):,})", va="center", fontsize=8, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "02_loro_by_city.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["loro_city"] = p

    # ---- 3. Probe comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    g = loro.groupby(["model", "k"])["macro_f1"].agg(["mean", "std"]).reset_index()
    g["label"] = g.apply(lambda r: f"{r['model']}(k={int(r['k'])})", axis=1)
    order = ["lr64(k=64)", "pca(k=2)", "pca(k=9)", "pls(k=2)", "pls(k=9)"]
    g = g.set_index("label").reindex(order).reset_index()
    ax = axes[0]
    x = np.arange(len(g))
    ax.bar(
        x,
        g["mean"],
        yerr=g["std"],
        color=[ORANGE if "lr" in l else (PURPLE if "pls" in l else TEAL) for l in g["label"]],
        edgecolor="none",
        width=0.65,
        capsize=3,
        error_kw={"ecolor": MUTED, "lw": 1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(g["label"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean LORO macro-F1")
    ax.set_title("Probes over 10 holdouts", loc="left", color=INK)
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    _style(ax)

    ax = axes[1]
    city_order = list(loro64.sort_values("macro_f1")["held_out"])
    xpos = {c: i for i, c in enumerate(city_order)}
    for model, color, marker in [
        ("lr64", ORANGE, "o"),
        ("pca", TEAL, "s"),
        ("pls", PURPLE, "D"),
    ]:
        sub = loro[loro.model == model]
        for k, grp in sub.groupby("k"):
            xs = [xpos[c] for c in grp["held_out"]]
            ax.scatter(
                xs,
                grp["macro_f1"],
                label=f"{model}(k={int(k)})",
                color=color,
                marker=marker,
                s=36,
                alpha=0.85,
            )
    ax.set_xticks(range(len(city_order)))
    ax.set_xticklabels([_short(c) for c in city_order], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1")
    ax.set_title("Per holdout · all probes", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "03_probe_comparison.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["probes"] = p

    # ---- 4. City transfer heatmap ----
    mat = city_lr.pivot(index="source", columns="target", values="macro_f1")
    order = sorted(mat.index.tolist(), key=lambda c: _short(c))
    mat = mat.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    arr = mat.to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=0.3, vmax=0.85, cmap=_f1_cmap(), aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels([_short(c) for c in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_short(c) for c in order], fontsize=8)
    ax.set_xlabel("Target city")
    ax.set_ylabel("Source city (train)")
    ax.set_title("Within-India city→city transfer (lr64)", loc="left", color=INK, pad=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v < 0.55 else INK,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("macro-F1")
    fig.tight_layout()
    p = fig_dir / "04_city_transfer_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["city_heat"] = p

    # ---- 5. Sample sizes ----
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    c = counts.sort_values("n_balanced_total").copy()
    y = np.arange(len(c))
    ax.barh(y, c["n_balanced_total"], color=ORANGE, edgecolor="none", height=0.7, label="balanced total")
    ax.barh(y, c["n_balanced_per_class"], color=TEAL, edgecolor="none", height=0.38, label="per class")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(x) for x in c["city_id"]], fontsize=9)
    ax.set_xlabel("cells")
    ax.set_title("Balanced extract sizes (1:1 per city)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for yi, n in zip(y, c["n_balanced_total"]):
        ax.text(n + max(c["n_balanced_total"]) * 0.01, yi, f"{int(n):,}", va="center", fontsize=8, color=MUTED)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "05_sample_sizes.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["sizes"] = p

    # ---- 6. F1 vs AUROC ----
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(loro64["auroc"], loro64["macro_f1"], s=70, color=ORANGE, edgecolor="white", zorder=3)
    for _, r in loro64.iterrows():
        ax.annotate(
            _short(r["held_out"]),
            (r["auroc"], r["macro_f1"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color=MUTED,
        )
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.axvline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_xlim(0.4, 0.9)
    ax.set_ylim(0.3, 0.8)
    ax.set_xlabel("AUROC (held-out city)")
    ax.set_ylabel("macro-F1 (held-out city)")
    ax.set_title("LORO discrimination vs decision F1", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    p = fig_dir / "06_f1_vs_auroc.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["f1_auroc"] = p

    # also copy into week4 notebook-relative figures/ for stable markdown paths
    nb_fig = run_dir.parent.parent / "figures" / "india_loro"
    nb_fig.mkdir(parents=True, exist_ok=True)
    for key, src in paths.items():
        dst = nb_fig / src.name
        dst.write_bytes(src.read_bytes())
        paths[key] = dst

    # week5 mirror
    w5 = ROOT / "week5" / "figures" / "india_loro"
    w5.mkdir(parents=True, exist_ok=True)
    for src in (run_dir / "figures").glob("*.png"):
        (w5 / src.name).write_bytes(src.read_bytes())

    return paths, {
        "mean_loro": mean_loro,
        "within": within,
        "mean_city": mean_city,
        "hardest": loro64.loc[loro64["macro_f1"].idxmin()],
        "easiest": loro64.loc[loro64["macro_f1"].idxmax()],
        "n_samples": int(summary.get("n_samples", counts["n_balanced_total"].sum())),
        "n_cities": int(summary.get("n_cities", len(counts))),
        "runtime": float(summary.get("runtime_seconds", float("nan"))),
        "probe_means": g,
        "loro64": loro64,
        "off": off,
    }


def build_notebook(run_dir: Path, stats: dict, fig_rel_w4: str, fig_rel_w5: str, out_w4: Path, out_w5: Path):
    run_name = run_dir.name
    mean_loro = stats["mean_loro"]
    within = stats["within"]
    mean_city = stats["mean_city"]
    hard = stats["hardest"]
    easy = stats["easiest"]
    probe = stats["probe_means"]
    loro64 = stats["loro64"].sort_values("held_out")
    off = stats["off"]

    probe_rows = "\n".join(
        f"| {r['label']} | {r['mean']:.3f} | {r['std']:.3f} |" for _, r in probe.iterrows()
    )
    city_rows = "\n".join(
        f"| {_short(r.held_out)} | {r.macro_f1:.3f} | {r.auroc:.3f} | {int(r.n):,} |"
        for _, r in loro64.iterrows()
    )
    worst = off.nsmallest(8, "macro_f1")
    best = off.nlargest(8, "macro_f1")
    worst_rows = "\n".join(
        f"| {_short(r.source)} → {_short(r.target)} | {r.macro_f1:.3f} |" for _, r in worst.iterrows()
    )
    best_rows = "\n".join(
        f"| {_short(r.source)} → {_short(r.target)} | {r.macro_f1:.3f} |" for _, r in best.iterrows()
    )

    def cells_for(fig_rel: str):
        return [
            nbf.v4.new_markdown_cell(
                f"""# India AlphaEarth LORO — 10-city panel

Week4-parity leave-one-city-out on the expanded India metro set (5 original + 5 new).

**Run:** `{run_name}`  
**n cells (balanced):** {stats['n_samples']:,} across {stats['n_cities']} cities  
**Runtime:** {stats['runtime']:.0f}s · workers=20  
**Protocol:** pure-exclusive clear cells · per-city 1:1 balance (seed 42) · `StandardScaler` on train only · threshold 0.5 · lr64 (+ PCA/PLS probes)

Figures are pre-rendered under `{fig_rel}/` — open this notebook without re-running to view results.
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## 1. Headline

| Metric | Value |
|---|---:|
| India 10-city LORO (mean lr64 F1) | **{mean_loro:.3f}** |
| Within-India 80/20 split | **{within:.3f}** |
| City→city off-diagonal mean | **{mean_city:.3f}** |
| Prior 5-city India LORO (intl run) | 0.437 |
| USA LORO (intl run) | 0.860 |

![headline]({fig_rel}/01_headline.png)

**Read:** Expanding from 5→10 cities lifts mean LORO (~0.44→{mean_loro:.2f}) but India remains far below USA. The within-country pooled head ({within:.2f}) is much stronger than leave-one-city-out — geographic fragmentation, not a weak classifier.
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## 2. Per-city LORO

Train on the other 9 Indian cities; evaluate held-out metro.

| Held-out | macro-F1 | AUROC | n |
|---|---:|---:|---:|
{city_rows}

Hardest: **{_short(hard.held_out)}** ({hard.macro_f1:.3f}). Easiest: **{_short(easy.held_out)}** ({easy.macro_f1:.3f}).

![loro by city]({fig_rel}/02_loro_by_city.png)
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## 3. PCA / PLS probes

Mean macro-F1 over the same 10 holdouts.

| model | mean F1 | std |
|---|---:|---:|
{probe_rows}

![probes]({fig_rel}/03_probe_comparison.png)

PLS-2 / PCA-9 beat raw lr64 on mean LORO — low-rank supervised / unsupervised projections help a bit, but do not close the gap to USA (~0.86).
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## 4. City → city transfer heatmap

Train on one city, test on each other (diagonal = train/test same city). Off-diagonal mean F1 = **{mean_city:.3f}** (n={len(off)} pairs).

![city heat]({fig_rel}/04_city_transfer_heatmap.png)

**Worst pairs**

| pair | F1 |
|---|---:|
{worst_rows}

**Best pairs**

| pair | F1 |
|---|---:|
{best_rows}
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## 5. Sample sizes & F1–AUROC

Balanced extract is 1:1 per city; metros differ ~50× in n (Jaipur/Lucknow small, Bengaluru/Mumbai large).

![sizes]({fig_rel}/05_sample_sizes.png)

![f1 auroc]({fig_rel}/06_f1_vs_auroc.png)

Several cities show mid AUROC with low decision F1 (class-conditional score shift / threshold mismatch under a shared Indian head).
"""
            ),
            nbf.v4.new_markdown_cell(
                f"""## Takeaways

1. **10-city India LORO mean lr64 F1 = {mean_loro:.3f}** — better than the 5-city panel (~0.44), still well below USA (~0.86) and below within-India 80/20 ({within:.3f}).
2. **Fragmentation, not capacity:** a pooled Indian head works in-distribution; leave-one-city-out and city→city transfer (mean {mean_city:.3f}) show metro-specific decision boundaries.
3. **Hard / easy poles:** {_short(hard.held_out)} ({hard.macro_f1:.3f}) vs {_short(easy.held_out)} ({easy.macro_f1:.3f}).
4. **Probes:** PCA-9 / PLS-2 raise mean LORO into the mid-0.5s–0.6s; still not USA-like sharing.
5. **Next:** city-conditioned heads, AOI/label audits on weak metros, or subspace (global⊕local) before claiming India↔USA transfer.

Artifacts: `{run_dir}/artifacts/` · report: `{run_dir}/RESULTS.md`
"""
            ),
            nbf.v4.new_markdown_cell("## Reproduce / reload tables"),
            nbf.v4.new_code_cell(
                f"""from pathlib import Path
import json
import pandas as pd

RUN = Path("{run_dir.as_posix()}").resolve()
if not RUN.exists():
    # fallback relative to this notebook
    RUN = (Path.cwd() / "../runs/{run_name}").resolve()
ART = RUN / "artifacts"
loro = pd.read_csv(ART / "loro_IND_metrics.csv")
city = pd.read_csv(ART / "city_transfer_metrics.csv")
country = pd.read_csv(ART / "country_transfer_metrics.csv")
counts = pd.read_csv(RUN / "extract" / "city_counts.csv")
summary = json.loads((RUN / "summary.json").read_text())
print(summary)
display(loro[loro.model == "lr64"][["held_out", "macro_f1", "auroc", "n"]].sort_values("macro_f1"))
"""
            ),
            nbf.v4.new_markdown_cell("### Optional: regenerate figures + this notebook"),
            nbf.v4.new_code_cell(
                f"""# !../../.venv/bin/python ../scripts/build_india_loro_notebook.py --run-dir ../runs/{run_name}
"""
            ),
        ]

    for out, fig_rel in [(out_w4, fig_rel_w4), (out_w5, fig_rel_w5)]:
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells_for(fig_rel)
        nb["metadata"] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, out)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    paths, stats = make_figures(run_dir)
    print("figures:", {k: str(v) for k, v in paths.items()})

    out_w4 = ROOT / "week4" / "international_geo_generalization" / "notebooks" / "02_india_10city_loro.ipynb"
    out_w5 = ROOT / "week5" / "notebooks" / "05_india_10city_loro.ipynb"
    build_notebook(
        run_dir,
        stats,
        fig_rel_w4="../figures/india_loro",
        fig_rel_w5="../figures/india_loro",
        out_w4=out_w4,
        out_w5=out_w5,
    )


if __name__ == "__main__":
    main()
