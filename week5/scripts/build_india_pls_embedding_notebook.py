#!/usr/bin/env python3
"""India PLS embedding-space figures + notebook (week5).

Fits PLS on the international balanced extract (India cities only) and writes:
  week5/figures/india_pls/*.png
  week5/notebooks/02_india_pls_embedding_space.ipynb

Views:
  1) India-pooled PLS t1–t2 (class + city coloring)
  2) Per-city PLS t1–t2 panels
  3) LORO OOF PLS t1–t2 (train on other India cities → project held-out)
  4) Class separation on t1 / 2D F1 summaries
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#1a1a1a"
MUTED = "#555555"

CITY_SHORT = {
    "IND_BENGALURU": "Bengaluru",
    "IND_CHENNAI": "Chennai",
    "IND_DELHI": "Delhi",
    "IND_KOLKATA": "Kolkata",
    "IND_MUMBAI": "Mumbai",
}
CITY_COLORS = {
    "IND_BENGALURU": "#0072B2",
    "IND_CHENNAI": "#009E73",
    "IND_DELHI": "#D55E00",
    "IND_KOLKATA": "#CC79A7",
    "IND_MUMBAI": "#F0E442",
}


def pls_rotations(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    yv = yv - yv.mean()
    n_comp = int(min(n_components, X.shape[1], max(X.shape[0] - 1, 1)))
    XtX = X.T @ X
    XtY = X.T @ yv
    p = X.shape[1]
    W = np.zeros((p, n_comp))
    P = np.zeros((p, n_comp))
    for k in range(n_comp):
        w = XtY.copy()
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            W, P = W[:, :k], P[:, :k]
            break
        w = w / nrm
        for _ in range(80):
            t_var = float(w @ (XtX @ w))
            if t_var < 1e-18:
                break
            c_scalar = float(XtY @ w) / t_var
            if abs(c_scalar) < 1e-18:
                break
            c_unit = 1.0 if c_scalar >= 0 else -1.0
            w_new = XtY * c_unit
            nrm = np.linalg.norm(w_new)
            if nrm < 1e-12:
                break
            w_new = w_new / nrm
            if np.linalg.norm(w_new - w) < 1e-8:
                w = w_new
                break
            w = w_new
        t_var = max(float(w @ (XtX @ w)), 1e-18)
        p_k = (XtX @ w) / t_var
        c_scalar = float(XtY @ w) / t_var
        Xtw = XtX @ w
        XtX = XtX - np.outer(p_k, Xtw) - np.outer(Xtw, p_k) + t_var * np.outer(p_k, p_k)
        XtY = XtY - c_scalar * Xtw - p_k * float(w @ XtY) + t_var * c_scalar * p_k
        W[:, k] = w
        P[:, k] = p_k
    if W.shape[1] == 0:
        raise RuntimeError("PLS produced no components")
    return W @ np.linalg.pinv(P.T @ W)


def fit_pls_scores(X: np.ndarray, y: np.ndarray, n_components: int = 2):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    R = pls_rotations(Xs, y, n_components=n_components)
    Z = Xs @ R
    return scaler, R, Z


def project(scaler, R, X):
    return scaler.transform(X) @ R


def subsample_by_class(y: np.ndarray, n_per: int, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        if len(idx) == 0:
            continue
        take = min(n_per, len(idx))
        parts.append(rng.choice(idx, size=take, replace=False))
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def scatter_t1_t2(
    ax,
    t1,
    t2,
    y,
    *,
    n_hex=80_000,
    n_pts=6_000,
    title="",
    f1=None,
    clf=None,
):
    rng = np.random.default_rng(SEED)
    if len(t1) > n_hex:
        big = rng.choice(len(t1), size=n_hex, replace=False)
    else:
        big = np.arange(len(t1))
    ax.hexbin(t1[big], t2[big], gridsize=70, cmap="Greys", mincnt=3, linewidths=0, alpha=0.35, zorder=0)
    take = subsample_by_class(y, n_pts // 2, rng)
    ax.scatter(
        t1[take][y[take] == 0],
        t2[take][y[take] == 0],
        s=5,
        alpha=0.35,
        c=ORANGE,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label="NRES",
    )
    ax.scatter(
        t1[take][y[take] == 1],
        t2[take][y[take] == 1],
        s=5,
        alpha=0.35,
        c=BLUE,
        linewidths=0,
        rasterized=True,
        zorder=2,
        label="RES",
    )
    if clf is None:
        clf = LogisticRegression(max_iter=500, random_state=SEED).fit(np.c_[t1, t2], y)
        pred = clf.predict(np.c_[t1, t2])
        f1_2d = float(f1_score(y, pred, average="macro")) if f1 is None else float(f1)
    else:
        f1_2d = float(f1) if f1 is not None else float(
            f1_score(y, clf.predict(np.c_[t1, t2]), average="macro")
        )
    x0, x1 = np.percentile(t1, [0.5, 99.5])
    y0, y1 = np.percentile(t2, [0.5, 99.5])
    xx, yy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    zz = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[0.5], colors="#333333", linewidths=1.2, zorder=3)
    ax.scatter([t1[y == 0].mean()], [t2[y == 0].mean()], s=80, c=ORANGE, edgecolors="black", lw=0.8, zorder=4, marker="X")
    ax.scatter([t1[y == 1].mean()], [t2[y == 1].mean()], s=80, c=BLUE, edgecolors="black", lw=0.8, zorder=4, marker="X")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ttl = title if title else "PLS $t_1$–$t_2$"
    ax.set_title(f"{ttl}  ·  2D F1={f1_2d:.2f}", loc="left", fontsize=10, color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return f1_2d, clf


def load_india(extract_dir: Path):
    X = np.load(extract_dir / "X.npy", mmap_mode="r")
    y = np.load(extract_dir / "y.npy")
    regions = np.load(extract_dir / "regions.npy", allow_pickle=True).astype(str)
    countries = np.load(extract_dir / "countries.npy", allow_pickle=True).astype(str)
    mask = countries == "IND"
    X_i = np.asarray(X[mask], dtype=np.float64)
    y_i = y[mask].astype(np.int64)
    r_i = regions[mask]
    cities = sorted(np.unique(r_i).tolist())
    return X_i, y_i, r_i, cities


def make_figures(extract_dir: Path, fig_dir: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("loading India extract…", flush=True)
    X, y, regions, cities = load_india(extract_dir)
    print(f"India n={len(y):,} cities={cities}", flush=True)
    paths = {}
    metrics = {"n": int(len(y)), "cities": cities}

    # ---- 1. India-pooled PLS ----
    print("fitting India-pooled PLS…", flush=True)
    scaler, R, Z = fit_pls_scores(X, y, n_components=2)
    t1, t2 = Z[:, 0], Z[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    f1_pool, _ = scatter_t1_t2(axes[0], t1, t2, y, title="India-pooled PLS (class)")
    # city-colored (subsample)
    ax = axes[1]
    rng = np.random.default_rng(SEED)
    take = []
    for c in cities:
        idx = np.where(regions == c)[0]
        take.append(rng.choice(idx, size=min(2500, len(idx)), replace=False))
    take = np.concatenate(take)
    for c in cities:
        m = regions[take] == c
        ax.scatter(
            t1[take][m],
            t2[take][m],
            s=4,
            alpha=0.35,
            c=CITY_COLORS[c],
            linewidths=0,
            rasterized=True,
            label=CITY_SHORT[c],
        )
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    x0, x1 = np.percentile(t1, [0.5, 99.5])
    y0, y1 = np.percentile(t2, [0.5, 99.5])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title("Same PLS space, colored by city", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, markerscale=2.5, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(
        "India embedding space under a single national PLS (fit on all 5 cities)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = fig_dir / "01_india_pooled_pls_t1_t2.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["pooled"] = p
    metrics["pooled_f1_2d"] = f1_pool
    metrics["pooled_delta_t1"] = float(t1[y == 1].mean() - t1[y == 0].mean())

    # ---- 2. Per-city PLS panels ----
    print("fitting per-city PLS…", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    axes = axes.ravel()
    per_city = []
    for i, city in enumerate(cities):
        m = regions == city
        sc, Rc, Zc = fit_pls_scores(X[m], y[m], n_components=2)
        f1c, _ = scatter_t1_t2(
            axes[i],
            Zc[:, 0],
            Zc[:, 1],
            y[m],
            n_hex=40_000,
            n_pts=5_000,
            title=CITY_SHORT[city],
        )
        dt1 = float(Zc[y[m] == 1, 0].mean() - Zc[y[m] == 0, 0].mean())
        per_city.append(
            {
                "city_id": city,
                "city": CITY_SHORT[city],
                "n": int(m.sum()),
                "f1_2d": f1c,
                "delta_t1": dt1,
                "mean_t1_res": float(Zc[y[m] == 1, 0].mean()),
                "mean_t1_nres": float(Zc[y[m] == 0, 0].mean()),
            }
        )
    axes[-1].axis("off")
    axes[-1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Residential"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=8, label="Non-residential"),
            Line2D([0], [0], color="#333", lw=1.2, label="2D LR P=0.5"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="#444", markersize=9, label="Class mean"),
        ],
        frameon=False,
        loc="center",
        fontsize=11,
    )
    fig.suptitle(
        "Per-city PLS: each metro gets its own scaler + PLS fit (in-city separation)",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = fig_dir / "02_india_per_city_pls_t1_t2.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["per_city"] = p
    metrics["per_city"] = per_city

    # ---- 3. LORO OOF PLS ----
    print("computing LORO OOF PLS scores…", flush=True)
    oof_t1 = np.zeros(len(y), dtype=np.float64)
    oof_t2 = np.zeros(len(y), dtype=np.float64)
    loro_rows = []
    loro_clfs = {}
    for held in cities:
        te = regions == held
        tr = ~te
        sc, Rc, _ = fit_pls_scores(X[tr], y[tr], n_components=2)
        Zte = project(sc, Rc, X[te])
        oof_t1[te] = Zte[:, 0]
        oof_t2[te] = Zte[:, 1]
        Ztr = project(sc, Rc, X[tr])
        clf = LogisticRegression(max_iter=500, random_state=SEED).fit(Ztr, y[tr])
        loro_clfs[held] = clf
        f1_oof = float(f1_score(y[te], clf.predict(Zte), average="macro"))
        loro_rows.append(
            {
                "city_id": held,
                "city": CITY_SHORT[held],
                "n_test": int(te.sum()),
                "f1_2d_loro": f1_oof,
                "delta_t1_test": float(Zte[y[te] == 1, 0].mean() - Zte[y[te] == 0, 0].mean()),
            }
        )
        print(f"  LORO {held}: F1_loro={f1_oof:.3f} Δt1={loro_rows[-1]['delta_t1_test']:.2f}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    axes = axes.ravel()
    for i, city in enumerate(cities):
        m = regions == city
        scatter_t1_t2(
            axes[i],
            oof_t1[m],
            oof_t2[m],
            y[m],
            n_hex=40_000,
            n_pts=5_000,
            title=f"{CITY_SHORT[city]} (OOF)",
            f1=loro_rows[i]["f1_2d_loro"],
            clf=loro_clfs[city],
        )
    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.55,
        "LORO OOF PLS\n\n"
        "For each held-out city:\n"
        "• fit scaler+PLS on the other 4\n"
        "• project held-out cells\n"
        "• LR fit on train scores only\n\n"
        "Collapse / overlap here means the\n"
        "national India direction does not\n"
        "align with that city's RES/NRES axis.",
        transform=axes[-1].transAxes,
        fontsize=10,
        color=MUTED,
        va="center",
    )
    fig.suptitle(
        "India LORO OOF PLS space — held-out city projected into other cities' PLS",
        fontsize=12,
        fontweight="bold",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = fig_dir / "03_india_loro_oof_pls_t1_t2.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["loro_oof"] = p
    metrics["loro"] = loro_rows

    # ---- 4. Separation summary bars ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    names = [r["city"] for r in per_city]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [r["delta_t1"] for r in per_city], width=0.36, color=BLUE, label="Per-city PLS Δt₁")
    ax.bar(x + 0.18, [r["delta_t1_test"] for r in loro_rows], width=0.36, color=ORANGE, label="LORO OOF Δt₁")
    ax.axhline(0, color="#AAAAAA", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean(t₁|RES) − mean(t₁|NRES)")
    ax.set_title("Class separation on $t_1$", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(x - 0.18, [r["f1_2d"] for r in per_city], width=0.36, color=BLUE, label="Per-city PLS 2D F1")
    ax.bar(x + 0.18, [r["f1_2d_loro"] for r in loro_rows], width=0.36, color=ORANGE, label="LORO OOF 2D F1")
    ax.axhline(0.5, color="#BBBBBB", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1 in $(t_1,t_2)$")
    ax.set_title("2D PLS probe F1: in-city fit vs LORO transfer", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = fig_dir / "04_india_pls_separation_summary.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["summary"] = p

    # ---- 5. Global OOF cloud (all India in LORO coordinates) ----
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    scatter_t1_t2(ax, oof_t1, oof_t2, y, title="All India LORO OOF PLS")
    fig.tight_layout()
    p = fig_dir / "05_india_loro_oof_global.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths["loro_global"] = p

    # save metrics + optional sample parquet for interactive use
    (fig_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    sample_idx = subsample_by_class(y, 4000, np.random.default_rng(SEED))
    # stratified also by city roughly
    pd.DataFrame(
        {
            "city_id": regions[sample_idx],
            "city": [CITY_SHORT[c] for c in regions[sample_idx]],
            "y": y[sample_idx],
            "t1_pooled": t1[sample_idx],
            "t2_pooled": t2[sample_idx],
            "t1_loro_oof": oof_t1[sample_idx],
            "t2_loro_oof": oof_t2[sample_idx],
        }
    ).to_parquet(fig_dir / "india_pls_scatter_sample.parquet", index=False)

    return paths, metrics


def build_notebook(fig_dir: Path, metrics: dict, out_nb: Path, *, fig_prefix: str = "../figures/india_pls") -> Path:
    files = {
        "pooled": "01_india_pooled_pls_t1_t2.png",
        "per_city": "02_india_per_city_pls_t1_t2.png",
        "loro_oof": "03_india_loro_oof_pls_t1_t2.png",
        "summary": "04_india_pls_separation_summary.png",
        "loro_global": "05_india_loro_oof_global.png",
    }
    fig_rel = {k: f"{fig_prefix}/{name}" for k, name in files.items()}

    nb = nbf.v4.new_notebook()
    cells = []

    def md(s: str):
        cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

    def code(s: str):
        cells.append(nbf.v4.new_code_cell(s.strip() + "\n"))

    md(
        f"""
# India PLS embedding space

How residential vs non-residential sit in **PLS $(t_1, t_2)$** for the five Indian metros
(Bengaluru, Chennai, Delhi, Kolkata, Mumbai).

**Data.** Balanced pure-exclusive AlphaEarth 10 m / 2024 / 64-D cells from the international extract
(`n = {metrics['n']:,}`). Same protocol as the week5 transfer study.

**Why three views**
| View | What is fit | What it answers |
|---|---|---|
| India-pooled PLS | scaler+PLS on all 5 cities | Is there a shared national RES/NRES axis? |
| Per-city PLS | scaler+PLS inside each city | Can each city separate RES/NRES *locally*? |
| LORO OOF PLS | fit on other 4 → project held-out | Does that local axis transfer? |

Black curve = 2D logistic $P=0.5$ in $(t_1,t_2)$. Crosses = class means.
"""
    )

    md(
        f"""
## 1. India-pooled PLS

National fit F1 in $(t_1,t_2)$ = **{metrics['pooled_f1_2d']:.2f}** ·
$\\Delta t_1$ (RES − NRES) = **{metrics['pooled_delta_t1']:.2f}**

![pooled]({fig_rel['pooled']})

Left: class coloring. Right: same coordinates colored by city — city clumps that don't align with the
RES/NRES cut foreshadow weak leave-one-city-out transfer.
"""
    )

    md(
        f"""
## 2. Per-city PLS (in-city fit)

Each panel refits scaler + PLS on that city only. High 2D F1 ⇒ the city *does* have a usable local direction.

| City | n | 2D F1 | Δt₁ |
|---|---:|---:|---:|
"""
        + "\n".join(
            f"| {r['city']} | {r['n']:,} | {r['f1_2d']:.2f} | {r['delta_t1']:.2f} |"
            for r in metrics["per_city"]
        )
        + f"""

![per city]({fig_rel['per_city']})
"""
    )

    md(
        f"""
## 3. LORO OOF PLS (transfer geometry)

Held-out city projected into the PLS fit of the other four. Low F1 / collapsed Δt₁ ⇒ that city's
RES/NRES axis is **not** the one learned from the rest of India.

| City | LORO 2D F1 | OOF Δt₁ |
|---|---:|---:|
"""
        + "\n".join(
            f"| {r['city']} | {r['f1_2d_loro']:.2f} | {r['delta_t1_test']:.2f} |"
            for r in metrics["loro"]
        )
        + f"""

![loro oof]({fig_rel['loro_oof']})

![loro global]({fig_rel['loro_global']})
"""
    )

    md(
        f"""
## 4. Separation summary

In-city PLS (blue) vs LORO OOF (orange). Gaps between bars are the geometric signature of India's
weak within-country transfer in the week5 analysis.

![summary]({fig_rel['summary']})
"""
    )

    md(
        """
## Takeaways

1. **Local PLS works** — most Indian cities separate RES/NRES cleanly when PLS is fit in-city.
2. **National / LORO PLS often collapses** — projecting a held-out city into the other cities' PLS
   shrinks Δt₁ and drops 2D F1, matching the weak India LORO macro-F1 (~0.44) from the transfer study.
3. **Embeddings carry city-specific structure in India** more than a single shared RES/NRES line;
   treat India as multi-domain, not one national head.
"""
    )

    md("## Reload tables / sample")
    code(
        f'''from pathlib import Path
import json
import pandas as pd

FIG = Path("../figures/india_pls")
metrics = json.loads((FIG / "metrics.json").read_text())
sample = pd.read_parquet(FIG / "india_pls_scatter_sample.parquet")
print(json.dumps(metrics, indent=2)[:1200])
print(sample.head())
print("sample rows", len(sample))
'''
    )

    md("### Optional: replot from sample parquet")
    code(
        '''import matplotlib.pyplot as plt
from IPython.display import Image, display

for p in sorted(FIG.glob("*.png")):
    print(p.name)
    display(Image(filename=str(p), width=980))
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
        "--extract-dir",
        type=Path,
        default=ROOT
        / "week4/international_geo_generalization/runs/intl_transfer_20261005_142709/extract",
    )
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "week5/figures/india_pls")
    ap.add_argument(
        "--notebook",
        type=Path,
        default=ROOT / "week5/notebooks/02_india_pls_embedding_space.ipynb",
    )
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    paths, metrics = make_figures(args.extract_dir.resolve(), args.fig_dir.resolve())
    nb = build_notebook(args.fig_dir.resolve(), metrics, args.notebook.resolve())
    mirror = ROOT / "notebooks/week5/02_india_pls_embedding_space.ipynb"
    build_notebook(
        args.fig_dir.resolve(),
        metrics,
        mirror,
        fig_prefix="../../week5/figures/india_pls",
    )

    print("Figures:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"Notebook: {nb}")
    print(f"Mirror:   {mirror}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
