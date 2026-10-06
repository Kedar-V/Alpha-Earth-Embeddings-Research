#!/usr/bin/env python3
"""Plot OOF PLS-DA t1 vs t2 for RES / NRES cluster visualization.

Writes:
  outputs/week4_pls_loro/figures/pls_t1_t2_{global,by_region,marginals}.png
  outputs/week4_pls_loro/pls_t1_t2_viz.json
  outputs/week4_pls_loro/pls_t1_t2_by_region.csv
  outputs/week4_pls_loro/pls_t1_t2_scatter_sample.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_pls_loro"
FIG = OUT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(
        OUT / "oof_predictions.parquet",
        columns=["pixel_id", "region", "y", "stratum", "pixel_lon", "pixel_lat"],
    )
    scores = np.load(OUT / "oof_pls_da_scores_k32.npy")
    assert len(oof) == len(scores)
    t1, t2 = scores[:, 0], scores[:, 1]
    oof = oof.copy()
    oof["t1"] = t1
    oof["t2"] = t2
    y = oof["y"].to_numpy()

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(np.c_[t1, t2], y)
    pred = clf.predict(np.c_[t1, t2])
    f1_2d = float(f1_score(y, pred, average="macro"))
    acc_2d = float(accuracy_score(y, pred))
    clf1 = LogisticRegression(max_iter=1000, random_state=42)
    clf1.fit(t1.reshape(-1, 1), y)
    f1_1d = float(f1_score(y, clf1.predict(t1.reshape(-1, 1)), average="macro"))
    print(
        f"OOF 2D(t1,t2) LR macro-F1={f1_2d:.3f}  acc={acc_2d:.3f}  |  1D(t1) F1={f1_1d:.3f}",
        flush=True,
    )

    rng = np.random.default_rng(42)
    n_per = 8000
    take_res = rng.choice(np.where(y == 1)[0], size=min(n_per, int((y == 1).sum())), replace=False)
    take_nres = rng.choice(np.where(y == 0)[0], size=min(n_per, int((y == 0).sum())), replace=False)
    take = np.concatenate([take_res, take_nres])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    big = rng.choice(len(oof), size=min(200_000, len(oof)), replace=False)
    ax.hexbin(t1[big], t2[big], gridsize=80, cmap="Greys", mincnt=5, linewidths=0, alpha=0.35, zorder=0)
    colors = np.where(y[take] == 1, "#0072B2", "#E69F00")
    ax.scatter(t1[take], t2[take], c=colors, s=6, alpha=0.35, linewidths=0, rasterized=True, zorder=2)
    xlim = np.percentile(t1, [0.5, 99.5])
    ylim = np.percentile(t2, [0.5, 99.5])
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    zz = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=[0.5], colors="#333333", linewidths=1.4, zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("PLS-DA score $t_1$ (1st orthogonal component)")
    ax.set_ylabel("PLS-DA score $t_2$ (2nd orthogonal component)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markersize=8, label="Residential"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00", markersize=8, label="Non-residential"),
            Line2D([0], [0], color="#333333", lw=1.4, label="2D LR $P=0.5$"),
        ],
        frameon=False,
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(
        0.01,
        0.98,
        f"Two PLS scores already separate RES / NRES (OOF 2D F1={f1_2d:.2f}; t₁ alone={f1_1d:.2f})",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.01,
        0.935,
        f"Stratified scatter n={len(take):,}; grey hexbin = density. Scores are OOF from LORO fold rotations.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "pls_t1_t2_global.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    regions = sorted(oof["region"].astype(str).unique())
    ncols, nrows = 5, int(np.ceil(len(regions) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    per_region = []
    for i, region in enumerate(tqdm(regions, desc="region panels")):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = oof.loc[oof.region.astype(str) == region]
        parts = []
        for lab, nmax in ((1, 2500), (0, 2500)):
            s = sub.loc[sub.y == lab]
            if len(s) > nmax:
                s = s.sample(nmax, random_state=42)
            parts.append(s)
        s = pd.concat(parts)
        ax.scatter(s.loc[s.y == 0, "t1"], s.loc[s.y == 0, "t2"], s=4, alpha=0.3, c="#E69F00", linewidths=0, rasterized=True)
        ax.scatter(s.loc[s.y == 1, "t1"], s.loc[s.y == 1, "t2"], s=4, alpha=0.3, c="#0072B2", linewidths=0, rasterized=True)
        Xr = sub[["t1", "t2"]].to_numpy()
        yr = sub["y"].to_numpy()
        clf_r = LogisticRegression(max_iter=1000, random_state=42).fit(Xr, yr)
        f1r = float(f1_score(yr, clf_r.predict(Xr), average="macro"))
        x0, x1 = np.percentile(sub.t1, [1, 99])
        y0, y1 = np.percentile(sub.t2, [1, 99])
        xx, yy = np.meshgrid(np.linspace(x0, x1, 100), np.linspace(y0, y1, 100))
        zz = clf_r.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0.5], colors="#333", linewidths=1.0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"{region.replace('_', ' ')}  ·  2D F1={f1r:.2f}", loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        per_region.append(
            {
                "region": region,
                "n": int(len(sub)),
                "f1_2d_inregion": f1r,
                "mean_t1_res": float(sub.loc[sub.y == 1, "t1"].mean()),
                "mean_t1_nres": float(sub.loc[sub.y == 0, "t1"].mean()),
            }
        )
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.text(
        0.01,
        0.98,
        "Per-metro $(t_1,t_2)$: clean split vs FN-heavy (blue on left) vs FP-heavy (orange on right)",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.text(
        0.01,
        0.955,
        "Blue=RES, orange=NRES. SF/Chicago/Houston pull RES left (high FN); NYC/Boston leak NRES right (low Prec). Black = in-region P=0.5.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "pls_t1_t2_by_region.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    bins = np.linspace(np.percentile(t1, 0.5), np.percentile(t1, 99.5), 80)
    ax.hist(t1[y == 0], bins=bins, density=True, alpha=0.55, color="#E69F00", label="NRES")
    ax.hist(t1[y == 1], bins=bins, density=True, alpha=0.55, color="#0072B2", label="RES")
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("density")
    ax.set_title(f"$t_1$ marginal  ·  1D LR F1={f1_1d:.2f}", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    bins2 = np.linspace(np.percentile(t2, 0.5), np.percentile(t2, 99.5), 80)
    ax.hist(t2[y == 0], bins=bins2, density=True, alpha=0.55, color="#E69F00", label="NRES")
    ax.hist(t2[y == 1], bins=bins2, density=True, alpha=0.55, color="#0072B2", label="RES")
    ax.set_xlabel("$t_2$")
    ax.set_ylabel("density")
    ax.set_title("$t_2$ marginal (weaker alone)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        "Most of the class split is along $t_1$; $t_2$ adds a thinner orthogonal residual",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    fig.savefig(FIG / "pls_t1_t2_marginals.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    oof.iloc[take][["pixel_id", "region", "y", "stratum", "t1", "t2"]].to_parquet(
        OUT / "pls_t1_t2_scatter_sample.parquet", index=False
    )
    pd.DataFrame(per_region).to_csv(OUT / "pls_t1_t2_by_region.csv", index=False)
    meta = {
        "oof_f1_2d_t1_t2": f1_2d,
        "oof_acc_2d_t1_t2": acc_2d,
        "oof_f1_1d_t1": f1_1d,
        "n_scatter": int(len(take)),
        "n_total": int(len(oof)),
        "note": "2D/1D F1 here refit on pooled OOF scores (descriptive). Honest LORO k=2 is in fine_k_sweep.csv (~0.738).",
        "paths": {
            "global": "outputs/week4_pls_loro/figures/pls_t1_t2_global.png",
            "by_region": "outputs/week4_pls_loro/figures/pls_t1_t2_by_region.png",
            "marginals": "outputs/week4_pls_loro/figures/pls_t1_t2_marginals.png",
        },
    }
    (OUT / "pls_t1_t2_viz.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
