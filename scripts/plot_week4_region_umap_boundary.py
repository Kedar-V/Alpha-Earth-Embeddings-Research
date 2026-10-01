#!/usr/bin/env python3
"""Week-2-style region UMAP + 2D LR decision boundary for Week 4 AE pixels.

Matches the visual language of:
  outputs/week2_cnn_bbox/umap_cnn_by_region_panels_residential_boundary.png

Same ten metros, same title / legend / RdYlBu wash / \"LR P=0.5\" contour,
so the briefing can compare CNN (building) vs AE (pixel) separability.

Uses cached coords from outputs/week4_top3_umap/umap_coords.parquet (no refit).

Usage:
  .venv/bin/python -u scripts/plot_week4_region_umap_boundary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COORDS = ROOT / "outputs" / "week4_top3_umap" / "umap_coords.parquet"
OUT = ROOT / "outputs" / "week4_region_umap"
FIG = OUT / "figures"
SEED = 42
# Cap scatter density per metro; LR fit / F1 / % wrong still use all points.
MAX_PLOT_POINTS = 400

REGIONS = [
    "Atlanta_SE",
    "Boston_NE",
    "Chicago_MW",
    "Denver_W",
    "Houston_S",
    "LA_W",
    "NYC_NE",
    "Phoenix_SW",
    "SF_W",
    "Seattle_NW",
]

RES_COLORS = {1: "#2c7fb8", 0: "#d95f0e"}
RES_LABELS = {1: "residential", 0: "non-residential"}
MUTED = "#B0B0B0"

PANELS = [
    {
        "key": "gt",
        "y_col": "y",
        "title": "Week 4 AE UMAP by region · residential vs non-residential + 2D LR decision boundary",
        "out_name": "umap_gt_by_region_panels_residential_boundary.png",
        "indiv_dir": "umap_gt_region_panels",
    },
    {
        "key": "pred_balanced_train",
        "y_col": "pred_balanced_train",
        "title": "Week 4 AE UMAP by region · W4 LR predictions + 2D LR decision boundary",
        "out_name": "umap_pred_w4lr_by_region_panels_residential_boundary.png",
        "indiv_dir": "umap_pred_w4lr_region_panels",
    },
]


def sample_for_display(
    df: pd.DataFrame,
    n_max: int = MAX_PLOT_POINTS,
    strata: pd.Series | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Stratified subsample for plotting only (metrics stay on full data)."""
    if len(df) <= n_max:
        return df
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.choice(len(df), size=n_max, replace=False)
        return df.iloc[np.sort(idx)]

    strata = strata.reindex(df.index)
    parts = []
    counts = strata.value_counts()
    # Proportional allocation with at least 1 per present stratum when possible
    alloc = (counts / counts.sum() * n_max).round().astype(int).clip(lower=1)
    # Fix rounding so sum == n_max
    while int(alloc.sum()) > n_max:
        alloc.loc[alloc.idxmax()] -= 1
    while int(alloc.sum()) < n_max:
        alloc.loc[alloc.idxmax()] += 1
    for key, k in alloc.items():
        block = df.loc[strata == key]
        take = min(int(k), len(block))
        if take <= 0:
            continue
        if take >= len(block):
            parts.append(block)
        else:
            pick = rng.choice(len(block), size=take, replace=False)
            parts.append(block.iloc[np.sort(pick)])
    return pd.concat(parts) if parts else df.iloc[:0]


def fit_umap_lr_boundary(sub: pd.DataFrame, y_col: str):
    X2 = sub[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    yr = sub[y_col].to_numpy(dtype=int)
    if len(np.unique(yr)) < 2 or min(int((yr == 0).sum()), int((yr == 1).sum())) < 5:
        return None
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )
    pipe.fit(X2, yr)
    f1 = float(f1_score(yr, pipe.predict(X2), average="macro"))
    return pipe, f1


def plot_region_resi_boundary(
    ax,
    sub: pd.DataFrame,
    title: str,
    y_col: str,
    s: int = 12,
    legend: bool = True,
    plot_df: pd.DataFrame | None = None,
):
    """Week-2 style: RdYlBu wash, LR P=0.5. Fit on full `sub`; scatter from `plot_df`."""
    draw = plot_df if plot_df is not None else sub
    for val in (1, 0):
        m = draw[y_col].astype(int) == val
        if not m.any():
            continue
        ax.scatter(
            draw.loc[m, "umap_1"],
            draw.loc[m, "umap_2"],
            s=s,
            alpha=0.75,
            c=RES_COLORS[val],
            edgecolors="none",
            zorder=2,
            rasterized=True,
        )

    fitted = fit_umap_lr_boundary(sub, y_col=y_col)
    f1 = None
    if fitted is not None:
        pipe, f1 = fitted
        x0, x1 = float(sub["umap_1"].min()), float(sub["umap_1"].max())
        y0, y1 = float(sub["umap_2"].min()), float(sub["umap_2"].max())
        pad_x = 0.05 * (x1 - x0 + 1e-6)
        pad_y = 0.05 * (y1 - y0 + 1e-6)
        xx, yy = np.meshgrid(
            np.linspace(x0 - pad_x, x1 + pad_x, 300),
            np.linspace(y0 - pad_y, y1 + pad_y, 300),
        )
        zz = pipe.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(
            xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdYlBu", alpha=0.22, zorder=0
        )
        cs = ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.8, zorder=3)
        ax.clabel(cs, fmt={0.5: "LR P=0.5"}, inline=True, fontsize=7)
        title = f"{title} · in-UMAP LR F1={f1:.2f}"

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=RES_COLORS[1],
                markersize=7,
                label="residential",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=RES_COLORS[0],
                markersize=7,
                label="non-residential",
            ),
            Line2D([0], [0], color="k", lw=1.8, label="LR boundary"),
        ]
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")
    return f1


def write_panel_grid(coords: pd.DataFrame, panel: dict) -> pd.DataFrame:
    y_col = panel["y_col"]
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.4 * nrows), squeeze=False
    )
    rows = []
    for ax, region in zip(axes.ravel(), REGIONS):
        sub = coords.loc[coords["region"] == region].dropna(
            subset=[y_col, "umap_1", "umap_2"]
        )
        sub = sub.copy()
        sub[y_col] = sub[y_col].astype(int)
        draw = sample_for_display(
            sub, strata=sub[y_col], seed=SEED + (sum(ord(c) for c in region) % 10_000)
        )
        f1 = plot_region_resi_boundary(
            ax, sub, region, y_col=y_col, s=10, legend=True, plot_df=draw
        )
        rows.append(
            {
                "region": region,
                "n": int(len(sub)),
                "n_plotted": int(len(draw)),
                "in_umap_lr_f1": f1,
                "y_col": y_col,
                "panel": panel["key"],
            }
        )

    for ax in axes.ravel()[len(REGIONS) :]:
        ax.axis("off")

    fig.suptitle(panel["title"], fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = FIG / panel["out_name"]
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    indiv = FIG / panel["indiv_dir"]
    indiv.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        sub = coords.loc[coords["region"] == region].dropna(
            subset=[y_col, "umap_1", "umap_2"]
        )
        sub = sub.copy()
        sub[y_col] = sub[y_col].astype(int)
        draw = sample_for_display(
            sub, strata=sub[y_col], seed=SEED + (sum(ord(c) for c in region) % 10_000)
        )
        fig_i, ax_i = plt.subplots(figsize=(5.4, 4.8))
        plot_region_resi_boundary(
            ax_i, sub, region, y_col=y_col, s=14, legend=True, plot_df=draw
        )
        fig_i.tight_layout()
        fig_i.savefig(
            indiv / f"umap_{region}_residential_boundary.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_i)

    return pd.DataFrame(rows)


def _wrong_side_stats(sub: pd.DataFrame, y_col: str):
    """Fit 2D LR; return (pipe, f1, wrong_mask, pct_wrong) or None."""
    fitted = fit_umap_lr_boundary(sub, y_col=y_col)
    if fitted is None:
        return None
    pipe, f1 = fitted
    y_true = sub[y_col].to_numpy(dtype=int)
    y_hat = pipe.predict(sub[["umap_1", "umap_2"]].to_numpy(dtype=np.float64))
    wrong = y_hat != y_true
    pct = 100.0 * float(wrong.mean())
    return pipe, f1, wrong, pct


def plot_region_wrong_side(
    ax, sub: pd.DataFrame, region: str, y_col: str, s_ok: int = 6, s_err: int = 14
):
    """Mute correct points; accent wrong-side by true class; label % wrong.

    Fit / % wrong on full `sub`. Scatter uses a stratified subsample.
    """
    stats = _wrong_side_stats(sub, y_col)
    if stats is None:
        ax.set_title(region, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    pipe, f1, wrong, pct = stats
    sub = sub.copy()
    sub["_wrong"] = wrong
    # Stratify display by (wrong, class) so errors stay visible after downsampling
    strata = sub["_wrong"].astype(int).astype(str) + "_" + sub[y_col].astype(int).astype(str)
    draw = sample_for_display(
        sub,
        strata=strata,
        seed=SEED + (sum(ord(c) for c in region) % 10_000),
    )
    x = draw["umap_1"].to_numpy()
    y = draw["umap_2"].to_numpy()
    y_true = draw[y_col].to_numpy(dtype=int)
    draw_wrong = draw["_wrong"].to_numpy(dtype=bool)
    ok = ~draw_wrong

    if ok.any():
        ax.scatter(
            x[ok],
            y[ok],
            s=s_ok,
            alpha=0.35,
            c=MUTED,
            edgecolors="none",
            zorder=1,
            rasterized=True,
        )
    for val in (1, 0):
        m = draw_wrong & (y_true == val)
        if not m.any():
            continue
        ax.scatter(
            x[m],
            y[m],
            s=s_err,
            alpha=0.85,
            c=RES_COLORS[val],
            edgecolors="none",
            zorder=2,
            rasterized=True,
        )

    x0, x1 = float(sub["umap_1"].min()), float(sub["umap_1"].max())
    y0, y1 = float(sub["umap_2"].min()), float(sub["umap_2"].max())
    pad_x = 0.05 * (x1 - x0 + 1e-6)
    pad_y = 0.05 * (y1 - y0 + 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(x0 - pad_x, x1 + pad_x, 300),
        np.linspace(y0 - pad_y, y1 + pad_y, 300),
    )
    zz = pipe.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    ax.contourf(
        xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdYlBu", alpha=0.22, zorder=0
    )
    cs = ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.6, zorder=3)
    ax.clabel(cs, fmt={0.5: "LR P=0.5"}, inline=True, fontsize=7)

    n_err = int(wrong.sum())
    ax.set_title(f"{region}  ·  {pct:.0f}% wrong  (n={n_err})", fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
    return {
        "region": region,
        "n": int(len(sub)),
        "n_plotted": int(len(draw)),
        "n_wrong": n_err,
        "pct_wrong": pct,
        "f1": f1,
    }


def write_wrong_side_figures(coords: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    """Two figures: (1) error-spotlight panels (2) sorted %wrong bars."""
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.4 * nrows), squeeze=False
    )
    rows = []
    for ax, region in zip(axes.ravel(), REGIONS):
        sub = coords.loc[coords["region"] == region].dropna(
            subset=[y_col, "umap_1", "umap_2"]
        ).copy()
        sub[y_col] = sub[y_col].astype(int)
        stats = plot_region_wrong_side(ax, sub, region, y_col=y_col)
        if stats is not None:
            rows.append(stats)

    for ax in axes.ravel()[len(REGIONS) :]:
        ax.axis("off")

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=MUTED, markersize=6,
            label="correct side",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=RES_COLORS[1], markersize=8,
            label="wrong · residential",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=RES_COLORS[0], markersize=8,
            label="wrong · non-residential",
        ),
        Line2D([0], [0], color="k", lw=1.6, label="LR P=0.5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    # Active title states the finding; subtitle gives context
    fig.suptitle(
        "NYC puts nearly half of AE pixels on the wrong side of a 2D LR cut; Phoenix ~20%",
        fontsize=13,
        y=1.04,
    )
    fig.text(
        0.5,
        0.995,
        f"Grey = correct side; color = true class of wrong-side pixels. "
        f"Blue/red wash = P(residential). Scatter ≤{MAX_PLOT_POINTS}/metro (stratified); "
        f"% wrong from full n.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_panels = FIG / "umap_gt_wrong_side_by_region.png"
    fig.savefig(out_panels, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_panels}")

    df = pd.DataFrame(rows)
    df = df.sort_values("pct_wrong", ascending=True)
    fig_b, ax_b = plt.subplots(figsize=(7.2, 4.6))
    y_pos = np.arange(len(df))
    # Highlight worst metro; mute the rest (color has a job)
    colors = ["#D55E00" if r == df["region"].iloc[-1] else "#4C78A8" for r in df["region"]]
    ax_b.barh(y_pos, df["pct_wrong"], color=colors, height=0.7)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(df["region"].tolist())
    ax_b.set_xlabel("% of pixels on wrong side of 2D LR boundary")
    ax_b.set_xlim(0, max(50.0, float(df["pct_wrong"].max()) * 1.15))
    for i, (_, r) in enumerate(df.iterrows()):
        ax_b.text(
            float(r["pct_wrong"]) + 0.6,
            i,
            f"{r['pct_wrong']:.0f}%  (n={int(r['n_wrong']):,})",
            va="center",
            fontsize=9,
            color="#222222",
        )
    ax_b.set_title(
        "Wrong-side rate by metro (sorted) — NYC worst, Phoenix best",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    fig_b.tight_layout()
    out_bars = FIG / "umap_gt_wrong_side_rate_by_region.png"
    fig_b.savefig(out_bars, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig_b)
    print(f"Wrote {out_bars}")

    df.to_csv(OUT / "wrong_side_by_region.csv", index=False)
    return df


def main() -> None:
    if not COORDS.exists():
        raise SystemExit(
            f"Missing {COORDS}. Run scripts/plot_week4_top3_umap.py first."
        )

    FIG.mkdir(parents=True, exist_ok=True)
    coords = pd.read_parquet(COORDS)
    missing = [r for r in REGIONS if r not in set(coords["region"].unique())]
    if missing:
        raise SystemExit(f"Coords missing regions: {missing}")

    all_rows = []
    for panel in PANELS:
        if panel["y_col"] not in coords.columns:
            print(f"Skip {panel['key']}: column {panel['y_col']} absent")
            continue
        df = write_panel_grid(coords, panel)
        all_rows.append(df)
        mean_f1 = float(df["in_umap_lr_f1"].mean())
        print(
            f"  {panel['key']}: mean in-UMAP LR F1={mean_f1:.3f}  "
            f"min={df['in_umap_lr_f1'].min():.3f}  max={df['in_umap_lr_f1'].max():.3f}"
        )

    f1_df = pd.concat(all_rows, ignore_index=True)
    f1_df.to_csv(OUT / "in_umap_lr_f1.csv", index=False)

    wrong_df = write_wrong_side_figures(coords, y_col="y")

    gt = f1_df.loc[f1_df["panel"] == "gt", ["region", "in_umap_lr_f1"]].rename(
        columns={"in_umap_lr_f1": "w4_ae_gt_in_umap_f1"}
    )
    pred = f1_df.loc[
        f1_df["panel"] == "pred_balanced_train", ["region", "in_umap_lr_f1"]
    ].rename(columns={"in_umap_lr_f1": "w4_ae_pred_in_umap_f1"})
    cmp = gt.merge(pred, on="region", how="outer")
    cmp = cmp.merge(
        wrong_df[["region", "pct_wrong", "n_wrong"]].rename(
            columns={"pct_wrong": "w4_ae_gt_pct_wrong", "n_wrong": "w4_ae_gt_n_wrong"}
        ),
        on="region",
        how="left",
    )

    # Recompute Week 2 CNN in-UMAP F1 from cached coords + y_res (same 2D LR recipe)
    cnn_coords_path = (
        ROOT / "outputs" / "week2_cnn_bbox" / "cnn_embeddings_umap_coords.parquet"
    )
    y_res_path = ROOT / "outputs" / "week2_cnn_bbox" / "cnn_oof_predictions.parquet"
    if cnn_coords_path.exists() and y_res_path.exists():
        cnn = pd.read_parquet(cnn_coords_path)
        yres = pd.read_parquet(y_res_path, columns=["id", "y_res"])
        cnn = cnn.merge(yres, on="id", how="inner")
        cnn["y"] = cnn["y_res"].astype(int)
        w2_rows = []
        for region in REGIONS:
            sub = cnn.loc[cnn["region"] == region].dropna(subset=["y", "umap_1", "umap_2"])
            if len(sub) == 0:
                continue
            fitted = fit_umap_lr_boundary(sub, "y")
            w2_rows.append(
                {
                    "region": region,
                    "w2_cnn_in_umap_f1": None if fitted is None else fitted[1],
                    "w2_n": int(len(sub)),
                }
            )
        if w2_rows:
            cmp = cmp.merge(pd.DataFrame(w2_rows), on="region", how="left")

    cmp = cmp.set_index("region").reindex(REGIONS).reset_index()
    cmp.to_csv(OUT / "w2_vs_w4_in_umap_f1.csv", index=False)
    print(cmp.round(3).to_string(index=False))

    summary = {
        "name": "week4_region_umap_w2style",
        "coords": str(COORDS.relative_to(ROOT)),
        "regions": REGIONS,
        "n_pixels": int(len(coords)),
        "max_plot_points_per_region": MAX_PLOT_POINTS,
        "figures": {
            **{p["key"]: str((FIG / p["out_name"]).relative_to(ROOT)) for p in PANELS},
            "wrong_side_panels": str(
                (FIG / "umap_gt_wrong_side_by_region.png").relative_to(ROOT)
            ),
            "wrong_side_bars": str(
                (FIG / "umap_gt_wrong_side_rate_by_region.png").relative_to(ROOT)
            ),
        },
        "week2_reference": "outputs/week2_cnn_bbox/umap_cnn_by_region_panels_residential_boundary.png",
        "in_umap_lr_f1_gt_mean": float(
            f1_df.loc[f1_df["panel"] == "gt", "in_umap_lr_f1"].mean()
        ),
        "in_umap_lr_f1_pred_mean": float(
            f1_df.loc[f1_df["panel"] == "pred_balanced_train", "in_umap_lr_f1"].mean()
        )
        if (f1_df["panel"] == "pred_balanced_train").any()
        else None,
        "pct_wrong_mean": float(wrong_df["pct_wrong"].mean()),
        "pct_wrong_max_region": str(
            wrong_df.loc[wrong_df["pct_wrong"].idxmax(), "region"]
        ),
        "pct_wrong_max": float(wrong_df["pct_wrong"].max()),
        "pct_wrong_min_region": str(
            wrong_df.loc[wrong_df["pct_wrong"].idxmin(), "region"]
        ),
        "pct_wrong_min": float(wrong_df["pct_wrong"].min()),
    }
    if "w2_cnn_in_umap_f1" in cmp.columns:
        summary["in_umap_lr_f1_w2_cnn_mean"] = float(cmp["w2_cnn_in_umap_f1"].mean())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
