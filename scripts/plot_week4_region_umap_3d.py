#!/usr/bin/env python3
"""3D UMAP + in-UMAP LR plane for Week 4 AE pixels (same metros as 2D panels).

Fits UMAP(n_components=3) on the same pixel set as outputs/week4_top3_umap,
draws wrong-side spotlight panels with a P=0.5 decision plane, and writes a
Plotly HTML for interactive rotation.

Usage:
  .venv/bin/python -u scripts/plot_week4_region_umap_3d.py
  .venv/bin/python -u scripts/plot_week4_region_umap_3d.py --refit-umap
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COORDS_2D = ROOT / "outputs" / "week4_top3_umap" / "umap_coords.parquet"
# Embeddings live on the full pixel GT (UMAP set is not limited to balanced train)
PIXELS = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_region_umap_3d"
FIG = OUT / "figures"
SEED = 42
BANDS = [f"A{i:02d}" for i in range(64)]
MAX_PLOT_POINTS = 400
N_JOBS = max(1, os.cpu_count() or 1)

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
MUTED = "#B0B0B0"


def sample_for_display(
    df: pd.DataFrame,
    n_max: int = MAX_PLOT_POINTS,
    strata: pd.Series | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    if len(df) <= n_max:
        return df
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.choice(len(df), size=n_max, replace=False)
        return df.iloc[np.sort(idx)]
    strata = strata.reindex(df.index)
    counts = strata.value_counts()
    alloc = (counts / counts.sum() * n_max).round().astype(int).clip(lower=1)
    while int(alloc.sum()) > n_max:
        alloc.loc[alloc.idxmax()] -= 1
    while int(alloc.sum()) < n_max:
        alloc.loc[alloc.idxmax()] += 1
    parts = []
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


def fit_umap_3d(X: np.ndarray) -> np.ndarray:
    print(
        f"Fitting 3D UMAP on {X.shape[0]:,}×{X.shape[1]} "
        f"(n_jobs={N_JOBS}) ...",
        flush=True,
    )
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=3,
        metric="cosine",
        random_state=None,
        n_jobs=N_JOBS,
        verbose=False,
    )
    Z = reducer.fit_transform(X)
    print(f"  UMAP done in {time.time() - t0:.1f}s", flush=True)
    return Z.astype(np.float64)


def fit_lr_3d(sub: pd.DataFrame, y_col: str = "y"):
    X = sub[["umap_1", "umap_2", "umap_3"]].to_numpy(dtype=np.float64)
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
    pipe.fit(X, yr)
    f1 = float(f1_score(yr, pipe.predict(X), average="macro"))
    y_hat = pipe.predict(X)
    wrong = y_hat != yr
    pct = 100.0 * float(wrong.mean())
    return pipe, f1, wrong, pct


def _decision_plane_mesh(pipe, sub: pd.DataFrame, n: int = 18):
    """Mesh for P=0.5 plane in original UMAP coords (after undoing StandardScaler)."""
    sc = pipe.named_steps["sc"]
    lr = pipe.named_steps["lr"]
    # In scaled space: w·z + b = 0  →  z' = StandardScaler(x)
    # w · ((x - mean) / scale) + b = 0
    # (w / scale) · x + (b - w·mean/scale) = 0
    w = lr.coef_.ravel()
    b = float(lr.intercept_[0])
    mean = sc.mean_
    scale = sc.scale_
    a = w / scale
    d = b - float(np.dot(w, mean / scale))
    # a1 x + a2 y + a3 z + d = 0  →  solve for z if |a3| large enough
    x0, x1 = float(sub["umap_1"].min()), float(sub["umap_1"].max())
    y0, y1 = float(sub["umap_2"].min()), float(sub["umap_2"].max())
    z0, z1 = float(sub["umap_3"].min()), float(sub["umap_3"].max())
    pad = 0.05
    xx, yy = np.meshgrid(
        np.linspace(x0 - pad * (x1 - x0), x1 + pad * (x1 - x0), n),
        np.linspace(y0 - pad * (y1 - y0), y1 + pad * (y1 - y0), n),
    )
    if abs(a[2]) < 1e-8:
        return None
    zz = -(a[0] * xx + a[1] * yy + d) / a[2]
    # Clip plane to data z-range (±pad) so it doesn't dominate the view
    z_lo, z_hi = z0 - pad * (z1 - z0), z1 + pad * (z1 - z0)
    zz = np.clip(zz, z_lo, z_hi)
    return xx, yy, zz


def plot_region_3d(ax, sub: pd.DataFrame, region: str, y_col: str = "y"):
    stats = fit_lr_3d(sub, y_col=y_col)
    if stats is None:
        ax.set_title(region, fontsize=9)
        return None
    pipe, f1, wrong, pct = stats
    sub = sub.copy()
    sub["_wrong"] = wrong
    strata = (
        sub["_wrong"].astype(int).astype(str) + "_" + sub[y_col].astype(int).astype(str)
    )
    draw = sample_for_display(
        sub, strata=strata, seed=SEED + (sum(ord(c) for c in region) % 10_000)
    )
    x = draw["umap_1"].to_numpy()
    y = draw["umap_2"].to_numpy()
    z = draw["umap_3"].to_numpy()
    y_true = draw[y_col].to_numpy(dtype=int)
    dw = draw["_wrong"].to_numpy(dtype=bool)

    ok = ~dw
    if ok.any():
        ax.scatter(
            x[ok], y[ok], z[ok], s=6, alpha=0.25, c=MUTED, depthshade=False, linewidths=0
        )
    for val in (1, 0):
        m = dw & (y_true == val)
        if not m.any():
            continue
        ax.scatter(
            x[m],
            y[m],
            z[m],
            s=12,
            alpha=0.85,
            c=RES_COLORS[val],
            depthshade=False,
            linewidths=0,
        )

    mesh = _decision_plane_mesh(pipe, sub)
    if mesh is not None:
        xx, yy, zz = mesh
        ax.plot_surface(xx, yy, zz, alpha=0.18, color="#444444", linewidth=0, shade=False)

    n_err = int(wrong.sum())
    ax.set_title(f"{region}  ·  {pct:.0f}% wrong  (n={n_err})", fontsize=9, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    return {
        "region": region,
        "n": int(len(sub)),
        "n_plotted": int(len(draw)),
        "n_wrong": n_err,
        "pct_wrong": pct,
        "in_umap_lr_f1": f1,
    }


def write_matplotlib_panels(coords: pd.DataFrame) -> pd.DataFrame:
    fig = plt.figure(figsize=(3.6 * 5, 3.5 * 2))
    rows = []
    for i, region in enumerate(REGIONS):
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        sub = coords.loc[coords["region"] == region].dropna(
            subset=["y", "umap_1", "umap_2", "umap_3"]
        ).copy()
        sub["y"] = sub["y"].astype(int)
        stats = plot_region_3d(ax, sub, region)
        if stats is not None:
            rows.append(stats)

    rows_df = pd.DataFrame(rows)
    worst = rows_df.loc[rows_df["pct_wrong"].idxmax()]
    best = rows_df.loc[rows_df["pct_wrong"].idxmin()]
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED, markersize=6, label="correct side"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RES_COLORS[1], markersize=8, label="wrong · residential"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RES_COLORS[0], markersize=8, label="wrong · non-residential"),
        Line2D([0], [0], color="#444444", lw=6, alpha=0.4, label="LR P=0.5 plane"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"3D AE UMAP: {worst['region']} messiest ({worst['pct_wrong']:.0f}% wrong); "
        f"{best['region']} cleanest ({best['pct_wrong']:.0f}%)",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"Same pixels as 2D panels. Grey = correct side of 3D LR cut; color = true class of wrong-side. "
        f"Scatter ≤{MAX_PLOT_POINTS}/metro; % wrong from full n.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    out = FIG / "umap3d_gt_wrong_side_by_region.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    return rows_df


def write_bars(df: pd.DataFrame) -> None:
    d = df.sort_values("pct_wrong", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_pos = np.arange(len(d))
    colors = ["#D55E00" if r == d["region"].iloc[-1] else "#4C78A8" for r in d["region"]]
    ax.barh(y_pos, d["pct_wrong"], color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["region"].tolist())
    ax.set_xlabel("% of pixels on wrong side of 3D LR boundary")
    ax.set_xlim(0, max(50.0, float(d["pct_wrong"].max()) * 1.15))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(
            float(r["pct_wrong"]) + 0.6,
            i,
            f"{r['pct_wrong']:.0f}%  (n={int(r['n_wrong']):,})",
            va="center",
            fontsize=9,
        )
    ax.set_title(
        "3D wrong-side rate by metro (sorted) — NYC worst, Phoenix best",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "umap3d_gt_wrong_side_rate_by_region.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def write_plotly_html(coords: pd.DataFrame) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skip interactive HTML", flush=True)
        return None

    fig = make_subplots(
        rows=2,
        cols=5,
        specs=[[{"type": "scatter3d"}] * 5] * 2,
        subplot_titles=REGIONS,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    for i, region in enumerate(REGIONS):
        r, c = divmod(i, 5)
        sub = coords.loc[coords["region"] == region].copy()
        sub["y"] = sub["y"].astype(int)
        stats = fit_lr_3d(sub)
        if stats is None:
            continue
        pipe, f1, wrong, pct = stats
        sub["_wrong"] = wrong
        strata = (
            sub["_wrong"].astype(int).astype(str)
            + "_"
            + sub["y"].astype(int).astype(str)
        )
        draw = sample_for_display(
            sub, strata=strata, seed=SEED + (sum(ord(c) for c in region) % 10_000)
        )
        ok = ~draw["_wrong"].to_numpy(dtype=bool)
        fig.add_trace(
            go.Scatter3d(
                x=draw.loc[ok, "umap_1"],
                y=draw.loc[ok, "umap_2"],
                z=draw.loc[ok, "umap_3"],
                mode="markers",
                marker=dict(size=2, color=MUTED, opacity=0.35),
                name="correct",
                showlegend=(i == 0),
                hoverinfo="skip",
            ),
            row=r + 1,
            col=c + 1,
        )
        for val, name, color in (
            (1, "wrong · res", RES_COLORS[1]),
            (0, "wrong · nres", RES_COLORS[0]),
        ):
            m = draw["_wrong"].to_numpy(dtype=bool) & (draw["y"].to_numpy(dtype=int) == val)
            if not m.any():
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=draw.loc[m, "umap_1"],
                    y=draw.loc[m, "umap_2"],
                    z=draw.loc[m, "umap_3"],
                    mode="markers",
                    marker=dict(size=3, color=color, opacity=0.9),
                    name=name,
                    showlegend=(i == 0),
                    hoverinfo="skip",
                ),
                row=r + 1,
                col=c + 1,
            )
        mesh = _decision_plane_mesh(pipe, sub, n=12)
        if mesh is not None:
            xx, yy, zz = mesh
            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=zz,
                    opacity=0.25,
                    colorscale=[[0, "#888"], [1, "#888"]],
                    showscale=False,
                    name="P=0.5",
                    showlegend=(i == 0),
                ),
                row=r + 1,
                col=c + 1,
            )
        # subplot title with metric
        fig.layout.annotations[i].update(
            text=f"{region} · {pct:.0f}% wrong", font=dict(size=11)
        )

    fig.update_layout(
        title=dict(
            text="3D AE UMAP by region · wrong-side of LR P=0.5 plane (rotate in browser)",
            x=0.5,
        ),
        height=780,
        width=1600,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=0, x=0.5, xanchor="center"),
    )
    for trace_key in [k for k in fig.layout if k.startswith("scene")]:
        fig.layout[trace_key].update(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            aspectmode="cube",
        )
    out = FIG / "umap3d_gt_wrong_side_by_region.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")
    return out


def load_or_fit_coords(refit: bool) -> pd.DataFrame:
    coords_path = OUT / "umap3d_coords.parquet"
    if coords_path.exists() and not refit:
        print(f"Reusing {coords_path.relative_to(ROOT)}", flush=True)
        return pd.read_parquet(coords_path)

    if not COORDS_2D.exists():
        raise SystemExit(f"Missing {COORDS_2D}; run plot_week4_top3_umap.py first")

    meta = pd.read_parquet(
        COORDS_2D, columns=["pixel_id", "region", "stratum", "y", "pixel_lon", "pixel_lat"]
    )
    print(f"Loading AE bands for {len(meta):,} pixels ...", flush=True)
    bands = pd.read_parquet(PIXELS, columns=["pixel_id"] + BANDS)
    df = meta.merge(bands, on="pixel_id", how="inner")
    if len(df) != len(meta):
        print(f"  warning: band join {len(meta):,} → {len(df):,}", flush=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    # random_state locks umap to n_jobs=1; seed via numpy for reproducibility + parallel
    np.random.seed(SEED)
    Z = fit_umap_3d(X)
    out = df[["pixel_id", "region", "stratum", "y", "pixel_lon", "pixel_lat"]].copy()
    out["umap_1"] = Z[:, 0]
    out["umap_2"] = Z[:, 1]
    out["umap_3"] = Z[:, 2]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(coords_path, index=False)
    print(f"Wrote {coords_path}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refit-umap", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"CPUs={N_JOBS}", flush=True)

    coords = load_or_fit_coords(refit=args.refit_umap)
    missing = [r for r in REGIONS if r not in set(coords["region"].unique())]
    if missing:
        raise SystemExit(f"Missing regions: {missing}")

    rows = write_matplotlib_panels(coords)
    write_bars(rows)
    rows.to_csv(OUT / "wrong_side_by_region_3d.csv", index=False)
    html = write_plotly_html(coords)

    # Compare to 2D if available
    cmp_2d = ROOT / "outputs" / "week4_region_umap" / "wrong_side_by_region.csv"
    if cmp_2d.exists():
        d2 = pd.read_csv(cmp_2d)[["region", "pct_wrong", "f1"]].rename(
            columns={"pct_wrong": "pct_wrong_2d", "f1": "f1_2d"}
        )
        d3 = rows[["region", "pct_wrong", "in_umap_lr_f1"]].rename(
            columns={"pct_wrong": "pct_wrong_3d", "in_umap_lr_f1": "f1_3d"}
        )
        cmp = d2.merge(d3, on="region", how="outer")
        cmp = cmp.set_index("region").reindex(REGIONS).reset_index()
        cmp.to_csv(OUT / "umap_2d_vs_3d_wrong_side.csv", index=False)
        print(cmp.round(3).to_string(index=False))

    summary = {
        "name": "week4_region_umap_3d",
        "n_pixels": int(len(coords)),
        "max_plot_points_per_region": MAX_PLOT_POINTS,
        "umap": {
            "n_neighbors": 30,
            "min_dist": 0.1,
            "n_components": 3,
            "metric": "cosine",
            "seed": SEED,
            "n_jobs": N_JOBS,
        },
        "pct_wrong_mean": float(rows["pct_wrong"].mean()),
        "pct_wrong_max_region": str(rows.loc[rows["pct_wrong"].idxmax(), "region"]),
        "pct_wrong_max": float(rows["pct_wrong"].max()),
        "pct_wrong_min_region": str(rows.loc[rows["pct_wrong"].idxmin(), "region"]),
        "pct_wrong_min": float(rows["pct_wrong"].min()),
        "in_umap_lr_f1_mean": float(rows["in_umap_lr_f1"].mean()),
        "figures": {
            "panels": str((FIG / "umap3d_gt_wrong_side_by_region.png").relative_to(ROOT)),
            "bars": str(
                (FIG / "umap3d_gt_wrong_side_rate_by_region.png").relative_to(ROOT)
            ),
            "html": None
            if html is None
            else str(html.relative_to(ROOT)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {OUT / 'summary.json'}  ({summary['elapsed_s']}s)")


if __name__ == "__main__":
    main()
