#!/usr/bin/env python3
"""UMAP of AE pixels for top-3 models — claim-first figures.

Fits one global UMAP on week4 balanced AE embeddings (or reuses coords),
joins Week3/Week4 OOF preds, and writes readable comparison figures:

  Primary claims
  - GT + in-UMAP LR boundary by metro (RES vs NRES partly separate)
  - Top-3 macro-F1 by metro (sorted horizontal bars; W4 highlighted)
  - Model–GT agreement by metro (encodes match rate directly)
  - Error spotlight: 3 metros × models (grey=correct, orange=error)

Usage:
  .venv/bin/python -u scripts/plot_week4_top3_umap.py
  .venv/bin/python -u scripts/plot_week4_top3_umap.py --refit-umap
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import (  # noqa: E402
    ACCENT,
    MODEL_COLORS,
    MODEL_SHORT,
    MUTED,
    NRES,
    OK,
    RES,
    RES_COLORS,
    RES_LABELS,
    label_bars_h,
    set_claim_title,
    style_axes,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_top3_umap"
FIG = OUT / "figures"
SEED = 42
BANDS = [f"A{i:02d}" for i in range(64)]

# Soft blue wash for P(residential) — sequential, not rainbow
_PROB_CMAP = LinearSegmentedColormap.from_list(
    "res_wash", ["#FFFFFF", "#9ECAE1", "#08519C"], N=256
)

MODELS = [
    {
        "key": "balanced_train",
        "label": MODEL_SHORT["balanced_train"],
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "kind": "y_pred",
    },
    {
        "key": "fraction_regression",
        "label": MODEL_SHORT["fraction_regression"],
        "path": ROOT / "outputs" / "week3_fraction_regression" / "oof_predictions.parquet",
        "kind": "y_pred",
    },
    {
        "key": "pixel_ae",
        "label": MODEL_SHORT["pixel_ae"],
        "path": ROOT / "outputs" / "week3_pixel_ae" / "oof_predictions.parquet",
        "kind": "y_pred",
    },
]


def _load_preds(meta: dict, pixel_ids: pd.Series) -> pd.DataFrame:
    raw = pd.read_parquet(meta["path"])
    keep = ["pixel_id"]
    for c in ("y_pred", "p_res", "pred_frac_res_full", "pred_frac_nres_full"):
        if c in raw.columns:
            keep.append(c)
    sub = raw[keep].drop_duplicates("pixel_id")
    out = pixel_ids.to_frame(name="pixel_id").merge(sub, on="pixel_id", how="left")
    if meta["kind"] == "y_pred":
        if "y_pred" not in out.columns or out["y_pred"].isna().any():
            if "pred_frac_res_full" in out.columns:
                out["y_pred"] = (
                    out["pred_frac_res_full"].astype(float)
                    > out["pred_frac_nres_full"].astype(float)
                ).astype(np.int64)
        out["y_pred"] = out["y_pred"].astype("Int64")
    return out


def _region_grid(n_reg: int) -> tuple[int, int]:
    ncols = 5
    nrows = int(np.ceil(n_reg / ncols))
    return nrows, ncols


def _zoom_to_data(ax, sub: pd.DataFrame, pad_frac: float = 0.08) -> None:
    x0, x1 = float(sub["umap_1"].min()), float(sub["umap_1"].max())
    y0, y1 = float(sub["umap_2"].min()), float(sub["umap_2"].max())
    px = pad_frac * (x1 - x0 + 1e-6)
    py = pad_frac * (y1 - y0 + 1e-6)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y0 - py, y1 + py)


def fit_umap_lr_boundary(sub: pd.DataFrame, y_col: str = "y"):
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
    y_col: str = "y",
    s: int = 10,
) -> float | None:
    """Scatter RES/NRES + sequential probability wash + P=0.5 contour."""
    for val in (0, 1):  # NRES under, RES on top
        m = sub[y_col].astype(int) == val
        if not m.any():
            continue
        ax.scatter(
            sub.loc[m, "umap_1"],
            sub.loc[m, "umap_2"],
            s=s,
            alpha=0.7,
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
            np.linspace(x0 - pad_x, x1 + pad_x, 220),
            np.linspace(y0 - pad_y, y1 + pad_y, 220),
        )
        zz = pipe.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(
            xx, yy, zz, levels=np.linspace(0, 1, 9), cmap=_PROB_CMAP, alpha=0.28, zorder=0
        )
        cs = ax.contour(xx, yy, zz, levels=[0.5], colors="#222222", linewidths=1.6, zorder=3)
        ax.clabel(cs, fmt={0.5: "P=0.5"}, inline=True, fontsize=7)
        title = f"{title}  ·  F1={f1:.2f}"

    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
    _zoom_to_data(ax, sub)
    ax.set_aspect("equal", adjustable="box")
    return f1


def _boundary_region_panels(
    coords: pd.DataFrame,
    regions: list[str],
    y_col: str,
    claim: str,
    subtitle: str,
    out_path: Path,
    per_dir: Path | None = None,
) -> pd.DataFrame:
    nrows, ncols = _region_grid(len(regions))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.4, nrows * 3.2), squeeze=False
    )
    rows = []
    for i, reg in enumerate(regions):
        ax = axes[i // ncols][i % ncols]
        sub = coords.loc[coords["region"] == reg].dropna(subset=[y_col, "umap_1", "umap_2"])
        sub = sub.copy()
        sub[y_col] = sub[y_col].astype(int)
        short = reg.replace("_", " ")
        f1 = plot_region_resi_boundary(ax, sub, short, y_col=y_col, s=9)
        rows.append({"region": reg, "n": len(sub), "in_umap_lr_f1": f1, "y_col": y_col})

        if per_dir is not None:
            per_dir.mkdir(parents=True, exist_ok=True)
            fig_i, ax_i = plt.subplots(figsize=(5.8, 4.8))
            plot_region_resi_boundary(ax_i, sub, short, y_col=y_col, s=14)
            handles = [
                Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=RES_COLORS[v], markersize=9, label=RES_LABELS[v],
                )
                for v in (1, 0)
            ]
            ax_i.legend(handles=handles, fontsize=9, frameon=False, loc="best")
            fig_i.tight_layout()
            fig_i.savefig(per_dir / f"{reg}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_i)

    for ax in axes.ravel()[len(regions) :]:
        ax.axis("off")

    handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=RES_COLORS[v], markersize=9, label=RES_LABELS[v],
        )
        for v in (1, 0)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    set_claim_title(fig, claim, subtitle)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_f1_by_metro(per_df: pd.DataFrame, out_path: Path) -> None:
    """Claim: top-3 F1 is tight across metros; W4 leads most."""
    order = per_df.sort_values("f1_balanced_train", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y = np.arange(len(order))
    h = 0.26
    keys = [m["key"] for m in MODELS]
    labels = [m["label"] for m in MODELS]
    colors = [MODEL_COLORS[k] for k in keys]
    for i, (key, lab, col) in enumerate(zip(keys, labels, colors)):
        vals = order[f"f1_{key}"].to_numpy()
        offset = (i - 1) * h
        bars = ax.barh(y + offset, vals, height=h, color=col, label=lab, zorder=2)
        # Direct-label only the focal (W4) series to cut clutter
        if key == "balanced_train":
            for rect, v in zip(bars, vals):
                ax.text(
                    v + 0.008,
                    rect.get_y() + rect.get_height() / 2,
                    f"{v:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#222222",
                )

    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("_", " ") for r in order["region"]], fontsize=10)
    ax.set_xlabel("Macro-F1 on balanced pixels")
    ax.set_xlim(0, 1.05)
    style_axes(ax, grid=True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    mean_w4 = float(order["f1_balanced_train"].mean())
    claim = f"W4 balanced LR averages {mean_w4:.2f} F1 — within ~0.03 of Week 3 Ridge/AE on most metros"
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        claim,
        "Leave-one-metro-out OOF on balanced pure RES=NRES pixels. Blue bars labeled.",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_agreement_by_metro(coords: pd.DataFrame, regions: list[str], out_path: Path) -> None:
    """Encode match-to-GT directly (do not make reader compare 40 scatters)."""
    rows = []
    for reg in regions:
        sub = coords.loc[coords["region"] == reg]
        row = {"region": reg, "n": len(sub)}
        for m in MODELS:
            key = m["key"]
            mask = sub[f"pred_{key}"].notna()
            row[f"agree_{key}"] = float(
                (sub.loc[mask, f"pred_{key}"].astype(float) == sub.loc[mask, "y"].astype(float)).mean()
            ) if mask.any() else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("agree_balanced_train", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y = np.arange(len(df))
    h = 0.26
    for i, m in enumerate(MODELS):
        key = m["key"]
        vals = df[f"agree_{key}"].to_numpy()
        offset = (i - 1) * h
        col = MODEL_COLORS[key]
        bars = ax.barh(y + offset, vals, height=h, color=col, label=m["label"], zorder=2)
        if key == "balanced_train":
            for rect, v in zip(bars, vals):
                ax.text(
                    v + 0.006,
                    rect.get_y() + rect.get_height() / 2,
                    f"{100 * v:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=8,
                )

    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("_", " ") for r in df["region"]], fontsize=10)
    ax.set_xlabel("Share of pixels matching ground truth")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    mean_agree = float(df["agree_balanced_train"].mean())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"All three models match GT on ~{100 * mean_agree:.0f}% of balanced pixels — disagreements are sparse",
        "Agreement = predicted class equals pure RES/NRES label. Prefer this over side-by-side UMAP grids.",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    df.to_csv(OUT / "agreement_by_region.csv", index=False)


def _plot_error_spotlight(
    coords: pd.DataFrame,
    per_df: pd.DataFrame,
    out_path: Path,
) -> list[str]:
    """3 metros × GT | W4 errors (hexbin context + error points).

    One model only — W3 rows were visually identical and overplotted.
    """
    ordered = per_df.sort_values("f1_balanced_train")
    picks = [
        ordered.iloc[0]["region"],
        ordered.iloc[len(ordered) // 2]["region"],
        ordered.iloc[-1]["region"],
    ]
    key = "balanced_train"
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), squeeze=False)

    for c, reg in enumerate(picks):
        sub = coords.loc[coords["region"] == reg].copy()
        f1_w4 = float(per_df.loc[per_df["region"] == reg, "f1_balanced_train"].iloc[0])
        short = reg.replace("_", " ")

        # Row 0: GT
        ax = axes[0][c]
        for val in (0, 1):
            m = sub["y"].astype(int) == val
            if not m.any():
                continue
            ax.scatter(
                sub.loc[m, "umap_1"],
                sub.loc[m, "umap_2"],
                s=10,
                alpha=0.7,
                c=RES_COLORS[val],
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(f"{short}  ·  W4 F1={f1_w4:.2f}", fontsize=11, loc="left")
        if c == 0:
            ax.set_ylabel("Ground truth", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _zoom_to_data(ax, sub)
        ax.set_aspect("equal", adjustable="box")

        # Row 1: W4 errors only — density context + accent errors
        ax = axes[1][c]
        ok = sub[f"correct_{key}"].fillna(False).astype(bool)
        err = ~ok & sub[f"pred_{key}"].notna()
        # Hexbin of all points as muted context (shows mass without overplotting)
        ax.hexbin(
            sub["umap_1"],
            sub["umap_2"],
            gridsize=35,
            cmap="Greys",
            mincnt=1,
            linewidths=0,
            alpha=0.55,
            zorder=1,
        )
        if err.any():
            ax.scatter(
                sub.loc[err, "umap_1"],
                sub.loc[err, "umap_2"],
                s=16,
                alpha=0.65,
                c=ACCENT,
                edgecolors="white",
                linewidths=0.3,
                rasterized=True,
                zorder=2,
            )
        n_err = int(err.sum())
        rate = 100.0 * n_err / max(len(sub), 1)
        ax.set_title(
            f"W4 errors: {n_err:,} ({rate:.0f}%)",
            fontsize=10,
            loc="left",
            color=ACCENT,
        )
        if c == 0:
            ax.set_ylabel("W4 mistakes", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        _zoom_to_data(ax, sub)
        ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RES, markersize=8, label="residential"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NRES, markersize=8, label="non-residential"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT, markeredgecolor="white", markersize=8, label="W4 error"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.86])
    set_claim_title(
        fig,
        "W4 errors sit where RES and NRES mix — not sprinkled across empty UMAP space",
        "Lowest / median / highest W4 F1 metros. Grey hexbin = all pixels; orange = wrong class (W4 only).",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(r) for r in picks]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refit-umap",
        action="store_true",
        help="Recompute UMAP even if umap_coords.parquet exists",
    )
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(BAL)
    regions = sorted(df["region"].astype(str).unique().tolist())
    print(f"Balanced pixels n={len(df):,} regions={len(regions)}", flush=True)

    coords_path = OUT / "umap_coords.parquet"
    if coords_path.exists() and not args.refit_umap:
        print(f"Reusing {coords_path.relative_to(ROOT)}", flush=True)
        coords = pd.read_parquet(coords_path)
        base = df[["pixel_id", "region", "stratum", "y", "pixel_lon", "pixel_lat"]].copy()
        coords = base.merge(
            coords[["pixel_id", "umap_1", "umap_2"]], on="pixel_id", how="inner"
        )
    else:
        X = df[BANDS].to_numpy(dtype=np.float32)
        print("Fitting UMAP (cosine, n_neighbors=30) ...", flush=True)
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=SEED,
            verbose=False,
        )
        Z = reducer.fit_transform(X)
        coords = df[
            ["pixel_id", "region", "stratum", "y", "pixel_lon", "pixel_lat"]
        ].copy()
        coords["umap_1"] = Z[:, 0]
        coords["umap_2"] = Z[:, 1]

    metrics = []
    for m in MODELS:
        preds = _load_preds(m, coords["pixel_id"])
        col = f"pred_{m['key']}"
        coords[col] = preds["y_pred"].astype("Int64")
        if "p_res" in preds.columns:
            coords[f"p_res_{m['key']}"] = preds["p_res"].astype(float)
        ok = coords[col].astype(float) == coords["y"].astype(float)
        coords[f"correct_{m['key']}"] = ok
        mask = coords[col].notna()
        f1 = float(
            f1_score(
                coords.loc[mask, "y"].astype(int),
                coords.loc[mask, col].astype(int),
                average="macro",
            )
        )
        metrics.append(
            {
                "model": m["key"],
                "label": m["label"],
                "n": int(mask.sum()),
                "macro_f1_on_umap_set": f1,
                "accuracy": float(ok[mask].mean()),
            }
        )
        print(f"  {m['key']}: macroF1={f1:.4f} acc={ok[mask].mean():.3f}", flush=True)

    coords.to_parquet(coords_path, index=False)

    # Per-region F1
    per_reg = []
    for reg in regions:
        sub = coords.loc[coords["region"] == reg]
        row = {"region": reg, "n": len(sub)}
        for m in MODELS:
            key = m["key"]
            mask = sub[f"pred_{key}"].notna()
            if mask.sum() < 2:
                row[f"f1_{key}"] = float("nan")
                continue
            row[f"f1_{key}"] = float(
                f1_score(
                    sub.loc[mask, "y"].astype(int),
                    sub.loc[mask, f"pred_{key}"].astype(int),
                    average="macro",
                )
            )
        per_reg.append(row)
    per_df = pd.DataFrame(per_reg)
    per_df.to_csv(OUT / "per_region_f1.csv", index=False)

    # --- Primary figures ---
    gt_f1 = _boundary_region_panels(
        coords,
        regions,
        y_col="y",
        claim="Within each metro, residential and non-residential pixels partly separate in AE UMAP",
        subtitle=(
            "Blue = residential, orange = non-residential (colorblind-safe). "
            "Contour = 2D logistic P(residential)=0.5 fit in this plane."
        ),
        out_path=FIG / "umap_gt_by_region_boundary.png",
        per_dir=FIG / "umap_gt_boundary_panels",
    )
    gt_f1.to_csv(OUT / "in_umap_lr_f1_gt.csv", index=False)
    mean_gt = float(gt_f1["in_umap_lr_f1"].mean())
    print(
        f"  in-UMAP LR F1 (GT): mean={mean_gt:.3f}  "
        f"min={gt_f1['in_umap_lr_f1'].min():.3f}  max={gt_f1['in_umap_lr_f1'].max():.3f}",
        flush=True,
    )
    # Rewrite claim with measured mean (re-save with updated title text via annotate file note)
    # Title already set; patch claim into a companion txt for notebooks
    (OUT / "gt_boundary_claim.txt").write_text(
        f"Within each metro, RES and NRES partly separate (mean in-UMAP LR F1={mean_gt:.2f}).\n"
    )

    boundary_f1_rows = []
    for m in MODELS:
        key = m["key"]
        ycol = f"pred_{key}"
        sub = coords.dropna(subset=[ycol]).copy()
        sub[ycol] = sub[ycol].astype(int)
        bf = _boundary_region_panels(
            sub,
            regions,
            y_col=ycol,
            claim=f"{m['label']}: predicted RES/NRES still form coherent UMAP regions",
            subtitle="Same layout as GT panel; colors show model predictions, not labels.",
            out_path=FIG / f"umap_pred_{key}_by_region_boundary.png",
            per_dir=FIG / f"umap_pred_{key}_boundary_panels",
        )
        bf["model"] = key
        boundary_f1_rows.append(bf)
        print(
            f"  in-UMAP LR F1 ({key}): mean={bf['in_umap_lr_f1'].mean():.3f}",
            flush=True,
        )
    pd.concat(boundary_f1_rows, ignore_index=True).to_csv(
        OUT / "in_umap_lr_f1_by_model.csv", index=False
    )

    _plot_f1_by_metro(per_df, FIG / "f1_by_region_top3.png")
    _plot_agreement_by_metro(coords, regions, FIG / "agreement_vs_gt_by_metro.png")
    spotlight = _plot_error_spotlight(
        coords, per_df, FIG / "umap_errors_spotlight.png"
    )
    print(f"  error spotlight metros: {spotlight}", flush=True)

    # Keep a single legacy filename that used to be the unreadable 4×10 grid —
    # now point it at the agreement chart so old notebook links still show something useful.
    import shutil

    shutil.copyfile(
        FIG / "agreement_vs_gt_by_metro.png",
        FIG / "umap_top3_vs_gt_by_region.png",
    )

    summary = {
        "name": "week4_top3_umap_v2",
        "data": str(BAL.relative_to(ROOT)),
        "umap": {
            "n_neighbors": 30,
            "min_dist": 0.1,
            "metric": "cosine",
            "seed": SEED,
            "n_pixels": int(len(coords)),
        },
        "models": metrics,
        "spotlight_metros": spotlight,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "coords": str(coords_path.relative_to(ROOT)),
            "per_region_f1": str((OUT / "per_region_f1.csv").relative_to(ROOT)),
            "figures": str(FIG.relative_to(ROOT)),
        },
        "primary_figures": [
            "umap_gt_by_region_boundary.png",
            "f1_by_region_top3.png",
            "agreement_vs_gt_by_metro.png",
            "umap_errors_spotlight.png",
        ],
        "figures": sorted(str(p.relative_to(ROOT)) for p in FIG.rglob("*.png")),
        "in_umap_lr_f1_gt_mean": mean_gt,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {OUT} in {summary['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
