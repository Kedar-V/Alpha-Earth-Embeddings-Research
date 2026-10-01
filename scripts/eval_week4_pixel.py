#!/usr/bin/env python3
"""Evaluate pixel-level fraction predictions against week4 GT.

Joins model OOF preds to datasets/week4_pixel_gt on pixel_id and scores
MAE / RMSE / R² for frac_res, frac_nres, frac_empty (LORO + pooled).

Usage:
  .venv/bin/python scripts/eval_week4_pixel.py
  .venv/bin/python scripts/eval_week4_pixel.py --run-id week3_rescore
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
GT_PATH_DEFAULT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
OUT_ROOT = ROOT / "outputs" / "week4_pixel_eval"

FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_block(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """y_* shape (n, 3) in order frac_res, frac_nres, frac_empty."""
    out: dict = {}
    maes = []
    rmses = []
    for i, name in enumerate(FRAC_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(mean_absolute_error(yt, yp))
        rmse = _rmse(yt, yp)
        r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
        out[f"{prefix}mae_{name}"] = mae
        out[f"{prefix}rmse_{name}"] = rmse
        out[f"{prefix}r2_{name}"] = r2
        maes.append(mae)
        rmses.append(rmse)
    out[f"{prefix}mae_mean"] = float(np.mean(maes))
    out[f"{prefix}rmse_mean"] = float(np.mean(rmses))
    return out


def normalize_preds(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return frame with pixel_id, region, pred_frac_{res,nres,empty}."""
    out = df.copy()
    if model == "pixel_ae":
        # Pure binary LR: map p_res → soft fractions (no empty mass).
        p = out["p_res"].astype(float).clip(0, 1)
        out["pred_frac_res"] = p
        out["pred_frac_nres"] = 1.0 - p
        out["pred_frac_empty"] = 0.0
    elif model == "fraction_regression":
        pr = out["pred_frac_res_full"].astype(float).clip(0, 1)
        pn = out["pred_frac_nres_full"].astype(float).clip(0, 1)
        # Renormalize mildly if sum > 1, then empty = remainder.
        s = pr + pn
        over = s > 1.0
        pr = pr.copy()
        pn = pn.copy()
        pr.loc[over] = pr.loc[over] / s.loc[over]
        pn.loc[over] = pn.loc[over] / s.loc[over]
        out["pred_frac_res"] = pr
        out["pred_frac_nres"] = pn
        out["pred_frac_empty"] = (1.0 - pr - pn).clip(0, 1)
    elif model in ("pixel_3class_reg", "neighborhood_k3_reg", "balanced_train"):
        # Prefer full-precision preds when present.
        for short, full in [
            ("pred_frac_res", "pred_frac_res_full"),
            ("pred_frac_nres", "pred_frac_nres_full"),
            ("pred_frac_empty", "pred_frac_empty_full"),
        ]:
            if full in out.columns:
                out[short] = out[full].astype(float).clip(0, 1)
            elif short in out.columns:
                out[short] = out[short].astype(float).clip(0, 1)
            else:
                raise KeyError(f"{model} missing {short}")
        s = out["pred_frac_res"] + out["pred_frac_nres"] + out["pred_frac_empty"]
        s = s.replace(0, np.nan)
        for c in ("pred_frac_res", "pred_frac_nres", "pred_frac_empty"):
            out[c] = (out[c] / s).fillna(0.0).clip(0, 1)
    else:
        raise ValueError(f"Unknown model key: {model}")

    keep = ["pixel_id", "region", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]
    return out[keep].copy()


MODELS = {
    "pixel_ae": {
        "path": ROOT / "outputs" / "week3_pixel_ae" / "oof_predictions.parquet",
        "desc": "Week3 Pixel AE LR (p_res → soft fracs, empty=0)",
        "week3_summary": ROOT / "outputs" / "week3_pixel_ae" / "summary.json",
    },
    "fraction_regression": {
        "path": ROOT / "outputs" / "week3_fraction_regression" / "oof_predictions.parquet",
        "desc": "Week3 Ridge fraction regression (+ derived empty)",
        "week3_summary": ROOT / "outputs" / "week3_fraction_regression" / "summary.json",
    },
    "pixel_3class_reg": {
        "path": ROOT / "outputs" / "week3_pixel_3class" / "oof_predictions.parquet",
        "desc": "Week3 3-class soft-fraction MLP (reg head)",
        "week3_summary": ROOT / "outputs" / "week3_pixel_3class" / "summary.json",
    },
    "neighborhood_k3_reg": {
        "path": ROOT / "outputs" / "week3_pixel_neighborhood_k3" / "oof_predictions.parquet",
        "desc": "Week3 3x3 neighborhood CNN soft-fraction (reg head)",
        "week3_summary": ROOT / "outputs" / "week3_pixel_neighborhood_k3" / "summary.json",
    },
    "balanced_train": {
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "desc": "Week4 Ridge+LR trained on per-region balanced pure RES/NRES",
        "week3_summary": ROOT / "outputs" / "week4_pixel_balanced_train" / "summary.json",
    },
}


def evaluate_model(model: str, gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = MODELS[model]
    raw = pd.read_parquet(meta["path"])
    preds = normalize_preds(raw, model)
    joined = preds.merge(
        gt,
        on="pixel_id",
        how="inner",
        suffixes=("_pred", ""),
    )
    # Prefer GT region if both present.
    if "region_pred" in joined.columns:
        joined = joined.drop(columns=["region_pred"])
    if "region" not in joined.columns and "region_x" in joined.columns:
        joined = joined.rename(columns={"region_x": "region"})

    y_true = joined[list(FRAC_COLS)].to_numpy(dtype=np.float64)
    y_pred = joined[[f"pred_{c}" for c in FRAC_COLS]].to_numpy(dtype=np.float64)

    pooled = score_block(y_true, y_pred, prefix="oof_")
    pooled["model"] = model
    pooled["desc"] = meta["desc"]
    pooled["n_pred_rows"] = int(len(raw))
    pooled["n_joined"] = int(len(joined))
    pooled["n_gt"] = int(len(gt))

    per_region = []
    for region, sub in joined.groupby("region"):
        yt = sub[list(FRAC_COLS)].to_numpy(dtype=np.float64)
        yp = sub[[f"pred_{c}" for c in FRAC_COLS]].to_numpy(dtype=np.float64)
        row = score_block(yt, yp)
        row["model"] = model
        row["region"] = str(region)
        row["n"] = int(len(sub))
        per_region.append(row)
    per_df = pd.DataFrame(per_region).sort_values("region").reset_index(drop=True)

    loro = {
        "model": model,
        "loro_mae_mean_mean": float(per_df["mae_mean"].mean()),
        "loro_mae_mean_std": float(per_df["mae_mean"].std(ddof=0)),
        "loro_mae_frac_res_mean": float(per_df["mae_frac_res"].mean()),
        "loro_mae_frac_nres_mean": float(per_df["mae_frac_nres"].mean()),
        "loro_mae_frac_empty_mean": float(per_df["mae_frac_empty"].mean()),
        "loro_rmse_mean_mean": float(per_df["rmse_mean"].mean()),
    }
    pooled.update(loro)

    detail = joined[
        [
            "pixel_id",
            "region",
            *FRAC_COLS,
            "pred_frac_res",
            "pred_frac_nres",
            "pred_frac_empty",
        ]
    ].copy()
    detail["model"] = model
    return per_df, detail, pooled


def write_figures(summary_df: pd.DataFrame, per_region: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    C_RES, C_NRES, C_EMPTY, MUTED = "#0072B2", "#E69F00", "#7F7F7F", "#B0B0B0"

    # Ladder: mean MAE (lower better), sorted, best highlighted
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    s = summary_df.sort_values("oof_mae_mean", ascending=False).copy()
    best = float(s["oof_mae_mean"].min())
    colors = ["#009E73" if abs(v - best) < 1e-12 else MUTED for v in s["oof_mae_mean"]]
    labels = [str(m).replace("_", " ") for m in s["model"]]
    bars = ax.barh(labels, s["oof_mae_mean"], color=colors, height=0.7)
    for rect, v in zip(bars, s["oof_mae_mean"]):
        ax.text(v + 0.008, rect.get_y() + rect.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlabel("Mean absolute error across three coverage fractions")
    ax.set_xlim(0, float(s["oof_mae_mean"].max()) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(
        0.01, 0.98,
        f"Best mean fraction MAE is {best:.3f} (green)",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        "Lower is better. OOF vs Week 4 coverage GT.",
        fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "mae_ladder.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Per-fraction MAE — horizontal grouped, sorted by mean
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    s2 = summary_df.sort_values("oof_mae_mean", ascending=True).copy()
    y = np.arange(len(s2))
    h = 0.25
    specs = [
        ("oof_mae_frac_res", C_RES, "residential"),
        ("oof_mae_frac_nres", C_NRES, "non-residential"),
        ("oof_mae_frac_empty", C_EMPTY, "empty"),
    ]
    for i, (col, color, lab) in enumerate(specs):
        ax.barh(y + (i - 1) * h, s2[col], height=h, color=color, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([str(m).replace("_", " ") for m in s2["model"]])
    ax.set_xlabel("OOF MAE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(
        0.01, 0.98,
        "Empty-fraction error is usually the smallest of the three",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "mae_by_fraction.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Per-region: show delta from best model per region to cut clutter — or sorted heatmap-style
    # Prefer: one focal model (best mean) vs others muted — actually show sorted mean MAE for best model only + note
    pivot = per_region.pivot(index="region", columns="model", values="mae_mean")
    # Sort regions by mean across models
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(pivot))
    n_m = len(pivot.columns)
    h = 0.8 / max(n_m, 1)
    cmap_m = plt.cm.get_cmap("tab10", max(n_m, 1))
    for i, model in enumerate(pivot.columns):
        offset = (i - (n_m - 1) / 2) * h
        ax.barh(
            y + offset, pivot[model].values, height=h * 0.9,
            color=cmap_m(i), label=str(model).replace("_", " "), alpha=0.9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([str(r).replace("_", " ") for r in pivot.index])
    ax.set_xlabel("Mean MAE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(
        0.01, 0.98,
        "Fraction error varies by metro — ranking is not uniform",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "mae_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="week3_rescore_v1")
    ap.add_argument(
        "--gt",
        type=Path,
        default=GT_PATH_DEFAULT,
        help="GT pixels.parquet (full or balanced)",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS.keys()),
        help=f"Subset of {list(MODELS)}",
    )
    args = ap.parse_args()

    t0 = time.time()
    out_dir = OUT_ROOT / args.run_id
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.gt if args.gt.is_absolute() else (ROOT / args.gt)
    gt = pd.read_parquet(
        gt_path,
        columns=["pixel_id", "region", "frac_res_full", "frac_nres_full", "frac_empty_full"],
    ).rename(
        columns={
            "frac_res_full": "frac_res",
            "frac_nres_full": "frac_nres",
            "frac_empty_full": "frac_empty",
        }
    )
    print(f"GT={gt_path} pixels={len(gt):,}  models={args.models}", flush=True)

    per_parts = []
    detail_parts = []
    summaries = []
    for model in args.models:
        print(f"  scoring {model} ...", flush=True)
        per_df, detail, pooled = evaluate_model(model, gt)
        per_parts.append(per_df)
        detail_parts.append(detail)
        summaries.append(pooled)
        print(
            f"    joined={pooled['n_joined']:,}  "
            f"oof_mae_mean={pooled['oof_mae_mean']:.4f}  "
            f"loro_mae_mean={pooled['loro_mae_mean_mean']:.4f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values("oof_mae_mean").reset_index(drop=True)
    per_region = pd.concat(per_parts, ignore_index=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    per_region.to_csv(out_dir / "per_region_metrics.csv", index=False)

    fig_paths = write_figures(summary_df, per_region, fig_dir)

    try:
        gt_rel = str(gt_path.relative_to(ROOT))
    except ValueError:
        gt_rel = str(gt_path)

    manifest = {
        "name": "week4_pixel_eval",
        "run_id": args.run_id,
        "gt": gt_rel,
        "label_definition": "resi_def_v1_pixel_frac_v1",
        "metrics": "MAE / RMSE / R2 on frac_res, frac_nres, frac_empty vs week4 GT",
        "models": args.models,
        "n_gt": int(len(gt)),
        "elapsed_s": round(time.time() - t0, 1),
        "leaderboard": summary_df[
            [
                "model",
                "n_joined",
                "oof_mae_mean",
                "oof_mae_frac_res",
                "oof_mae_frac_nres",
                "oof_mae_frac_empty",
                "loro_mae_mean_mean",
            ]
        ].to_dict(orient="records"),
        "paths": {
            "summary_metrics": str((out_dir / "summary_metrics.csv").relative_to(ROOT)),
            "per_region_metrics": str((out_dir / "per_region_metrics.csv").relative_to(ROOT)),
            "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_dir}", flush=True)
    print(summary_df[["model", "oof_mae_mean", "loro_mae_mean_mean", "n_joined"]].to_string(index=False))


if __name__ == "__main__":
    main()
