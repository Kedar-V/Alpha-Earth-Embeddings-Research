#!/usr/bin/env python3
"""RES vs NRES fraction MAE only (ignore empty) — separate from 3-class ladder.

Models (binary / two-share heads only):
  - W4 balanced Ridge (pred_frac_res / pred_frac_nres)
  - W3 fraction Ridge
  - W3 pixel AE LR (p_res → (p, 1-p))
  - GHSL FUN one-hot: RES→(1,0), NRES→(0,1), NONBUILT/NODATA→(0,0)

Metric: mean(|frac_res−pred|, |frac_nres−pred|) — empty is never scored.

Usage:
  .venv/bin/python -u scripts/eval_week4_res_nres_fractions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import ACCENT, MUTED, OK, RES, label_bars_h, set_claim_title, style_axes  # noqa: E402

BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
GHSL = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
OUT = ROOT / "outputs" / "week4_ghsl_pixel"
FIG = OUT / "figures"

MODELS = [
    {
        "key": "balanced_train",
        "label": "W4 balanced Ridge",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "kind": "frac",
    },
    {
        "key": "fraction_regression",
        "label": "W3 Ridge",
        "path": ROOT / "outputs" / "week3_fraction_regression" / "oof_predictions.parquet",
        "kind": "frac_full",
    },
    {
        "key": "pixel_ae",
        "label": "W3 AE LR",
        "path": ROOT / "outputs" / "week3_pixel_ae" / "oof_predictions.parquet",
        "kind": "p_res",
    },
    {
        "key": "ghsl",
        "label": "GHSL (one-hot)",
        "path": GHSL,
        "kind": "ghsl",
    },
]

GHSL_TO_RN = {
    "RES": (1.0, 0.0),
    "NRES": (0.0, 1.0),
    "NONBUILT": (0.0, 0.0),
    "NODATA": (0.0, 0.0),
}


def _preds(meta: dict, pixel_ids: pd.Series) -> pd.DataFrame:
    raw = pd.read_parquet(meta["path"])
    base = pixel_ids.to_frame("pixel_id")
    if meta["kind"] == "ghsl":
        sub = raw[["pixel_id", "ghsl_label"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="left")
        out["ghsl_label"] = out["ghsl_label"].fillna("NODATA")
        tup = out["ghsl_label"].map(GHSL_TO_RN)
        tup = tup.apply(lambda t: t if isinstance(t, tuple) else (0.0, 0.0))
        out["pred_frac_res"] = tup.map(lambda t: float(t[0]))
        out["pred_frac_nres"] = tup.map(lambda t: float(t[1]))
        return out[["pixel_id", "pred_frac_res", "pred_frac_nres"]]

    if meta["kind"] == "p_res":
        sub = raw[["pixel_id", "p_res"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="left")
        p = out["p_res"].astype(float).clip(0, 1)
        out["pred_frac_res"] = p
        out["pred_frac_nres"] = 1.0 - p
        return out[["pixel_id", "pred_frac_res", "pred_frac_nres"]]

    if meta["kind"] == "frac_full":
        keep = ["pixel_id"]
        for c in ("pred_frac_res_full", "pred_frac_nres_full", "pred_frac_res", "pred_frac_nres"):
            if c in raw.columns:
                keep.append(c)
        sub = raw[keep].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="left")
        if "pred_frac_res_full" in out.columns:
            pr = out["pred_frac_res_full"].astype(float).clip(0, 1)
            pn = out["pred_frac_nres_full"].astype(float).clip(0, 1)
        else:
            pr = out["pred_frac_res"].astype(float).clip(0, 1)
            pn = out["pred_frac_nres"].astype(float).clip(0, 1)
        s = (pr + pn).replace(0, np.nan)
        # Keep as predicted shares (do not force sum=1); clip overshoot
        over = (pr + pn) > 1.0
        pr2, pn2 = pr.copy(), pn.copy()
        pr2.loc[over] = pr.loc[over] / (pr.loc[over] + pn.loc[over])
        pn2.loc[over] = pn.loc[over] / (pr.loc[over] + pn.loc[over])
        out["pred_frac_res"] = pr2
        out["pred_frac_nres"] = pn2
        return out[["pixel_id", "pred_frac_res", "pred_frac_nres"]]

    # frac (balanced_train)
    keep = ["pixel_id"]
    for c in ("pred_frac_res", "pred_frac_nres", "pred_frac_res_full", "pred_frac_nres_full"):
        if c in raw.columns:
            keep.append(c)
    sub = raw[keep].drop_duplicates("pixel_id")
    out = base.merge(sub, on="pixel_id", how="left")
    if "pred_frac_res_full" in out.columns and out["pred_frac_res_full"].notna().any():
        out["pred_frac_res"] = out["pred_frac_res_full"].astype(float)
        out["pred_frac_nres"] = out["pred_frac_nres_full"].astype(float)
    out["pred_frac_res"] = out["pred_frac_res"].astype(float)
    out["pred_frac_nres"] = out["pred_frac_nres"].astype(float)
    return out[["pixel_id", "pred_frac_res", "pred_frac_nres"]]


def score_set(gt: pd.DataFrame, name: str, model_keys: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    detail_parts = []
    for m in MODELS:
        if model_keys is not None and m["key"] not in model_keys:
            continue
        preds = _preds(m, gt["pixel_id"])
        j = gt[["pixel_id", "region", "frac_res", "frac_nres"]].merge(preds, on="pixel_id", how="inner")
        j = j.dropna(subset=["pred_frac_res", "pred_frac_nres"])
        # Skip if model only covers a small fraction of this eval set (e.g. W4 OOF on full GT)
        if len(j) < 0.5 * len(gt):
            print(
                f"  [{name}] skip {m['label']}: only {len(j):,}/{len(gt):,} joined",
                flush=True,
            )
            continue
        mae_res = float(mean_absolute_error(j["frac_res"], j["pred_frac_res"]))
        mae_nres = float(mean_absolute_error(j["frac_nres"], j["pred_frac_nres"]))
        mae_mean = 0.5 * (mae_res + mae_nres)
        rows.append(
            {
                "model": m["key"],
                "label": m["label"],
                "eval": name,
                "n": int(len(j)),
                "mae_frac_res": mae_res,
                "mae_frac_nres": mae_nres,
                "mae_mean_res_nres": mae_mean,
            }
        )
        j = j.copy()
        j["model"] = m["key"]
        detail_parts.append(j)
        print(
            f"  [{name}] {m['label']}: MAE_mean={mae_mean:.4f}  "
            f"res={mae_res:.4f}  nres={mae_nres:.4f}  n={len(j):,}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()


def plot_ladder(summary: pd.DataFrame, eval_name: str, out_path: Path, claim: str) -> None:
    s = summary.loc[summary["eval"] == eval_name].sort_values(
        "mae_mean_res_nres", ascending=False
    )
    best = float(s["mae_mean_res_nres"].min())
    colors = []
    for _, r in s.iterrows():
        if r["model"] == "ghsl":
            colors.append(ACCENT)
        elif abs(r["mae_mean_res_nres"] - best) < 1e-12:
            colors.append(OK)
        else:
            colors.append(MUTED)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.barh(s["label"], s["mae_mean_res_nres"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.008)
    ax.set_xlabel("Mean absolute error · (MAE_res + MAE_nres) / 2")
    ax.set_xlim(0, float(s["mae_mean_res_nres"].max()) * 1.2)
    style_axes(ax, grid=True)
    ghsl_mae = float(s.loc[s["model"] == "ghsl", "mae_mean_res_nres"].iloc[0])
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        claim.format(best=best, ghsl=ghsl_mae, gap=ghsl_mae - best),
        "Empty ignored. RES/NRES models only (no soft 3-class). Orange = GHSL one-hot.",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_components(summary: pd.DataFrame, eval_name: str, out_path: Path) -> None:
    s = summary.loc[summary["eval"] == eval_name].sort_values("mae_mean_res_nres")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(s))
    h = 0.35
    ax.barh(y + h / 2, s["mae_frac_res"], height=h, color=RES, label="residential")
    ax.barh(y - h / 2, s["mae_frac_nres"], height=h, color="#E69F00", label="non-residential")
    ax.set_yticks(y)
    ax.set_yticklabels(s["label"])
    ax.set_xlabel("MAE")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        "RES vs NRES error by model (empty excluded)",
        f"Eval: {eval_name.replace('_', ' ')}. Blue = frac_res MAE; orange = frac_nres MAE.",
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not GHSL.exists():
        raise SystemExit(f"Missing {GHSL}; run eval_week4_ghsl_pixel.py first.")

    FIG.mkdir(parents=True, exist_ok=True)
    bal = pd.read_parquet(BAL, columns=["pixel_id", "region", "frac_res", "frac_nres"])
    full = pd.read_parquet(FULL, columns=["pixel_id", "region", "frac_res", "frac_nres"])

    print("RES/NRES-only fraction MAE (ignore empty)", flush=True)
    bal_sum, bal_det = score_set(bal, "balanced")
    full_sum, full_det = score_set(full, "full")
    summary = pd.concat([bal_sum, full_sum], ignore_index=True)
    summary.to_csv(OUT / "res_nres_fraction_mae.csv", index=False)
    bal_det.to_parquet(OUT / "predictions_balanced_res_nres_frac.parquet", index=False)

    plot_ladder(
        summary,
        "balanced",
        FIG / "res_nres_frac_mae_ladder_balanced.png",
        "On balanced pixels, best RES/NRES MAE is {best:.3f}; GHSL is {ghsl:.3f} ({gap:+.3f})",
    )
    plot_ladder(
        summary,
        "full",
        FIG / "res_nres_frac_mae_ladder_full.png",
        "On full Week 4 GT, best RES/NRES MAE is {best:.3f}; GHSL is {ghsl:.3f} ({gap:+.3f})",
    )
    plot_components(summary, "balanced", FIG / "res_nres_frac_mae_components_balanced.png")

    meta = {
        "name": "week4_res_nres_fraction_mae_v1",
        "metric": "(mae_frac_res + mae_frac_nres) / 2 — empty ignored",
        "models": [m["key"] for m in MODELS],
        "excluded": ["pixel_3class_reg", "neighborhood_k3_reg"],
        "summary_csv": "outputs/week4_ghsl_pixel/res_nres_fraction_mae.csv",
        "figures": [
            "outputs/week4_ghsl_pixel/figures/res_nres_frac_mae_ladder_balanced.png",
            "outputs/week4_ghsl_pixel/figures/res_nres_frac_mae_ladder_full.png",
            "outputs/week4_ghsl_pixel/figures/res_nres_frac_mae_components_balanced.png",
        ],
        "results": summary.to_dict(orient="records"),
    }
    (OUT / "res_nres_fraction_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT / 'res_nres_fraction_mae.csv'}", flush=True)


if __name__ == "__main__":
    main()
