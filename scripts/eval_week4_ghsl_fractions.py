#!/usr/bin/env python3
"""GHSL as a fraction / regression baseline on Week 4 GT.

GHSL FUN is categorical. Encode each AE-center sample as a one-hot soft
fraction vector and score MAE/RMSE/R² vs Overture coverage fractions:

  RES      → (1, 0, 0)
  NRES     → (0, 1, 0)
  NONBUILT → (0, 0, 1)
  NODATA   → (0, 0, 1)

Eval sets:
  - full week4_pixel_gt (all AE cells with fraction GT)
  - week4_pixel_gt_balanced (pure RES=NRES subset)

Requires outputs/week4_ghsl_pixel/ghsl_at_ae_centers.parquet
(from scripts/eval_week4_ghsl_pixel.py).

Usage:
  .venv/bin/python -u scripts/eval_week4_ghsl_fractions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import (  # noqa: E402
    ACCENT,
    MUTED,
    OK,
    RES,
    label_bars_h,
    set_claim_title,
    style_axes,
)

BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
GHSL_CACHE = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
OUT = ROOT / "outputs" / "week4_ghsl_pixel"
FIG = OUT / "figures"
CMP = ROOT / "outputs" / "week4_model_comparison"
EVAL_BAL = ROOT / "outputs" / "week4_pixel_eval" / "balanced_rescore_v1" / "summary_metrics.csv"
EVAL_FULL = ROOT / "outputs" / "week4_pixel_eval" / "week3_rescore_v1" / "summary_metrics.csv"

FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")
GHSL_TO_FRAC = {
    "RES": (1.0, 0.0, 0.0),
    "NRES": (0.0, 1.0, 0.0),
    "NONBUILT": (0.0, 0.0, 1.0),
    "NODATA": (0.0, 0.0, 1.0),
}


def attach_fracs(pixels: pd.DataFrame, ghsl: pd.DataFrame) -> pd.DataFrame:
    m = pixels.merge(ghsl, on=["pixel_id", "region"], how="left")
    m["ghsl_label"] = m["ghsl_label"].fillna("NODATA")
    tup = m["ghsl_label"].map(GHSL_TO_FRAC)
    tup = tup.apply(lambda t: t if isinstance(t, tuple) else (0.0, 0.0, 1.0))
    m["pred_frac_res"] = tup.map(lambda t: float(t[0]))
    m["pred_frac_nres"] = tup.map(lambda t: float(t[1]))
    m["pred_frac_empty"] = tup.map(lambda t: float(t[2]))
    m["ghsl_known"] = m["ghsl_label"].isin(["RES", "NRES", "NONBUILT"])
    return m


def frac_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    out: dict = {"n": int(len(yt))}
    maes, rmses = [], []
    for i, name in enumerate(FRAC_COLS):
        if len(yt) == 0:
            mae = rmse = r2 = float("nan")
        else:
            mae = float(mean_absolute_error(yt[:, i], yp[:, i]))
            rmse = float(np.sqrt(mean_squared_error(yt[:, i], yp[:, i])))
            r2 = float(r2_score(yt[:, i], yp[:, i])) if len(yt) > 1 else float("nan")
        out[f"mae_{name}"] = mae
        out[f"rmse_{name}"] = rmse
        out[f"r2_{name}"] = r2
        maes.append(mae)
        rmses.append(rmse)
    out["mae_mean"] = float(np.mean(maes))
    out["rmse_mean"] = float(np.mean(rmses))
    return out


def score(pixels: pd.DataFrame, ghsl: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    cols = ["pixel_id", "region", *FRAC_COLS]
    # balanced parquet may lack frac_empty — derive if needed
    p = pixels.copy()
    if "frac_empty" not in p.columns:
        if "frac_empty_full" in p.columns:
            p["frac_empty"] = p["frac_empty_full"]
        else:
            p["frac_empty"] = (1.0 - p["frac_res"] - p["frac_nres"]).clip(0, 1)
    m = attach_fracs(p[cols], ghsl)
    yt = m[list(FRAC_COLS)].to_numpy(dtype=np.float64)
    yp = m[["pred_frac_res", "pred_frac_nres", "pred_frac_empty"]].to_numpy(dtype=np.float64)
    pooled = frac_metrics(yt, yp)
    known = m["ghsl_known"].to_numpy()
    known_m = frac_metrics(yt[known], yp[known]) if known.any() else pooled

    per = []
    for reg, sub in m.groupby("region"):
        yt_r = sub[list(FRAC_COLS)].to_numpy(dtype=np.float64)
        yp_r = sub[["pred_frac_res", "pred_frac_nres", "pred_frac_empty"]].to_numpy(dtype=np.float64)
        row = frac_metrics(yt_r, yp_r)
        row["region"] = reg
        row["pct_known"] = float(sub["ghsl_known"].mean())
        per.append(row)
    per_df = pd.DataFrame(per).sort_values("mae_mean")

    summary = {
        "eval": name,
        "encoding": "GHSL FUN one-hot → (frac_res, frac_nres, frac_empty)",
        "n_pixels": int(len(m)),
        "pct_known_label": float(m["ghsl_known"].mean()),
        "all_pixels": pooled,
        "known_label_only": known_m,
        "ghsl_label_counts": m["ghsl_label"].value_counts().to_dict(),
    }
    return m, summary, per_df


def write_figures(
    full_sum: dict,
    bal_sum: dict,
    bal_per: pd.DataFrame,
    ae_bal: pd.DataFrame | None,
    ae_full: pd.DataFrame | None,
) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    mae_bal = bal_sum["all_pixels"]["mae_mean"]
    mae_full = full_sum["all_pixels"]["mae_mean"]

    # Ladder vs AE on balanced
    rows = []
    if ae_bal is not None and len(ae_bal):
        for _, r in ae_bal.iterrows():
            rows.append({"model": str(r["model"]), "mae_mean": float(r["oof_mae_mean"]), "kind": "ae"})
    rows.append({"model": "GHSL (one-hot)", "mae_mean": mae_bal, "kind": "ghsl"})
    lad = pd.DataFrame(rows).sort_values("mae_mean", ascending=False)
    best = float(lad["mae_mean"].min())
    colors = [
        ACCENT if k == "ghsl" else (OK if abs(v - best) < 1e-12 else MUTED)
        for k, v in zip(lad["kind"], lad["mae_mean"])
    ]
    short = {
        "balanced_train": "W4 Ridge",
        "fraction_regression": "W3 Ridge",
        "pixel_ae": "W3 AE LR",
        "pixel_3class_reg": "W3 soft 3-class",
        "neighborhood_k3_reg": "W3 3×3 CNN",
        "GHSL (one-hot)": "GHSL (one-hot)",
    }
    lad["label"] = lad["model"].map(lambda x: short.get(x, x))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(lad["label"], lad["mae_mean"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.008)
    ax.set_xlabel("Mean absolute error (res + nres + empty) / 3")
    ax.set_xlim(0, float(lad["mae_mean"].max()) * 1.2)
    style_axes(ax, grid=True)
    gap = mae_bal - best
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"GHSL one-hot fraction MAE is {mae_bal:.3f} — {gap:+.3f} vs best AE on balanced pixels",
        "Lower is better. Orange = GHSL hard label as (1,0,0)/(0,1,0)/(0,0,1). Green = best AE.",
    )
    p = FIG / "ghsl_frac_mae_ladder_balanced.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Full GT ladder
    rows_f = []
    if ae_full is not None and len(ae_full):
        for _, r in ae_full.iterrows():
            rows_f.append({"model": str(r["model"]), "mae_mean": float(r["oof_mae_mean"]), "kind": "ae"})
    rows_f.append({"model": "GHSL (one-hot)", "mae_mean": mae_full, "kind": "ghsl"})
    lad_f = pd.DataFrame(rows_f).sort_values("mae_mean", ascending=False)
    best_f = float(lad_f["mae_mean"].min())
    colors_f = [
        ACCENT if k == "ghsl" else (OK if abs(v - best_f) < 1e-12 else MUTED)
        for k, v in zip(lad_f["kind"], lad_f["mae_mean"])
    ]
    lad_f["label"] = lad_f["model"].map(lambda x: short.get(x, x))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(lad_f["label"], lad_f["mae_mean"], color=colors_f, height=0.7)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.008)
    ax.set_xlabel("Mean absolute error (three coverage fractions)")
    ax.set_xlim(0, float(lad_f["mae_mean"].max()) * 1.2)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"On full Week 4 GT, GHSL one-hot MAE={mae_full:.3f} (best AE={best_f:.3f})",
        "All AE cells with Overture fraction labels. Hard GHSL → one-hot soft target.",
    )
    p = FIG / "ghsl_frac_mae_ladder_full.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Per-component MAE on full
    comps = full_sum["all_pixels"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names = ["residential", "non-residential", "empty"]
    vals = [comps["mae_frac_res"], comps["mae_frac_nres"], comps["mae_frac_empty"]]
    cols = [RES, "#E69F00", MUTED]
    bars = ax.barh(names, vals, color=cols, height=0.65)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.01)
    ax.set_xlabel("MAE vs Overture fraction")
    ax.set_xlim(0, max(vals) * 1.2)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        "GHSL one-hot errs most on empty and NRES coverage",
        "Full Week 4 fraction GT. One hard FUN class cannot express mixed AE cells.",
    )
    p = FIG / "ghsl_frac_mae_by_component_full.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Per metro MAE balanced
    s = bal_per.sort_values("mae_mean", ascending=False).copy()
    s["lab"] = s["region"].astype(str).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    mean_m = float(s["mae_mean"].mean())
    colors = [OK if v <= mean_m else MUTED for v in s["mae_mean"]]
    bars = ax.barh(s["lab"], s["mae_mean"], color=colors, height=0.7)
    ax.axvline(mean_m, color="#333", ls="--", lw=1)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.008)
    ax.set_xlabel("Mean fraction MAE")
    ax.set_xlim(0, float(s["mae_mean"].max()) * 1.2)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"GHSL fraction MAE averages {mean_m:.3f} across metros (balanced set)",
        "Green = at/below mean. One-hot FUN vs Overture coverage fractions.",
    )
    p = FIG / "ghsl_frac_mae_by_metro_balanced.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    return paths


def update_comparison(mae_bal: float) -> None:
    path = CMP / "comparison_balanced_eval.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    mask = df["model"].astype(str).str.contains("GHSL", case=False)
    if mask.any():
        df.loc[mask, "mae_mean_balanced"] = mae_bal
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "model": "GHSL (pixel, W4 GT)",
                            "n": 59596,
                            "protocol_train": "GHSL FUN one-hot fractions",
                            "macro_f1_balanced_eval": float("nan"),
                            "mae_mean_balanced": mae_bal,
                            "macro_f1_week3_reported": float("nan"),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    df.to_csv(path, index=False)

    summary_path = CMP / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for row in summary.get("leaderboard_balanced_f1", []):
            if "GHSL" in str(row.get("model", "")):
                row["mae_mean_balanced"] = round(mae_bal, 4)
        takes = [t for t in summary.get("takeaways", []) if "fraction MAE" not in t.lower() or "GHSL" not in t]
        # find best AE mae from csv
        ae = df.loc[~df["model"].astype(str).str.contains("GHSL", case=False), "mae_mean_balanced"].dropna()
        best_ae = float(ae.min()) if len(ae) else float("nan")
        takes.insert(
            0,
            f"GHSL one-hot fraction MAE on balanced={mae_bal:.3f}; best AE={best_ae:.3f} (Δ={mae_bal - best_ae:+.3f}).",
        )
        summary["takeaways"] = takes
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    from plot_week4_model_comparison import main as plot_cmp

    plot_cmp()


def main() -> None:
    if not GHSL_CACHE.exists():
        raise SystemExit(
            f"Missing {GHSL_CACHE}. Run scripts/eval_week4_ghsl_pixel.py first."
        )

    ghsl = pd.read_parquet(GHSL_CACHE)
    bal = pd.read_parquet(BAL)
    full = pd.read_parquet(FULL, columns=["pixel_id", "region", *FRAC_COLS])

    print(f"GHSL samples={len(ghsl):,}  bal={len(bal):,}  full={len(full):,}", flush=True)

    bal_j, bal_sum, bal_per = score(bal, ghsl, "week4_balanced_fractions")
    full_j, full_sum, full_per = score(full, ghsl, "week4_full_fractions")

    bal_j.to_parquet(OUT / "predictions_balanced_fractions.parquet", index=False)
    full_j[
        [
            "pixel_id",
            "region",
            *FRAC_COLS,
            "ghsl_label",
            "pred_frac_res",
            "pred_frac_nres",
            "pred_frac_empty",
            "ghsl_known",
        ]
    ].to_parquet(OUT / "predictions_full_fractions.parquet", index=False)
    bal_per.to_csv(OUT / "per_region_fractions_balanced.csv", index=False)
    full_per.to_csv(OUT / "per_region_fractions_full.csv", index=False)

    ae_bal = pd.read_csv(EVAL_BAL) if EVAL_BAL.exists() else None
    ae_full = pd.read_csv(EVAL_FULL) if EVAL_FULL.exists() else None
    # Ensure balanced_train in ae_bal for ladder
    fig_paths = write_figures(full_sum, bal_sum, bal_per, ae_bal, ae_full)

    mae_bal = float(bal_sum["all_pixels"]["mae_mean"])
    mae_full = float(full_sum["all_pixels"]["mae_mean"])
    update_comparison(mae_bal)

    # Merge into existing summary.json
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["name"] = "week4_ghsl_pixel_v2"
    summary["fractions"] = {
        "balanced": bal_sum,
        "full": full_sum,
        "method": (
            "Categorical GHSL FUN at AE center encoded as one-hot "
            "(RES→(1,0,0), NRES→(0,1,0), NONBUILT/NODATA→(0,0,1)); "
            "MAE vs Overture frac_res/nres/empty."
        ),
    }
    summary["figures"] = sorted(
        set(summary.get("figures", []) + [str(p.relative_to(ROOT)) for p in fig_paths])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\nFraction MAE  balanced={mae_bal:.4f}  "
        f"(res={bal_sum['all_pixels']['mae_frac_res']:.3f} "
        f"nres={bal_sum['all_pixels']['mae_frac_nres']:.3f} "
        f"empty={bal_sum['all_pixels']['mae_frac_empty']:.3f})",
        flush=True,
    )
    print(
        f"Fraction MAE  full={mae_full:.4f}  "
        f"(res={full_sum['all_pixels']['mae_frac_res']:.3f} "
        f"nres={full_sum['all_pixels']['mae_frac_nres']:.3f} "
        f"empty={full_sum['all_pixels']['mae_frac_empty']:.3f})",
        flush=True,
    )
    if ae_bal is not None:
        print("AE balanced MAE:", flush=True)
        print(ae_bal[["model", "oof_mae_mean"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
