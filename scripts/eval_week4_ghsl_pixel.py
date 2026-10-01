#!/usr/bin/env python3
"""Score GHSL GHS-BUILT-C FUN at Week 4 AE pixel centers.

Samples nearest GHSL 10 m cell at each AE pixel lon/lat (WGS84 → Mollweide),
maps FUN codes to RES/NRES, and evaluates against Week 4 Overture fraction GT:

  - balanced pure RES=NRES set (fair vs W4/W3 binary heads)
  - full fraction GT with majority building label (frac_res > frac_nres)

Writes outputs/week4_ghsl_pixel/{predictions, metrics, figures}.

Usage:
  .venv/bin/python -u scripts/eval_week4_ghsl_pixel.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from validation.ghsl import map_ghsl_value, open_ghsl  # noqa: E402
from validation.io import load_config  # noqa: E402
from week4_viz import (  # noqa: E402
    ACCENT,
    MUTED,
    NRES,
    OK,
    RES,
    label_bars_h,
    set_claim_title,
    style_axes,
)

BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_ghsl_pixel"
FIG = OUT / "figures"
CMP = ROOT / "outputs" / "week4_model_comparison"

LAB2Y = {"RES": 1, "NRES": 0}


def _sample_region(payload: tuple[str, np.ndarray, np.ndarray, np.ndarray]) -> pd.DataFrame:
    """Sample GHSL at lon/lat arrays for one region (own mosaic handle)."""
    region, pixel_ids, lons, lats = payload
    cfg = load_config()
    mosaic = open_ghsl(cfg)
    try:
        transformer = Transformer.from_crs("EPSG:4326", mosaic.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        raw = np.full(len(xs), 255, dtype=np.uint8)
        # Assign each point to a covering tile; sample in batches per tile
        assigned = np.full(len(xs), -1, dtype=np.int32)
        for ti, ds in enumerate(mosaic._datasets):
            b = ds.bounds
            m = (
                (xs >= b.left)
                & (xs <= b.right)
                & (ys >= b.bottom)
                & (ys <= b.top)
                & (assigned < 0)
            )
            assigned[m] = ti

        for ti, ds in enumerate(mosaic._datasets):
            idx = np.where(assigned == ti)[0]
            if len(idx) == 0:
                continue
            coords = list(zip(xs[idx], ys[idx]))
            vals = np.fromiter(
                (int(v[0]) for v in ds.sample(coords, indexes=1)),
                dtype=np.uint8,
                count=len(idx),
            )
            raw[idx] = vals

        labels = [map_ghsl_value(int(v), cfg) for v in raw]
        return pd.DataFrame(
            {
                "pixel_id": pixel_ids,
                "region": region,
                "ghsl_raw": raw.astype(np.int16),
                "ghsl_label": labels,
            }
        )
    finally:
        mosaic.close()


def sample_ghsl(df: pd.DataFrame, n_workers: int | None = None) -> pd.DataFrame:
    payloads = []
    for region, sub in df.groupby("region", sort=True):
        payloads.append(
            (
                str(region),
                sub["pixel_id"].to_numpy(),
                sub["pixel_lon"].to_numpy(dtype=np.float64),
                sub["pixel_lat"].to_numpy(dtype=np.float64),
            )
        )
    n_cpu = os.cpu_count() or 1
    workers = n_workers or min(n_cpu, len(payloads))
    print(f"Sampling GHSL · {len(df):,} pixels · {len(payloads)} metros · workers={workers}", flush=True)

    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sample_region, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            reg = futs[fut]
            part = fut.result()
            parts.append(part)
            covered = part["ghsl_label"].isin(["RES", "NRES"]).mean()
            print(
                f"  {reg}: n={len(part):,}  RES/NRES coverage={100 * covered:.1f}%",
                flush=True,
            )
    return pd.concat(parts, ignore_index=True)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if len(y_true)
        else float("nan"),
        "res_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        if len(y_true)
        else float("nan"),
        "nres_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
        if len(y_true)
        else float("nan"),
    }


def score_balanced(bal: pd.DataFrame, ghsl: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    m = bal.merge(ghsl, on=["pixel_id", "region"], how="left")
    m["ghsl_covered"] = m["ghsl_label"].isin(["RES", "NRES"])
    m["ghsl_y"] = m["ghsl_label"].map(LAB2Y)
    m["correct"] = (m["ghsl_y"] == m["y"]) & m["ghsl_covered"]

    covered = m.loc[m["ghsl_covered"]].copy()
    functional = _binary_metrics(
        covered["y"].to_numpy(dtype=int),
        covered["ghsl_y"].to_numpy(dtype=int),
    )
    # End-to-end: uncovered counts as wrong
    y_e2e_pred = m["ghsl_y"].fillna(-1).to_numpy()
    y_true = m["y"].to_numpy(dtype=int)
    e2e_correct = y_e2e_pred == y_true
    e2e = {
        "n": int(len(m)),
        "coverage": float(m["ghsl_covered"].mean()),
        "effective_accuracy": float(e2e_correct.mean()),
        "selective_macro_f1": functional["macro_f1"],
    }

    per_reg = []
    for reg, sub in covered.groupby("region"):
        met = _binary_metrics(sub["y"].to_numpy(dtype=int), sub["ghsl_y"].to_numpy(dtype=int))
        met["region"] = reg
        met["coverage"] = float(m.loc[m["region"] == reg, "ghsl_covered"].mean())
        per_reg.append(met)
    per_df = pd.DataFrame(per_reg).sort_values("macro_f1", ascending=False)

    summary = {
        "eval": "week4_balanced_pure",
        "n_pixels": int(len(m)),
        "functional": functional,
        "end_to_end": e2e,
        "label_counts_gt": m["y"].value_counts().to_dict(),
        "ghsl_label_counts": m["ghsl_label"].value_counts().to_dict(),
    }
    return m, summary, per_df


def score_full_majority(full: pd.DataFrame, ghsl: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Majority building class among cells with any RES or NRES coverage."""
    m = full.merge(ghsl, on=["pixel_id", "region"], how="left")
    m["ghsl_covered"] = m["ghsl_label"].isin(["RES", "NRES"])
    m["ghsl_y"] = m["ghsl_label"].map(LAB2Y)
    # GT binary: majority of building coverage; skip empty-only cells
    has_bldg = (m["frac_res"] + m["frac_nres"]) > 0
    m = m.loc[has_bldg].copy()
    m["y"] = (m["frac_res"] > m["frac_nres"]).astype(np.int64)
    # Ties (equal frac): drop
    tie = m["frac_res"] == m["frac_nres"]
    m = m.loc[~tie].copy()

    covered = m.loc[m["ghsl_covered"]].copy()
    functional = _binary_metrics(
        covered["y"].to_numpy(dtype=int),
        covered["ghsl_y"].to_numpy(dtype=int),
    )
    e2e_correct = (m["ghsl_y"].fillna(-1).to_numpy() == m["y"].to_numpy(dtype=int))
    e2e = {
        "n": int(len(m)),
        "coverage": float(m["ghsl_covered"].mean()),
        "effective_accuracy": float(e2e_correct.mean()),
        "selective_macro_f1": functional["macro_f1"],
    }
    per_reg = []
    for reg, sub in covered.groupby("region"):
        met = _binary_metrics(sub["y"].to_numpy(dtype=int), sub["ghsl_y"].to_numpy(dtype=int))
        met["region"] = reg
        met["coverage"] = float(m.loc[m["region"] == reg, "ghsl_covered"].mean())
        per_reg.append(met)
    per_df = pd.DataFrame(per_reg).sort_values("macro_f1", ascending=False)
    summary = {
        "eval": "week4_full_majority_building",
        "n_pixels": int(len(m)),
        "functional": functional,
        "end_to_end": e2e,
        "label_counts_gt": m["y"].value_counts().to_dict(),
        "ghsl_label_counts": m["ghsl_label"].value_counts().to_dict(),
    }
    return m, summary, per_df


def write_figures(
    bal_joined: pd.DataFrame,
    bal_summary: dict,
    bal_per: pd.DataFrame,
    cmp_ladder: pd.DataFrame | None,
) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    f1 = bal_summary["functional"]["macro_f1"]
    cov = bal_summary["end_to_end"]["coverage"]

    # Per-metro F1
    s = bal_per.sort_values("macro_f1", ascending=True).copy()
    s["lab"] = s["region"].astype(str).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = [RES if v >= f1 else MUTED for v in s["macro_f1"]]
    bars = ax.barh(s["lab"], s["macro_f1"], color=colors, height=0.7)
    ax.axvline(f1, color="#333333", ls="--", lw=1)
    label_bars_h(ax, bars, fmt="{:.2f}", pad=0.01)
    ax.set_xlabel("Macro-F1 (GHSL vs Week 4 balanced GT)")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"GHSL hits {f1:.2f} macro-F1 on balanced AE pixels (coverage {100 * cov:.0f}%)",
        "Functional score: only cells where GHSL predicts RES or NRES. Blue = at/above mean.",
    )
    p = FIG / "ghsl_f1_by_metro_balanced.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Confusion
    covered = bal_joined.loc[bal_joined["ghsl_covered"]]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        covered["y"].astype(int),
        covered["ghsl_y"].astype(int),
        display_labels=["NRES", "RES"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title("")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01,
        0.98,
        f"GHSL on balanced pixels · macro-F1={f1:.2f}",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        transform=fig.transFigure,
    )
    p = FIG / "ghsl_confusion_balanced.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Ladder vs AE models on balanced eval
    if cmp_ladder is not None and len(cmp_ladder):
        rows = cmp_ladder[["model", "macro_f1_balanced_eval"]].copy()
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "model": "GHSL (pixel, W4 GT)",
                            "macro_f1_balanced_eval": f1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        rows = rows.sort_values("macro_f1_balanced_eval", ascending=True)
        best = float(rows["macro_f1_balanced_eval"].max())
        short = {
            "balanced_train (W4)": "W4 balanced LR",
            "fraction_regression (W3)": "W3 Ridge",
            "pixel_ae (W3)": "W3 AE LR",
            "pixel_3class_reg (W3)": "W3 soft 3-class",
            "neighborhood_k3_reg (W3)": "W3 3×3 CNN",
            "GHSL (pixel, W4 GT)": "GHSL (pixel)",
        }
        rows["label"] = rows["model"].map(lambda x: short.get(x, x))
        colors = []
        for _, r in rows.iterrows():
            if "GHSL" in str(r["model"]):
                colors.append(ACCENT)
            elif abs(r["macro_f1_balanced_eval"] - best) < 1e-9:
                colors.append(RES)
            else:
                colors.append(MUTED)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        bars = ax.barh(rows["label"], rows["macro_f1_balanced_eval"], color=colors, height=0.7)
        label_bars_h(ax, bars, fmt="{:.2f}", pad=0.012)
        ax.set_xlabel("Macro-F1 on Week 4 balanced pixels")
        ax.set_xlim(0, 1.08)
        style_axes(ax, grid=True)
        gap = best - f1
        fig.tight_layout(rect=[0, 0, 1, 0.84])
        set_claim_title(
            fig,
            f"AE heads beat GHSL by +{gap:.2f} F1 on the same balanced pixels",
            "Orange = GHSL at AE cell centers. Blue = best AE model. Same Week 4 pure RES=NRES eval.",
        )
        p = FIG / "ghsl_vs_ae_balanced_ladder.png"
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    return paths


def update_comparison(f1_bal: float, cov: float) -> None:
    """Append GHSL pixel row into week4 model comparison CSVs + regenerate plots."""
    if not (CMP / "comparison_balanced_eval.csv").exists():
        print("No week4_model_comparison yet — skip ladder update", flush=True)
        return

    df = pd.read_csv(CMP / "comparison_balanced_eval.csv")
    df = df.loc[~df["model"].astype(str).str.contains("GHSL", case=False)].copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "model": "GHSL (pixel, W4 GT)",
                        "n": int(pd.read_parquet(BAL, columns=["pixel_id"]).shape[0]),
                        "protocol_train": "GHSL GHS-BUILT-C FUN @ AE centers",
                        "macro_f1_balanced_eval": f1_bal,
                        "mae_mean_balanced": float("nan"),
                        "macro_f1_week3_reported": 0.5121,  # week3 validation pixel GHSL
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df = df.sort_values("macro_f1_balanced_eval", ascending=False)
    df.to_csv(CMP / "comparison_balanced_eval.csv", index=False)

    ladder = pd.read_csv(CMP / "f1_ladder_cross_week.csv")
    ladder = ladder.loc[
        ~((ladder["model"].astype(str).str.contains("GHSL")) & (ladder["unit"].astype(str).str.contains("pixel")))
    ].copy()
    # Keep building GHSL; add pixel GHSL on W4 balanced
    ladder = pd.concat(
        [
            ladder,
            pd.DataFrame(
                [
                    {
                        "model": "GHSL (pixel, W4 balanced)",
                        "unit": "pixel_balanced",
                        "week": 4.0,
                        "macro_f1": f1_bal,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ladder.to_csv(CMP / "f1_ladder_cross_week.csv", index=False)

    summary_path = CMP / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    board = [
        r
        for r in summary.get("leaderboard_balanced_f1", [])
        if "GHSL" not in str(r.get("model", ""))
    ]
    board.append(
        {
            "model": "GHSL (pixel, W4 GT)",
            "macro_f1_balanced_eval": round(f1_bal, 4),
            "mae_mean_balanced": None,
            "macro_f1_week3_reported": 0.5121,
            "ghsl_coverage": round(cov, 4),
        }
    )
    board = sorted(board, key=lambda r: r["macro_f1_balanced_eval"], reverse=True)
    summary["leaderboard_balanced_f1"] = board
    takes = summary.get("takeaways", [])
    takes = [t for t in takes if "GHSL" not in t]
    best_ae = max(
        (r["macro_f1_balanced_eval"] for r in board if "GHSL" not in r["model"]),
        default=0.0,
    )
    takes.insert(
        0,
        f"GHSL at AE centers on balanced pixels: F1={f1_bal:.3f} (cov={100 * cov:.0f}%); AE best={best_ae:.3f} (Δ={best_ae - f1_bal:+.3f}).",
    )
    summary["takeaways"] = takes
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    # Regenerate comparison figures with GHSL included
    from plot_week4_model_comparison import main as plot_cmp

    # Ensure SHORT map has GHSL
    import plot_week4_model_comparison as pcm

    pcm.SHORT["GHSL (pixel, W4 GT)"] = "GHSL (pixel)"
    pcm.SHORT["GHSL (pixel, W4 balanced)"] = "GHSL (pixel, balanced)"
    plot_cmp()


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    bal = pd.read_parquet(
        BAL,
        columns=["pixel_id", "region", "pixel_lon", "pixel_lat", "y", "stratum", "frac_res", "frac_nres"],
    )
    full = pd.read_parquet(
        FULL,
        columns=["pixel_id", "region", "pixel_lon", "pixel_lat", "frac_res", "frac_nres", "frac_empty"],
    )

    # Sample unique pixel_ids across both sets to avoid double IO
    need = pd.concat(
        [
            bal[["pixel_id", "region", "pixel_lon", "pixel_lat"]],
            full[["pixel_id", "region", "pixel_lon", "pixel_lat"]],
        ],
        ignore_index=True,
    ).drop_duplicates("pixel_id")
    print(f"Unique AE cells to sample: {len(need):,}", flush=True)

    cache = OUT / "ghsl_at_ae_centers.parquet"
    if cache.exists():
        print(f"Reusing {cache.relative_to(ROOT)}", flush=True)
        ghsl = pd.read_parquet(cache)
        missing = need.loc[~need["pixel_id"].isin(ghsl["pixel_id"])]
        if len(missing):
            print(f"Sampling {len(missing):,} missing cells …", flush=True)
            extra = sample_ghsl(missing)
            ghsl = pd.concat([ghsl, extra], ignore_index=True).drop_duplicates("pixel_id")
            ghsl.to_parquet(cache, index=False)
    else:
        ghsl = sample_ghsl(need)
        ghsl.to_parquet(cache, index=False)

    bal_j, bal_sum, bal_per = score_balanced(bal, ghsl)
    bal_j.to_parquet(OUT / "predictions_balanced.parquet", index=False)
    bal_per.to_csv(OUT / "per_region_balanced.csv", index=False)

    full_j, full_sum, full_per = score_full_majority(full, ghsl)
    full_j[
        ["pixel_id", "region", "y", "ghsl_raw", "ghsl_label", "ghsl_covered", "ghsl_y", "frac_res", "frac_nres"]
    ].to_parquet(OUT / "predictions_full_majority.parquet", index=False)
    full_per.to_csv(OUT / "per_region_full_majority.csv", index=False)

    cmp_df = None
    if (CMP / "comparison_balanced_eval.csv").exists():
        cmp_df = pd.read_csv(CMP / "comparison_balanced_eval.csv")

    fig_paths = write_figures(bal_j, bal_sum, bal_per, cmp_df)

    f1_bal = float(bal_sum["functional"]["macro_f1"])
    cov = float(bal_sum["end_to_end"]["coverage"])
    update_comparison(f1_bal, cov)

    summary = {
        "name": "week4_ghsl_pixel_v1",
        "method": (
            "Nearest GHSL GHS-BUILT-C FUN value at AE pixel center "
            "(EPSG:4326→ESRI:54009). Functional = RES/NRES GHSL only."
        ),
        "balanced": bal_sum,
        "full_majority": full_sum,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "ghsl_samples": str(cache.relative_to(ROOT)),
            "predictions_balanced": "outputs/week4_ghsl_pixel/predictions_balanced.parquet",
            "figures": str(FIG.relative_to(ROOT)),
        },
        "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
        "week3_ghsl_pixel_macro_f1_reference": 0.5121,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\nBalanced functional macro-F1={f1_bal:.4f}  coverage={100 * cov:.1f}%  "
        f"e2e acc={bal_sum['end_to_end']['effective_accuracy']:.3f}",
        flush=True,
    )
    print(
        f"Full-majority functional macro-F1={full_sum['functional']['macro_f1']:.4f}  "
        f"coverage={100 * full_sum['end_to_end']['coverage']:.1f}%",
        flush=True,
    )
    print(f"Wrote {OUT} in {summary['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
