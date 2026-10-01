#!/usr/bin/env python3
"""Train LORO AE models on week4 balanced pure RES/NRES pixels.

No SMOTE (counts already balanced per region).
- Soft targets: (frac_res, frac_nres, frac_empty) via Ridge multi-output
- Binary head: LogisticRegression on stratum y

Writes outputs/week4_pixel_balanced_train/.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_pixel_balanced_train"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
CKPT = OUT / "checkpoints"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
FRAC_COLS = ("frac_res", "frac_nres", "frac_empty")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def _fit_fold(payload: tuple) -> dict:
    te_region, x_path, y_path, ybin_path, reg_path, cache_dir, ckpt_dir = payload
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    y_bin = np.load(ybin_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = np.asarray(X[tr], dtype=np.float64)
    Y_tr = np.asarray(Y[tr], dtype=np.float64)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float64)
    y_te = np.asarray(Y[te], dtype=np.float64)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(Ridge(alpha=1.0, random_state=SEED), n_jobs=1)),
        ]
    )
    reg.fit(X_tr, Y_tr)
    pred = np.clip(reg.predict(X_te), 0.0, 1.0)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    p_res = clf.predict_proba(X_te)[:, 1]

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    pred_path = cache / f"oof_pred_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    ypred_path = cache / f"oof_ypred_{te_region}.npy"
    pres_path = cache / f"oof_pres_{te_region}.npy"
    np.save(pred_path, pred.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))
    np.save(ypred_path, y_pred.astype(np.int64))
    np.save(pres_path, p_res.astype(np.float64))
    joblib.dump(
        {
            "region_held_out": te_region,
            "regressor": reg,
            "classifier": clf,
            "seed": SEED,
            "n_train": int(tr.sum()),
            "n_train_res": int((y_tr == 1).sum()),
            "n_train_nres": int((y_tr == 0).sum()),
        },
        ckpt / f"fold_{te_region}.joblib",
    )

    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(3)]
    return {
        "region": te_region,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "y_pred_path": str(ypred_path),
        "p_res_path": str(pres_path),
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
        "n_train_res": int((y_tr == 1).sum()),
        "n_train_nres": int((y_tr == 0).sum()),
        "mae_frac_res": mae[0],
        "mae_frac_nres": mae[1],
        "mae_frac_empty": mae[2],
        "mae_mean": float(np.mean(mae)),
        "macro_f1": float(f1_score(yb_te, y_pred, average="macro", zero_division=0)),
    }


def score_full_gt_oof(ckpt_dir: Path, out_path: Path) -> dict:
    """Apply each fold's held-out model to ALL pure cells in that region (full GT)."""
    if not FULL_GT.exists():
        raise FileNotFoundError(FULL_GT)
    print(f"Scoring full GT pure cells with fold checkpoints → {out_path}", flush=True)
    full = pd.read_parquet(FULL_GT)
    pure = full.loc[
        ((full["frac_res_full"] > 0) & (full["frac_nres_full"] == 0))
        | ((full["frac_nres_full"] > 0) & (full["frac_res_full"] == 0))
    ].copy()
    pure["y"] = (pure["frac_res_full"] > 0).astype(np.int64)
    pure["stratum"] = np.where(pure["y"] == 1, "residential", "non_residential")

    rows = []
    for region in sorted(pure["region"].astype(str).unique()):
        ck = ckpt_dir / f"fold_{region}.joblib"
        if not ck.exists():
            print(f"  skip {region}: missing {ck.name}", flush=True)
            continue
        bundle = joblib.load(ck)
        sub = pure.loc[pure["region"].astype(str) == region]
        X = sub[BANDS].to_numpy(dtype=np.float64)
        pred = np.clip(bundle["regressor"].predict(X), 0.0, 1.0)
        p_res = bundle["classifier"].predict_proba(X)[:, 1]
        y_pred = bundle["classifier"].predict(X)
        part = sub[
            [
                "pixel_id",
                "region",
                "stratum",
                "y",
                "frac_res",
                "frac_nres",
                "frac_empty",
                "frac_res_full",
                "frac_nres_full",
                "frac_empty_full",
                "pixel_lon",
                "pixel_lat",
            ]
        ].copy()
        part["pred_frac_res"] = np.round(pred[:, 0], 2)
        part["pred_frac_nres"] = np.round(pred[:, 1], 2)
        part["pred_frac_empty"] = np.round(pred[:, 2], 2)
        part["pred_frac_res_full"] = pred[:, 0]
        part["pred_frac_nres_full"] = pred[:, 1]
        part["pred_frac_empty_full"] = pred[:, 2]
        part["y_pred"] = y_pred
        part["p_res"] = p_res
        rows.append(part)
        f1 = float(f1_score(part["y"], y_pred, average="macro", zero_division=0))
        print(f"  {region:12s} n={len(part):,}  macroF1={f1:.3f}", flush=True)

    if not rows:
        raise RuntimeError("No full-GT OOF rows produced")
    oof = pd.concat(rows, ignore_index=True)
    oof.to_parquet(out_path, index=False)
    summary = {
        "n": int(len(oof)),
        "n_res": int((oof["y"] == 1).sum()),
        "n_nres": int((oof["y"] == 0).sum()),
        "macro_f1": float(f1_score(oof["y"], oof["y_pred"], average="macro", zero_division=0)),
        "path": str(out_path.relative_to(ROOT)),
    }
    print(
        f"Full-GT OOF n={summary['n']:,} RES={summary['n_res']:,} NRES={summary['n_nres']:,} "
        f"macroF1={summary['macro_f1']:.3f}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--also-score-full-gt",
        action="store_true",
        help="After balanced OOF, apply fold models to all pure cells in week4_pixel_gt",
    )
    args = ap.parse_args()

    t0 = time.time()
    for d in (OUT, FIG, CACHE, CKPT):
        d.mkdir(parents=True, exist_ok=True)

    if not PIXELS.exists():
        raise FileNotFoundError(
            f"Missing {PIXELS}; run scripts/build_week4_pixel_gt_balanced.py first"
        )

    df = pd.read_parquet(PIXELS).reset_index(drop=True)
    X = df[BANDS].to_numpy(dtype=np.float32)
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float32)
    y_bin = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].to_numpy()
    region_list = sorted(df["region"].astype(str).unique().tolist())

    x_path = str(CACHE / "X.npy")
    y_path = str(CACHE / "Y.npy")
    ybin_path = str(CACHE / "y_bin.npy")
    reg_path = str(CACHE / "regions.npy")
    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(ybin_path, y_bin)
    np.save(reg_path, regions)

    n_workers = args.workers or min(len(region_list), os.cpu_count() or 1)
    print(
        f"Balanced LORO · n={len(df):,} RES={int((y_bin==1).sum()):,} "
        f"NRES={int((y_bin==0).sum()):,} folds={len(region_list)} workers={n_workers}",
        flush=True,
    )

    jobs = [
        (r, x_path, y_path, ybin_path, reg_path, str(CACHE), str(CKPT)) for r in region_list
    ]
    oof_pred = np.full((len(df), 3), np.nan, dtype=np.float64)
    oof_ypred = np.full(len(df), -1, dtype=np.int64)
    oof_pres = np.full(len(df), np.nan, dtype=np.float64)
    rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_fit_fold, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            r = fut.result()
            te = np.load(r["idx_path"])
            oof_pred[te] = np.load(r["pred_path"])
            oof_ypred[te] = np.load(r["y_pred_path"])
            oof_pres[te] = np.load(r["p_res_path"])
            rows.append(r)
            print(
                f"  {r['region']:12s}  mae_mean={r['mae_mean']:.3f}  "
                f"macroF1={r['macro_f1']:.3f}  "
                f"train RES/NRES={r['n_train_res']}/{r['n_train_nres']}",
                flush=True,
            )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)

    oof = df[
        [
            "pixel_id",
            "region",
            "stratum",
            "y",
            "frac_res",
            "frac_nres",
            "frac_empty",
            "frac_res_full",
            "frac_nres_full",
            "frac_empty_full",
            "pixel_lon",
            "pixel_lat",
        ]
    ].copy()
    oof["pred_frac_res"] = np.round(oof_pred[:, 0], 2)
    oof["pred_frac_nres"] = np.round(oof_pred[:, 1], 2)
    oof["pred_frac_empty"] = np.round(oof_pred[:, 2], 2)
    oof["pred_frac_res_full"] = oof_pred[:, 0]
    oof["pred_frac_nres_full"] = oof_pred[:, 1]
    oof["pred_frac_empty_full"] = oof_pred[:, 2]
    oof["y_pred"] = oof_ypred
    oof["p_res"] = oof_pres
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)

    yt, yp = Y, oof_pred
    pooled: dict = {}
    for i, name in enumerate(FRAC_COLS):
        pooled[f"oof_mae_{name}"] = float(mean_absolute_error(yt[:, i], yp[:, i]))
        pooled[f"oof_rmse_{name}"] = _rmse(yt[:, i], yp[:, i])
        pooled[f"oof_r2_{name}"] = float(r2_score(yt[:, i], yp[:, i]))
    pooled["oof_mae_mean"] = float(np.mean([pooled[f"oof_mae_{c}"] for c in FRAC_COLS]))
    pooled["oof_macro_f1"] = float(f1_score(y_bin, oof_ypred, average="macro", zero_division=0))
    pooled["loro_mae_mean_mean"] = float(metrics["mae_mean"].mean())
    pooled["loro_mae_mean_std"] = float(metrics["mae_mean"].std(ddof=0))
    pooled["loro_macro_f1_mean"] = float(metrics["macro_f1"].mean())
    pooled["loro_macro_f1_std"] = float(metrics["macro_f1"].std(ddof=0))

    # --- Figures (claim-first) ---
    metrics_s = metrics.sort_values("macro_f1", ascending=True).copy()
    metrics_s["region_lab"] = metrics_s["region"].astype(str).str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    mean_f1 = pooled["loro_macro_f1_mean"]
    colors = ["#0072B2" if v >= mean_f1 else "#B0B0B0" for v in metrics_s["macro_f1"]]
    bars = ax.barh(metrics_s["region_lab"], metrics_s["macro_f1"], color=colors, height=0.7)
    ax.axvline(mean_f1, color="#333333", ls="--", lw=1)
    ax.text(mean_f1 + 0.01, len(metrics_s) - 0.5, f"mean {mean_f1:.2f}", fontsize=8, color="#333333")
    for rect, v in zip(bars, metrics_s["macro_f1"]):
        ax.text(v + 0.01, rect.get_y() + rect.get_height() / 2, f"{v:.2f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(
        0.01, 0.98,
        f"LORO macro-F1 averages {mean_f1:.2f} on balanced pure cells",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        "Blue = at/above mean; grey = below. Binary head on Ridge features.",
        fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / "loro_f1_by_region.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics_m = metrics.sort_values("mae_mean", ascending=False).copy()
    metrics_m["region_lab"] = metrics_m["region"].astype(str).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    mean_mae = pooled["loro_mae_mean_mean"]
    colors = ["#009E73" if v <= mean_mae else "#B0B0B0" for v in metrics_m["mae_mean"]]
    bars = ax.barh(metrics_m["region_lab"], metrics_m["mae_mean"], color=colors, height=0.7)
    ax.axvline(mean_mae, color="#333333", ls="--", lw=1)
    for rect, v in zip(bars, metrics_m["mae_mean"]):
        ax.text(v + 0.005, rect.get_y() + rect.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlabel("Mean MAE (three coverage fractions)")
    ax.set_xlim(0, float(metrics_m["mae_mean"].max()) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.text(
        0.01, 0.98,
        f"Fraction MAE averages {mean_mae:.3f} across held-out metros",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / "loro_mae_by_region.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    colors = ["#0072B2", "#E69F00", "#7F7F7F"]
    titles = ["Residential", "Non-residential", "Empty"]
    for ax, i, name, c, t in zip(axes, range(3), FRAC_COLS, colors, titles):
        ax.scatter(yt[:, i], yp[:, i], s=4, alpha=0.15, c=c, rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel(f"True {t.lower()}")
        ax.set_ylabel(f"Predicted")
        ax.set_title(f"{t}  ·  MAE={pooled[f'oof_mae_{name}']:.3f}", loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        "Predictions track the diagonal; empty is easiest, RES/NRES noisier",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / "oof_true_vs_pred.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_bin, oof_ypred, display_labels=["NRES", "RES"], ax=ax, colorbar=False,
        cmap="Blues",
    )
    ax.set_title("")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.text(
        0.01, 0.98,
        f"OOF confusion · macro-F1={pooled['oof_macro_f1']:.2f}",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.savefig(FIG / "oof_confusion.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "name": "week4_pixel_balanced_loro_v1",
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "data": str(PIXELS.relative_to(ROOT)),
        "protocol": "leave-one-region-out",
        "balance": "per-region equal pure RES/NRES counts; no SMOTE; undersample majority (RES)",
        "features": BANDS,
        "regressor": "StandardScaler + MultiOutputRegressor(Ridge)",
        "classifier": "StandardScaler + LogisticRegression",
        "seed": SEED,
        "workers": n_workers,
        "n_pixels": int(len(df)),
        "n_residential": int((y_bin == 1).sum()),
        "n_non_residential": int((y_bin == 0).sum()),
        "regions": region_list,
        **pooled,
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "oof": str((OUT / "oof_predictions.parquet").relative_to(ROOT)),
            "metrics": str((OUT / "loro_metrics.csv").relative_to(ROOT)),
            "checkpoints": str(CKPT.relative_to(ROOT)),
            "figures": str(FIG.relative_to(ROOT)),
        },
    }
    if args.also_score_full_gt:
        full_path = OUT / "oof_predictions_full_gt.parquet"
        full_sum = score_full_gt_oof(CKPT, full_path)
        summary["full_gt_oof"] = full_sum
        summary["paths"]["oof_full_gt"] = full_sum["path"]
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"OOF mae_mean={pooled['oof_mae_mean']:.4f}  "
        f"macroF1={pooled['oof_macro_f1']:.4f}  "
        f"LORO F1={pooled['loro_macro_f1_mean']:.4f}±{pooled['loro_macro_f1_std']:.4f}  "
        f"{summary['elapsed_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
