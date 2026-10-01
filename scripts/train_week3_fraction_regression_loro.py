#!/usr/bin/env python3
"""Week 3 — AE fraction regression (LORO) on pure pixels + SMOTE, CPU-only.

Fair vs Stage-2 LR: drop mixed, SMOTE-balance RES/NRES on each train fold,
Ridge on (frac_res, frac_nres) with hard 0/1 targets from the binary label
(pure cells), majority head → RES/NRES. Eval on pure holdout only.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week3_pixel_gt" / "pixels.parquet"
STAGE2 = ROOT / "outputs" / "week3_pixel_ae" / "summary.json"
OUT = ROOT / "outputs" / "week3_fraction_regression"
FIG = OUT / "figures"
CACHE = OUT / "_cache"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
N_WORKERS = 1
SMOTE_K = 5


def majority_label(frac_res: np.ndarray, frac_nres: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.full(len(frac_res), "non_residential", dtype=object)
    out[frac_res > frac_nres + eps] = "residential"
    out[np.abs(frac_res - frac_nres) <= eps] = "tie"
    # ties → NRES (rare); keep binary-only vocabulary for pure eval
    out[out == "tie"] = "non_residential"
    return out


def _fit_fold(payload: tuple[str, str, str, str, str, str]) -> dict:
    te_region, x_path, y_path, lab_path, reg_path, cache_dir = payload
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    y_bin = np.load(lab_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te

    X_tr = np.asarray(X[tr], dtype=np.float64)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    n_before = int(tr.sum())
    k = min(SMOTE_K, max(1, int((y_tr == 1).sum()) - 1), max(1, int((y_tr == 0).sum()) - 1))
    sm = SMOTE(random_state=SEED, k_neighbors=k)
    X_bal, y_bal = sm.fit_resample(X_tr, y_tr)
    # Pure-pixel hard fraction targets from the (possibly synthetic) binary label
    Y_bal = np.column_stack([y_bal.astype(np.float64), 1.0 - y_bal.astype(np.float64)])

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "ridge",
                MultiOutputRegressor(
                    Ridge(alpha=1.0, random_state=SEED),
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_bal, Y_bal)
    pred = pipe.predict(np.asarray(X[te], dtype=np.float64))
    pred = np.clip(pred, 0.0, 1.0)

    y_te = np.asarray(Y[te], dtype=np.float64)
    mae_res = float(mean_absolute_error(y_te[:, 0], pred[:, 0]))
    mae_nres = float(mean_absolute_error(y_te[:, 1], pred[:, 1]))
    rmse_res = float(root_mean_squared_error(y_te[:, 0], pred[:, 0]))
    rmse_nres = float(root_mean_squared_error(y_te[:, 1], pred[:, 1]))

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"oof_pred_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    np.save(pred_path, pred.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))

    return {
        "region": te_region,
        "pred_path": str(pred_path),
        "idx_path": str(idx_path),
        "n_test": int(te.sum()),
        "n_train_before": n_before,
        "n_train_after_smote": int(len(y_bal)),
        "mae_res": mae_res,
        "mae_nres": mae_nres,
        "rmse_res": rmse_res,
        "rmse_nres": rmse_nres,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"CPU-only LORO + SMOTE · pure only · workers={N_WORKERS}", flush=True)

    df_raw = pd.read_parquet(PIXELS)
    n_raw = len(df_raw)
    n_mixed = int((df_raw["label"] == "mixed").sum())
    df = df_raw.loc[df_raw["label"] != "mixed"].copy().reset_index(drop=True)
    df["y"] = (df["label"] == "residential").astype(np.int64)

    X = df[BANDS].to_numpy(dtype=np.float32)
    # Keep true coverage for MAE reporting; training uses hard labels after SMOTE
    Y = df[["frac_res_full", "frac_nres_full"]].to_numpy(dtype=np.float32)
    y_bin = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].to_numpy()
    region_list = sorted(df["region"].unique().tolist())

    x_path = str(CACHE / "X.npy")
    y_path = str(CACHE / "Y.npy")
    lab_path = str(CACHE / "y_bin.npy")
    reg_path = str(CACHE / "regions.npy")
    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(lab_path, y_bin)
    np.save(reg_path, regions)
    print(
        f"raw={n_raw}  mixed_dropped={n_mixed}  pure={len(df)}  "
        f"RES={int((y_bin==1).sum())}  NRES={int((y_bin==0).sum())}  "
        f"regions={len(region_list)}",
        flush=True,
    )

    oof_pred = np.full((len(df), 2), np.nan, dtype=np.float64)
    reg_rows: list[dict] = []
    jobs = [(r, x_path, y_path, lab_path, reg_path, str(CACHE)) for r in region_list]

    print(f"LORO Ridge + SMOTE · {len(jobs)} folds", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_fit_fold, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            r = fut.result()
            te = np.load(r["idx_path"])
            pred = np.load(r["pred_path"])
            oof_pred[te] = pred
            reg_rows.append(
                {
                    "region": r["region"],
                    "n_test": r["n_test"],
                    "n_train_before": r["n_train_before"],
                    "n_train_after_smote": r["n_train_after_smote"],
                    "mae_res": r["mae_res"],
                    "mae_nres": r["mae_nres"],
                    "rmse_res": r["rmse_res"],
                    "rmse_nres": r["rmse_nres"],
                }
            )
            print(
                f"  done · {r['region']:12s}  "
                f"MAE_res={r['mae_res']:.3f} MAE_nres={r['mae_nres']:.3f}  "
                f"train {r['n_train_before']}→{r['n_train_after_smote']}",
                flush=True,
            )

    reg_metrics = pd.DataFrame(reg_rows).sort_values("region").reset_index(drop=True)
    reg_metrics.to_csv(OUT / "loro_regression_metrics.csv", index=False)

    pred_res = np.clip(oof_pred[:, 0], 0, 1)
    pred_nres = np.clip(oof_pred[:, 1], 0, 1)
    maj = majority_label(pred_res, pred_nres)

    oof = df[
        [
            "pixel_id",
            "region",
            "label",
            "y",
            "frac_res",
            "frac_nres",
            "frac_res_full",
            "frac_nres_full",
            "pixel_lon",
            "pixel_lat",
        ]
    ].copy()
    oof["pred_frac_res"] = np.round(pred_res, 2)
    oof["pred_frac_nres"] = np.round(pred_nres, 2)
    oof["pred_frac_res_full"] = pred_res
    oof["pred_frac_nres_full"] = pred_nres
    oof["pred_label_majority"] = maj
    oof["y_pred"] = (maj == "residential").astype(np.int64)
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)

    maj_rows = []
    for region in region_list:
        m = oof["region"] == region
        y_tb = oof.loc[m, "y"].to_numpy()
        y_pb = oof.loc[m, "y_pred"].to_numpy()
        maj_rows.append(
            {
                "region": region,
                "n_test": int(m.sum()),
                "n_pure": int(m.sum()),
                "accuracy_3class": float(accuracy_score(oof.loc[m, "label"], oof.loc[m, "pred_label_majority"])),
                "f1_macro_3class": float(
                    f1_score(oof.loc[m, "label"], oof.loc[m, "pred_label_majority"], average="macro", zero_division=0)
                ),
                "f1_macro_pure_binary": float(f1_score(y_tb, y_pb, average="macro", zero_division=0)),
                "accuracy_pure_binary": float(accuracy_score(y_tb, y_pb)),
            }
        )

    maj_metrics = pd.DataFrame(maj_rows).sort_values("region").reset_index(drop=True)
    maj_metrics.to_csv(OUT / "loro_majority_metrics.csv", index=False)

    mae_res = float(mean_absolute_error(Y[:, 0], pred_res))
    mae_nres = float(mean_absolute_error(Y[:, 1], pred_nres))
    rmse_res = float(root_mean_squared_error(Y[:, 0], pred_res))
    rmse_nres = float(root_mean_squared_error(Y[:, 1], pred_nres))

    pure_f1 = float(f1_score(oof["y"], oof["y_pred"], average="macro", zero_division=0))

    stage2_f1 = None
    if STAGE2.exists():
        stage2_f1 = float(json.loads(STAGE2.read_text())["loro_macro_f1_mean"])

    summary = {
        "name": "week3_fraction_regression_loro_smote_pure_v1",
        "device": "cpu",
        "features": BANDS,
        "model": "SMOTE + StandardScaler + MultiOutputRegressor(Ridge)",
        "smote": {"k_neighbors": SMOTE_K, "sampling_strategy": "auto"},
        "train_set": "pure RES/NRES only (mixed dropped)",
        "protocol": "leave-one-region-out",
        "workers": N_WORKERS,
        "seed": SEED,
        "n_raw": n_raw,
        "n_mixed_excluded": n_mixed,
        "n_pixels": len(df),
        "n_res": int((y_bin == 1).sum()),
        "n_nres": int((y_bin == 0).sum()),
        "oof_mae_res": mae_res,
        "oof_mae_nres": mae_nres,
        "oof_rmse_res": rmse_res,
        "oof_rmse_nres": rmse_nres,
        "loro_mae_res_mean": float(reg_metrics["mae_res"].mean()),
        "loro_mae_nres_mean": float(reg_metrics["mae_nres"].mean()),
        "loro_f1_macro_3class_mean": float(maj_metrics["f1_macro_3class"].mean()),
        "loro_f1_macro_pure_binary_mean": float(maj_metrics["f1_macro_pure_binary"].mean()),
        "oof_f1_macro_pure_binary": pure_f1,
        "stage2_pixel_lr_loro_macro_f1_mean": stage2_f1,
        "design_doc": "docs/week3/pixel_fraction_regression_design.md",
        "paths": {
            "regression_metrics": "outputs/week3_fraction_regression/loro_regression_metrics.csv",
            "majority_metrics": "outputs/week3_fraction_regression/loro_majority_metrics.csv",
            "oof": "outputs/week3_fraction_regression/oof_predictions.parquet",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(8000, len(df)), replace=False)
    for ax, ti, name, color, mae in [
        (axes[0], 0, "frac_res", "#d95f0e", mae_res),
        (axes[1], 1, "frac_nres", "#2c7fb8", mae_nres),
    ]:
        ax.scatter(Y[idx, ti], oof_pred[idx, ti], s=4, alpha=0.25, c=color)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"pred {name}")
        ax.set_title(f"OOF {name} (MAE={mae:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "oof_true_vs_pred_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        oof["y"],
        oof["y_pred"],
        display_labels=["NRES", "RES"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("OOF majority head (pure + SMOTE)")
    fig.tight_layout()
    fig.savefig(FIG / "oof_majority_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = maj_metrics.sort_values("f1_macro_pure_binary")
    ax.barh(m["region"], m["f1_macro_pure_binary"], color="#d95f0e", label="Regression→majority")
    if stage2_f1 is not None:
        ax.axvline(stage2_f1, color="#2c7fb8", ls="--", label=f"Stage2 pixel LR mean={stage2_f1:.3f}")
    ax.axvline(
        maj_metrics["f1_macro_pure_binary"].mean(),
        color="#d95f0e",
        ls=":",
        label=f"majority mean={maj_metrics['f1_macro_pure_binary'].mean():.3f}",
    )
    ax.set_xlabel("LORO macro-F1 (pure binary)")
    ax.set_title("Regression+SMOTE vs Stage-2 LR+SMOTE")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "majority_vs_stage2_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r = reg_metrics.sort_values("mae_res")
    x = np.arange(len(r))
    ax.bar(x - 0.2, r["mae_res"], 0.4, label="MAE frac_res", color="#d95f0e")
    ax.bar(x + 0.2, r["mae_nres"], 0.4, label="MAE frac_nres", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(r["region"], rotation=35, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Per-region regression MAE (pure + SMOTE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "loro_mae_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Regression (LORO) ===", flush=True)
    print(reg_metrics.to_string(index=False), flush=True)
    print("\n=== Majority (LORO) ===", flush=True)
    print(maj_metrics.to_string(index=False), flush=True)
    print(
        f"\nOOF MAE res={mae_res:.3f} nres={mae_nres:.3f}  "
        f"pure-binary F1={pure_f1:.3f}  stage2_lr={stage2_f1}",
        flush=True,
    )
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
