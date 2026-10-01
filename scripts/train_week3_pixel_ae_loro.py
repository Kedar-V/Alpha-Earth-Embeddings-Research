#!/usr/bin/env python3
"""Week 3 Stage 2 — AE-only LORO logistic regression on pure pixel GT + SMOTE.

Reads datasets/week3_pixel_gt/pixels.parquet, drops mixed, applies SMOTE on each
train fold, fits StandardScaler + LogisticRegression leave-one-region-out,
writes outputs/week3_pixel_ae/.
"""

from __future__ import annotations

import os

# Must be set before numpy/OpenBLAS init (CPU-only, no thread oversubscription)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PIXELS = ROOT / "datasets" / "week3_pixel_gt" / "pixels.parquet"
OUT = ROOT / "outputs" / "week3_pixel_ae"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
LABEL_DEFINITION = "resi_def_v1_pixel_overlap_v1"
# Serial folds on CPU — SMOTE + OpenBLAS are unstable under nested parallelism
N_WORKERS = 1
SMOTE_K = 5


def _fit_fold(payload: tuple[str, str, str, str, str]) -> dict:
    te_region, x_path, y_path, reg_path, cache_dir = payload
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    regions = np.load(reg_path, allow_pickle=True)
    te = regions == te_region
    tr = ~te
    X_tr = np.asarray(X[tr], dtype=np.float64)
    y_tr = np.asarray(y[tr], dtype=np.int64)
    n_before = int(tr.sum())
    k = min(SMOTE_K, max(1, int((y_tr == 1).sum()) - 1), max(1, int((y_tr == 0).sum()) - 1))
    sm = SMOTE(random_state=SEED, k_neighbors=k)
    X_bal, y_bal = sm.fit_resample(X_tr, y_tr)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight=None,
                    max_iter=2000,
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_bal, y_bal)
    X_te = np.asarray(X[te], dtype=np.float64)
    proba = pipe.predict_proba(X_te)
    classes = list(pipe.named_steps["lr"].classes_)
    pos = classes.index(1) if 1 in classes else 0
    p_res = proba[:, pos]
    pred = (p_res >= 0.5).astype(np.int64)
    y_te = np.asarray(y[te], dtype=np.int64)

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pred_path = cache / f"oof_pred_{te_region}.npy"
    p_path = cache / f"oof_p_{te_region}.npy"
    idx_path = cache / f"oof_idx_{te_region}.npy"
    np.save(pred_path, pred)
    np.save(p_path, p_res.astype(np.float64))
    np.save(idx_path, np.where(te)[0].astype(np.int64))

    return {
        "region": te_region,
        "pred_path": str(pred_path),
        "p_path": str(p_path),
        "idx_path": str(idx_path),
        "n_test": int(te.sum()),
        "n_pos": int((y_te == 1).sum()),
        "n_neg": int((y_te == 0).sum()),
        "n_train_before": n_before,
        "n_train_after_smote": int(len(y_bal)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PIXELS}", flush=True)
    df = pd.read_parquet(PIXELS)
    n_raw = len(df)
    mixed = df["label"] == "mixed"
    n_mixed = int(mixed.sum())
    pure = df.loc[~mixed].copy().reset_index(drop=True)
    pure["y"] = (pure["label"] == "residential").astype(np.int64)
    assert set(pure["label"].unique()) <= {"residential", "non_residential"}

    print(
        f"raw={n_raw}  mixed_dropped={n_mixed}  pure={len(pure)}  "
        f"RES={int((pure['y']==1).sum())}  NRES={int((pure['y']==0).sum())}  "
        f"SMOTE k={SMOTE_K}  workers={N_WORKERS}",
        flush=True,
    )

    X = pure[BANDS].to_numpy(dtype=np.float32)
    y = pure["y"].to_numpy(dtype=np.int64)
    regions = pure["region"].to_numpy()
    region_list = sorted(pure["region"].unique().tolist())

    x_path = str(CACHE / "X.npy")
    y_path = str(CACHE / "y.npy")
    reg_path = str(CACHE / "regions.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    np.save(reg_path, regions)

    oof_pred = np.full(len(pure), -1, dtype=np.int64)
    oof_p = np.full(len(pure), np.nan, dtype=np.float64)
    rows: list[dict] = []
    jobs = [(r, x_path, y_path, reg_path, str(CACHE)) for r in region_list]

    print(f"LORO LR + SMOTE · {len(jobs)} folds · {N_WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_fit_fold, job): job[0] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="LORO folds"):
            r = fut.result()
            te = np.load(r["idx_path"])
            oof_pred[te] = np.load(r["pred_path"])
            oof_p[te] = np.load(r["p_path"])
            rows.append(
                {
                    "region": r["region"],
                    "n_test": r["n_test"],
                    "n_pos": r["n_pos"],
                    "n_neg": r["n_neg"],
                    "n_train_before": r["n_train_before"],
                    "n_train_after_smote": r["n_train_after_smote"],
                    "accuracy": r["accuracy"],
                    "f1_macro": r["f1_macro"],
                }
            )
            print(
                f"  done · {r['region']:12s}  n={r['n_test']:6d}  "
                f"macro-F1={r['f1_macro']:.3f}  acc={r['accuracy']:.3f}  "
                f"train {r['n_train_before']}→{r['n_train_after_smote']}",
                flush=True,
            )

    metrics = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    metrics.to_csv(OUT / "loro_metrics.csv", index=False)

    oof = pure[
        ["pixel_id", "region", "label", "y", "frac_res", "frac_nres", "pixel_lon", "pixel_lat"]
    ].copy()
    oof["y_pred"] = oof_pred
    oof["p_res"] = oof_p
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)

    mean_f1 = float(metrics["f1_macro"].mean())
    std_f1 = float(metrics["f1_macro"].std(ddof=1))
    mean_acc = float(metrics["accuracy"].mean())
    summary = {
        "name": "week3_pixel_ae_loro_smote_v1",
        "label_definition": LABEL_DEFINITION,
        "label_definition_doc": "docs/residential_definition_v1.md",
        "design_doc": "docs/week3/pixel_ae_classifier_design.md",
        "features": BANDS,
        "model": "SMOTE + StandardScaler + LogisticRegression",
        "smote": {"k_neighbors": SMOTE_K, "sampling_strategy": "auto"},
        "protocol": "leave-one-region-out",
        "seed": SEED,
        "n_raw": n_raw,
        "n_mixed_excluded": n_mixed,
        "n_train_eval": len(pure),
        "n_res": int((pure["y"] == 1).sum()),
        "n_nres": int((pure["y"] == 0).sum()),
        "regions": region_list,
        "loro_macro_f1_mean": mean_f1,
        "loro_macro_f1_std": std_f1,
        "loro_accuracy_mean": mean_acc,
        "oof_macro_f1": float(f1_score(y, oof_pred, average="macro", zero_division=0)),
        "oof_accuracy": float(accuracy_score(y, oof_pred)),
        "workers": N_WORKERS,
        "paths": {
            "pixels": str(PIXELS.relative_to(ROOT)),
            "metrics": "outputs/week3_pixel_ae/loro_metrics.csv",
            "oof": "outputs/week3_pixel_ae/oof_predictions.parquet",
            "figures": "outputs/week3_pixel_ae/figures",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = metrics.sort_values("f1_macro")
    ax.barh(m["region"], m["f1_macro"], color="#2c7fb8")
    ax.axvline(mean_f1, color="#d95f0e", ls="--", label=f"mean={mean_f1:.3f}")
    ax.set_xlabel("LORO macro-F1")
    ax.set_title("Pixel AE LogReg + SMOTE — per-region holdout")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "loro_macro_f1_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        oof_pred,
        display_labels=["NRES", "RES"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("OOF confusion (pure + SMOTE train)")
    fig.tight_layout()
    fig.savefig(FIG / "oof_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y, oof_pred, target_names=["non_residential", "residential"], digits=3
    )
    (OUT / "oof_classification_report.txt").write_text(report)

    print("\n=== LORO metrics ===", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print(f"\nmean macro-F1={mean_f1:.3f} ± {std_f1:.3f}", flush=True)
    print(f"OOF macro-F1={summary['oof_macro_f1']:.3f}  acc={summary['oof_accuracy']:.3f}", flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
