#!/usr/bin/env python3
"""Week 3 — 3-class (RES/NRES/empty) LORO on all GPUs.

Mirrors notebooks/week3/05_pixel_res_nres_empty.ipynb training path.

- Classifier: SMOTE + hard labels + CE
- Regression: soft GT frac_res/nres/empty + RandomOverSampler (carry soft Y) + softmax MSE
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week3_pixel_3class_trainlib import (  # noqa: E402
    BUILDING_SHARE_MAX,
    CLASSES,
    clf_fold_worker,
    decode_fractions,
    reg_fold_worker,
)

MP_CTX = mp.get_context("spawn")
PIXELS = ROOT / "datasets" / "week3_pixel_gt_v2" / "pixels.parquet"
OUT = ROOT / "outputs" / "week3_pixel_3class"
FIG = OUT / "figures"
CACHE = OUT / "_cache"
BANDS = [f"A{i:02d}" for i in range(64)]


def main() -> None:
    for p in (OUT, FIG, CACHE):
        p.mkdir(parents=True, exist_ok=True)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"CUDA={torch.cuda.is_available()} n_gpu={n_gpu}", flush=True)
    if n_gpu:
        for i in range(n_gpu):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}", flush=True)

    raw = pd.read_parquet(PIXELS)
    df = raw.loc[raw["label"] != "mixed"].copy().reset_index(drop=True)
    if "frac_empty_full" not in df.columns:
        df["frac_empty_full"] = np.clip(1.0 - df["frac_res_full"] - df["frac_nres_full"], 0, 1)
    print(df["label"].value_counts().to_dict(), flush=True)

    X = df[BANDS].to_numpy(dtype=np.float32)
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(df["label"].to_numpy())
    Y_frac = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float32)
    regions = df["region"].to_numpy()
    region_list = sorted(df["region"].unique().tolist())
    np.save(CACHE / "X.npy", X)
    np.save(CACHE / "y.npy", y)
    np.save(CACHE / "Y_frac.npy", Y_frac)
    np.save(CACHE / "regions.npy", regions)

    x_path, y_path, reg_path = str(CACHE / "X.npy"), str(CACHE / "y.npy"), str(CACHE / "regions.npy")
    yf_path = str(CACHE / "Y_frac.npy")
    max_workers = max(1, n_gpu) if n_gpu else min(4, os.cpu_count() or 1)

    # --- classifier ---
    jobs = [
        {
            "te_region": r,
            "x_path": x_path,
            "y_path": y_path,
            "reg_path": reg_path,
            "cache_dir": str(CACHE),
            "gpu_id": (i % n_gpu) if n_gpu else None,
        }
        for i, r in enumerate(region_list)
    ]
    oof_pred = np.full(len(df), -1, dtype=np.int64)
    oof_proba = np.full((len(df), 3), np.nan)
    clf_rows = []
    print(f"clf LORO · {len(jobs)} folds · workers={max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CTX) as pool:
        futs = {pool.submit(clf_fold_worker, job): job["te_region"] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="clf"):
            r = fut.result()
            te = np.load(CACHE / f"clf_idx_{r['region']}.npy")
            oof_pred[te] = np.load(CACHE / f"clf_pred_{r['region']}.npy")
            oof_proba[te] = np.load(CACHE / f"clf_proba_{r['region']}.npy")
            clf_rows.append(r)
            print(f"  {r['region']:12s} F1={r['f1_macro']:.3f} [{r['device']}]", flush=True)

    clf_metrics = pd.DataFrame(clf_rows).sort_values("region")
    clf_metrics.to_csv(OUT / "loro_clf_metrics.csv", index=False)
    print("mean clf F1", float(clf_metrics.f1_macro.mean()), flush=True)

    # --- regression ---
    jobs = [
        {
            "te_region": r,
            "x_path": x_path,
            "y_path": y_path,
            "yf_path": yf_path,
            "reg_path": reg_path,
            "cache_dir": str(CACHE),
            "gpu_id": (i % n_gpu) if n_gpu else None,
        }
        for i, r in enumerate(region_list)
    ]
    oof_frac = np.full((len(df), 3), np.nan)
    reg_rows = []
    print(f"reg LORO · {len(jobs)} folds · workers={max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CTX) as pool:
        futs = {pool.submit(reg_fold_worker, job): job["te_region"] for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="reg"):
            r = fut.result()
            te = np.load(CACHE / f"reg_idx_{r['region']}.npy")
            oof_frac[te] = np.load(CACHE / f"reg_pred_{r['region']}.npy")
            reg_rows.append(r)
            print(f"  {r['region']:12s} F1={r['f1_macro']:.3f} [{r['device']}]", flush=True)

    reg_metrics = pd.DataFrame(reg_rows).sort_values("region")
    reg_metrics.to_csv(OUT / "loro_reg_metrics.csv", index=False)
    decoded = decode_fractions(
        oof_frac[:, 0], oof_frac[:, 1], oof_frac[:, 2],
        building_share_max=BUILDING_SHARE_MAX,
    )
    class_to_i = {c: i for i, c in enumerate(le.classes_)}
    oof_reg_pred = np.array([class_to_i[s] for s in decoded], dtype=np.int64)
    print("mean reg F1", float(reg_metrics.f1_macro.mean()), flush=True)
    print("=== CLF ===")
    print(classification_report(y, oof_pred, target_names=list(le.classes_), digits=3))
    print("=== REG ===")
    print(classification_report(y, oof_reg_pred, target_names=list(le.classes_), digits=3))

    oof = df[
        ["pixel_id", "region", "label", "pixel_lon", "pixel_lat", "frac_res", "frac_nres", "frac_empty"]
    ].copy()
    oof["y"] = y
    oof["y_pred_clf"] = oof_pred
    oof["y_pred_reg"] = oof_reg_pred
    oof["pred_label_clf"] = le.inverse_transform(oof_pred)
    oof["pred_label_reg"] = decoded
    oof["pred_frac_res"] = np.round(oof_frac[:, 0], 2)
    oof["pred_frac_nres"] = np.round(oof_frac[:, 1], 2)
    oof["pred_frac_empty"] = np.round(oof_frac[:, 2], 2)
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title in [
        (axes[0], oof_pred, "Classifier OOF"),
        (axes[1], oof_reg_pred, "Regression→decode OOF"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y, pred, display_labels=["empty", "NRES", "RES"], cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / "oof_confusion_clf_vs_reg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    m = clf_metrics.merge(reg_metrics[["region", "f1_macro"]], on="region", suffixes=("_clf", "_reg")).sort_values(
        "f1_macro_clf"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(m))
    ax.barh(ypos - 0.2, m["f1_macro_clf"], 0.4, label="Classifier", color="#c45c26")
    ax.barh(ypos + 0.2, m["f1_macro_reg"], 0.4, label="Regression→decode", color="#2b6a8f")
    ax.set_yticks(ypos)
    ax.set_yticklabels(m["region"])
    ax.set_xlabel("LORO macro-F1 (3-class)")
    ax.set_title("Classifier vs regression head by metro")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "loro_f1_clf_vs_reg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    m = m.copy()
    m["delta"] = m["f1_macro_reg"] - m["f1_macro_clf"]
    m = m.sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2a6f4e" if v >= 0 else "#8b3a3a" for v in m["delta"]]
    bars = ax.barh(m["region"], m["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#222", lw=1)
    ax.set_xlabel("macro-F1 gap (regression head − classifier)")
    ax.set_title(f"Regression head beats classifier in {int((m.delta >= 0).sum())}/{len(m)} metros")
    for bar, v in zip(bars, m["delta"]):
        ax.text(
            v + (0.004 if v >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIG / "delta_reg_minus_clf_by_city.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "name": "week3_pixel_3class_res_nres_empty_v1",
        "n_gpu": n_gpu,
        "n_pixels": int(len(df)),
        "building_share_max": BUILDING_SHARE_MAX,
        "decode": "empty if frac_res+frac_nres < building_share_max else argmax(res,nres)",
        "clf_targets": "hard_label_smote",
        "reg_targets": "soft_frac_res_nres_empty_ros",
        "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "clf_loro_macro_f1_mean": float(clf_metrics["f1_macro"].mean()),
        "clf_loro_macro_f1_std": float(clf_metrics["f1_macro"].std(ddof=1)),
        "reg_loro_macro_f1_mean": float(reg_metrics["f1_macro"].mean()),
        "reg_loro_macro_f1_std": float(reg_metrics["f1_macro"].std(ddof=1)),
        "oof_clf_macro_f1": float(f1_score(y, oof_pred, average="macro", zero_division=0)),
        "oof_reg_macro_f1": float(f1_score(y, oof_reg_pred, average="macro", zero_division=0)),
        "reg_beats_clf_metros": int((m["delta"] >= 0).sum()),
        "n_metros": int(len(m)),
        "artifacts": {
            "oof": str(OUT / "oof_predictions.parquet"),
            "clf_metrics": str(OUT / "loro_clf_metrics.csv"),
            "reg_metrics": str(OUT / "loro_reg_metrics.csv"),
            "figures": str(FIG),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
