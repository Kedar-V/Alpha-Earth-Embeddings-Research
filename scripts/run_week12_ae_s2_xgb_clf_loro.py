#!/usr/bin/env python3
"""Week 12: AE+S2 → XGBClassifier LORO (binary soft-argmax RES vs NRES).

Label: y = 1 if soft_gt_frac_res >= soft_gt_frac_nres else 0
  (mixed ~4% assigned to dominant soft class).

Eval: locked Section 3/4 building IDs (docs/eval_protocol_apples_to_apples.md).
Headline side metrics: RES F1, NRES F1, macro F1 vs GHS-BUILT-S soft-argmax
and AE+S2 regression soft-argmax.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
S2_TRAIN = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "train_buildings_ae_s2.parquet"
S2_OOF = ROOT / "outputs" / "week9_ae_s2_nres_loro" / "latest" / "artifacts" / "buildings_oof_ae_s2.parquet"
EMPTY_OOF = (
    ROOT
    / "outputs"
    / "week11_ae_s2_empty_head_loro"
    / "latest"
    / "artifacts"
    / "buildings_oof_ae_s2_empty.parquet"
)
XGB_BUILD = (
    ROOT
    / "outputs"
    / "week7_section3_xgb_mae"
    / "20261113_033609"
    / "artifacts"
    / "buildings_section3_with_xgb.parquet"
)
RING_OOF = ROOT / "outputs" / "week8_ae_rings_loro" / "latest" / "artifacts" / "buildings_oof_ring.parquet"

BANDS = [f"A{i:02d}" for i in range(64)]
S2_FEATS = [
    "s2_blue",
    "s2_green",
    "s2_red",
    "s2_nir",
    "s2_brightness",
    "s2_blue_dom",
    "s2_ndvi",
    "s2_contrast_small",
    "s2_contrast_large",
    "s2_contrast_ratio",
    "fp_area_m2",
    "fp_equiv_diam_m",
    "fp_compactness",
    "s2_nres_score",
    "s2_n_pix",
    "s2_ok",
]
SEED = 42

METRO_META = [
    ("Boston_NE", "Boston NE", "#ECE8F4"),
    ("NYC_NE", "NYC NE", "#ECE8F4"),
    ("Atlanta_SE", "Atlanta SE", "#E2EFE6"),
    ("Houston_S", "Houston S", "#F3EDD5"),
    ("Chicago_MW", "Chicago MW", "#F3E3E7"),
    ("Phoenix_SW", "Phoenix SW", "#EFE4D9"),
    ("Denver_W", "Denver W", "#E1ECEE"),
    ("LA_W", "LA W", "#E1ECEE"),
    ("SF_W", "SF W", "#E1ECEE"),
    ("Seattle_NW", "Seattle NW", "#E9EDDC"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    if "_t0" not in prev and "_t0" not in kwargs:
        kwargs["_t0"] = time.time()
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0.0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def detect_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []


def feat_colnames() -> list[str]:
    return BANDS + S2_FEATS


def soft_argmax_res(res: np.ndarray, nres: np.ndarray) -> np.ndarray:
    return (np.asarray(res, float) >= np.asarray(nres, float)).astype(np.int32)


def clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    m = np.isfinite(y_true.astype(float)) & np.isfinite(y_pred.astype(float))
    yt, yp = y_true[m], y_pred[m]
    if len(yt) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "res_f1": float("nan"), "nres_f1": float("nan")}
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "nres_precision": float(p[0]),
        "nres_recall": float(r[0]),
        "nres_f1": float(f[0]),
        "nres_support": int(s[0]),
        "res_precision": float(p[1]),
        "res_recall": float(r[1]),
        "res_f1": float(f[1]),
        "res_support": int(s[1]),
        "report": classification_report(yt, yp, target_names=["NRES", "RES"], digits=4, zero_division=0),
    }


def decorate_metro(rows: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in rows}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        out.append(r)
    return out


def build_training_table(run_dir: Path, s2_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(s2_path)
    df["id"] = df["id"].astype(str)
    df["y_soft"] = soft_argmax_res(df["soft_gt_frac_res"], df["soft_gt_frac_nres"])
    both = (df["soft_gt_frac_res"] > 0) & (df["soft_gt_frac_nres"] > 0)
    df["is_mixed_both"] = both.astype(np.int8)
    for c in feat_colnames():
        if c not in df.columns:
            raise RuntimeError(f"missing feature {c}")
        df[c] = df[c].astype(np.float32)
    out = run_dir / "artifacts" / "train_buildings_ae_s2_clf.parquet"
    df.to_parquet(out, index=False)
    log(
        run_dir,
        f"train n={len(df):,} RES%={100*df['y_soft'].mean():.1f} "
        f"mixed_both%={100*df['is_mixed_both'].mean():.2f} features={len(feat_colnames())}",
    )
    return df


def _loro_fold(payload: dict[str, Any]) -> dict[str, Any]:
    from xgboost import XGBClassifier

    hold = payload["hold"]
    train_path = Path(payload["train_path"])
    cols = payload["cols"]
    n_jobs = payload["n_jobs"]
    device = payload.get("device", "cpu")
    t0 = time.time()

    df = pd.read_parquet(train_path, columns=["region", "y_soft", *cols])
    te = (df["region"] == hold).to_numpy()
    tr = ~te
    te_idx = np.where(te)[0].astype(np.int64)
    Xtr = np.nan_to_num(df.loc[tr, cols].to_numpy(np.float32), nan=0.0)
    Ytr = df.loc[tr, "y_soft"].to_numpy(np.int32)
    Xte = np.nan_to_num(df.loc[te, cols].to_numpy(np.float32), nan=0.0)
    Yte = df.loc[te, "y_soft"].to_numpy(np.int32)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # scale_pos_weight for RES minority/majority imbalance per fold
    n_pos = max(1, int((Ytr == 1).sum()))
    n_neg = max(1, int((Ytr == 0).sum()))
    spw = n_neg / n_pos

    kwargs: dict[str, Any] = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=n_jobs,
        random_state=SEED,
        tree_method="hist",
        scale_pos_weight=spw,
    )
    if device.startswith("cuda"):
        kwargs["device"] = device
    model = XGBClassifier(**kwargs)
    model.fit(Xtr_s, Ytr)
    proba = model.predict_proba(Xte_s)[:, 1].astype(np.float32)
    pred = (proba >= 0.5).astype(np.int32)
    met = clf_metrics(Yte, pred)
    return {
        "hold": hold,
        "te_idx": te_idx,
        "pred": pred,
        "proba": proba,
        "metrics": met,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "seconds": round(time.time() - t0, 1),
        "device": device,
        "scale_pos_weight": float(spw),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
    }


def train_loro_clf(df: pd.DataFrame, run_dir: Path, fold_workers: int) -> pd.DataFrame:
    cols = feat_colnames()
    regions = sorted(df["region"].unique())
    train_path = run_dir / "artifacts" / "train_buildings_ae_s2_clf.parquet"
    oof_pred = np.full(len(df), -1, dtype=np.int32)
    oof_proba = np.full(len(df), np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    ckpt = run_dir / "checkpoints" / "folds"
    ckpt.mkdir(parents=True, exist_ok=True)

    n_gpus = len(detect_gpus())
    log(run_dir, f"LORO XGBClassifier folds={len(regions)} workers={fold_workers} gpus={n_gpus}")

    payloads = []
    for i, hold in enumerate(regions):
        device = f"cuda:{i % n_gpus}" if n_gpus > 0 else "cpu"
        n_jobs = 1 if n_gpus > 0 else max(1, (os.cpu_count() or 8) // max(1, fold_workers))
        payloads.append(
            {
                "hold": hold,
                "train_path": str(train_path),
                "cols": cols,
                "n_jobs": n_jobs,
                "device": device,
            }
        )

    write_progress(run_dir, status="running", stage="train_loro", completed=0, total=len(regions), failed=0)
    max_w = min(fold_workers, n_gpus) if n_gpus > 0 else fold_workers
    max_w = max(1, min(max_w, len(payloads)))
    done = 0
    failed = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
        futs = {pool.submit(_loro_fold, p): p["hold"] for p in payloads}
        for fut in as_completed(futs):
            hold = futs[fut]
            try:
                res = fut.result()
                oof_pred[res["te_idx"]] = res["pred"]
                oof_proba[res["te_idx"]] = res["proba"]
                fold_metrics.append(
                    {
                        "holdout": res["hold"],
                        "n_train": res["n_train"],
                        "n_test": res["n_test"],
                        "macro_f1": res["metrics"]["macro_f1"],
                        "res_f1": res["metrics"]["res_f1"],
                        "nres_f1": res["metrics"]["nres_f1"],
                        "accuracy": res["metrics"]["accuracy"],
                        "seconds": res["seconds"],
                        "device": res["device"],
                        "scale_pos_weight": res["scale_pos_weight"],
                    }
                )
                np.savez_compressed(
                    ckpt / f"fold_{hold}.npz",
                    pred=res["pred"],
                    proba=res["proba"],
                    te_idx=res["te_idx"],
                    scaler_mean=res["scaler_mean"],
                    scaler_scale=res["scaler_scale"],
                )
                log(
                    run_dir,
                    f"LORO hold={hold}: macroF1={res['metrics']['macro_f1']:.4f} "
                    f"RES_F1={res['metrics']['res_f1']:.4f} NRES_F1={res['metrics']['nres_f1']:.4f} "
                    f"acc={res['metrics']['accuracy']:.4f} n_te={res['n_test']:,} "
                    f"device={res['device']} in {res['seconds']}s",
                )
            except Exception as e:
                failed += 1
                log(run_dir, f"LORO FAIL {hold}: {e}")
                atomic_json(run_dir / "failures" / f"fold_{hold}.json", {"hold": hold, "error": str(e)})
            done += 1
            write_progress(
                run_dir,
                status="running",
                stage="train_loro",
                completed=done,
                total=len(regions),
                failed=failed,
            )
    if failed:
        raise RuntimeError(f"{failed} folds failed")
    if (oof_pred < 0).any():
        raise RuntimeError("incomplete OOF preds")

    keep = [
        "id",
        "region",
        "soft_gt_frac_res",
        "soft_gt_frac_nres",
        "ae_frac_res",
        "ae_frac_nres",
        "ghsl_frac_res",
        "ghsl_frac_nres",
        "pls_frac_res",
        "pls_frac_nres",
        "y_soft",
        "is_mixed_both",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["clf_pred"] = oof_pred
    out["clf_proba_res"] = oof_proba

    if S2_OOF.exists():
        s2 = pd.read_parquet(S2_OOF.resolve(), columns=["id", "s2_frac_res", "s2_frac_nres"])
        s2["id"] = s2["id"].astype(str)
        out = out.merge(s2, on="id", how="left")
    if EMPTY_OOF.exists():
        e = pd.read_parquet(
            EMPTY_OOF.resolve(),
            columns=["id", "s2e_frac_res", "s2e_frac_nres", "s2e_renorm_frac_res", "s2e_renorm_frac_nres"],
        )
        e["id"] = e["id"].astype(str)
        out = out.merge(e, on="id", how="left")
    if RING_OOF.exists():
        rk = pd.read_parquet(RING_OOF.resolve(), columns=["id", "ring_frac_res", "ring_frac_nres"])
        rk["id"] = rk["id"].astype(str)
        out = out.merge(rk, on="id", how="left")

    out.to_parquet(run_dir / "artifacts" / "buildings_oof_ae_s2_clf.parquet", index=False)
    pd.DataFrame(fold_metrics).sort_values("holdout").to_csv(run_dir / "artifacts" / "fold_metrics.csv", index=False)
    atomic_json(run_dir / "artifacts" / "fold_metrics.json", sorted(fold_metrics, key=lambda x: x["holdout"]))
    return out


def score_buildings(run_dir: Path, oof: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    oof = oof.copy()
    oof["id"] = oof["id"].astype(str)
    m = cohort.merge(oof, on=["id", "region"], how="left")
    assert len(m) == 71497, f"building cohort size {len(m)} != 71497"

    if "y_soft" not in m.columns or m["y_soft"].isna().any():
        m["y_soft"] = soft_argmax_res(m["soft_gt_frac_res"], m["soft_gt_frac_nres"])

    if XGB_BUILD.exists():
        xb = pd.read_parquet(XGB_BUILD, columns=["id", "xgb_frac_res", "xgb_frac_nres"])
        xb["id"] = xb["id"].astype(str)
        m = m.merge(xb, on="id", how="left")

    y = m["y_soft"].to_numpy(np.int32)
    models: dict[str, np.ndarray] = {
        "ghsl_built_s": soft_argmax_res(m["ghsl_frac_res"], m["ghsl_frac_nres"]),
        "ae_ridge": soft_argmax_res(m["ae_frac_res"], m["ae_frac_nres"]),
        "ae_pls": soft_argmax_res(m["pls_frac_res"], m["pls_frac_nres"]),
    }
    if "xgb_frac_res" in m.columns:
        models["ae_xgb"] = soft_argmax_res(m["xgb_frac_res"], m["xgb_frac_nres"])
    if "ring_frac_res" in m.columns:
        models["ae_ring"] = soft_argmax_res(m["ring_frac_res"], m["ring_frac_nres"])
    if "s2_frac_res" in m.columns:
        models["ae_s2_reg"] = soft_argmax_res(m["s2_frac_res"], m["s2_frac_nres"])
    if "s2e_frac_res" in m.columns:
        models["ae_s2_empty_reg"] = soft_argmax_res(m["s2e_frac_res"], m["s2e_frac_nres"])
    models["ae_s2_clf"] = m["clf_pred"].to_numpy(np.int32)

    pooled = {}
    for name, pred in models.items():
        pooled[name] = {**clf_metrics(y, pred), "model": name}
        r = pooled[name]
        log(
            run_dir,
            f"building {name:16s} macroF1={r['macro_f1']:.4f} RES_F1={r['res_f1']:.4f} "
            f"NRES_F1={r['nres_f1']:.4f} acc={r['accuracy']:.4f} n={r['n']}",
        )

    metro = []
    for region, sub in m.groupby("region", sort=False):
        yt = sub["y_soft"].to_numpy(np.int32)
        row: dict[str, Any] = {"region": region, "n": len(sub), "res_rate": float(yt.mean())}
        for name, pred_all in models.items():
            # recompute from sub columns
            if name == "ae_s2_clf":
                pred = sub["clf_pred"].to_numpy(np.int32)
            elif name == "ghsl_built_s":
                pred = soft_argmax_res(sub["ghsl_frac_res"], sub["ghsl_frac_nres"])
            elif name == "ae_ridge":
                pred = soft_argmax_res(sub["ae_frac_res"], sub["ae_frac_nres"])
            elif name == "ae_pls":
                pred = soft_argmax_res(sub["pls_frac_res"], sub["pls_frac_nres"])
            elif name == "ae_xgb" and "xgb_frac_res" in sub.columns:
                pred = soft_argmax_res(sub["xgb_frac_res"], sub["xgb_frac_nres"])
            elif name == "ae_ring" and "ring_frac_res" in sub.columns:
                pred = soft_argmax_res(sub["ring_frac_res"], sub["ring_frac_nres"])
            elif name == "ae_s2_reg" and "s2_frac_res" in sub.columns:
                pred = soft_argmax_res(sub["s2_frac_res"], sub["s2_frac_nres"])
            elif name == "ae_s2_empty_reg" and "s2e_frac_res" in sub.columns:
                pred = soft_argmax_res(sub["s2e_frac_res"], sub["s2e_frac_nres"])
            else:
                continue
            met = clf_metrics(yt, pred)
            row[name] = {
                "macro_f1": met["macro_f1"],
                "res_f1": met["res_f1"],
                "nres_f1": met["nres_f1"],
                "accuracy": met["accuracy"],
                "n": met["n"],
            }
        metro.append(row)

    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_ae_s2_clf.parquet", index=False)
    # drop bulky report strings from JSON
    pooled_json = {}
    for k, v in pooled.items():
        pooled_json[k] = {kk: vv for kk, vv in v.items() if kk != "report"}
        (run_dir / "artifacts" / f"clf_report_{k}.txt").write_text(v.get("report", ""))

    return {
        "n_shared": len(m),
        "label": "soft_argmax(soft_gt_frac_res >= soft_gt_frac_nres)",
        "res_rate": float(y.mean()),
        "mixed_both_rate": float(((m["soft_gt_frac_res"] > 0) & (m["soft_gt_frac_nres"] > 0)).mean()),
        "pooled": pooled_json,
        "metro": decorate_metro(metro),
        "clf_beats_ghsl_macro": int(
            pooled_json["ae_s2_clf"]["macro_f1"] + 1e-12 > pooled_json["ghsl_built_s"]["macro_f1"]
        ),
        "clf_beats_s2reg_macro": int(
            "ae_s2_reg" in pooled_json
            and pooled_json["ae_s2_clf"]["macro_f1"] + 1e-12 > pooled_json["ae_s2_reg"]["macro_f1"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week12_ae_s2_xgb_clf_loro")
    ap.add_argument("--resume-dir", type=Path, default=None)
    ap.add_argument("--s2-train", type=Path, default=S2_TRAIN)
    ap.add_argument("--fold-workers", type=int, default=2)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.resume_dir or (args.out_root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    for sub in ("artifacts", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    latest = args.out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    gpu_names = detect_gpus()
    n_gpus = len(gpu_names)
    (run_dir / "launch.txt").write_text(
        f"started={utc_now()}\npid={os.getpid()}\nfold_workers={args.fold_workers}\n"
        f"n_gpus={n_gpus}\ngpus={gpu_names}\n"
        f"protocol=docs/eval_protocol_apples_to_apples.md\n"
        f"task=binary_soft_argmax_RES_vs_NRES\n"
        f"model=XGBClassifier\n"
    )
    (run_dir / "environment.txt").write_text(
        f"cpus={os.cpu_count()}\nn_gpus={n_gpus}\ngpus={gpu_names}\npython={os.sys.executable}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="init",
        completed=0,
        total=4,
        failed=0,
        _t0=time.time(),
        output=str(run_dir),
        n_cpus=os.cpu_count(),
        n_gpus=n_gpus,
        fold_workers=args.fold_workers,
    )
    log(run_dir, f"Week12 AE+S2 XGBClassifier LORO start → {run_dir} · GPUs={n_gpus} {gpu_names}")

    s2_path = args.s2_train.resolve()
    if not s2_path.exists():
        raise FileNotFoundError(s2_path)
    df = build_training_table(run_dir, s2_path)
    write_progress(run_dir, stage="train_table", completed=1, total=4, train_n=len(df))

    oof = train_loro_clf(df, run_dir, fold_workers=args.fold_workers)
    write_progress(run_dir, stage="score", completed=3, total=4)

    b_metrics = score_buildings(run_dir, oof)
    metrics = {
        "protocol": "docs/eval_protocol_apples_to_apples.md",
        "task": "binary_classification_soft_argmax_RES_vs_NRES",
        "train_n_buildings": int(len(df)),
        "eval_n_buildings": 71497,
        "features": feat_colnames(),
        "buildings": b_metrics,
        "checklist": {
            "eval_ids_locked": True,
            "soft_gt_from_week6": True,
            "label": "soft_argmax res>=nres",
            "spatial_split": "LORO",
            "metric": "RES_F1 / NRES_F1 / macro_F1",
            "baselines_joined": True,
            "s2_features_reused_from_week9": True,
        },
    }
    atomic_json(run_dir / "artifacts" / "section9_ae_s2_clf_metrics.json", metrics)

    clf = b_metrics["pooled"]["ae_s2_clf"]
    ghsl = b_metrics["pooled"]["ghsl_built_s"]
    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "finished_at": utc_now(),
            "output": str(run_dir),
            "clf_macro_f1": clf["macro_f1"],
            "clf_res_f1": clf["res_f1"],
            "clf_nres_f1": clf["nres_f1"],
            "ghsl_macro_f1": ghsl["macro_f1"],
            "ghsl_res_f1": ghsl["res_f1"],
            "ghsl_nres_f1": ghsl["nres_f1"],
            "s2reg_macro_f1": b_metrics["pooled"].get("ae_s2_reg", {}).get("macro_f1"),
            "metrics": str(run_dir / "artifacts" / "section9_ae_s2_clf_metrics.json"),
        },
    )
    write_progress(run_dir, status="completed", stage="done", completed=4, total=4, failed=0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, "DONE")

    try:
        board = ROOT / "scripts" / "rebuild_section9_ae_s2_clf_board.py"
        if board.exists():
            subprocess.check_call(
                [
                    os.sys.executable,
                    str(board),
                    "--metrics",
                    str(run_dir / "artifacts" / "section9_ae_s2_clf_metrics.json"),
                ],
                cwd=str(ROOT),
            )
            log(run_dir, "Section 9 canvas board injected")
    except Exception as e:
        log(run_dir, f"board inject skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
