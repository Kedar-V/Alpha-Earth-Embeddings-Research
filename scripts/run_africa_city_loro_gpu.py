#!/usr/bin/env python3
"""Africa city-LORO: AE Ridge + GPU XGB soft-fraction MAE for selected countries.

Leave-one-city-out within each country (EGY/NGA by default). Uses Postgres
pure-clear labels + AE-64. XGB folds run as isolated subprocesses (one GPU
each), matching week7_xgb_loro_gpu. Ridge is CPU baseline.

Outputs under outputs/africa_city_loro_gpu/<stamp>/:
  progress.json, job.log, summary.json, DONE
  artifacts/balanced_pure.parquet, loro_metrics.csv, oof_*.parquet
  checkpoints/fold_<city>.joblib + metrics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week4" / "international_geo_generalization"))
from lib.db import connect, load_dotenv, wait_until_ready  # noqa: E402

BANDS = [f"A{i:02d}" for i in range(64)]
AE_COLS = [f"a{i:02d}" for i in range(64)]
LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
SEED = 42
FRAC_COLS = ("frac_res_full", "frac_nres_full", "frac_empty_full")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
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
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def mae_rn(yt_r, yt_n, yp_r, yp_n) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yt_r) & np.isfinite(yt_n)
    if m.sum() == 0:
        return {"n": 0, "mae_mean_res_nres": float("nan"), "mae_frac_res": float("nan"), "mae_frac_nres": float("nan")}
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_r[m], yp_r[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_n[m], yp_n[m])),
        "mae_mean_res_nres": float(
            0.5 * (mean_absolute_error(yt_r[m], yp_r[m]) + mean_absolute_error(yt_n[m], yp_n[m]))
        ),
    }


def list_cities(conn, countries: set[str]) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT country_id, city_id
            FROM cities
            WHERE country_id = ANY(%s) AND inclusion_status = 'selected'
            ORDER BY country_id, city_id
            """,
            (sorted(countries),),
        )
        return [(r["country_id"], r["city_id"]) for r in cur.fetchall()]


def export_city_pure(conn, city_id: str, country: str, out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            return {
                "city_id": city_id,
                "country": country,
                "status": "cached",
                "n": int(len(df)),
                "n_res": int((df.stratum == "res").sum()) if "stratum" in df.columns else 0,
                "n_nres": int((df.stratum == "nres").sum()) if "stratum" in df.columns else 0,
            }
        except Exception:
            out_path.unlink(missing_ok=True)

    ae_sel = ", ".join(f"c.{c}" for c in AE_COLS)
    sql = f"""
        SELECT
          c.cell_id, c.city_id,
          ST_X(ST_Centroid(c.geom)) AS pixel_lon,
          ST_Y(ST_Centroid(c.geom)) AS pixel_lat,
          cl.frac_res, cl.frac_nres, cl.frac_empty, cl.binary_label,
          COALESCE(m.frac_res_full, cl.frac_res) AS frac_res_full,
          COALESCE(m.frac_nres_full, cl.frac_nres) AS frac_nres_full,
          COALESCE(m.frac_empty_full, cl.frac_empty) AS frac_empty_full,
          {ae_sel}
        FROM alphaearth_cells c
        JOIN cell_labels cl
          ON cl.cell_id = c.cell_id AND cl.label_version = %s
        LEFT JOIN cell_label_metrics m
          ON m.cell_id = c.cell_id AND m.label_version = cl.label_version
        WHERE c.city_id = %s AND c.year = 2024
          AND cl.binary_label IS NOT NULL
          AND cl.ambiguity_status = 'clear'
          AND cl.purity_status IN ('pure_residential', 'pure_non_residential')
          AND c.a00 IS NOT NULL AND c.a63 IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (LABEL_VERSION, city_id))
        rows = cur.fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        tmp = out_path.with_suffix(".tmp")
        pd.DataFrame().to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"city_id": city_id, "country": country, "status": "empty", "n": 0, "n_res": 0, "n_nres": 0}

    df = pd.DataFrame(rows)
    df["country"] = country
    df["pixel_id"] = df["city_id"].astype(str) + ":" + df["cell_id"].astype(str)
    df["stratum"] = np.where(df["binary_label"].astype(int) == 1, "res", "nres")
    df["y"] = df["binary_label"].astype(np.int64)
    for a, A in zip(AE_COLS, BANDS):
        df[A] = df[a].astype(np.float64)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return {
        "city_id": city_id,
        "country": country,
        "status": "exported",
        "n": int(len(df)),
        "n_res": int((df.stratum == "res").sum()),
        "n_nres": int((df.stratum == "nres").sum()),
    }


def balance_pure(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    parts = []
    rng = np.random.default_rng(seed)
    for city_id, g in df.groupby("city_id", sort=False):
        res = g[g.stratum == "res"]
        nres = g[g.stratum == "nres"]
        n = min(len(res), len(nres))
        if n == 0:
            continue
        res_i = rng.choice(res.index.to_numpy(), size=n, replace=False)
        nres_i = rng.choice(nres.index.to_numpy(), size=n, replace=False)
        parts.append(g.loc[np.concatenate([res_i, nres_i])])
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def city_loro_ridge(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    cities = sorted(df.city_id.unique())
    X_all = df[BANDS].to_numpy(dtype=np.float64)
    Y_all = df[list(FRAC_COLS)].to_numpy(dtype=np.float64)
    oof = np.full_like(Y_all, np.nan)
    for hold in cities:
        te = (df.city_id == hold).to_numpy()
        tr = ~te
        if tr.sum() < 50 or te.sum() == 0:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_all[tr])
        Xte = scaler.transform(X_all[te])
        est = Ridge(alpha=alpha)
        est.fit(Xtr, Y_all[tr])
        pred = np.clip(est.predict(Xte), 0.0, 1.0)
        s = pred.sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        oof[te] = pred / s
    out = df.copy()
    out["ridge_frac_res"] = oof[:, 0]
    out["ridge_frac_nres"] = oof[:, 1]
    out["ridge_frac_empty"] = oof[:, 2]
    return out


def _xgb_regressor(device: str):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        multi_strategy="one_output_per_tree",
        tree_method="hist",
        device=device,
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


def _xgb_classifier(device: str):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=SEED,
        n_jobs=4,
        verbosity=0,
    )


def run_fold_worker(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    te_region = args.fold
    device = args.device
    cache = run_dir / "_cache"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    X = np.load(cache / "X.npy", mmap_mode="r")
    Y = np.load(cache / "Y.npy", mmap_mode="r")
    y_bin = np.load(cache / "y_bin.npy", mmap_mode="r")
    regions = np.load(cache / "regions.npy", allow_pickle=True)
    countries = np.load(cache / "countries.npy", allow_pickle=True)

    country = str(countries[regions == te_region][0])
    same = countries == country
    te = regions == te_region
    tr = same & ~te
    if tr.sum() < 50 or te.sum() == 0:
        raise SystemExit(f"fold {te_region}: insufficient train/test ({tr.sum()}/{te.sum()})")

    X_tr = np.asarray(X[tr], dtype=np.float32)
    Y_tr = np.asarray(Y[tr], dtype=np.float32)
    y_tr = np.asarray(y_bin[tr], dtype=np.int64)
    X_te = np.asarray(X[te], dtype=np.float32)
    y_te = np.asarray(Y[te], dtype=np.float32)
    yb_te = np.asarray(y_bin[te], dtype=np.int64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    reg = _xgb_regressor(device)
    reg.fit(X_tr_s, Y_tr)
    pred = np.clip(np.asarray(reg.predict(X_te_s), dtype=np.float64), 0.0, 1.0)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    s = pred.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    pred = pred / s

    clf = _xgb_classifier(device)
    clf.fit(X_tr_s, y_tr)
    y_pred = np.asarray(clf.predict(X_te_s), dtype=np.int64)
    p_res = np.asarray(clf.predict_proba(X_te_s)[:, 1], dtype=np.float64)

    pipe_reg = Pipeline([("scaler", scaler), ("xgb", reg)])
    joblib.dump(
        {
            "region_held_out": te_region,
            "country": country,
            "regressor": pipe_reg,
            "scaler": scaler,
            "classifier": clf,
            "device": device,
            "seed": SEED,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "model": "africa_city_loro_xgb_gpu_v1",
        },
        ckpt / f"fold_{te_region}.joblib",
    )
    np.save(cache / f"oof_pred_{te_region}.npy", pred)
    np.save(cache / f"oof_idx_{te_region}.npy", np.where(te)[0].astype(np.int64))
    np.save(cache / f"oof_ypred_{te_region}.npy", y_pred)
    np.save(cache / f"oof_pres_{te_region}.npy", p_res)

    mae = [float(mean_absolute_error(y_te[:, i], pred[:, i])) for i in range(3)]
    rn = mae_rn(y_te[:, 0], y_te[:, 1], pred[:, 0], pred[:, 1])
    metrics = {
        "region": te_region,
        "country": country,
        "device": device,
        "n_test": int(te.sum()),
        "n_train": int(tr.sum()),
        "mae_frac_res": mae[0],
        "mae_frac_nres": mae[1],
        "mae_frac_empty": mae[2],
        "mae_mean": float(np.mean(mae)),
        "mae_mean_res_nres": rn["mae_mean_res_nres"],
        "macro_f1": float(f1_score(yb_te, y_pred, average="macro", zero_division=0)),
    }
    atomic_json(ckpt / f"fold_{te_region}_metrics.json", metrics)
    print(json.dumps(metrics), flush=True)
    return 0


def _launch_fold(py: str, script: Path, run_dir: Path, region: str, device: str) -> dict:
    log_path = run_dir / "logs" / f"fold_{region}.log"
    env = os.environ.copy()
    worker_device = "cuda:0"
    if device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    cmd = [
        py, "-u", str(script),
        "--fold-worker", "--run-dir", str(run_dir),
        "--fold", region, "--device", worker_device,
    ]
    with log_path.open("w") as fout:
        proc = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT, env=env)
    metrics_path = run_dir / "checkpoints" / f"fold_{region}_metrics.json"
    if proc.returncode != 0 or not metrics_path.exists():
        raise RuntimeError(f"fold {region} failed rc={proc.returncode}; see {log_path}")
    metrics = json.loads(metrics_path.read_text())
    metrics["device"] = device
    return metrics


def plot_country_bars(metrics_df: pd.DataFrame, run_dir: Path) -> Path:
    country_tbl = (
        metrics_df.groupby("country")["mae_mean_res_nres"]
        .mean()
        .rename("ae_xgb")
        .to_frame()
    )
    ridge_path = run_dir / "artifacts" / "ridge_city_metrics.csv"
    if ridge_path.exists():
        ridge = pd.read_csv(ridge_path)
        country_tbl["ae_ridge"] = ridge.groupby("country")["mae_mean_res_nres"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    countries = sorted(country_tbl.index)
    x = np.arange(len(countries))
    w = 0.35
    if "ae_ridge" in country_tbl.columns:
        ax.bar(x - w / 2, [country_tbl.loc[c, "ae_ridge"] for c in countries], w, label="AE Ridge LORO")
    ax.bar(x + w / 2, [country_tbl.loc[c, "ae_xgb"] for c in countries], w, label="AE XGB GPU LORO")
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel("MAE mean(res, nres)")
    ax.set_title("Africa city-LORO · soft fractions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = run_dir / "figures" / "country_mae_loro.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _count_gpus() -> int:
    """Prefer nvidia-smi (fast); avoid importing torch (can hang on NFS)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL, timeout=15
        )
        n = sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))
        if n > 0:
            return n
    except Exception:
        pass
    # Fallback: visible CUDA_VISIBLE_DEVICES list
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return len([x for x in cvd.split(",") if x.strip() != ""])
    return 0


def main_orchestrator(args: argparse.Namespace) -> int:
    import xgboost

    load_dotenv()
    wait_until_ready(timeout_s=60)

    countries = {c.strip().upper() for c in args.countries.replace(",", " ").split() if c.strip()}
    n_gpus = _count_gpus()
    if n_gpus < 1:
        raise SystemExit("No CUDA GPUs visible — refuse CPU fallback for this job")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else (ROOT / "outputs" / "africa_city_loro_gpu" / stamp)
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures", "_cache", "exports"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    (run_dir / "launch.txt").write_text(
        f"countries={sorted(countries)}\nn_gpus={n_gpus}\nxgboost={xgboost.__version__}\n"
        f"argv={sys.argv}\nstarted={utc_now()}\n"
    )
    write_progress(
        run_dir,
        status="running",
        stage="export",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=1,
        countries=sorted(countries),
        n_gpus=n_gpus,
        resource_allocation={"gpus": n_gpus, "cpus": os.cpu_count(), "fold_workers": args.workers or n_gpus},
        output_location=str(run_dir),
    )
    log(run_dir, f"Africa city-LORO GPU start countries={sorted(countries)} gpus={n_gpus}")

    with connect() as conn:
        cities = list_cities(conn, countries)
        if not cities:
            raise SystemExit(f"no selected cities for {countries}")
        export_rows = []
        for i, (ctry, cid) in enumerate(cities):
            outp = run_dir / "exports" / f"{cid}.parquet"
            rec = export_city_pure(conn, cid, ctry, outp)
            export_rows.append(rec)
            log(run_dir, f"export {rec['status']:8s} {cid}: n={rec['n']:,} res={rec['n_res']:,} nres={rec['n_nres']:,}")
            write_progress(run_dir, stage="export", completed=i + 1, total=len(cities), last_city=cid)

    parts = []
    for rec in export_rows:
        p = run_dir / "exports" / f"{rec['city_id']}.parquet"
        if rec["n"] > 0 and p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise SystemExit("no pure cells exported")
    raw = pd.concat(parts, ignore_index=True)
    bal = balance_pure(raw)
    bal.to_parquet(run_dir / "artifacts" / "balanced_pure.parquet", index=False)
    pd.DataFrame(export_rows).to_csv(run_dir / "artifacts" / "export_city_counts.csv", index=False)
    log(run_dir, f"balanced n={len(bal):,} cities={sorted(bal.city_id.unique())}")

    # --- Ridge city-LORO (CPU) ---
    write_progress(run_dir, stage="ridge_loro", completed=0, total=bal.city_id.nunique())
    ridge_oof = city_loro_ridge(bal)
    ridge_rows = []
    for (ctry, city), sub in ridge_oof.groupby(["country", "city_id"]):
        m = mae_rn(
            sub["frac_res_full"].to_numpy(),
            sub["frac_nres_full"].to_numpy(),
            sub["ridge_frac_res"].to_numpy(),
            sub["ridge_frac_nres"].to_numpy(),
        )
        ridge_rows.append({"country": ctry, "region": city, "model": "ae_ridge", **m})
    ridge_df = pd.DataFrame(ridge_rows)
    ridge_df.to_csv(run_dir / "artifacts" / "ridge_city_metrics.csv", index=False)
    ridge_oof.to_parquet(run_dir / "artifacts" / "oof_ridge.parquet", index=False)
    for ctry, g in ridge_df.groupby("country"):
        log(run_dir, f"Ridge LORO {ctry}: mean mae_rn={g['mae_mean_res_nres'].mean():.4f}")

    # --- cache for XGB fold workers ---
    cache = run_dir / "_cache"
    X = bal[BANDS].to_numpy(dtype=np.float32)
    Y = bal[list(FRAC_COLS)].to_numpy(dtype=np.float32)
    y_bin = bal["y"].to_numpy(dtype=np.int64)
    regions = bal["city_id"].astype(str).to_numpy()
    countries_arr = bal["country"].astype(str).to_numpy()
    np.save(cache / "X.npy", X)
    np.save(cache / "Y.npy", Y)
    np.save(cache / "y_bin.npy", y_bin)
    np.save(cache / "regions.npy", regions)
    np.save(cache / "countries.npy", countries_arr)
    bal[
        [
            "pixel_id", "city_id", "country", "stratum", "y",
            "frac_res", "frac_nres", "frac_empty",
            "frac_res_full", "frac_nres_full", "frac_empty_full",
            "pixel_lon", "pixel_lat",
        ]
    ].rename(columns={"city_id": "region"}).to_parquet(cache / "meta.parquet", index=False)

    region_list = sorted(bal.city_id.astype(str).unique().tolist())
    # Drop cities that would leave <2 cities in their country for train
    eligible = []
    for r in region_list:
        ctry = str(countries_arr[regions == r][0])
        n_sib = len(set(regions[countries_arr == ctry]))
        if n_sib >= 2:
            eligible.append(r)
        else:
            log(run_dir, f"XGB skip {r}: country {ctry} has <2 cities")

    n_parallel = args.workers or n_gpus
    total = len(eligible) + 2  # folds + assemble + plots
    write_progress(
        run_dir,
        stage="xgb_loro",
        completed=0,
        total=total,
        cities_pending=eligible,
        n_gpus=n_gpus,
        workers=n_parallel,
        xgboost_version=xgboost.__version__,
    )
    log(run_dir, f"XGB city-LORO folds={len(eligible)} parallel={n_parallel} on {n_gpus} GPUs")

    script = Path(__file__).resolve()
    py = sys.executable
    jobs = [(r, f"cuda:{i % n_gpus}") for i, r in enumerate(eligible)]
    rows = []
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futs = {pool.submit(_launch_fold, py, script, run_dir, region, device): region for region, device in jobs}
        for fut in as_completed(futs):
            region = futs[fut]
            try:
                metrics = fut.result()
                rows.append(metrics)
                done += 1
                log(
                    run_dir,
                    f"  {metrics['region']:22s} {metrics['country']} {metrics['device']} "
                    f"mae_rn={metrics['mae_mean_res_nres']:.4f} F1={metrics['macro_f1']:.3f}",
                )
                write_progress(
                    run_dir,
                    stage=f"fold_{region}",
                    completed=done,
                    failed=failed,
                    last_region=region,
                    success=len(rows),
                )
            except Exception as e:
                failed += 1
                done += 1
                log(run_dir, f"FAIL {region}: {e}")
                (run_dir / "failures" / f"fold_{region}.txt").write_text(str(e))
                write_progress(run_dir, stage=f"fail_{region}", completed=done, failed=failed)

    if not rows:
        write_progress(run_dir, status="failed", stage="no_folds")
        (run_dir / "FAILED").write_text("no folds\n")
        return 1

    metrics_df = pd.DataFrame(rows).sort_values(["country", "region"])
    metrics_df.to_csv(run_dir / "artifacts" / "loro_metrics.csv", index=False)

    meta = pd.read_parquet(cache / "meta.parquet")
    oof_pred = np.full((len(meta), 3), np.nan)
    oof_ypred = np.full(len(meta), -1, dtype=np.int64)
    oof_pres = np.full(len(meta), np.nan)
    for r in rows:
        region = r["region"]
        idx = np.load(cache / f"oof_idx_{region}.npy")
        oof_pred[idx] = np.load(cache / f"oof_pred_{region}.npy")
        oof_ypred[idx] = np.load(cache / f"oof_ypred_{region}.npy")
        oof_pres[idx] = np.load(cache / f"oof_pres_{region}.npy")
    oof = meta.copy()
    oof["pred_frac_res"] = oof_pred[:, 0]
    oof["pred_frac_nres"] = oof_pred[:, 1]
    oof["pred_frac_empty"] = oof_pred[:, 2]
    oof["y_pred"] = oof_ypred
    oof["p_res"] = oof_pres
    # join ridge preds
    ridge_join = ridge_oof[["pixel_id", "ridge_frac_res", "ridge_frac_nres", "ridge_frac_empty"]]
    oof = oof.merge(ridge_join, on="pixel_id", how="left")
    oof.to_parquet(run_dir / "artifacts" / "oof_predictions.parquet", index=False)

    country_summary = []
    for ctry in sorted(countries):
        m = oof["country"] == ctry
        if m.sum() == 0:
            continue
        xgb_m = mae_rn(
            oof.loc[m, "frac_res_full"], oof.loc[m, "frac_nres_full"],
            oof.loc[m, "pred_frac_res"], oof.loc[m, "pred_frac_nres"],
        )
        ridge_m = mae_rn(
            oof.loc[m, "frac_res_full"], oof.loc[m, "frac_nres_full"],
            oof.loc[m, "ridge_frac_res"], oof.loc[m, "ridge_frac_nres"],
        )
        f1 = float(f1_score(oof.loc[m, "y"], oof.loc[m, "y_pred"], average="macro", zero_division=0))
        country_summary.append({
            "country": ctry,
            "n": int(m.sum()),
            "n_cities": int(oof.loc[m, "region"].nunique()),
            "ae_ridge_mae_mean_res_nres": ridge_m["mae_mean_res_nres"],
            "ae_xgb_mae_mean_res_nres": xgb_m["mae_mean_res_nres"],
            "ae_xgb_macro_f1": f1,
        })
        log(
            run_dir,
            f"COUNTRY {ctry}: ridge={ridge_m['mae_mean_res_nres']:.4f} "
            f"xgb={xgb_m['mae_mean_res_nres']:.4f} F1={f1:.3f}",
        )
    pd.DataFrame(country_summary).to_csv(run_dir / "artifacts" / "country_summary.csv", index=False)

    fig = plot_country_bars(metrics_df, run_dir)
    results_md = [
        "# Africa city-LORO (AE Ridge + GPU XGB)",
        "",
        f"Countries: {', '.join(sorted(countries))} · soft GT MAE mean(res,nres) · leave-one-city-out",
        "",
        "## Country summary",
        "",
        "| Country | n | cities | AE Ridge | AE XGB | XGB macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in country_summary:
        results_md.append(
            f"| {r['country']} | {r['n']:,} | {r['n_cities']} | "
            f"{r['ae_ridge_mae_mean_res_nres']:.4f} | {r['ae_xgb_mae_mean_res_nres']:.4f} | "
            f"{r['ae_xgb_macro_f1']:.3f} |"
        )
    results_md += ["", "## Per-city XGB holdout", "", "| Country | Held-out | MAE(res,nres) | macro-F1 | n_test |", "|---|---|---:|---:|---:|"]
    for r in metrics_df.itertuples():
        results_md.append(
            f"| {r.country} | {r.region} | {r.mae_mean_res_nres:.4f} | {r.macro_f1:.3f} | {r.n_test:,} |"
        )
    results_md += ["", f"![country MAE]({fig.name})", ""]
    (run_dir / "RESULTS.md").write_text("\n".join(results_md) + "\n")

    summary = {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "countries": sorted(countries),
        "n_gpus": n_gpus,
        "failed_folds": failed,
        "n_balanced": int(len(bal)),
        "country_summary": country_summary,
        "runtime_seconds": round(time.time() - t0, 1),
        "run_dir": str(run_dir),
        "figure": str(fig),
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        completed=total,
        total=total,
        percent_complete=100.0,
        failed=failed,
        success=len(rows),
    )
    if failed == 0:
        (run_dir / "DONE").write_text(utc_now() + "\n")
    else:
        (run_dir / "FAILED").write_text(json.dumps(summary) + "\n")
    log(run_dir, f"DONE · failed={failed} · {run_dir}")
    return 0 if failed == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-worker", action="store_true")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--countries", default="EGY,NGA")
    ap.add_argument("--workers", type=int, default=0, help="concurrent fold subprocesses (default=n_gpus)")
    args = ap.parse_args()
    if args.fold_worker:
        if not args.run_dir or not args.fold:
            raise SystemExit("--fold-worker requires --run-dir and --fold")
        return run_fold_worker(args)
    return main_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
