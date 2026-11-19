#!/usr/bin/env python3
"""International BUILT-S vs AE fraction MAE — city & country × protocols.

Protocols
  city_loro  Leave-one-city-out within each country (Ridge + PLS)
  loco       Leave-one-country-out: train on other countries, test on held-out

Reports MAE mean(res, nres) at:
  - city level (per protocol for AE; BUILT-S once)
  - country level
  - overall

Inputs (default): pure/balanced caches from an intl export run.
BUILT-S: sample at AE centers (cached).

  .venv/bin/python -u scripts/eval_intl_built_s_protocols.py \\
    --source-run outputs/ghsl_built_s_vs_ae_intl/20261103_pgdata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BANDS = [f"A{i:02d}" for i in range(64)]
S_TILE_DIR = ROOT / "data" / "ghsl" / "built_s_tiles"
NRES_TILE_DIR = ROOT / "data" / "ghsl" / "built_s_nres_tiles"
S_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10/"
    "V1-0/tiles/GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R{row}_C{col}.zip"
)
NRES_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_NRES_E2018_GLOBE_R2023A_54009_10/"
    "V1-0/tiles/GHS_BUILT_S_NRES_E2018_GLOBE_R2023A_54009_10_V1_0_R{row}_C{col}.zip"
)
ORIGIN_LEFT, ORIGIN_TOP, TILE_M = -18_041_000.0, 10_000_000.0, 1_000_000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(0.0, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done = int(prev["completed"])
            total = int(prev["total"])
            prev["percent_complete"] = round(100.0 * done / total, 2)
            rate = done / max(elapsed, 1e-6)
            prev["rate_per_second"] = round(rate, 4)
            rem = max(0, total - done)
            prev["eta_seconds"] = round(rem / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def mae_rn(yt_r, yt_n, yp_r, yp_n) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yt_r) & np.isfinite(yt_n)
    if int(m.sum()) == 0:
        return {"n": 0, "mae_mean_res_nres": float("nan"), "mae_frac_res": float("nan"), "mae_frac_nres": float("nan")}
    er = float(mean_absolute_error(yt_r[m], yp_r[m]))
    en = float(mean_absolute_error(yt_n[m], yp_n[m]))
    return {"n": int(m.sum()), "mae_frac_res": er, "mae_frac_nres": en, "mae_mean_res_nres": 0.5 * (er + en)}


def fit_predict_ridge_pls(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    model: str,
    alpha: float = 1.0,
    pls_k: int = 10,
) -> np.ndarray:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    if model == "ridge":
        est = Ridge(alpha=alpha)
        est.fit(Xtr_s, Ytr)
        pred = est.predict(Xte_s)
    else:
        k = min(pls_k, Xtr_s.shape[1], max(1, Xtr_s.shape[0] - 1))
        est = PLSRegression(n_components=k, scale=False)
        est.fit(Xtr_s, Ytr)
        pred = est.predict(Xte_s)
    pred = np.clip(pred, 0.0, 1.0)
    s = pred.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return pred / s


def city_loro_oof(df: pd.DataFrame, model: str, pls_k: int = 10) -> np.ndarray:
    """Within-country leave-one-city-out. Returns (n, 3) preds aligned to df."""
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(np.float64)
    X = df[BANDS].to_numpy(np.float64)
    oof = np.full_like(Y, np.nan)
    for ctry, g in df.groupby("country", sort=False):
        cities = sorted(g.city_id.unique())
        idx_all = g.index.to_numpy()
        for hold in cities:
            te_mask = (g.city_id == hold).to_numpy()
            tr_mask = ~te_mask
            if tr_mask.sum() < 100 or te_mask.sum() == 0:
                continue
            pred = fit_predict_ridge_pls(
                X[idx_all][tr_mask],
                Y[idx_all][tr_mask],
                X[idx_all][te_mask],
                model=model,
                pls_k=pls_k,
            )
            oof[idx_all[te_mask]] = pred
    return oof


def loco_predict(df: pd.DataFrame, model: str, pls_k: int = 10) -> np.ndarray:
    """Leave-one-country-out. Train on all other countries' cells."""
    Y = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(np.float64)
    X = df[BANDS].to_numpy(np.float64)
    oof = np.full_like(Y, np.nan)
    countries = sorted(df.country.unique())
    for hold in countries:
        te = (df.country == hold).to_numpy()
        tr = ~te
        if tr.sum() < 100 or te.sum() == 0:
            continue
        oof[te] = fit_predict_ridge_pls(X[tr], Y[tr], X[te], model=model, pls_k=pls_k)
    return oof


def lonlat_to_tile(lon: float, lat: float) -> tuple[int, int]:
    tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    x, y = tr.transform(lon, lat)
    c = int(np.floor((x - ORIGIN_LEFT) / TILE_M)) + 1
    r = int(np.floor((ORIGIN_TOP - y) / TILE_M))
    return r, c


def tiles_for_df(df: pd.DataFrame) -> list[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for _, g in df.groupby("city_id"):
        for lon in np.linspace(g.pixel_lon.min(), g.pixel_lon.max(), 5):
            for lat in np.linspace(g.pixel_lat.min(), g.pixel_lat.max(), 5):
                tiles.add(lonlat_to_tile(float(lon), float(lat)))
    return sorted(tiles)


def _download_one(url: str, dest_zip: Path, dest_tif: Path, retries: int = 4) -> str:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_tif.exists() and dest_tif.stat().st_size > 1000:
        return "skip"
    last = ""
    for attempt in range(1, retries + 1):
        try:
            if dest_zip.exists():
                dest_zip.unlink()
            req = Request(url, headers={"User-Agent": "AlphaEarthResearch/1.0 (intl protocols)"})
            with urlopen(req, timeout=600) as resp, dest_zip.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            with zipfile.ZipFile(dest_zip) as zf:
                members = [m for m in zf.namelist() if m.endswith(".tif") and ".ovr" not in m]
                zf.extract(members[0], path=dest_tif.parent)
                extracted = dest_tif.parent / members[0]
                if extracted.resolve() != dest_tif.resolve():
                    if dest_tif.exists():
                        dest_tif.unlink()
                    extracted.replace(dest_tif)
            dest_zip.unlink(missing_ok=True)
            return "done"
        except Exception as e:
            last = str(e)
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(last)


def download_tiles(run_dir: Path, tiles: list[tuple[int, int]]) -> None:
    S_TILE_DIR.mkdir(parents=True, exist_ok=True)
    NRES_TILE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r, c in tiles:
        sn = f"GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        nn = f"GHS_BUILT_S_NRES_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        jobs.append((S_URL.format(row=r, col=c), S_TILE_DIR / f"{sn}.zip", S_TILE_DIR / sn))
        jobs.append((NRES_URL.format(row=r, col=c), NRES_TILE_DIR / f"{nn}.zip", NRES_TILE_DIR / nn))
    log(run_dir, f"download {len(jobs)} products for {len(tiles)} tiles")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_download_one, u, z, t): t for u, z, t in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            st = fut.result()
            log(run_dir, f"{st:4s} {t.name}")
            write_progress(run_dir, stage="download", completed=i, total=len(jobs), last_progress_at=utc_now())


def _sample_city(payload: tuple) -> pd.DataFrame:
    city_id, pixel_ids, lons, lats = payload
    s_paths = sorted(S_TILE_DIR.glob("GHS_BUILT_S_E2018_*.tif"))
    n_paths = sorted(NRES_TILE_DIR.glob("GHS_BUILT_S_NRES_E2018_*.tif"))
    s_ds = [rasterio.open(p) for p in s_paths]
    n_ds = [rasterio.open(p) for p in n_paths]
    try:
        tr = Transformer.from_crs("EPSG:4326", s_ds[0].crs, always_xy=True)
        xs, ys = tr.transform(lons, lats)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        bu = np.full(len(xs), 255, dtype=np.uint8)
        nr = np.full(len(xs), 255, dtype=np.uint8)
        assigned = np.full(len(xs), -1, dtype=np.int32)
        for ti, ds in enumerate(s_ds):
            b = ds.bounds
            m = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top) & (assigned < 0)
            assigned[m] = ti
        for ti, (ds_s, ds_n) in enumerate(zip(s_ds, n_ds)):
            idx = np.where(assigned == ti)[0]
            if len(idx) == 0:
                continue
            coords = list(zip(xs[idx], ys[idx]))
            bu[idx] = np.fromiter((int(v[0]) for v in ds_s.sample(coords, indexes=1)), dtype=np.uint8, count=len(idx))
            nr[idx] = np.fromiter((int(v[0]) for v in ds_n.sample(coords, indexes=1)), dtype=np.uint8, count=len(idx))
        return pd.DataFrame(
            {"pixel_id": pixel_ids, "city_id": city_id, "built_s_raw": bu.astype(np.int16), "built_s_nres_raw": nr.astype(np.int16)}
        )
    finally:
        for ds in s_ds + n_ds:
            ds.close()


def sample_built_s(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    workers = workers or max(1, min(os.cpu_count() or 4, df.city_id.nunique()))
    payloads = [
        (cid, g.pixel_id.to_numpy(), g.pixel_lon.to_numpy(np.float64), g.pixel_lat.to_numpy(np.float64))
        for cid, g in df.groupby("city_id", sort=False)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_sample_city, payloads))
    out = pd.concat(parts, ignore_index=True)
    bu = out.built_s_raw.astype(float)
    nr = out.built_s_nres_raw.astype(float)
    valid = (bu != 255) & (nr != 255)
    buf = np.clip(bu / 100.0, 0, 1)
    is_nres = nr == 1
    out["bs_frac_res"] = np.where(valid, np.where(is_nres, 0.0, buf), np.nan)
    out["bs_frac_nres"] = np.where(valid, np.where(is_nres, buf, 0.0), np.nan)
    out["built_s_valid"] = valid
    return out


def score_frame(df: pd.DataFrame, protocol: str, pred_cols: dict[str, tuple[str, str]]) -> list[dict]:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", df)]
    scopes += [(f"country:{c}", g) for c, g in df.groupby("country")]
    scopes += [(f"city:{c}", g) for c, g in df.groupby("city_id")]
    for scope, sub in scopes:
        yt_r = sub["frac_res_full"].to_numpy()
        yt_n = sub["frac_nres_full"].to_numpy()
        for model, (a, b) in pred_cols.items():
            met = mae_rn(yt_r, yt_n, sub[a].to_numpy(), sub[b].to_numpy())
            met.update({"model": model, "protocol": protocol, "scope": scope})
            rows.append(met)
    return rows


def load_balanced(source: Path) -> pd.DataFrame:
    bal = source / "artifacts" / "balanced_pure.parquet"
    if bal.exists():
        return pd.read_parquet(bal)
    # fall back: concat pure_* and balance
    parts = list((source / "checkpoints").glob("pure_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no balanced_pure or pure_* under {source}")
    labeled = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    rng = np.random.default_rng(42)
    bal_parts = []
    for _, g in labeled.groupby("city_id", sort=False):
        res, nres = g[g.stratum == "res"], g[g.stratum == "nres"]
        n = min(len(res), len(nres))
        if n == 0:
            continue
        bal_parts.append(
            g.loc[
                np.concatenate(
                    [
                        rng.choice(res.index.to_numpy(), n, replace=False),
                        rng.choice(nres.index.to_numpy(), n, replace=False),
                    ]
                )
            ]
        )
    return pd.concat(bal_parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", type=Path, required=True, help="intl export run with pure_/balanced caches")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--pls-k", type=int, default=10)
    ap.add_argument("--reuse-city-loro-oof", type=Path, default=None, help="optional oof_ae_fractions.parquet")
    ap.add_argument(
        "--wait-oof-seconds",
        type=int,
        default=0,
        help="poll source-run artifacts/oof_ae_fractions.parquet this many seconds before training city-LORO",
    )
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "ghsl_built_s_vs_ae_intl_protocols" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"cmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n")
    (run_dir / "environment.txt").write_text(
        f"python={sys.executable}\ncpus={os.cpu_count()}\nsource_run={args.source_run}\n"
    )

    write_progress(
        run_dir,
        status="running",
        stage="load",
        started_at=utc_now(),
        _t0=time.time(),
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count(), "gpus": 0},
        source_run=str(args.source_run),
    )
    log(run_dir, f"Run {run_dir} source={args.source_run}")

    bal = load_balanced(args.source_run)
    # reset index for oof alignment
    bal = bal.reset_index(drop=True)
    bal.to_parquet(run_dir / "artifacts" / "balanced_pure.parquet", index=False)
    log(run_dir, f"balanced n={len(bal):,} countries={sorted(bal.country.unique())} cities={bal.city_id.nunique()}")

    # --- BUILT-S sample ---
    bs_cache = run_dir / "checkpoints" / "built_s_at_ae_centers.parquet"
    src_bs = args.source_run / "checkpoints" / "built_s_at_ae_centers.parquet"
    india_bs = args.source_run / "india_early" / "scored.parquet"
    if bs_cache.exists():
        bs = pd.read_parquet(bs_cache)
        log(run_dir, f"reuse local BUILT-S cache n={len(bs):,}")
    elif src_bs.exists():
        bs = pd.read_parquet(src_bs)
        bs.to_parquet(bs_cache, index=False)
        log(run_dir, f"reuse source BUILT-S cache n={len(bs):,}")
    else:
        write_progress(run_dir, stage="download", completed=0, total=1, last_progress_at=utc_now())
        tiles = tiles_for_df(bal)
        log(run_dir, f"tiles needed: {tiles}")
        if not args.skip_download:
            download_tiles(run_dir, tiles)
        write_progress(run_dir, stage="sample", last_progress_at=utc_now())
        log(run_dir, "sampling BUILT-S")
        bs = sample_built_s(bal[["pixel_id", "city_id", "pixel_lon", "pixel_lat"]])
        # merge IND from india_early if partial overlap and valid
        if india_bs.exists():
            ie = pd.read_parquet(india_bs, columns=["pixel_id", "bs_frac_res", "bs_frac_nres", "built_s_valid"])
            log(run_dir, f"india_early scored available n={len(ie):,} (not required; full sample used)")
        bs.to_parquet(bs_cache, index=False)
        log(run_dir, f"BUILT-S sampled n={len(bs):,} valid={float(bs.built_s_valid.mean()):.3f}")

    # --- city_loro ---
    write_progress(run_dir, stage="city_loro", completed=0, total=2, last_progress_at=utc_now())
    oof_path = args.reuse_city_loro_oof
    if oof_path is None:
        cand = args.source_run / "artifacts" / "oof_ae_fractions.parquet"
        if args.wait_oof_seconds > 0 and not cand.exists():
            log(run_dir, f"waiting up to {args.wait_oof_seconds}s for {cand}")
            deadline = time.time() + args.wait_oof_seconds
            while time.time() < deadline and not cand.exists():
                write_progress(
                    run_dir,
                    stage="wait_oof",
                    last_progress_at=utc_now(),
                    waiting_for=str(cand),
                )
                time.sleep(30)
        if cand.exists():
            oof_path = cand
    if oof_path is not None and Path(oof_path).exists():
        log(run_dir, f"reuse city-LORO OOF {oof_path}")
        oof_loro = pd.read_parquet(oof_path)
        # align by pixel_id
        keep = ["pixel_id", "ridge_frac_res", "ridge_frac_nres", "pls_frac_res", "pls_frac_nres"]
        missing = [c for c in keep if c not in oof_loro.columns]
        if missing:
            raise RuntimeError(f"OOF missing cols {missing}")
        df = bal.merge(oof_loro[keep], on="pixel_id", how="left")
    else:
        log(run_dir, "city-LORO Ridge")
        pred_r = city_loro_oof(bal, "ridge", args.pls_k)
        write_progress(run_dir, stage="city_loro", completed=1, total=2, last_progress_at=utc_now())
        log(run_dir, f"city-LORO PLS k={args.pls_k}")
        pred_p = city_loro_oof(bal, "pls", args.pls_k)
        write_progress(run_dir, stage="city_loro", completed=2, total=2, last_progress_at=utc_now())
        df = bal.copy()
        df["ridge_frac_res"], df["ridge_frac_nres"] = pred_r[:, 0], pred_r[:, 1]
        df["pls_frac_res"], df["pls_frac_nres"] = pred_p[:, 0], pred_p[:, 1]
        df.to_parquet(run_dir / "artifacts" / "oof_city_loro.parquet", index=False)

    # --- loco ---
    write_progress(run_dir, stage="loco", completed=0, total=2, last_progress_at=utc_now())
    log(run_dir, "LOCO Ridge")
    loco_r = loco_predict(bal, "ridge", args.pls_k)
    write_progress(run_dir, stage="loco", completed=1, total=2, last_progress_at=utc_now())
    log(run_dir, f"LOCO PLS k={args.pls_k}")
    loco_p = loco_predict(bal, "pls", args.pls_k)
    write_progress(run_dir, stage="loco", completed=2, total=2, last_progress_at=utc_now())
    df["loco_ridge_frac_res"], df["loco_ridge_frac_nres"] = loco_r[:, 0], loco_r[:, 1]
    df["loco_pls_frac_res"], df["loco_pls_frac_nres"] = loco_p[:, 0], loco_p[:, 1]

    # join BUILT-S
    bs_cols = ["pixel_id", "bs_frac_res", "bs_frac_nres", "built_s_valid"]
    df = df.merge(bs[bs_cols], on="pixel_id", how="left")
    df.to_parquet(run_dir / "artifacts" / "scored_all_protocols.parquet", index=False)
    log(run_dir, f"scored frame n={len(df):,} built_s_valid={float(df.built_s_valid.mean()):.3f}")

    # --- metrics ---
    write_progress(run_dir, stage="score", last_progress_at=utc_now())
    rows = []
    rows += score_frame(
        df,
        "city_loro",
        {
            "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
            "ae_pls": ("pls_frac_res", "pls_frac_nres"),
            "ghsl_built_s": ("bs_frac_res", "bs_frac_nres"),
        },
    )
    rows += score_frame(
        df,
        "loco",
        {
            "ae_ridge": ("loco_ridge_frac_res", "loco_ridge_frac_nres"),
            "ae_pls": ("loco_pls_frac_res", "loco_pls_frac_nres"),
            "ghsl_built_s": ("bs_frac_res", "bs_frac_nres"),
        },
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)

    # tidy tables
    def level_table(level_prefix: str) -> pd.DataFrame:
        s = summary[summary.scope.str.startswith(level_prefix)].copy()
        s["unit"] = s.scope.str.replace(level_prefix, "", regex=False)
        return s.pivot_table(
            index=["protocol", "unit"],
            columns="model",
            values="mae_mean_res_nres",
            aggfunc="first",
        ).reset_index()

    country_tbl = level_table("country:")
    city_tbl = level_table("city:")
    overall = (
        summary[summary.scope == "all"]
        .pivot_table(index="protocol", columns="model", values="mae_mean_res_nres", aggfunc="first")
        .reset_index()
    )
    country_tbl.to_csv(run_dir / "mae_by_country_protocol.csv", index=False)
    city_tbl.to_csv(run_dir / "mae_by_city_protocol.csv", index=False)
    overall.to_csv(run_dir / "mae_overall_protocol.csv", index=False)
    log(run_dir, "country table:\n" + country_tbl.to_string(index=False))
    log(run_dir, "overall:\n" + overall.to_string(index=False))

    # figures: country bars per protocol
    for protocol in ["city_loro", "loco"]:
        sub = country_tbl[country_tbl.protocol == protocol]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.8))
        units = list(sub["unit"])
        x = np.arange(len(units))
        w = 0.25
        for i, (model, color) in enumerate(
            [("ghsl_built_s", "#ff7f0e"), ("ae_ridge", "#2ca02c"), ("ae_pls", "#1f77b4")]
        ):
            vals = [float(sub.loc[sub.unit == u, model].iloc[0]) for u in units]
            ax.bar(x + (i - 1) * w, vals, w, label=model, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(units)
        ax.set_ylabel("MAE mean(res, nres) ↓")
        ax.set_title(f"{protocol} · by country")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / f"mae_country_{protocol}.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # per-city heatmap-ish barh by country
        cities_sub = city_tbl[city_tbl.protocol == protocol].copy()
        for ctry in sorted({u.split("_")[0] for u in cities_sub["unit"]}):
            c = cities_sub[cities_sub.unit.str.startswith(ctry + "_")].sort_values("unit")
            if c.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(c))))
            y = np.arange(len(c))
            for model, color, off in [
                ("ghsl_built_s", "#ff7f0e", -0.25),
                ("ae_ridge", "#2ca02c", 0.0),
                ("ae_pls", "#1f77b4", 0.25),
            ]:
                ax.barh(y + off, c[model].to_numpy(float), height=0.25, color=color, label=model)
            ax.set_yticks(y)
            ax.set_yticklabels(c["unit"])
            ax.set_xlabel("MAE mean(res, nres) ↓")
            ax.set_title(f"{protocol} · {ctry} cities")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                run_dir / "figures" / f"mae_city_{protocol}_{ctry}.png",
                dpi=160,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    # notebook + README
    nb_dir = ROOT / "notebooks" / "ghsl_built_s_vs_ae"
    nb_dir.mkdir(exist_ok=True)
    (nb_dir / "figures").mkdir(exist_ok=True)
    for p in (run_dir / "figures").glob("*.png"):
        (nb_dir / "figures" / p.name).write_bytes(p.read_bytes())

    # headline lines for README
    def cell(protocol: str, unit_scope: str, model: str) -> float:
        r = summary[(summary.protocol == protocol) & (summary.scope == unit_scope) & (summary.model == model)]
        return float(r.mae_mean_res_nres.iloc[0]) if len(r) else float("nan")

    lines = []
    for protocol in ["city_loro", "loco"]:
        for ctry in sorted(bal.country.unique()):
            lines.append(
                f"{protocol}/{ctry}: BUILT-S={cell(protocol, f'country:{ctry}', 'ghsl_built_s'):.3f} "
                f"Ridge={cell(protocol, f'country:{ctry}', 'ae_ridge'):.3f} "
                f"PLS={cell(protocol, f'country:{ctry}', 'ae_pls'):.3f}"
            )
    verdict = " · ".join(lines)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BUILT-S vs AE — city / country × city-LORO & leave-one-country-out\n\n",
                    f"Run: `{run_dir}`\n\n",
                    f"{verdict}\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from IPython.display import Image, display, Markdown\n",
                    f"RUN = Path({str(run_dir)!r})\n",
                    "display(pd.read_csv(RUN / 'mae_overall_protocol.csv'))\n",
                    "display(pd.read_csv(RUN / 'mae_by_country_protocol.csv'))\n",
                    "display(pd.read_csv(RUN / 'mae_by_city_protocol.csv'))\n",
                    "for p in sorted((RUN / 'figures').glob('*.png')):\n",
                    "    display(Markdown(f'### {p.name}'))\n",
                    "    display(Image(filename=str(p)))\n",
                ],
            },
        ],
    }
    nb_path = nb_dir / "05_intl_city_country_loro_loco.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))

    # README append/replace protocols section
    readme = nb_dir / "README.md"
    section = [
        "",
        "---",
        "",
        "## Results — city & country × city-LORO / leave-one-country-out",
        "",
        "Cohort: `intl_city_loro` balanced pure (Postgres). Protocols:",
        "",
        "- **city_loro** — leave-one-city-out within country",
        "- **loco** — leave-one-country-out (train on other countries, test on held-out)",
        "",
        "BUILT-S is protocol-independent (same soft surface at AE centers).",
        "",
        "### Country-level MAE mean(res, nres)",
        "",
        "| Date | Protocol | Country | n | BUILT-S | Ridge | PLS | Run |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    date = datetime.now().strftime("%Y-%m-%d")
    for protocol in ["city_loro", "loco"]:
        for ctry in sorted(bal.country.unique()):
            n = int(
                summary[
                    (summary.protocol == protocol)
                    & (summary.scope == f"country:{ctry}")
                    & (summary.model == "ae_ridge")
                ].n.iloc[0]
            )
            section.append(
                f"| {date} | {protocol} | {ctry} | {n:,} | "
                f"**{cell(protocol, f'country:{ctry}', 'ghsl_built_s'):.3f}** | "
                f"{cell(protocol, f'country:{ctry}', 'ae_ridge'):.3f} | "
                f"{cell(protocol, f'country:{ctry}', 'ae_pls'):.3f} | "
                f"[`{run_dir.name}`](../../{run_dir}) |"
            )
    section += [
        "",
        f"City-level CSV: [`mae_by_city_protocol.csv`](../../{run_dir}/mae_by_city_protocol.csv)",
        f"Notebook: [`05_intl_city_country_loro_loco.ipynb`](05_intl_city_country_loro_loco.ipynb)",
        "",
        f"Takeaway: {verdict}",
        "",
    ]
    if readme.exists():
        text = readme.read_text()
        marker = "## Results — city & country × city-LORO"
        block = "\n".join(section)
        if marker in text:
            pre, _, rest = text.partition(marker)
            if "## Changelog" in rest:
                _, _, post = rest.partition("## Changelog")
                text = pre.rstrip() + "\n" + block + "\n## Changelog" + post
            else:
                text = pre.rstrip() + "\n" + block
        elif "## Changelog" in text:
            text = text.replace("## Changelog", block + "\n## Changelog")
        else:
            text = text.rstrip() + "\n" + block
        if "## Changelog" in text and "city/country LORO+LOCO" not in text:
            text = text.replace(
                "## Changelog\n\n| Date | Change |\n|---|---|\n",
                "## Changelog\n\n| Date | Change |\n|---|---|\n"
                f"| {date} | city/country LORO+LOCO fraction MAE tables |\n",
            )
        readme.write_text(text)

    atomic_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "finished_at": utc_now(),
            "run_dir": str(run_dir),
            "source_run": str(args.source_run),
            "n_balanced": int(len(bal)),
            "protocols": ["city_loro", "loco"],
            "levels": ["city", "country", "all"],
            "verdict": verdict,
            "notebook": str(nb_path),
            "mae_by_country": country_tbl.to_dict(orient="records"),
            "mae_overall": overall.to_dict(orient="records"),
        },
    )
    write_progress(run_dir, status="completed", stage="done", last_progress_at=utc_now(), verdict=verdict)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, verdict)
    log(run_dir, f"DONE · {nb_path}")


if __name__ == "__main__":
    main()
