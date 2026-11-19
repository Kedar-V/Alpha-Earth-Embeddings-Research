#!/usr/bin/env python3
"""GHS-BUILT-S vs AE fraction MAE for India + Japan.

Uses workspace Postgres at pgdata/ (aef_international) for Overture soft labels
+ AE embeddings. Then:

  1. Export pure clear cells; balance 1:1 res/nres per city
  2. City-LORO Ridge + PLS multi-output fraction OOF (per country)
  3. Download BUILT-S / NRES tiles covering city bboxes
  4. Sample at AE centers; score MAE mean(res, nres) by country + city

Start DB first if needed:
  DB_RUNTIME=local ./week4/international_geo_generalization/scripts/init_db.sh

Artifacts: outputs/ghsl_built_s_vs_ae_intl/<timestamp>/
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "week4" / "international_geo_generalization"))
from lib.db import connect, load_dotenv, wait_until_ready  # noqa: E402

BANDS = [f"A{i:02d}" for i in range(64)]
AE_COLS = [f"a{i:02d}" for i in range(64)]
LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
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
ORIGIN_LEFT = -18_041_000.0
ORIGIN_TOP = 10_000_000.0
TILE_M = 1_000_000.0


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
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        done = int(prev.get("completed", 0) or 0)
        total = int(prev.get("total", 0) or 0)
        if total > 0:
            prev["percent_complete"] = round(100.0 * done / total, 2)
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            rem = max(0, total - done)
            prev["eta_seconds"] = round(rem / rate, 1) if rate > 0 and done > 0 else None
            prev["items_remaining"] = rem
    atomic_json(path, prev)


def write_environment(run_dir: Path) -> None:
    lines = [
        f"cwd={ROOT}",
        f"python={sys.executable}",
        f"pid={os.getpid()}",
        f"cpus={os.cpu_count()}",
        f"hostname={os.uname().nodename}",
        f"POSTGRES_HOST={os.environ.get('POSTGRES_HOST', '')}",
        f"POSTGRES_PORT={os.environ.get('POSTGRES_PORT', '')}",
        f"POSTGRES_DB={os.environ.get('POSTGRES_DB', '')}",
        f"POSTGRES_USER={os.environ.get('POSTGRES_USER', '')}",
        f"PGDATA_DIR={os.environ.get('PGDATA_DIR', str(ROOT / 'pgdata'))}",
        f"started={utc_now()}",
    ]
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n")


def append_failure(run_dir: Path, rec: dict[str, Any]) -> None:
    with (run_dir / "failures.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def lonlat_to_tile(lon: float, lat: float) -> tuple[int, int]:
    tr = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    x, y = tr.transform(lon, lat)
    c = int(np.floor((x - ORIGIN_LEFT) / TILE_M)) + 1
    r = int(np.floor((ORIGIN_TOP - y) / TILE_M))
    return r, c


def tiles_for_bbox(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[tuple[int, int]]:
    tiles = set()
    for lon in np.linspace(lon_min, lon_max, 5):
        for lat in np.linspace(lat_min, lat_max, 5):
            tiles.add(lonlat_to_tile(float(lon), float(lat)))
    return sorted(tiles)


def _download_one(url: str, dest_zip: Path, dest_tif: Path, retries: int = 4) -> dict[str, Any]:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_tif.exists() and dest_tif.stat().st_size > 1000:
        return {
            "item_id": dest_tif.name,
            "status": "skipped_exists",
            "attempts": 0,
            "size_bytes": dest_tif.stat().st_size,
            "error": "",
        }
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            if dest_zip.exists():
                dest_zip.unlink()
            req = Request(url, headers={"User-Agent": "AlphaEarthResearch/1.0 (built-s intl)"})
            with urlopen(req, timeout=600) as resp, dest_zip.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            with zipfile.ZipFile(dest_zip) as zf:
                members = [m for m in zf.namelist() if m.endswith(".tif") and ".ovr" not in m]
                if not members:
                    raise FileNotFoundError(f"no tif in {dest_zip}")
                zf.extract(members[0], path=dest_tif.parent)
                extracted = dest_tif.parent / members[0]
                if extracted.resolve() != dest_tif.resolve():
                    if dest_tif.exists():
                        dest_tif.unlink()
                    extracted.replace(dest_tif)
            dest_zip.unlink(missing_ok=True)
            return {
                "item_id": dest_tif.name,
                "status": "done",
                "attempts": attempt,
                "size_bytes": dest_tif.stat().st_size,
                "error": "",
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(min(60, 2**attempt))
    return {
        "item_id": dest_tif.name,
        "status": "failed",
        "attempts": retries,
        "size_bytes": 0,
        "error": last_err,
    }


def download_tiles(run_dir: Path, tiles: list[tuple[int, int]], workers: int = 4) -> None:
    S_TILE_DIR.mkdir(parents=True, exist_ok=True)
    NRES_TILE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r, c in tiles:
        s_name = f"GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        n_name = f"GHS_BUILT_S_NRES_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        jobs.append(("S", S_URL.format(row=r, col=c), S_TILE_DIR / f"{s_name}.zip", S_TILE_DIR / s_name))
        jobs.append(("NRES", NRES_URL.format(row=r, col=c), NRES_TILE_DIR / f"{n_name}.zip", NRES_TILE_DIR / n_name))
    log(run_dir, f"Download {len(jobs)} products for {len(tiles)} tiles")
    failed = done = 0
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, url, z, t): (kind, t) for kind, url, z, t in jobs}
        for fut in as_completed(futs):
            kind, tif = futs[fut]
            rec = fut.result()
            rec["product"] = kind
            rows.append(rec)
            if rec["status"] == "failed":
                failed += 1
                log(run_dir, f"FAIL {kind} {tif.name}: {rec['error']}")
            else:
                done += 1
                log(run_dir, f"{rec['status']:14s} {kind:4s} {tif.name}")
            write_progress(
                run_dir,
                stage="download",
                completed=done + failed,
                total=len(jobs),
                failed=failed,
                success=done,
                last_progress_at=utc_now(),
            )
    pd.DataFrame(rows).to_csv(run_dir / "download_manifest.csv", index=False)
    if failed:
        raise RuntimeError(f"{failed} tile downloads failed")


def list_cities(conn, countries: set[str]) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT country_id, city_id
            FROM cities
            WHERE country_id = ANY(%s)
            ORDER BY country_id, city_id
            """,
            (sorted(countries),),
        )
        return [(r["country_id"], r["city_id"]) for r in cur.fetchall()]


def export_city_pure(conn, city_id: str, country: str, out_path: Path) -> dict[str, Any]:
    """Export pure clear labeled cells with embeddings + lon/lat + full fracs."""
    if out_path.exists():
        try:
            df = pd.read_parquet(out_path)
            return {
                "city_id": city_id,
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
          c.cell_id,
          c.city_id,
          ST_X(ST_Centroid(c.geom)) AS pixel_lon,
          ST_Y(ST_Centroid(c.geom)) AS pixel_lat,
          cl.frac_res, cl.frac_nres, cl.frac_empty, cl.binary_label,
          cl.purity_status,
          COALESCE(m.frac_res_full, cl.frac_res) AS frac_res_full,
          COALESCE(m.frac_nres_full, cl.frac_nres) AS frac_nres_full,
          COALESCE(m.frac_empty_full, cl.frac_empty) AS frac_empty_full,
          {ae_sel}
        FROM alphaearth_cells c
        JOIN cell_labels cl
          ON cl.cell_id = c.cell_id AND cl.label_version = %s
        LEFT JOIN cell_label_metrics m
          ON m.cell_id = c.cell_id AND m.label_version = cl.label_version
        WHERE c.city_id = %s
          AND c.year = 2024
          AND cl.binary_label IS NOT NULL
          AND cl.ambiguity_status = 'clear'
          AND cl.purity_status IN ('pure_residential', 'pure_non_residential')
          AND c.a00 IS NOT NULL AND c.a63 IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (LABEL_VERSION, city_id))
        rows = cur.fetchall()
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
        pd.DataFrame().to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"city_id": city_id, "status": "empty", "n": 0, "n_res": 0, "n_nres": 0}

    df = pd.DataFrame(rows)
    df["country"] = country
    df["pixel_id"] = df["city_id"].astype(str) + ":" + df["cell_id"].astype(str)
    df["stratum"] = np.where(df["binary_label"].astype(int) == 1, "res", "nres")
    # Upper-case band aliases for training
    for a, A in zip(AE_COLS, BANDS):
        df[A] = df[a].astype(np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return {
        "city_id": city_id,
        "status": "exported",
        "n": int(len(df)),
        "n_res": int((df.stratum == "res").sum()),
        "n_nres": int((df.stratum == "nres").sum()),
    }


def balance_pure(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
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


def mae_rn(yt_r, yt_n, yp_r, yp_n) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yt_r) & np.isfinite(yt_n)
    if m.sum() == 0:
        return {
            "n": 0,
            "mae_mean_res_nres": float("nan"),
            "mae_frac_res": float("nan"),
            "mae_frac_nres": float("nan"),
        }
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_r[m], yp_r[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_n[m], yp_n[m])),
        "mae_mean_res_nres": float(
            0.5 * (mean_absolute_error(yt_r[m], yp_r[m]) + mean_absolute_error(yt_n[m], yp_n[m]))
        ),
    }


def _fit_predict_frac(Xtr, Ytr, Xte, model: str, alpha: float, pls_k: int) -> np.ndarray:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
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


def city_loro_fraction_oof(df: pd.DataFrame, model: str = "ridge", alpha: float = 1.0, pls_k: int = 10) -> pd.DataFrame:
    """City leave-one-out OOF; if <2 cities (e.g. Africa-5), fall back to 5-fold within-city CV."""
    cities = sorted(df.city_id.unique())
    X_all = df[BANDS].to_numpy(dtype=np.float64)
    Y_all = df[["frac_res_full", "frac_nres_full", "frac_empty_full"]].to_numpy(dtype=np.float64)
    oof = np.full_like(Y_all, np.nan)

    if len(cities) >= 2:
        for hold in cities:
            te = (df.city_id == hold).to_numpy()
            tr = ~te
            if tr.sum() < 100 or te.sum() == 0:
                continue
            oof[te] = _fit_predict_frac(X_all[tr], Y_all[tr], X_all[te], model, alpha, pls_k)
    else:
        # Single-city country: city-LORO is undefined; use stratified-ish row KFold OOF.
        n = len(df)
        n_splits = 5 if n >= 500 else max(2, min(5, n // 50 or 2))
        if n < 2 * n_splits:
            n_splits = 2 if n >= 4 else 0
        if n_splits >= 2:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for tr_idx, te_idx in kf.split(X_all):
                if len(tr_idx) < 50:
                    continue
                oof[te_idx] = _fit_predict_frac(
                    X_all[tr_idx], Y_all[tr_idx], X_all[te_idx], model, alpha, pls_k
                )

    out = df.copy()
    out["pred_frac_res"] = oof[:, 0]
    out["pred_frac_nres"] = oof[:, 1]
    out["pred_frac_empty"] = oof[:, 2]
    return out


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
            {
                "pixel_id": pixel_ids,
                "city_id": city_id,
                "built_s_raw": bu.astype(np.int16),
                "built_s_nres_raw": nr.astype(np.int16),
            }
        )
    finally:
        for ds in s_ds + n_ds:
            ds.close()


def sample_built_s(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    workers = workers or max(1, min(os.cpu_count() or 4, df.city_id.nunique()))
    payloads = []
    for city_id, g in df.groupby("city_id", sort=False):
        payloads.append(
            (
                city_id,
                g["pixel_id"].to_numpy(),
                g["pixel_lon"].to_numpy(dtype=np.float64),
                g["pixel_lat"].to_numpy(dtype=np.float64),
            )
        )
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_sample_city, payloads):
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def raw_to_frac(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bu = out["built_s_raw"].astype(np.float64)
    nr = out["built_s_nres_raw"].astype(np.float64)
    valid = (bu != 255) & (nr != 255)
    bu_f = np.clip(bu / 100.0, 0, 1)
    is_nres = nr == 1
    out["bs_frac_res"] = np.where(valid, np.where(is_nres, 0.0, bu_f), np.nan)
    out["bs_frac_nres"] = np.where(valid, np.where(is_nres, bu_f, 0.0), np.nan)
    out["bs_frac_empty"] = np.where(valid, 1.0 - bu_f, np.nan)
    out["built_s_valid"] = valid
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--countries", default="IND,JPN")
    ap.add_argument("--pls-k", type=int, default=10)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "ghsl_built_s_vs_ae_intl" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "failures").mkdir(exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"cmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n"
        f"submit_job=1\nresume_checkpoints={run_dir / 'checkpoints'}\n"
    )
    countries = {c.strip().upper() for c in args.countries.split(",") if c.strip()}

    load_dotenv(ROOT / "week4/international_geo_generalization/db/.env")
    write_environment(run_dir)
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="db",
        started_at=utc_now(),
        _t0=t0,
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count(), "gpus": 0, "workers": "cpu-bound export+sample"},
        countries=sorted(countries),
        countries_completed=[],
        cities_completed=[],
        failed=0,
        label_source="postgres:pgdata/aef_international",
    )
    log(run_dir, f"Run dir {run_dir} · waiting for Postgres (pgdata)")
    wait_until_ready(timeout_s=60)
    log(run_dir, "Postgres ready")

    with connect() as conn:
        cities = list_cities(conn, countries)
        log(run_dir, f"Cities ({len(cities)}): {[c for _, c in cities]}")
        if not cities:
            raise SystemExit("no cities in DB for requested countries")

        # --- export from Postgres (idempotent per-city parquet checkpoints) ---
        write_progress(
            run_dir,
            stage="export",
            total=len(cities),
            completed=0,
            failed=0,
            last_progress_at=utc_now(),
        )
        export_rows = []
        cities_completed: list[str] = []
        countries_completed: list[str] = []
        failed = 0
        for i, (ctry, cid) in enumerate(cities, 1):
            outp = run_dir / "checkpoints" / f"pure_{cid}.parquet"
            try:
                rec = export_city_pure(conn, cid, ctry, outp)
            except Exception as e:
                failed += 1
                rec = {"city_id": cid, "status": "failed", "n": 0, "n_res": 0, "n_nres": 0, "error": str(e)}
                append_failure(
                    run_dir,
                    {"item_id": cid, "stage": "export", "error": str(e), "at": utc_now()},
                )
                log(run_dir, f"export FAIL {cid}: {e}")
            else:
                cities_completed.append(cid)
                log(
                    run_dir,
                    f"export {rec['status']:8s} {cid}: n={rec['n']:,} res={rec['n_res']:,} nres={rec['n_nres']:,}",
                )
            export_rows.append(rec)
            # country marker when all cities of that country are done among planned list
            planned_ctry = [c for co, c in cities if co == ctry]
            done_ctry = [c for c in cities_completed if c.startswith(ctry + "_")]
            if len(done_ctry) == len(planned_ctry) and ctry not in countries_completed:
                countries_completed.append(ctry)
                (run_dir / "checkpoints" / f"COUNTRY_{ctry}.DONE").write_text(utc_now() + "\n")
                log(run_dir, f"COUNTRY_DONE {ctry} cities={done_ctry}")
            write_progress(
                run_dir,
                stage="export",
                completed=i,
                total=len(cities),
                failed=failed,
                success=len(cities_completed),
                cities_completed=cities_completed,
                countries_completed=countries_completed,
                current_city=cid,
                last_progress_at=utc_now(),
            )
            # live export manifest
            pd.DataFrame(export_rows).to_csv(run_dir / "export_city_counts.csv", index=False)
        pd.DataFrame(export_rows).to_csv(run_dir / "export_city_counts.csv", index=False)
        if failed:
            raise RuntimeError(f"{failed} city exports failed; see failures.jsonl")

    labeled = pd.concat(
        [pd.read_parquet(run_dir / "checkpoints" / f"pure_{cid}.parquet") for _, cid in cities if (run_dir / "checkpoints" / f"pure_{cid}.parquet").exists()],
        ignore_index=True,
    )
    if labeled.empty:
        raise SystemExit("no pure cells exported")
    labeled.to_parquet(run_dir / "artifacts" / "pure_all.parquet", index=False)
    log(run_dir, f"Pure clear cells n={len(labeled):,}")

    write_progress(run_dir, stage="balance", last_progress_at=utc_now())
    bal = balance_pure(labeled, seed=42)
    bal.to_parquet(run_dir / "artifacts" / "balanced_pure.parquet", index=False)
    log(run_dir, f"Balanced pure n={len(bal):,} cities={bal.city_id.nunique()}")
    for ctry, g in bal.groupby("country"):
        log(run_dir, f"  {ctry}: n={len(g):,} cities={g.city_id.nunique()}")

    write_progress(run_dir, stage="train_oof", last_progress_at=utc_now())
    oof_parts = []
    for ctry, g in bal.groupby("country", sort=False):
        n_cities = int(g.city_id.nunique())
        mode = "city-LORO" if n_cities >= 2 else "within-city-KFold"
        log(run_dir, f"{mode} Ridge {ctry} n={len(g):,} cities={n_cities}")
        ridge = city_loro_fraction_oof(g, model="ridge")
        log(run_dir, f"{mode} PLS k={args.pls_k} {ctry}")
        pls = city_loro_fraction_oof(g, model="pls", pls_k=args.pls_k)
        merged = ridge.rename(
            columns={
                "pred_frac_res": "ridge_frac_res",
                "pred_frac_nres": "ridge_frac_nres",
                "pred_frac_empty": "ridge_frac_empty",
            }
        )
        merged["pls_frac_res"] = pls["pred_frac_res"].to_numpy()
        merged["pls_frac_nres"] = pls["pred_frac_nres"].to_numpy()
        merged["pls_frac_empty"] = pls["pred_frac_empty"].to_numpy()
        oof_parts.append(merged)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(run_dir / "artifacts" / "oof_ae_fractions.parquet", index=False)

    write_progress(run_dir, stage="download", last_progress_at=utc_now())
    tiles = set()
    for _, g in oof.groupby("city_id"):
        tiles.update(
            tiles_for_bbox(
                float(g.pixel_lon.min()),
                float(g.pixel_lat.min()),
                float(g.pixel_lon.max()),
                float(g.pixel_lat.max()),
            )
        )
    tiles_l = sorted(tiles)
    log(run_dir, f"GHSL tiles needed: {tiles_l}")
    atomic_json(run_dir / "tiles_needed.json", {"tiles": [[r, c] for r, c in tiles_l]})
    if not args.skip_download:
        download_tiles(run_dir, tiles_l, workers=4)

    write_progress(run_dir, stage="sample", last_progress_at=utc_now())
    bs = sample_built_s(oof[["pixel_id", "city_id", "pixel_lon", "pixel_lat"]])
    bs = raw_to_frac(bs)
    bs.to_parquet(run_dir / "checkpoints" / "built_s_at_ae_centers.parquet", index=False)
    log(run_dir, f"BUILT-S sampled n={len(bs):,} valid={float(bs.built_s_valid.mean()):.3f}")

    j = oof.merge(
        bs[["pixel_id", "bs_frac_res", "bs_frac_nres", "bs_frac_empty", "built_s_valid"]],
        on="pixel_id",
        how="left",
    )
    j.to_parquet(run_dir / "artifacts" / "scored.parquet", index=False)

    write_progress(run_dir, stage="score", last_progress_at=utc_now())
    rows = []
    scopes = [("all", j)]
    scopes += [(f"country:{c}", g) for c, g in j.groupby("country")]
    scopes += [(f"city:{c}", g) for c, g in j.groupby("city_id")]
    for scope, sub in scopes:
        yt_r, yt_n = sub["frac_res_full"].to_numpy(), sub["frac_nres_full"].to_numpy()
        for name, a, b in [
            ("ae_ridge", "ridge_frac_res", "ridge_frac_nres"),
            ("ae_pls", "pls_frac_res", "pls_frac_nres"),
            ("ghsl_built_s", "bs_frac_res", "bs_frac_nres"),
        ]:
            met = mae_rn(yt_r, yt_n, sub[a].to_numpy(), sub[b].to_numpy())
            met.update({"model": name, "scope": scope})
            rows.append(met)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)

    country_tbl = (
        summary[summary.scope.str.startswith("country:")]
        .assign(country=lambda d: d.scope.str.replace("country:", "", regex=False))
        .pivot_table(index="country", columns="model", values="mae_mean_res_nres", aggfunc="first")
    )
    country_tbl.to_csv(run_dir / "mae_by_country.csv")
    log(run_dir, "MAE by country:\n" + country_tbl.to_string())

    def _tbl_val(tbl: pd.DataFrame, ctry: str, model: str) -> float:
        if model not in tbl.columns or ctry not in tbl.index:
            return float("nan")
        v = tbl.loc[ctry, model]
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    write_progress(run_dir, stage="figures", last_progress_at=utc_now())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    countries_sorted = sorted(country_tbl.index)
    x = np.arange(len(countries_sorted))
    w = 0.25
    for i, (model, color) in enumerate(
        [("ghsl_built_s", "#ff7f0e"), ("ae_ridge", "#2ca02c"), ("ae_pls", "#1f77b4")]
    ):
        vals = [_tbl_val(country_tbl, c, model) for c in countries_sorted]
        ax.bar(x + (i - 1) * w, vals, w, label=model, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(countries_sorted)
    ax.set_ylabel("MAE mean(res, nres) ↓")
    ax.set_title("BUILT-S vs AE city-LORO/KFold · balanced pure · Postgres labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mae_by_country.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    city_sum = summary[summary.scope.str.startswith("city:")].copy()
    city_sum["city_id"] = city_sum.scope.str.replace("city:", "", regex=False)
    for ctry in sorted(countries):
        cities_c = sorted(j.loc[j.country == ctry, "city_id"].unique())
        if not cities_c:
            continue
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(cities_c))))
        y = np.arange(len(cities_c))
        for model, color, off in [
            ("ghsl_built_s", "#ff7f0e", -0.25),
            ("ae_ridge", "#2ca02c", 0.0),
            ("ae_pls", "#1f77b4", 0.25),
        ]:
            vals = []
            for cid in cities_c:
                row = city_sum[(city_sum.city_id == cid) & (city_sum.model == model)]
                vals.append(float(row.mae_mean_res_nres.iloc[0]) if len(row) else np.nan)
            ax.barh(y + off, vals, height=0.25, color=color, label=model)
        ax.set_yticks(y)
        ax.set_yticklabels(cities_c)
        ax.set_xlabel("MAE mean(res, nres) ↓")
        ax.set_title(f"{ctry} · per-city")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / f"mae_by_city_{ctry}.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    nb_dir = ROOT / "notebooks" / "ghsl_built_s_vs_ae"
    nb_dir.mkdir(exist_ok=True)
    (nb_dir / "figures").mkdir(exist_ok=True)
    for p in (run_dir / "figures").glob("*.png"):
        (nb_dir / "figures" / p.name).write_bytes(p.read_bytes())

    lines = []
    for ctry in countries_sorted:
        bs_v = _tbl_val(country_tbl, ctry, "ghsl_built_s")
        ridge_v = _tbl_val(country_tbl, ctry, "ae_ridge")
        pls_v = _tbl_val(country_tbl, ctry, "ae_pls")
        ridge_rows = summary[(summary.scope == f"country:{ctry}") & (summary.model == "ae_ridge")]
        n_c = int(ridge_rows.n.iloc[0]) if len(ridge_rows) else 0
        lines.append(
            f"{ctry} n={n_c:,}: BUILT-S={bs_v:.3f} Ridge={ridge_v:.3f} PLS={pls_v:.3f}"
        )
    verdict = " | ".join(lines)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GHS-BUILT-S vs AE — India & Japan\n\n",
                    f"Run: `{run_dir}`\n\n",
                    "Labels from workspace Postgres (`pgdata/` → `aef_international`). "
                    "Balanced pure, city-LORO Ridge + PLS vs BUILT-S.\n\n",
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
                    "display(pd.read_csv(RUN / 'mae_by_country.csv'))\n",
                    "display(pd.read_csv(RUN / 'mae_summary.csv').query(\"scope.str.startswith('country:')\"))\n",
                    "for p in sorted((RUN / 'figures').glob('*.png')):\n",
                    "    display(Markdown(f'### {p.name}'))\n",
                    "    display(Image(filename=str(p)))\n",
                ],
            },
        ],
    }
    nb_path = nb_dir / "04_built_s_vs_ae_intl.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))

    # README append
    readme = nb_dir / "README.md"
    section_lines = [
        "",
        "---",
        "",
        "## Results — India & Japan (balanced pure, city-LORO, Postgres labels)",
        "",
        "Labels: `pgdata/` → `aef_international` (`resi_def_v1_pixel_frac_v1` pure clear). "
        "AE heads: city-LORO OOF Ridge + PLS. GHSL: BUILT-S sampled at AE centers.",
        "",
        "| Date | Country | n | BUILT-S | Ridge | PLS | Run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for ctry in countries_sorted:
        bs_v = _tbl_val(country_tbl, ctry, "ghsl_built_s")
        ridge_v = _tbl_val(country_tbl, ctry, "ae_ridge")
        pls_v = _tbl_val(country_tbl, ctry, "ae_pls")
        ridge_rows = summary[(summary.scope == f"country:{ctry}") & (summary.model == "ae_ridge")]
        n_c = int(ridge_rows.n.iloc[0]) if len(ridge_rows) else 0
        section_lines.append(
            f"| {datetime.now().strftime('%Y-%m-%d')} | {ctry} | {n_c:,} | **{bs_v:.3f}** | {ridge_v:.3f} | {pls_v:.3f} | [`{run_dir.name}`](../../{run_dir}) |"
        )
    section_lines += [
        "",
        f"Notebook: [`04_built_s_vs_ae_intl.ipynb`](04_built_s_vs_ae_intl.ipynb)",
        "",
        f"Takeaway: {verdict}",
        "",
    ]
    if readme.exists():
        text = readme.read_text()
        marker = "## Results — India & Japan"
        block = "\n".join(section_lines)
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
        if "## Changelog" in text and "India & Japan city-LORO" not in text:
            text = text.replace(
                "## Changelog\n\n| Date | Change |\n|---|---|\n",
                "## Changelog\n\n| Date | Change |\n|---|---|\n"
                f"| {datetime.now().strftime('%Y-%m-%d')} | India & Japan city-LORO via Postgres pgdata |\n",
            )
        readme.write_text(text)

    payload = {
        "status": "completed",
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "label_source": "postgres pgdata/aef_international",
        "countries": sorted(countries),
        "cities": [c for _, c in cities],
        "n_balanced": int(len(bal)),
        "verdict": verdict,
        "mae_by_country": country_tbl.reset_index().to_dict(orient="records"),
        "notebook": str(nb_path),
    }
    atomic_json(run_dir / "summary.json", payload)
    write_progress(run_dir, status="completed", stage="done", last_progress_at=utc_now(), verdict=verdict)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, verdict)
    log(run_dir, f"DONE · {nb_path}")


if __name__ == "__main__":
    main()
