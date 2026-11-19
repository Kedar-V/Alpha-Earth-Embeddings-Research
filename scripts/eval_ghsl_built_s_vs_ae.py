#!/usr/bin/env python3
"""Compare GHS-BUILT-S (soft built surface) vs AE fraction MAE models.

Downloads GHS-BUILT-S E2018 10 m (BUFRAC 0–100) and GHS-BUILT-S NRES E2018
10 m (boolean) for the same US tiles already used by GHS-BUILT-C FUN, samples
both at AE cell centers, converts to soft fractions:

  bu = BUFRAC / 100
  if NRES==1: (frac_res, frac_nres) = (0, bu)
  else:       (frac_res, frac_nres) = (bu, 0)
  frac_empty = 1 - bu

Scores MAE against Overture coverage GT alongside:
  - W4 balanced Ridge OOF (AE MAE model)
  - GHS-BUILT-C FUN one-hot (prior hard proxy)

Run dir (default):
  outputs/ghsl_built_s_vs_ae/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FUN_TILE_DIR = ROOT / "data" / "ghsl" / "tiles"
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

AE_OOF = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
SHARED_FRAC = ROOT / "outputs" / "week4_ghsl_pixel" / "predictions_balanced_fractions.parquet"
BAL_PIXELS = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
FULL_PIXELS = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
FUN_CACHE = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    if "started_at" in prev and "completed" in prev and "total" in prev:
        started = prev.get("_t0")
        if started:
            elapsed = max(1e-6, time.time() - float(started))
            done = int(prev.get("completed", 0))
            total = max(1, int(prev.get("total", 1)))
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            rem = max(0, total - done)
            prev["eta_seconds"] = round(rem / rate, 1) if rate > 0 else None
            prev["elapsed_seconds"] = round(elapsed, 1)
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def discover_tiles() -> list[tuple[int, int]]:
    tiles = []
    for p in sorted(FUN_TILE_DIR.glob("GHS_BUILT_C_FUN_*_R*_C*.tif")):
        name = p.stem
        # ..._R5_C11
        try:
            r = int(name.split("_R")[-1].split("_")[0])
            c = int(name.split("_C")[-1])
        except Exception:
            continue
        tiles.append((r, c))
    return sorted(set(tiles))


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
            req = Request(url, headers={"User-Agent": "AlphaEarthResearch/1.0 (built-s eval)"})
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


def download_tiles(run_dir: Path, tiles: list[tuple[int, int]], workers: int = 3) -> pd.DataFrame:
    jobs = []
    for r, c in tiles:
        s_name = f"GHS_BUILT_S_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        n_name = f"GHS_BUILT_S_NRES_E2018_GLOBE_R2023A_54009_10_V1_0_R{r}_C{c}.tif"
        jobs.append(("S", S_URL.format(row=r, col=c), S_TILE_DIR / f"{s_name}.zip", S_TILE_DIR / s_name))
        jobs.append(("NRES", NRES_URL.format(row=r, col=c), NRES_TILE_DIR / f"{n_name}.zip", NRES_TILE_DIR / n_name))

    manifest_path = run_dir / "manifest.csv"
    rows: list[dict[str, Any]] = []
    write_progress(
        run_dir,
        status="running",
        stage="download",
        completed=0,
        total=len(jobs),
        failed=0,
        _t0=time.time(),
        output_location=str(run_dir),
    )
    log(run_dir, f"Downloading {len(jobs)} tiles ({len(tiles)} S + {len(tiles)} NRES) workers={workers}")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, url, z, t): (kind, t) for kind, url, z, t in jobs}
        for fut in as_completed(futs):
            kind, tif = futs[fut]
            rec = fut.result()
            rec["product"] = kind
            rec["started_at"] = ""
            rec["completed_at"] = utc_now()
            rows.append(rec)
            if rec["status"] == "failed":
                failed += 1
                log(run_dir, f"FAIL {kind} {tif.name}: {rec['error']}")
                with (run_dir / "failures.jsonl").open("a") as ff:
                    ff.write(json.dumps(rec) + "\n")
            else:
                done += 1
                mb = rec["size_bytes"] / 1e6
                log(run_dir, f"{rec['status']:14s} {kind:4s} {tif.name} ({mb:.1f} MB)")
            write_progress(
                run_dir,
                status="running",
                stage="download",
                completed=done + failed,
                total=len(jobs),
                failed=failed,
                success=done,
                last_progress_at=utc_now(),
            )
            pd.DataFrame(rows).to_csv(manifest_path, index=False)

    if failed:
        raise RuntimeError(f"{failed} tile downloads failed; see {run_dir / 'failures.jsonl'}")
    return pd.DataFrame(rows)


@dataclass
class DualMosaic:
    s_paths: list[Path]
    nres_paths: list[Path]
    s_ds: list[Any]
    nres_ds: list[Any]

    @classmethod
    def open(cls) -> "DualMosaic":
        s_paths = sorted(S_TILE_DIR.glob("GHS_BUILT_S_E2018_*.tif"))
        nres_paths = sorted(NRES_TILE_DIR.glob("GHS_BUILT_S_NRES_E2018_*.tif"))
        if not s_paths or not nres_paths:
            raise FileNotFoundError("missing BUILT-S / NRES tiles")
        if len(s_paths) != len(nres_paths):
            raise RuntimeError(f"tile count mismatch S={len(s_paths)} NRES={len(nres_paths)}")
        return cls(
            s_paths=s_paths,
            nres_paths=nres_paths,
            s_ds=[rasterio.open(p) for p in s_paths],
            nres_ds=[rasterio.open(p) for p in nres_paths],
        )

    @property
    def crs(self):
        return self.s_ds[0].crs

    def close(self) -> None:
        for ds in self.s_ds + self.nres_ds:
            ds.close()


def _sample_region(payload: tuple) -> pd.DataFrame:
    region, pixel_ids, lons, lats = payload
    mosaic = DualMosaic.open()
    try:
        transformer = Transformer.from_crs("EPSG:4326", mosaic.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        bu_raw = np.full(len(xs), 255, dtype=np.uint8)
        nres_raw = np.full(len(xs), 255, dtype=np.uint8)
        assigned = np.full(len(xs), -1, dtype=np.int32)
        for ti, ds in enumerate(mosaic.s_ds):
            b = ds.bounds
            m = (
                (xs >= b.left)
                & (xs <= b.right)
                & (ys >= b.bottom)
                & (ys <= b.top)
                & (assigned < 0)
            )
            assigned[m] = ti
        for ti, (ds_s, ds_n) in enumerate(zip(mosaic.s_ds, mosaic.nres_ds)):
            idx = np.where(assigned == ti)[0]
            if len(idx) == 0:
                continue
            coords = list(zip(xs[idx], ys[idx]))
            bu_raw[idx] = np.fromiter(
                (int(v[0]) for v in ds_s.sample(coords, indexes=1)),
                dtype=np.uint8,
                count=len(idx),
            )
            nres_raw[idx] = np.fromiter(
                (int(v[0]) for v in ds_n.sample(coords, indexes=1)),
                dtype=np.uint8,
                count=len(idx),
            )
        return pd.DataFrame(
            {
                "pixel_id": pixel_ids,
                "region": region,
                "built_s_raw": bu_raw.astype(np.int16),
                "built_s_nres_raw": nres_raw.astype(np.int16),
            }
        )
    finally:
        mosaic.close()


def raw_to_fractions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bu = out["built_s_raw"].astype(np.float64)
    nres = out["built_s_nres_raw"].astype(np.float64)
    valid = (bu != 255) & (nres != 255)
    bu_frac = np.clip(bu / 100.0, 0.0, 1.0)
    # NRES boolean: assign all built mass to NRES or RES
    is_nres = nres == 1
    frac_res = np.where(is_nres, 0.0, bu_frac)
    frac_nres = np.where(is_nres, bu_frac, 0.0)
    frac_empty = 1.0 - bu_frac
    out["pred_frac_res"] = np.where(valid, frac_res, np.nan)
    out["pred_frac_nres"] = np.where(valid, frac_nres, np.nan)
    out["pred_frac_empty"] = np.where(valid, frac_empty, np.nan)
    out["built_s_valid"] = valid
    out["built_s_bu_frac"] = np.where(valid, bu_frac, np.nan)
    return out


def sample_all(run_dir: Path, pixels: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    cache = run_dir / "checkpoints" / "built_s_at_ae_centers.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        log(run_dir, f"Reusing sample cache {cache}")
        return pd.read_parquet(cache)

    payloads = []
    for region, sub in pixels.groupby("region", sort=True):
        payloads.append(
            (
                str(region),
                sub["pixel_id"].to_numpy(),
                sub["pixel_lon"].to_numpy(dtype=np.float64),
                sub["pixel_lat"].to_numpy(dtype=np.float64),
            )
        )
    n_cpu = os.cpu_count() or 1
    workers = workers or min(n_cpu, len(payloads))
    write_progress(
        run_dir,
        status="running",
        stage="sample",
        completed=0,
        total=len(payloads),
        failed=0,
        n_pixels=len(pixels),
        last_progress_at=utc_now(),
    )
    log(run_dir, f"Sampling BUILT-S at {len(pixels):,} AE centers · {len(payloads)} metros · workers={workers}")
    parts = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_sample_region, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            reg = futs[fut]
            part = fut.result()
            parts.append(part)
            done += 1
            log(run_dir, f"  sampled {reg}: n={len(part):,}")
            write_progress(
                run_dir,
                status="running",
                stage="sample",
                completed=done,
                total=len(payloads),
                last_progress_at=utc_now(),
            )
    out = raw_to_fractions(pd.concat(parts, ignore_index=True))
    out.to_parquet(cache, index=False)
    log(run_dir, f"Wrote {cache}")
    return out


def mae_pack(yt_res, yt_nres, yt_empty, yp_res, yp_nres, yp_empty) -> dict[str, float]:
    m = np.isfinite(yp_res) & np.isfinite(yp_nres) & np.isfinite(yp_empty)
    if m.sum() == 0:
        return {"n": 0}
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_res[m], yp_res[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_nres[m], yp_nres[m])),
        "mae_frac_empty": float(mean_absolute_error(yt_empty[m], yp_empty[m])),
        "mae_mean_3": float(
            np.mean(
                [
                    mean_absolute_error(yt_res[m], yp_res[m]),
                    mean_absolute_error(yt_nres[m], yp_nres[m]),
                    mean_absolute_error(yt_empty[m], yp_empty[m]),
                ]
            )
        ),
        "mae_mean_res_nres": float(
            0.5
            * (
                mean_absolute_error(yt_res[m], yp_res[m])
                + mean_absolute_error(yt_nres[m], yp_nres[m])
            )
        ),
        "r2_frac_res": float(r2_score(yt_res[m], yp_res[m])),
        "r2_frac_nres": float(r2_score(yt_nres[m], yp_nres[m])),
        "r2_frac_empty": float(r2_score(yt_empty[m], yp_empty[m])),
        "r2_built": float(r2_score(yt_res[m] + yt_nres[m], yp_res[m] + yp_nres[m])),
    }


def fun_onehot(labels: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    res = (labels == "RES").astype(float).to_numpy()
    nres = (labels == "NRES").astype(float).to_numpy()
    empty = (~labels.isin(["RES", "NRES"])).astype(float).to_numpy()
    return res, nres, empty


def score_eval(
    name: str,
    gt: pd.DataFrame,
    ae: pd.DataFrame,
    built_s: pd.DataFrame,
    fun: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    base = gt[["pixel_id", "region", "frac_res", "frac_nres", "frac_empty"]].drop_duplicates("pixel_id")
    j = base.merge(
        ae[["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]].rename(
            columns={
                "pred_frac_res": "ae_frac_res",
                "pred_frac_nres": "ae_frac_nres",
                "pred_frac_empty": "ae_frac_empty",
            }
        ),
        on="pixel_id",
        how="inner",
    )
    j = j.merge(
        built_s[
            [
                "pixel_id",
                "pred_frac_res",
                "pred_frac_nres",
                "pred_frac_empty",
                "built_s_valid",
                "built_s_bu_frac",
            ]
        ].rename(
            columns={
                "pred_frac_res": "bs_frac_res",
                "pred_frac_nres": "bs_frac_nres",
                "pred_frac_empty": "bs_frac_empty",
            }
        ),
        on="pixel_id",
        how="inner",
    )
    rows = []
    yt_r = j["frac_res"].to_numpy()
    yt_n = j["frac_nres"].to_numpy()
    yt_e = j["frac_empty"].to_numpy()
    for model, pr, pn, pe in [
        ("ae_w4_ridge", j["ae_frac_res"], j["ae_frac_nres"], j["ae_frac_empty"]),
        ("ghsl_built_s", j["bs_frac_res"], j["bs_frac_nres"], j["bs_frac_empty"]),
    ]:
        met = mae_pack(yt_r, yt_n, yt_e, pr.to_numpy(), pn.to_numpy(), pe.to_numpy())
        met.update({"model": model, "eval": name})
        rows.append(met)

    if fun is not None:
        jf = j.merge(fun[["pixel_id", "ghsl_label"]], on="pixel_id", how="left")
        fr, fn, fe = fun_onehot(jf["ghsl_label"].fillna("NODATA"))
        met = mae_pack(yt_r, yt_n, yt_e, fr, fn, fe)
        met.update({"model": "ghsl_fun_onehot", "eval": name})
        rows.append(met)

    # per-region for built-s and ae
    per = []
    for reg, sub in j.groupby("region"):
        for model, pr, pn, pe in [
            ("ae_w4_ridge", sub["ae_frac_res"], sub["ae_frac_nres"], sub["ae_frac_empty"]),
            ("ghsl_built_s", sub["bs_frac_res"], sub["bs_frac_nres"], sub["bs_frac_empty"]),
        ]:
            met = mae_pack(
                sub["frac_res"].to_numpy(),
                sub["frac_nres"].to_numpy(),
                sub["frac_empty"].to_numpy(),
                pr.to_numpy(),
                pn.to_numpy(),
                pe.to_numpy(),
            )
            met.update({"model": model, "eval": name, "region": reg})
            per.append(met)
    return rows, per, j


def make_figures(run_dir: Path, summary: pd.DataFrame, per_reg: pd.DataFrame) -> list[Path]:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Ladder on shared / full
    for eval_name in summary["eval"].unique():
        s = summary[summary["eval"] == eval_name].sort_values("mae_mean_res_nres")
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        colors = []
        for m in s["model"]:
            if m.startswith("ae"):
                colors.append("#2ca02c")
            elif "built_s" in m:
                colors.append("#ff7f0e")
            else:
                colors.append("#7f7f7f")
        ax.barh(s["model"], s["mae_mean_res_nres"], color=colors)
        for y, v in enumerate(s["mae_mean_res_nres"]):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("MAE mean(res, nres)  ↓ better")
        ax.set_title(f"Fraction MAE · {eval_name}")
        ax.set_xlim(0, max(0.55, float(s["mae_mean_res_nres"].max()) * 1.25))
        fig.tight_layout()
        p = fig_dir / f"mae_ladder_{eval_name}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    # Per-metro AE vs BUILT-S on shared
    if len(per_reg):
        sub = per_reg[per_reg["eval"] == "shared_59k"].copy()
        if len(sub):
            piv = sub.pivot_table(index="region", columns="model", values="mae_mean_res_nres")
            piv = piv.sort_values("ae_w4_ridge")
            fig, ax = plt.subplots(figsize=(9, 5))
            y = np.arange(len(piv))
            h = 0.35
            ax.barh(y - h / 2, piv["ae_w4_ridge"], height=h, color="#2ca02c", label="AE W4 Ridge")
            ax.barh(y + h / 2, piv["ghsl_built_s"], height=h, color="#ff7f0e", label="GHS-BUILT-S")
            ax.set_yticks(y)
            ax.set_yticklabels(piv.index)
            ax.set_xlabel("MAE mean(res, nres)")
            ax.set_title("Per-metro fraction MAE · shared 59k set")
            ax.legend()
            fig.tight_layout()
            p = fig_dir / "mae_by_metro_shared.png"
            fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths.append(p)

    # Component breakdown
    s = summary[summary["eval"] == "shared_59k"]
    if len(s):
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x = np.arange(len(s))
        w = 0.25
        ax.bar(x - w, s["mae_frac_res"], width=w, label="res", color="#4C78A8")
        ax.bar(x, s["mae_frac_nres"], width=w, label="nres", color="#F58518")
        ax.bar(x + w, s["mae_frac_empty"], width=w, label="empty", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels(s["model"], rotation=15)
        ax.set_ylabel("MAE")
        ax.set_title("Component MAE · shared 59k")
        ax.legend()
        fig.tight_layout()
        p = fig_dir / "mae_components_shared.png"
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)

    return paths


def build_notebook(run_dir: Path, summary: pd.DataFrame) -> Path:
    nb_dir = ROOT / "notebooks" / "ghsl_built_s_vs_ae"
    nb_dir.mkdir(parents=True, exist_ok=True)
    # relative paths from notebook to run_dir
    try:
        rel = os.path.relpath(run_dir, nb_dir)
    except Exception:
        rel = str(run_dir)

    cells = []
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GHS-BUILT-S vs AE fraction MAE\n",
                "\n",
                "Soft GHSL built-up surface (`GHS-BUILT-S` BUFRAC + NRES boolean) sampled at AE cell centers, "
                "compared to Week-4 Ridge OOF fraction predictions and the old FUN one-hot proxy.\n",
                "\n",
                f"Run artifacts: `{rel}`\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import pandas as pd\n",
                "from IPython.display import Image, display, Markdown\n",
                "\n",
                f"RUN = Path({str(run_dir)!r})\n",
                "summary = pd.read_csv(RUN / 'mae_summary.csv')\n",
                "per = pd.read_csv(RUN / 'mae_per_region.csv') if (RUN / 'mae_per_region.csv').exists() else None\n",
                "meta = json.loads((RUN / 'summary.json').read_text())\n",
                "summary",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "shared = summary[summary.eval == 'shared_59k'].sort_values('mae_mean_res_nres')\n",
                "display(Markdown('## Shared 59k head-to-head'))\n",
                "display(shared[['model','n','mae_mean_res_nres','mae_frac_res','mae_frac_nres','mae_frac_empty','r2_built']])\n",
                "for p in sorted((RUN / 'figures').glob('*.png')):\n",
                "    display(Markdown(f'### {p.name}'))\n",
                "    display(Image(filename=str(p)))\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Markdown('## Method'))\n",
                "print(json.dumps(meta.get('method', {}), indent=2))\n",
                "display(Markdown('## Verdict'))\n",
                "print(meta.get('verdict', ''))\n",
            ],
        }
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    out = nb_dir / "01_built_s_vs_ae_mae.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    # also copy figures into notebook folder for easy viewing
    fig_out = nb_dir / "figures"
    fig_out.mkdir(exist_ok=True)
    for p in (run_dir / "figures").glob("*.png"):
        target = fig_out / p.name
        target.write_bytes(p.read_bytes())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--download-workers", type=int, default=3)
    ap.add_argument("--sample-workers", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "ghsl_built_s_vs_ae" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"cwd={ROOT}\ncmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"python={sys.version}\ncpus={os.cpu_count()}\n"
        f"S_TILE_DIR={S_TILE_DIR}\nNRES_TILE_DIR={NRES_TILE_DIR}\n"
    )
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        completed=0,
        total=1,
        failed=0,
        _t0=time.time(),
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count(), "download_workers": args.download_workers},
    )
    log(run_dir, f"Run dir {run_dir}")

    tiles = discover_tiles()
    log(run_dir, f"Tiles from FUN mosaic: {tiles}")
    if not args.skip_download:
        download_tiles(run_dir, tiles, workers=args.download_workers)
    else:
        log(run_dir, "Skipping download (--skip-download)")

    # Probe encoding on first S/NRES tile
    s0 = next(S_TILE_DIR.glob("GHS_BUILT_S_E2018_*.tif"))
    n0 = next(NRES_TILE_DIR.glob("GHS_BUILT_S_NRES_E2018_*.tif"))
    with rasterio.open(s0) as ds:
        a = ds.read(1, window=((0, 256), (0, 256)))
        log(run_dir, f"S probe {s0.name}: dtype={a.dtype} unique_head={np.unique(a)[:12].tolist()} nodata={ds.nodata}")
    with rasterio.open(n0) as ds:
        a = ds.read(1, window=((0, 256), (0, 256)))
        log(run_dir, f"NRES probe {n0.name}: dtype={a.dtype} unique={np.unique(a).tolist()[:12]} nodata={ds.nodata}")

    # Load AE OOF + coords for all pixels we may score
    write_progress(run_dir, stage="load_ae", last_progress_at=utc_now())
    ae = pd.read_parquet(
        AE_OOF,
        columns=[
            "pixel_id",
            "region",
            "pixel_lon",
            "pixel_lat",
            "frac_res",
            "frac_nres",
            "frac_empty",
            "pred_frac_res",
            "pred_frac_nres",
            "pred_frac_empty",
        ],
    )
    log(run_dir, f"AE OOF rows={len(ae):,}")

    # Sample on union of shared 59k + AE OOF coords (dedupe) — actually sample all AE OOF
    # for reuse; shared is subset.
    sample_pixels = ae[["pixel_id", "region", "pixel_lon", "pixel_lat"]].drop_duplicates("pixel_id")
    built_s = sample_all(run_dir, sample_pixels, workers=args.sample_workers)

    fun = None
    if FUN_CACHE.exists():
        fun = pd.read_parquet(FUN_CACHE, columns=["pixel_id", "region", "ghsl_label"])
        log(run_dir, f"Loaded FUN cache n={len(fun):,}")

    write_progress(run_dir, stage="score", last_progress_at=utc_now())
    shared_ids = pd.read_parquet(SHARED_FRAC, columns=["pixel_id", "region", "frac_res", "frac_nres", "frac_empty"])
    # full building-overlapped: cells with any building cover from AE OOF GT
    full_gt = ae.loc[(ae["frac_res"] + ae["frac_nres"]) > 0, ["pixel_id", "region", "frac_res", "frac_nres", "frac_empty"]]

    all_rows = []
    all_per = []
    joined_shared = None
    for name, gt in [("shared_59k", shared_ids), ("full_building", full_gt)]:
        rows, per, joined = score_eval(name, gt, ae, built_s, fun)
        all_rows.extend(rows)
        all_per.extend(per)
        if name == "shared_59k":
            joined_shared = joined
        log(run_dir, f"Scored {name}: " + ", ".join(f"{r['model']}={r.get('mae_mean_res_nres', float('nan')):.4f}" for r in rows))

    summary = pd.DataFrame(all_rows)
    per_reg = pd.DataFrame(all_per)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)
    per_reg.to_csv(run_dir / "mae_per_region.csv", index=False)
    if joined_shared is not None:
        art = run_dir / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        joined_shared.to_parquet(art / "shared_59k_joined.parquet", index=False)

    figs = make_figures(run_dir, summary, per_reg)
    log(run_dir, f"Figures: {[str(p) for p in figs]}")

    shared = summary[summary["eval"] == "shared_59k"].set_index("model")
    ae_mae = float(shared.loc["ae_w4_ridge", "mae_mean_res_nres"])
    bs_mae = float(shared.loc["ghsl_built_s", "mae_mean_res_nres"])
    fun_mae = float(shared.loc["ghsl_fun_onehot", "mae_mean_res_nres"]) if "ghsl_fun_onehot" in shared.index else None
    if ae_mae < bs_mae:
        verdict = (
            f"On the shared 59k set, AE W4 Ridge MAE={ae_mae:.3f} beats GHS-BUILT-S MAE={bs_mae:.3f} "
            f"(Δ={bs_mae - ae_mae:.3f}). Soft BUILT-S is the fairer GHSL fraction compare than FUN one-hot"
            + (f" (FUN one-hot was {fun_mae:.3f})." if fun_mae is not None else ".")
        )
    else:
        verdict = (
            f"On the shared 59k set, GHS-BUILT-S MAE={bs_mae:.3f} beats AE W4 Ridge MAE={ae_mae:.3f} "
            f"(Δ={ae_mae - bs_mae:.3f})."
        )

    summary_payload = {
        "status": "completed",
        "started_at": json.loads((run_dir / "progress.json").read_text()).get("started_at"),
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "method": {
            "ghsl_product": "GHS-BUILT-S E2018 10m BUFRAC + GHS-BUILT-S NRES E2018 10m boolean",
            "sample": "Nearest BUILT-S / NRES value at AE pixel center (EPSG:4326→ESRI:54009)",
            "fraction_encoding": "bu=BUFRAC/100; NRES=1 → (0,bu); else (bu,0); empty=1-bu",
            "ae_model": "Week4 balanced Ridge OOF (outputs/week4_pixel_balanced_train/oof_predictions_full_gt.parquet)",
            "metrics": "MAE on frac_res / frac_nres / empty; primary mae_mean_res_nres",
        },
        "tiles": tiles,
        "shared_59k": shared.reset_index().to_dict(orient="records"),
        "verdict": verdict,
        "figures": [str(p) for p in figs],
        "notebook": None,
    }
    nb_path = build_notebook(run_dir, summary)
    summary_payload["notebook"] = str(nb_path)
    atomic_json(run_dir / "summary.json", summary_payload)
    write_progress(
        run_dir,
        status="completed",
        stage="done",
        completed=1,
        total=1,
        failed=0,
        last_progress_at=utc_now(),
        verdict=verdict,
    )
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, verdict)
    log(run_dir, f"DONE · notebook={nb_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # best-effort failure marker if run_dir known
        print(f"FATAL: {e}", flush=True)
        raise
