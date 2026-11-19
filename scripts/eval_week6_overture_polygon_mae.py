#!/usr/bin/env python3
"""Week 6: fair AE vs GHS-BUILT-S fraction MAE on Overture polygons.

Both predictors are summarized independently into each Track-A Overture
footprint (EPSG:4326). Neither product is warped onto the other's lattice.

  GT(building)     = (1,0) if y_res==1 else (0,1)
  AE(building)     = mean week4 Ridge LORO OOF over building_cells with
                     inside_polygon=True (cells stay in per-metro AE UTM)
  GHSL-S(building) = mean soft BUILT-S fractions under the footprint mask
                     in native Mollweide (ESRI:54009)

Headline metric: mae_mean_res_nres = 0.5 * (MAE_res + MAE_nres) on the
shared-coverage cohort (AE covered ∩ GHSL covered).

Outputs (default):
  outputs/week6_overture_polygon_mae/<timestamp>/
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window, from_bounds
from shapely import wkt as shapely_wkt
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
AE_OOF = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
PIXEL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
PLS_OOF = ROOT / "outputs" / "week4_pls_loro" / "oof_predictions.parquet"
S_TILE_DIR = ROOT / "data" / "ghsl" / "built_s_tiles"
NRES_TILE_DIR = ROOT / "data" / "ghsl" / "built_s_nres_tiles"
GHSL_CRS = "ESRI:54009"
LONLAT_DECIMALS = 7
PREV_GHSL_CKPT = (
    ROOT / "outputs" / "week6_overture_polygon_mae" / "20261110_034500" / "checkpoints" / "ghsl_by_building.parquet"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


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


def lonlat_key(lon: np.ndarray, lat: np.ndarray, nd: int = LONLAT_DECIMALS) -> np.ndarray:
    """Return join keys as a plain ndarray (avoid pandas index alignment bugs)."""
    return np.round(lon, nd).astype(str) + "_" + np.round(lat, nd).astype(str)


def raw_to_soft(bu_raw: np.ndarray, nres_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BUILT-S raw → soft (frac_res, frac_nres, valid)."""
    bu = bu_raw.astype(np.float64)
    nres = nres_raw.astype(np.float64)
    valid = (bu != 255) & (nres != 255)
    bu_frac = np.clip(bu / 100.0, 0.0, 1.0)
    is_nres = nres == 1
    frac_res = np.where(is_nres, 0.0, bu_frac)
    frac_nres = np.where(is_nres, bu_frac, 0.0)
    return (
        np.where(valid, frac_res, np.nan),
        np.where(valid, frac_nres, np.nan),
        valid,
    )


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
            raise FileNotFoundError(f"missing BUILT-S tiles under {S_TILE_DIR} / {NRES_TILE_DIR}")
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

    def datasets_for_bounds(self, minx: float, miny: float, maxx: float, maxy: float) -> list[int]:
        hits = []
        for i, ds in enumerate(self.s_ds):
            b = ds.bounds
            if maxx < b.left or minx > b.right or maxy < b.bottom or miny > b.top:
                continue
            hits.append(i)
        return hits


def _soft_under_geom(geom, ds_s, ds_n) -> dict[str, float]:
    empty = {
        "ghsl_frac_res": np.nan,
        "ghsl_frac_nres": np.nan,
        "ghsl_n_pixels": 0.0,
        "ghsl_n_valid": 0.0,
        "ghsl_covered": False,
    }
    if geom is None or geom.is_empty:
        return empty
    minx, miny, maxx, maxy = geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=ds_s.transform)
    window = Window(
        max(0, int(window.col_off) - 1),
        max(0, int(window.row_off) - 1),
        max(1, int(np.ceil(window.width)) + 2),
        max(1, int(np.ceil(window.height)) + 2),
    ).intersection(Window(0, 0, ds_s.width, ds_s.height))
    if window.width <= 0 or window.height <= 0:
        return empty
    bu = ds_s.read(1, window=window)
    nres = ds_n.read(1, window=window)
    transform = ds_s.window_transform(window)
    mask = features.rasterize(
        [(geom, 1)],
        out_shape=bu.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    sel = mask == 1
    if not np.any(sel):
        # centroid fallback (tiny footprints)
        c = geom.centroid
        try:
            from rasterio.transform import rowcol

            rr, cc = rowcol(transform, c.x, c.y)
            if 0 <= rr < bu.shape[0] and 0 <= cc < bu.shape[1]:
                sel = np.zeros_like(mask, dtype=bool)
                sel[rr, cc] = True
            else:
                return empty
        except Exception:
            return empty
    fr, fn, valid = raw_to_soft(bu[sel], nres[sel])
    n_pix = int(sel.sum())
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {
            "ghsl_frac_res": np.nan,
            "ghsl_frac_nres": np.nan,
            "ghsl_n_pixels": float(n_pix),
            "ghsl_n_valid": 0.0,
            "ghsl_covered": False,
        }
    return {
        "ghsl_frac_res": float(np.nanmean(fr)),
        "ghsl_frac_nres": float(np.nanmean(fn)),
        "ghsl_n_pixels": float(n_pix),
        "ghsl_n_valid": float(n_valid),
        "ghsl_covered": True,
    }


def _ghsl_chunk_worker(payload: tuple[str, bytes]) -> pd.DataFrame:
    """Score one parquet chunk of buildings (WGS84 WKT) against BUILT-S."""
    import geopandas as gpd

    region, parquet_bytes = payload
    sub = pd.read_parquet(io.BytesIO(parquet_bytes))
    mosaic = DualMosaic.open()
    try:
        # Batch-project all footprints in the chunk to Mollweide once
        geoms = []
        ids = []
        for _, r in sub.iterrows():
            try:
                g = shapely_wkt.loads(str(r["wkt"]))
                if not g.is_valid:
                    g = g.buffer(0)
            except Exception:
                g = None
            geoms.append(g)
            ids.append(str(r["id"]))
        gdf = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:4326")
        valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
        gdf_m = gdf.loc[valid_mask].to_crs(GHSL_CRS)
        valid_ids = set(gdf_m["id"].tolist())
        geom_by_id = dict(zip(gdf_m["id"].tolist(), gdf_m.geometry.tolist()))

        rows = []
        for bid in ids:
            if bid not in valid_ids:
                rows.append(
                    {
                        "id": bid,
                        "region": region,
                        "ghsl_frac_res": np.nan,
                        "ghsl_frac_nres": np.nan,
                        "ghsl_n_pixels": 0.0,
                        "ghsl_n_valid": 0.0,
                        "ghsl_covered": False,
                        "ghsl_error": "invalid_geom",
                    }
                )
                continue
            g_m = geom_by_id[bid]
            minx, miny, maxx, maxy = g_m.bounds
            idxs = mosaic.datasets_for_bounds(minx, miny, maxx, maxy)
            if not idxs:
                c = g_m.centroid
                idxs = mosaic.datasets_for_bounds(c.x, c.y, c.x, c.y)
            if not idxs:
                fracs = {
                    "ghsl_frac_res": np.nan,
                    "ghsl_frac_nres": np.nan,
                    "ghsl_n_pixels": 0.0,
                    "ghsl_n_valid": 0.0,
                    "ghsl_covered": False,
                }
            else:
                fracs = _soft_under_geom(g_m, mosaic.s_ds[idxs[0]], mosaic.nres_ds[idxs[0]])
            rows.append({"id": bid, "region": region, "ghsl_error": "", **fracs})
        return pd.DataFrame(rows)
    finally:
        mosaic.close()


def stage_ae(run_dir: Path, buildings: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Aggregate AE OOF + soft week4 cell GT into each Overture building."""
    ckpt = run_dir / "checkpoints" / "ae_by_building.parquet"
    if ckpt.exists():
        prev = pd.read_parquet(ckpt)
        if {"soft_gt_frac_res", "soft_gt_frac_nres", "soft_gt_n_cells"}.issubset(prev.columns):
            log(run_dir, f"Reusing AE checkpoint {ckpt}")
            return prev
        log(run_dir, f"AE checkpoint missing soft GT columns — rebuilding {ckpt}")

    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="ae_aggregate",
        completed=0,
        total=4,
        failed=0,
        _t0=t0,
        started_at=utc_now(),
        output_location=str(run_dir),
        workers=workers,
    )
    log(run_dir, f"Loading AE OOF from {AE_OOF}")
    oof = pd.read_parquet(
        AE_OOF,
        columns=["pixel_id", "region", "pixel_lon", "pixel_lat", "pred_frac_res", "pred_frac_nres"],
    )
    oof["k"] = lonlat_key(oof["pixel_lon"].to_numpy(), oof["pixel_lat"].to_numpy())
    oof = oof.drop_duplicates("k", keep="first")
    write_progress(run_dir, stage="ae_aggregate", completed=1, total=4, last_progress_at=utc_now())

    log(run_dir, f"Loading soft pixel GT from {PIXEL_GT}")
    pix = pd.read_parquet(
        PIXEL_GT,
        columns=["pixel_lon", "pixel_lat", "frac_res_full", "frac_nres_full"],
    )
    pix["k"] = lonlat_key(pix["pixel_lon"].to_numpy(), pix["pixel_lat"].to_numpy())
    pix = pix.drop_duplicates("k", keep="first")
    write_progress(run_dir, stage="ae_aggregate", completed=2, total=4, last_progress_at=utc_now())

    want_ids = set(buildings["id"].astype(str))
    log(run_dir, f"Loading inside_polygon cells from {CELLS} (filter {len(want_ids):,} buildings)")
    cells = pd.read_parquet(
        CELLS,
        columns=["id", "pixel_lon", "pixel_lat", "inside_polygon", "crs"],
    )
    cells = cells[cells["inside_polygon"]].copy()
    cells["id"] = cells["id"].astype(str)
    cells = cells[cells["id"].isin(want_ids)].copy()
    cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    write_progress(run_dir, stage="ae_aggregate", completed=3, total=4, last_progress_at=utc_now())

    joined = cells.merge(oof[["k", "pred_frac_res", "pred_frac_nres"]], on="k", how="left")
    joined = joined.merge(
        pix[["k", "frac_res_full", "frac_nres_full"]].rename(
            columns={"frac_res_full": "soft_gt_frac_res", "frac_nres_full": "soft_gt_frac_nres"}
        ),
        on="k",
        how="left",
    )
    bmeta = buildings.set_index("id")[["region", "crs", "y_res"]]
    joined["region"] = joined["id"].map(bmeta["region"])
    joined["crs"] = joined["id"].map(bmeta["crs"])

    agg = (
        joined.groupby("id", sort=False)
        .agg(
            ae_frac_res=("pred_frac_res", "mean"),
            ae_frac_nres=("pred_frac_nres", "mean"),
            soft_gt_frac_res=("soft_gt_frac_res", "mean"),
            soft_gt_frac_nres=("soft_gt_frac_nres", "mean"),
            ae_n_cells=("k", "size"),
            ae_n_matched=("pred_frac_res", "count"),
            soft_gt_n_cells=("soft_gt_frac_res", "count"),
            region=("region", "first"),
            crs=("crs", "first"),
        )
        .reset_index()
    )
    agg = agg.merge(
        buildings[["id", "y_res", "region", "crs"]].rename(
            columns={"region": "region_b", "crs": "crs_b"}
        ),
        on="id",
        how="right",
    )
    agg["region"] = agg["region"].fillna(agg["region_b"])
    agg["crs"] = agg["crs"].fillna(agg["crs_b"])
    agg = agg.drop(columns=["region_b", "crs_b"])
    agg["ae_n_cells"] = agg["ae_n_cells"].fillna(0).astype(int)
    agg["ae_n_matched"] = agg["ae_n_matched"].fillna(0).astype(int)
    agg["soft_gt_n_cells"] = agg["soft_gt_n_cells"].fillna(0).astype(int)
    agg["ae_covered"] = agg["ae_n_matched"] > 0
    agg["soft_gt_covered"] = agg["soft_gt_n_cells"] > 0

    agg.to_parquet(ckpt, index=False)
    log(
        run_dir,
        f"AE done: {int(agg.ae_covered.sum()):,}/{len(agg):,} AE-covered · "
        f"{int(agg.soft_gt_covered.sum()):,} soft-GT-covered · "
        f"matched cells mean={agg.loc[agg.ae_covered, 'ae_n_matched'].mean():.1f} · "
        f"{time.time() - t0:.1f}s → {ckpt}",
    )
    write_progress(run_dir, stage="ae_aggregate", completed=4, total=4, last_progress_at=utc_now())
    return agg


def stage_ghsl(
    run_dir: Path,
    buildings: pd.DataFrame,
    workers: int,
    chunk_size: int,
    reuse_ghsl_from: Path | None = None,
) -> pd.DataFrame:
    ckpt = run_dir / "checkpoints" / "ghsl_by_building.parquet"
    if ckpt.exists():
        log(run_dir, f"Reusing GHSL checkpoint {ckpt}")
        return pd.read_parquet(ckpt)

    src = reuse_ghsl_from or (PREV_GHSL_CKPT if PREV_GHSL_CKPT.exists() else None)
    if src is not None and Path(src).exists():
        import shutil

        shutil.copy2(src, ckpt)
        log(run_dir, f"Copied GHSL checkpoint from {src}")
        return pd.read_parquet(ckpt)

    parts_dir = run_dir / "checkpoints" / "ghsl_parts"
    chunks_dir = run_dir / "checkpoints" / "ghsl_chunks"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "failures").mkdir(parents=True, exist_ok=True)

    # Enumerate all chunk tags; skip those already on disk
    payloads: list[tuple[str, bytes, str]] = []
    all_tags: list[str] = []
    for region, sub in buildings.groupby("region", sort=True):
        sub = sub[["id", "wkt", "y_res"]].reset_index(drop=True)
        for start in range(0, len(sub), chunk_size):
            tag = f"{region}__{start}"
            all_tags.append(tag)
            chunk_path = chunks_dir / f"{tag}.parquet"
            if chunk_path.exists():
                continue
            chunk = sub.iloc[start : start + chunk_size]
            buf = io.BytesIO()
            chunk.to_parquet(buf)
            payloads.append((str(region), buf.getvalue(), tag))

    already = len(all_tags) - len(payloads)
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="ghsl_zonal",
        completed=already,
        total=max(1, len(all_tags)),
        failed=0,
        _t0=t0,
        output_location=str(run_dir),
        workers=workers,
        n_buildings=len(buildings),
        last_progress_at=utc_now(),
    )
    log(
        run_dir,
        f"GHSL zonal: {len(payloads)} chunks remaining / {len(all_tags)} total "
        f"(resume skipped={already}) · workers={workers} · chunk_size={chunk_size}",
    )

    failed = 0
    completed = already
    if payloads:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_ghsl_chunk_worker, (region, blob)): (region, tag)
                for region, blob, tag in payloads
            }
            for fut in as_completed(futs):
                region, tag = futs[fut]
                try:
                    part = fut.result()
                    part.to_parquet(chunks_dir / f"{tag}.parquet", index=False)
                    completed += 1
                    write_progress(
                        run_dir,
                        stage="ghsl_zonal",
                        completed=completed,
                        total=len(all_tags),
                        failed=failed,
                        last_progress_at=utc_now(),
                        last_chunk=tag,
                    )
                    if completed % 10 == 0 or completed == len(all_tags):
                        log(run_dir, f"GHSL chunks {completed}/{len(all_tags)} last={tag}")
                except Exception as e:
                    failed += 1
                    completed += 1
                    rec = {"tag": tag, "region": region, "error": str(e), "at": utc_now()}
                    with (run_dir / "failures" / "ghsl_chunks.jsonl").open("a") as ff:
                        ff.write(json.dumps(rec) + "\n")
                    log(run_dir, f"FAIL chunk {tag}: {e}")
                    write_progress(
                        run_dir,
                        stage="ghsl_zonal",
                        completed=completed,
                        total=len(all_tags),
                        failed=failed,
                        last_progress_at=utc_now(),
                    )

    missing = [t for t in all_tags if not (chunks_dir / f"{t}.parquet").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} GHSL chunks missing (e.g. {missing[:5]}); see failures/")

    # Assemble per-region + global
    frames = [pd.read_parquet(chunks_dir / f"{t}.parquet") for t in all_tags]
    ghsl = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="last")
    for region, sub in ghsl.groupby("region", sort=True):
        sub.to_parquet(parts_dir / f"{region}.parquet", index=False)
    ghsl.to_parquet(ckpt, index=False)
    log(
        run_dir,
        f"GHSL done: {int(ghsl.ghsl_covered.sum()):,}/{len(ghsl):,} covered · "
        f"{time.time() - t0:.1f}s → {ckpt}",
    )
    return ghsl


def mae_block(y_res: np.ndarray, y_nres: np.ndarray, p_res: np.ndarray, p_nres: np.ndarray) -> dict[str, float]:
    m = np.isfinite(p_res) & np.isfinite(p_nres) & np.isfinite(y_res) & np.isfinite(y_nres)
    if m.sum() == 0:
        return {
            "n": 0,
            "mae_frac_res": float("nan"),
            "mae_frac_nres": float("nan"),
            "mae_mean_res_nres": float("nan"),
            "r2_frac_res": float("nan"),
            "r2_frac_nres": float("nan"),
        }
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(y_res[m], p_res[m])),
        "mae_frac_nres": float(mean_absolute_error(y_nres[m], p_nres[m])),
        "mae_mean_res_nres": float(
            0.5
            * (
                mean_absolute_error(y_res[m], p_res[m])
                + mean_absolute_error(y_nres[m], p_nres[m])
            )
        ),
        "r2_frac_res": float(r2_score(y_res[m], p_res[m])) if m.sum() > 1 else float("nan"),
        "r2_frac_nres": float(r2_score(y_nres[m], p_nres[m])) if m.sum() > 1 else float("nan"),
    }


def _score_gt_mode(
    m: pd.DataFrame,
    gt_res_col: str,
    gt_nres_col: str,
    shared_mask: pd.Series,
    gt_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, float], pd.DataFrame]:
    """MAE tables + win-rate for one GT definition on a shared mask."""
    rows: list[dict[str, Any]] = []
    for cohort_name, mask in [
        (f"{gt_mode}::shared", shared_mask),
        (f"{gt_mode}::ae_only", m["ae_covered"]),
        (f"{gt_mode}::ghsl_only", m["ghsl_covered"]),
        (f"{gt_mode}::all_buildings", pd.Series(True, index=m.index)),
    ]:
        sub = m.loc[mask]
        for model, pr, pn in [
            ("ae_w4_ridge", "ae_frac_res", "ae_frac_nres"),
            ("ghsl_built_s", "ghsl_frac_res", "ghsl_frac_nres"),
        ]:
            metrics = mae_block(
                sub[gt_res_col].to_numpy(dtype=float),
                sub[gt_nres_col].to_numpy(dtype=float),
                sub[pr].to_numpy(dtype=float),
                sub[pn].to_numpy(dtype=float),
            )
            rows.append({"gt_mode": gt_mode, "cohort": cohort_name, "model": model, **metrics})

        if cohort_name == f"{gt_mode}::shared":
            for region, rsub in sub.groupby("region", sort=True):
                for model, pr, pn in [
                    ("ae_w4_ridge", "ae_frac_res", "ae_frac_nres"),
                    ("ghsl_built_s", "ghsl_frac_res", "ghsl_frac_nres"),
                ]:
                    metrics = mae_block(
                        rsub[gt_res_col].to_numpy(dtype=float),
                        rsub[gt_nres_col].to_numpy(dtype=float),
                        rsub[pr].to_numpy(dtype=float),
                        rsub[pn].to_numpy(dtype=float),
                    )
                    rows.append(
                        {
                            "gt_mode": gt_mode,
                            "cohort": f"{gt_mode}::shared::{region}",
                            "model": model,
                            **metrics,
                        }
                    )

    shared = m.loc[shared_mask].copy()
    err_cols = pd.DataFrame({"id": shared["id"]})
    if len(shared):
        ae_err = 0.5 * (
            np.abs(shared["ae_frac_res"] - shared[gt_res_col])
            + np.abs(shared["ae_frac_nres"] - shared[gt_nres_col])
        )
        gh_err = 0.5 * (
            np.abs(shared["ghsl_frac_res"] - shared[gt_res_col])
            + np.abs(shared["ghsl_frac_nres"] - shared[gt_nres_col])
        )
        err_cols[f"ae_err_{gt_mode}"] = ae_err.to_numpy()
        err_cols[f"ghsl_err_{gt_mode}"] = gh_err.to_numpy()
        winner = np.where(
            ae_err < gh_err - 1e-12,
            "ae",
            np.where(gh_err < ae_err - 1e-12, "ghsl", "tie"),
        )
        err_cols[f"winner_{gt_mode}"] = winner
        win = pd.Series(winner).value_counts(normalize=True).to_dict()
    else:
        win = {}
    return rows, win, err_cols


def stage_score(run_dir: Path, buildings: pd.DataFrame, ae: pd.DataFrame, ghsl: pd.DataFrame) -> pd.DataFrame:
    write_progress(run_dir, status="running", stage="score", last_progress_at=utc_now())
    base = buildings[["id", "region", "crs", "y_res", "label", "class", "lon", "lat"]].copy()
    base["hard_gt_frac_res"] = (base["y_res"] == 1).astype(float)
    base["hard_gt_frac_nres"] = (base["y_res"] == 0).astype(float)

    ae_cols = [
        "id",
        "ae_frac_res",
        "ae_frac_nres",
        "ae_n_cells",
        "ae_n_matched",
        "ae_covered",
        "soft_gt_frac_res",
        "soft_gt_frac_nres",
        "soft_gt_n_cells",
        "soft_gt_covered",
    ]
    m = base.merge(ae[ae_cols], on="id", how="left").merge(
        ghsl[["id", "ghsl_frac_res", "ghsl_frac_nres", "ghsl_n_pixels", "ghsl_n_valid", "ghsl_covered"]],
        on="id",
        how="left",
    )
    m["ae_covered"] = m["ae_covered"].fillna(False).astype(bool)
    m["ghsl_covered"] = m["ghsl_covered"].fillna(False).astype(bool)
    m["soft_gt_covered"] = m["soft_gt_covered"].fillna(False).astype(bool)
    m["shared_hard"] = m["ae_covered"] & m["ghsl_covered"]
    m["shared_soft"] = m["ae_covered"] & m["ghsl_covered"] & m["soft_gt_covered"]

    out_path = run_dir / "artifacts" / "per_building.parquet"
    m.to_parquet(out_path, index=False)
    log(
        run_dir,
        f"Wrote per-building table n={len(m):,} "
        f"shared_hard={int(m.shared_hard.sum()):,} shared_soft={int(m.shared_soft.sum()):,} → {out_path}",
    )

    hard_rows, win_hard, err_hard = _score_gt_mode(
        m, "hard_gt_frac_res", "hard_gt_frac_nres", m["shared_hard"], "hard"
    )
    soft_rows, win_soft, err_soft = _score_gt_mode(
        m, "soft_gt_frac_res", "soft_gt_frac_nres", m["shared_soft"], "soft"
    )
    rows = hard_rows + soft_rows
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "artifacts" / "mae_summary.csv", index=False)
    summary.to_parquet(run_dir / "artifacts" / "mae_summary.parquet", index=False)

    err = m.loc[m["shared_hard"], ["id", "region", "y_res"]].merge(err_hard, on="id", how="left")
    err = err.merge(err_soft, on="id", how="left")
    err.to_parquet(run_dir / "artifacts" / "per_building_errors.parquet", index=False)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for gt_mode, title_suffix in [("hard", "hard y_res GT"), ("soft", "soft cell-frac GT")]:
        shared_sum = summary[summary["cohort"] == f"{gt_mode}::shared"].sort_values("mae_mean_res_nres")
        if not len(shared_sum):
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        colors = ["#1d4ed8" if m_ == "ae_w4_ridge" else "#ea580c" for m_ in shared_sum["model"]]
        ax.barh(shared_sum["model"], shared_sum["mae_mean_res_nres"], color=colors)
        for y, v in enumerate(shared_sum["mae_mean_res_nres"]):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlabel("MAE mean(res, nres)")
        ax.set_title(f"Overture-polygon · {title_suffix} · n={int(shared_sum['n'].iloc[0]):,}")
        ax.set_xlim(0, max(0.55, float(shared_sum["mae_mean_res_nres"].max()) * 1.25))
        fig.tight_layout()
        fig.savefig(fig_dir / f"mae_shared_headline_{gt_mode}.png", dpi=140)
        plt.close(fig)

        piv = (
            summary[summary["cohort"].str.startswith(f"{gt_mode}::shared::")]
            .assign(region=lambda d: d["cohort"].str.replace(f"{gt_mode}::shared::", "", regex=False))
            .pivot_table(index="region", columns="model", values="mae_mean_res_nres")
        )
        if len(piv):
            fig, ax = plt.subplots(figsize=(8, 5))
            y = np.arange(len(piv))
            h = 0.35
            ax.barh(y - h / 2, piv.get("ae_w4_ridge", 0), height=h, color="#1d4ed8", label="AE Ridge OOF")
            ax.barh(y + h / 2, piv.get("ghsl_built_s", 0), height=h, color="#ea580c", label="GHS-BUILT-S")
            ax.set_yticks(y)
            ax.set_yticklabels(piv.index)
            ax.set_xlabel("MAE mean(res, nres)")
            ax.set_title(f"Per-metro · {title_suffix}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"mae_by_region_{gt_mode}.png", dpi=140)
            plt.close(fig)

    # Side-by-side hard vs soft headline
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, gt_mode, title in zip(axes, ["hard", "soft"], ["Hard y_res GT", "Soft cell-frac GT"]):
        s = summary[summary["cohort"] == f"{gt_mode}::shared"].sort_values("mae_mean_res_nres")
        if not len(s):
            continue
        colors = ["#1d4ed8" if m_ == "ae_w4_ridge" else "#ea580c" for m_ in s["model"]]
        ax.barh(s["model"], s["mae_mean_res_nres"], color=colors)
        for y, v in enumerate(s["mae_mean_res_nres"]):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("MAE mean(res, nres)")
        ax.set_xlim(0, max(0.55, float(s["mae_mean_res_nres"].max()) * 1.3))
    fig.suptitle("Fair Overture-polygon compare · hard vs soft GT", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_hard_vs_soft.png", dpi=140)
    plt.close(fig)

    def _pack(shared_cohort: str) -> dict[str, Any]:
        sr = summary[summary["cohort"] == shared_cohort].set_index("model")
        if not len(sr):
            return {}
        return {
            model: {
                "n": int(sr.loc[model, "n"]),
                "mae_frac_res": float(sr.loc[model, "mae_frac_res"]),
                "mae_frac_nres": float(sr.loc[model, "mae_frac_nres"]),
                "mae_mean_res_nres": float(sr.loc[model, "mae_mean_res_nres"]),
            }
            for model in sr.index
        }

    coverage = {
        "n_buildings": int(len(m)),
        "n_ae_covered": int(m.ae_covered.sum()),
        "n_ghsl_covered": int(m.ghsl_covered.sum()),
        "n_soft_gt_covered": int(m.soft_gt_covered.sum()),
        "n_shared_hard": int(m.shared_hard.sum()),
        "n_shared_soft": int(m.shared_soft.sum()),
        "win_rate_hard": win_hard,
        "win_rate_soft": win_soft,
    }
    atomic_json(run_dir / "artifacts" / "coverage.json", coverage)

    result = {
        "status": "completed",
        "protocol": "overture_polygon_zonal_v2_hard_and_soft",
        "gt_hard": "y_res → (1,0)/(0,1)",
        "gt_soft": "mean week4 frac_res_full/frac_nres_full over inside_polygon AE cells",
        "ae": "mean week4 Ridge LORO OOF on inside_polygon AE cells (per-metro UTM)",
        "ghsl": "mean GHS-BUILT-S soft fractions under footprint mask (ESRI:54009)",
        "metric": "mae_mean_res_nres = 0.5*(MAE_res+MAE_nres)",
        "coverage": coverage,
        "shared_mae_hard": _pack("hard::shared"),
        "shared_mae_soft": _pack("soft::shared"),
        "artifacts": {
            "per_building": str(out_path),
            "mae_summary": str(run_dir / "artifacts" / "mae_summary.csv"),
            "figures": str(fig_dir),
        },
        "finished_at": utc_now(),
    }
    atomic_json(run_dir / "summary.json", result)
    log(run_dir, "SUMMARY hard " + json.dumps(result["shared_mae_hard"]))
    log(run_dir, "SUMMARY soft " + json.dumps(result["shared_mae_soft"]))
    return summary


def balance_buildings_per_region(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Section-2 style 1:1 per metro by undersampling majority y_res class."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for region, sub in df.groupby("region", sort=True):
        res = sub.loc[sub["y_res"] == 1]
        nres = sub.loc[sub["y_res"] == 0]
        n = min(len(res), len(nres))
        if n == 0:
            continue
        if len(res) > n:
            res = res.loc[rng.choice(res.index.to_numpy(), size=n, replace=False)]
        if len(nres) > n:
            nres = nres.loc[rng.choice(nres.index.to_numpy(), size=n, replace=False)]
        bal = pd.concat([res, nres], ignore_index=True)
        bal["balance_n_per_class"] = n
        parts.append(bal)
    if not parts:
        raise RuntimeError("No balanced regions produced")
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_notebook(run_dir: Path, summary: pd.DataFrame) -> Path:
    nb_dir = ROOT / "notebooks" / "week6"
    nb_dir.mkdir(parents=True, exist_ok=True)
    rel_run = os.path.relpath(run_dir, nb_dir)

    def _table(cohort: str) -> str:
        shared = summary[summary["cohort"] == cohort].sort_values("mae_mean_res_nres")
        if not len(shared):
            return "_(no rows)_"
        return "\n".join(
            f"| `{r.model}` | {int(r.n):,} | {r.mae_frac_res:.4f} | {r.mae_frac_nres:.4f} | **{r.mae_mean_res_nres:.4f}** |"
            for r in shared.itertuples()
        )

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Week 6 — AE vs GHS-BUILT-S on Overture polygons\n",
                "\n",
                "Fair fraction MAE on **per-metro 1:1 balanced** Track-A buildings (Section-2 style). "
                "Both predictors aggregate independently into each footprint. "
                "Building GT = mean week-4 cell fractions inside the polygon.\n",
                "\n",
                f"Run: `{run_dir}`\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Protocol\n",
                "\n",
                "```text\n",
                "Balance          = per metro, undersample majority y_res to minority (seed 42)\n",
                "Building GT      = mean week4 frac_res_full / frac_nres_full on inside_polygon cells\n",
                "AE               = mean Ridge LORO OOF on inside_polygon cells\n",
                "GHSL-S           = mean BUILT-S fractions under footprint mask\n",
                "```\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Building GT (shared)\n",
                "\n",
                "| Model | n | MAE res | MAE nres | MAE mean |\n",
                "|---|---:|---:|---:|---:|\n",
                f"{_table('soft::shared')}\n",
                "\n",
                f"![building headline]({rel_run}/figures/mae_shared_headline_soft.png)\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import json\n",
                "import pandas as pd\n",
                f"RUN = Path({str(run_dir)!r})\n",
                "print(json.dumps(json.loads((RUN/'summary.json').read_text()), indent=2))\n",
                "display(pd.read_csv(RUN/'artifacts'/'mae_summary.csv'))\n",
            ],
        },
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = nb_dir / "01_ae_vs_ghsl_overture_polygons.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def write_briefing() -> Path:
    nb_dir = ROOT / "notebooks" / "week6"
    nb_dir.mkdir(parents=True, exist_ok=True)
    out = nb_dir / "00_briefing.ipynb"
    if out.exists():
        return out
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Week 6\n",
                    "\n",
                    "Fair comparison of **AlphaEarth fraction heads** vs **GHS-BUILT-S** "
                    "using **Overture building polygons** as the common spatial support.\n",
                    "\n",
                    "Motivation: sampling GHSL at AE cell centers forces GHSL onto AE's lattice "
                    "(~4 m origin shift). Here both products are summarized independently into "
                    "each Overture footprint — AE via per-metro UTM cells, GHSL via Mollweide — "
                    "then scored with MAE on `(frac_res, frac_nres)`.\n",
                    "\n",
                    "| Piece | Choice |\n",
                    "|---|---|\n",
                    "| Reference | Overture Track-A footprints (`resi_def_v1`) |\n",
                    "| GT | `y_res` → `(1,0)` / `(0,1)` |\n",
                    "| AE | Mean week4 Ridge LORO OOF on `inside_polygon` cells |\n",
                    "| GHSL | GHS-BUILT-S soft fractions under polygon mask |\n",
                    "| Metric | `mae_mean_res_nres` on shared coverage |\n",
                    "\n",
                    "Results notebook: [`01_ae_vs_ghsl_overture_polygons.ipynb`]"
                    "(01_ae_vs_ghsl_overture_polygons.ipynb).\n",
                ],
            }
        ],
    }
    out.write_text(json.dumps(nb, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPUs")
    ap.add_argument("--chunk-size", type=int, default=2000, help="buildings per GHSL worker task")
    ap.add_argument("--max-buildings", type=int, default=0, help="debug subsample (0=all)")
    ap.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-metro 1:1 y_res balance like Section 2 (default: on)",
    )
    ap.add_argument("--balance-seed", type=int, default=42)
    ap.add_argument(
        "--reuse-ghsl-from",
        type=Path,
        default=None,
        help="optional path to prior ghsl_by_building.parquet",
    )
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "week6_overture_polygon_mae" / ts)
    run_dir = Path(run_dir)
    for d in ["checkpoints", "artifacts", "figures", "failures", "logs"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    env = {
        "python": sys.executable,
        "cpus": os.cpu_count(),
        "workers": workers,
        "chunk_size": args.chunk_size,
        "balanced": args.balanced,
        "balance_seed": args.balance_seed,
        "buildings": str(BUILDINGS),
        "cells": str(CELLS),
        "ae_oof": str(AE_OOF),
        "pixel_gt": str(PIXEL_GT),
        "s_tiles": str(S_TILE_DIR),
        "nres_tiles": str(NRES_TILE_DIR),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }
    (run_dir / "environment.txt").write_text("\n".join(f"{k}={v}" for k, v in env.items()) + "\n")
    atomic_json(
        run_dir / "manifest.json",
        {
            "protocol": "overture_polygon_zonal_v2_hard_and_soft_balanced",
            "created_at": utc_now(),
            **env,
        },
    )

    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="load_buildings",
        completed=0,
        total=1,
        failed=0,
        started_at=utc_now(),
        _t0=t0,
        output_location=str(run_dir),
        workers=workers,
        balanced=args.balanced,
    )
    log(
        run_dir,
        f"Week6 Overture-polygon MAE · run_dir={run_dir} · workers={workers} · "
        f"balanced={args.balanced} soft+hard GT",
    )
    write_briefing()

    buildings = pd.read_parquet(BUILDINGS)
    buildings["id"] = buildings["id"].astype(str)
    n_raw = len(buildings)
    if args.balanced:
        before = buildings.groupby(["region", "y_res"]).size().to_dict()
        buildings = balance_buildings_per_region(buildings, seed=args.balance_seed)
        log(
            run_dir,
            f"Balanced buildings {n_raw:,} → {len(buildings):,} "
            f"(1:1 y_res per metro, seed={args.balance_seed})",
        )
        bal_path = run_dir / "artifacts" / "balanced_buildings.parquet"
        buildings.to_parquet(bal_path, index=False)
        atomic_json(
            run_dir / "artifacts" / "balance_counts.json",
            {
                "seed": args.balance_seed,
                "n_before": n_raw,
                "n_after": len(buildings),
                "before_region_yres": {f"{k[0]}|{k[1]}": int(v) for k, v in before.items()},
                "after_region_yres": {
                    f"{r}|{y}": int(n)
                    for (r, y), n in buildings.groupby(["region", "y_res"]).size().items()
                },
            },
        )
    if args.max_buildings and args.max_buildings > 0:
        buildings = buildings.sample(n=min(args.max_buildings, len(buildings)), random_state=42)
        log(run_dir, f"Subsampled to {len(buildings):,} buildings")
    log(
        run_dir,
        f"Buildings {len(buildings):,} · RES={int((buildings.y_res==1).sum()):,} "
        f"NRES={int((buildings.y_res==0).sum()):,} · "
        f"regions={sorted(buildings.region.unique().tolist())}",
    )

    ae = stage_ae(run_dir, buildings, workers=workers)
    ghsl = stage_ghsl(
        run_dir,
        buildings,
        workers=workers,
        chunk_size=args.chunk_size,
        reuse_ghsl_from=args.reuse_ghsl_from,
    )
    # If GHSL was copied from full run, restrict to this cohort ids
    want = set(buildings["id"])
    if len(ghsl) > len(want) * 1.05:
        ghsl = ghsl[ghsl["id"].astype(str).isin(want)].copy()
        log(run_dir, f"Filtered GHSL checkpoint to cohort n={len(ghsl):,}")
    summary = stage_score(run_dir, buildings, ae, ghsl)
    nb = write_notebook(run_dir, summary)
    log(run_dir, f"Notebook → {nb}")

    write_progress(
        run_dir,
        status="completed",
        stage="done",
        completed=1,
        total=1,
        failed=0,
        last_progress_at=utc_now(),
        elapsed_seconds=round(time.time() - t0, 1),
        percent_complete=100.0,
    )
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
