"""10 m pixel-level evaluation on the exact GHSL grid (AOI-parallel)."""

from __future__ import annotations

import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .ghsl import map_ghsl_value, open_ghsl, sample_window
from .metrics import compute_metrics, confusion, coverage_and_selective
from .rasterize import rasterize_buildings_on_window, window_pixel_table
from .spatial import to_ghsl_crs


def _worker_count(cfg: dict[str, Any], n_tasks: int) -> int:
    configured = cfg.get("workers")
    n_cpu = os.cpu_count() or 1
    if configured is None or int(configured) <= 0:
        workers = n_cpu
    else:
        workers = int(configured)
    return max(1, min(workers, n_tasks, n_cpu))


def _eval_aoi_pixels(payload: tuple[str, bytes, dict[str, Any]]) -> pd.DataFrame:
    aoi, parquet_bytes, cfg = payload
    sub = gpd.read_parquet(io.BytesIO(parquet_bytes)).reset_index(drop=True)
    res = cfg["labels"]["residential"]
    nres = cfg["labels"]["non_residential"]
    mosaic = open_ghsl(cfg)
    try:
        minx, miny, maxx, maxy = sub.total_bounds
        data, window, transform = sample_window(mosaic, minx, miny, maxx, maxy, pad_px=2)

        maj, frac, best_idx = rasterize_buildings_on_window(sub, "label", transform, data.shape, cfg)
        gt_pixels = window_pixel_table(
            str(aoi), maj, frac, best_idx, int(window.row_off), int(window.col_off), transform, sub, cfg
        )
        if gt_pixels.empty:
            return pd.DataFrame()

        pred_gdf = sub.copy()
        pred_gdf["label"] = pred_gdf["prediction"].astype(str)
        maj_m, _, best_m = rasterize_buildings_on_window(pred_gdf, "label", transform, data.shape, cfg)

        local_rows = (gt_pixels["row"] - int(window.row_off)).astype(int).to_numpy()
        local_cols = (gt_pixels["col"] - int(window.col_off)).astype(int).to_numpy()
        ghsl_raw = data[local_rows, local_cols]
        ghsl_lab = [map_ghsl_value(v, cfg) for v in ghsl_raw]

        model_code = maj_m[local_rows, local_cols]
        code_to_lab = {1: res, 2: nres}
        model_lab = [code_to_lab.get(int(c)) for c in model_code]
        model_p_res, model_p_nres = [], []
        for bi in best_m[local_rows, local_cols]:
            if bi < 0:
                model_p_res.append(np.nan)
                model_p_nres.append(np.nan)
            else:
                row = sub.iloc[int(bi)]
                model_p_res.append(float(row["p_res"]) if "p_res" in row.index and pd.notna(row.get("p_res")) else np.nan)
                model_p_nres.append(float(row["p_nres"]) if "p_nres" in row.index and pd.notna(row.get("p_nres")) else np.nan)

        out = gt_pixels.copy()
        out["model_prediction"] = model_lab
        out["model_p_res"] = model_p_res
        out["model_p_nres"] = model_p_nres
        out["ghsl_raw"] = ghsl_raw.astype(int)
        out["ghsl_prediction"] = ghsl_lab
        out["ghsl_covered"] = out["ghsl_prediction"].isin([res, nres])
        out["model_covered"] = out["model_prediction"].isin([res, nres])
        return out[out["model_covered"]].copy()
    finally:
        mosaic.close()


def evaluate_pixels(
    gt: gpd.GeoDataFrame,
    preds: pd.DataFrame,
    ghsl_ds,  # unused; workers open their own mosaics
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    res = cfg["labels"]["residential"]
    nres = cfg["labels"]["non_residential"]
    exclude = set(cfg["labels"]["exclude"])

    gdf = to_ghsl_crs(gt.copy())
    gdf = gdf[~gdf["label"].astype(str).isin(exclude)].copy()
    gdf = gdf[gdf["label"].astype(str).isin([res, nres])].copy()
    gdf = gdf.merge(preds, on="building_id", how="inner")

    payloads = []
    for aoi, sub in gdf.groupby("aoi_id"):
        buf = io.BytesIO()
        sub.reset_index(drop=True).to_parquet(buf)
        payloads.append((str(aoi), buf.getvalue(), cfg))
    workers = _worker_count(cfg, len(payloads))
    print(f"  pixel eval: {len(payloads)} AOIs · workers={workers}/{os.cpu_count()}", flush=True)

    pixel_parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_aoi_pixels, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            aoi = futs[fut]
            part = fut.result()
            if len(part):
                pixel_parts.append(part)
            print(f"  done pixels {aoi} n={len(part)}", flush=True)

    if not pixel_parts:
        return pd.DataFrame(), {"level": "pixel", "n_pixels": 0, "error": "no evaluation pixels", "workers": workers}

    pixels = pd.concat(pixel_parts, ignore_index=True)
    func = pixels[pixels["ghsl_covered"]].copy()
    model_func = compute_metrics(func["gt_label"], func["model_prediction"], cfg)
    ghsl_func = compute_metrics(func["gt_label"], func["ghsl_prediction"], cfg)

    e2e_model = coverage_and_selective(
        pixels["gt_label"], pixels["model_prediction"], pixels["model_covered"].to_numpy(), cfg
    )
    e2e_ghsl = coverage_and_selective(
        pixels["gt_label"], pixels["ghsl_prediction"], pixels["ghsl_covered"].to_numpy(), cfg
    )

    per_aoi_rows = []
    for aoi, sub in func.groupby("aoi_id"):
        mm = compute_metrics(sub["gt_label"], sub["model_prediction"], cfg)
        gg = compute_metrics(sub["gt_label"], sub["ghsl_prediction"], cfg)
        per_aoi_rows.append({"aoi_id": aoi, "system": "model", **mm})
        per_aoi_rows.append({"aoi_id": aoi, "system": "ghsl", **gg})

    summary = {
        "level": "pixel",
        "n_pixels_total": int(len(pixels)),
        "n_functional": int(len(func)),
        "workers": workers,
        "functional": {"model": model_func, "ghsl": ghsl_func},
        "end_to_end": {"model": e2e_model, "ghsl": e2e_ghsl},
        "confusion_functional": {
            "model": confusion(func["gt_label"], func["model_prediction"]).to_dict(),
            "ghsl": confusion(func["gt_label"], func["ghsl_prediction"]).to_dict(),
        },
        "per_aoi": per_aoi_rows,
    }
    return pixels, summary
