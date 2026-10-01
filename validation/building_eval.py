"""Building-level GHSL vs model evaluation (AOI-parallel)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import io

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds

from .ghsl import GhslMosaic, open_ghsl
from .metrics import compute_metrics, confusion, coverage_and_selective
from .spatial import to_ghsl_crs


def _majority_from_values(vals: np.ndarray, cfg: dict[str, Any]) -> tuple[str, dict[str, float]]:
    g = cfg["ghsl"]
    res_v = int(g["residential_value"])
    nres_v = int(g["non_residential_value"])
    non_v = int(g["nonbuilt_value"])
    nod_v = int(g["nodata_value"])
    empty = {"n_pixels": 0.0, "frac_res": 0.0, "frac_nres": 0.0, "frac_nonbuilt": 0.0, "frac_nodata": 0.0}
    if vals.size == 0:
        return "NODATA", empty
    n_res = int(np.sum(vals == res_v))
    n_nres = int(np.sum(vals == nres_v))
    n_non = int(np.sum(vals == non_v))
    n_nod = int(np.sum(vals == nod_v))
    fracs = {
        "frac_res": n_res / vals.size,
        "frac_nres": n_nres / vals.size,
        "frac_nonbuilt": n_non / vals.size,
        "frac_nodata": n_nod / vals.size,
        "n_pixels": float(vals.size),
    }
    if n_res == 0 and n_nres == 0:
        return ("NONBUILT" if n_non > 0 else "NODATA"), fracs
    return ("RES" if n_res >= n_nres else "NRES"), fracs


def _ghsl_label_for_geom(geom, ds, cfg: dict[str, Any]) -> tuple[str, dict[str, float]]:
    empty = {"n_pixels": 0.0, "frac_res": 0.0, "frac_nres": 0.0, "frac_nonbuilt": 0.0, "frac_nodata": 0.0}
    if geom is None or geom.is_empty:
        return "NODATA", empty
    minx, miny, maxx, maxy = geom.bounds
    window = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
    window = Window(
        max(0, int(window.col_off) - 1),
        max(0, int(window.row_off) - 1),
        max(1, int(np.ceil(window.width)) + 2),
        max(1, int(np.ceil(window.height)) + 2),
    ).intersection(Window(0, 0, ds.width, ds.height))
    if window.width <= 0 or window.height <= 0:
        return "NODATA", empty
    data = ds.read(1, window=window)
    transform = ds.window_transform(window)
    mask = features.rasterize(
        [(geom, 1)],
        out_shape=data.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    vals = data[mask == 1]
    if vals.size == 0:
        c = geom.centroid
        try:
            rr, cc = rowcol(transform, c.x, c.y)
            if 0 <= rr < data.shape[0] and 0 <= cc < data.shape[1]:
                vals = np.array([data[rr, cc]])
        except Exception:
            vals = np.array([], dtype=data.dtype)
    return _majority_from_values(vals, cfg)


def _eval_aoi_buildings(payload: tuple[str, bytes, dict[str, Any]]) -> pd.DataFrame:
    """Worker: open local GHSL tiles and score one AOI's buildings."""
    aoi_id, parquet_bytes, cfg = payload
    import io

    sub = gpd.read_parquet(io.BytesIO(parquet_bytes))
    mosaic = open_ghsl(cfg)
    try:
        rows = []
        res = cfg["labels"]["residential"]
        nres = cfg["labels"]["non_residential"]
        for _, r in sub.iterrows():
            geom = r.geometry
            c = geom.centroid if geom is not None and not geom.is_empty else None
            ds_list = mosaic.datasets_for_bounds(c.x, c.y, c.x, c.y) if c is not None else []
            if not ds_list and geom is not None:
                minx, miny, maxx, maxy = geom.bounds
                ds_list = mosaic.datasets_for_bounds(minx, miny, maxx, maxy)
            if not ds_list:
                lab, fracs = "NODATA", {
                    "n_pixels": 0.0, "frac_res": 0.0, "frac_nres": 0.0, "frac_nonbuilt": 0.0, "frac_nodata": 0.0
                }
            else:
                lab, fracs = _ghsl_label_for_geom(geom, ds_list[0], cfg)
            rows.append(
                {
                    "building_id": r["building_id"],
                    "aoi_id": aoi_id,
                    "gt_label": str(r["label"]),
                    "model_prediction": str(r["prediction"]),
                    "model_p_res": float(r["p_res"]) if "p_res" in sub.columns and pd.notna(r.get("p_res")) else np.nan,
                    "model_p_nres": float(r["p_nres"]) if "p_nres" in sub.columns and pd.notna(r.get("p_nres")) else np.nan,
                    "ghsl_prediction": lab,
                    "ghsl_covered": lab in (res, nres),
                    "model_covered": True,
                    **fracs,
                }
            )
        return pd.DataFrame(rows)
    finally:
        mosaic.close()


def _worker_count(cfg: dict[str, Any], n_tasks: int) -> int:
    configured = cfg.get("workers")
    n_cpu = os.cpu_count() or 1
    if configured is None or int(configured) <= 0:
        workers = n_cpu
    else:
        workers = int(configured)
    return max(1, min(workers, n_tasks, n_cpu))


def evaluate_buildings(
    gt: gpd.GeoDataFrame,
    preds: pd.DataFrame,
    ghsl_ds: GhslMosaic | None,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    res = cfg["labels"]["residential"]
    nres = cfg["labels"]["non_residential"]
    exclude = set(cfg["labels"]["exclude"])

    gdf = to_ghsl_crs(gt.copy())
    gdf = gdf[~gdf["label"].astype(str).isin(exclude)].copy()
    gdf = gdf[gdf["label"].astype(str).isin([res, nres])].copy()

    merged = gdf.merge(preds, on="building_id", how="left", suffixes=("", "_pred"))
    missing_pred = int(merged["prediction"].isna().sum())
    if missing_pred:
        merged = merged[merged["prediction"].notna()].copy()

    payloads = []
    for aoi, sub in merged.groupby("aoi_id"):
        buf = io.BytesIO()
        sub.reset_index(drop=True).to_parquet(buf)
        payloads.append((str(aoi), buf.getvalue(), cfg))

    workers = _worker_count(cfg, len(payloads))
    print(f"  building eval: {len(payloads)} AOIs · workers={workers}/{os.cpu_count()}", flush=True)

    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_eval_aoi_buildings, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            aoi = futs[fut]
            part = fut.result()
            parts.append(part)
            print(f"  done buildings {aoi} n={len(part)}", flush=True)

    per_building = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    func = per_building[per_building["ghsl_covered"] & per_building["gt_label"].isin([res, nres])]
    model_func = compute_metrics(func["gt_label"], func["model_prediction"], cfg)
    ghsl_func = compute_metrics(func["gt_label"], func["ghsl_prediction"], cfg)

    e2e_model = coverage_and_selective(
        per_building["gt_label"], per_building["model_prediction"], per_building["model_covered"].to_numpy(), cfg
    )
    e2e_ghsl = coverage_and_selective(
        per_building["gt_label"], per_building["ghsl_prediction"], per_building["ghsl_covered"].to_numpy(), cfg
    )

    per_aoi_rows = []
    for aoi, sub in func.groupby("aoi_id"):
        mm = compute_metrics(sub["gt_label"], sub["model_prediction"], cfg)
        gg = compute_metrics(sub["gt_label"], sub["ghsl_prediction"], cfg)
        per_aoi_rows.append({"aoi_id": aoi, "system": "model", **mm})
        per_aoi_rows.append({"aoi_id": aoi, "system": "ghsl", **gg})

    summary = {
        "level": "building",
        "n_buildings_total": int(len(per_building)),
        "n_functional": int(len(func)),
        "n_missing_model_preds_dropped": missing_pred,
        "workers": workers,
        "functional": {"model": model_func, "ghsl": ghsl_func},
        "end_to_end": {"model": e2e_model, "ghsl": e2e_ghsl},
        "confusion_functional": {
            "model": confusion(func["gt_label"], func["model_prediction"]).to_dict() if len(func) else {},
            "ghsl": confusion(func["gt_label"], func["ghsl_prediction"]).to_dict() if len(func) else {},
        },
        "per_aoi": per_aoi_rows,
    }
    return per_building, summary
