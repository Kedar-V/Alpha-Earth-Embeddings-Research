"""Spatial helpers: CRS, independence checks, AOI buffers."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union


GHSL_CRS = "ESRI:54009"


def to_ghsl_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS")
    out = gdf.to_crs(GHSL_CRS)
    return out


def aoi_bounds_wgs84(gdf: gpd.GeoDataFrame) -> dict[str, tuple[float, float, float, float]]:
    """Per-AOI WGS84 bounding boxes (minx, miny, maxx, maxy)."""
    g = gdf.to_crs("EPSG:4326")
    out = {}
    for aoi, sub in g.groupby("aoi_id"):
        minx, miny, maxx, maxy = sub.total_bounds
        out[str(aoi)] = (float(minx), float(miny), float(maxx), float(maxy))
    return out


def validate_spatial_independence(
    validation_gdf: gpd.GeoDataFrame,
    training_gdf: gpd.GeoDataFrame | None,
    buffer_km: float = 10.0,
) -> pd.DataFrame:
    """Check training footprints against buffered validation AOIs.

    Returns one row per validation AOI with nearest distance and leak counts.
    If training_gdf is None, returns pass=True with a note that the check was skipped.
    """
    rows = []
    val = validation_gdf.to_crs(GHSL_CRS)
    aois = sorted(val["aoi_id"].astype(str).unique())

    if training_gdf is None or len(training_gdf) == 0:
        for aoi in aois:
            rows.append(
                {
                    "aoi_id": aoi,
                    "nearest_training_distance_m": None,
                    "n_training_within_buffer": 0,
                    "pass": True,
                    "note": "training geometries not supplied; independence not verified",
                }
            )
        return pd.DataFrame(rows)

    train = training_gdf.to_crs(GHSL_CRS)
    train_pts = train.copy()
    train_pts["geometry"] = train_pts.geometry.centroid
    buf_m = float(buffer_km) * 1000.0

    for aoi in aois:
        sub = val[val["aoi_id"].astype(str) == aoi]
        hull = box(*sub.total_bounds)
        buffered = hull.buffer(buf_m)
        within = train_pts[train_pts.intersects(buffered)]
        if len(train_pts) == 0:
            nearest = None
        else:
            # Approximate nearest by centroid distance to AOI hull
            dists = train_pts.geometry.distance(hull)
            nearest = float(dists.min())
        n_within = int(len(within))
        rows.append(
            {
                "aoi_id": aoi,
                "nearest_training_distance_m": nearest,
                "n_training_within_buffer": n_within,
                "pass": n_within == 0,
                "note": "ok" if n_within == 0 else "spatial leakage within exclusion buffer",
            }
        )
    return pd.DataFrame(rows)


def held_out_claim_ok(independence: pd.DataFrame, refuse_on_leakage: bool) -> tuple[bool, str]:
    if independence.empty:
        return False, "no AOIs"
    if "note" in independence.columns and independence["note"].str.contains("not supplied", na=False).all():
        return True, "held-out claim deferred: training geometries not supplied (LORO folds assumed)"
    leaked = independence.loc[~independence["pass"]]
    if len(leaked) == 0:
        return True, "no training samples within exclusion buffers"
    msg = f"spatial leakage in AOIs: {leaked['aoi_id'].tolist()}"
    if refuse_on_leakage:
        return False, msg
    return True, "WARNING: " + msg
