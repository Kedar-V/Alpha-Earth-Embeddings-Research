"""Rasterize building polygons onto the exact GHSL pixel grid."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.transform import xy as transform_xy


def rasterize_buildings_on_window(
    buildings: gpd.GeoDataFrame,
    label_col: str,
    transform,
    out_shape: tuple[int, int],
    cfg: dict[str, Any],
    value_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast area-proxy rasterization onto a GHSL window.

    Burns all footprints in two ADD passes (RES / NRES) instead of one
    rasterize-per-building loop.

    Returns:
      majority_code: int array (0=empty, 1=RES, 2=NRES, -1=ambiguous)
      building_fraction: float array in [0,1] (coverage proxy)
      building_id_index: int array (-1 if none); last-wins index into buildings
    """
    h, w = out_shape
    res = cfg["labels"]["residential"]
    nres = cfg["labels"]["non_residential"]
    min_frac = float(cfg["pixel_validation"]["minimum_building_fraction"])
    tie_eps = float(cfg["pixel_validation"].get("ambiguous_tie_eps", 0.05))

    labels = buildings[label_col].astype(str).to_numpy()
    geoms = list(buildings.geometry)

    res_shapes = [(g, 1) for g, lab in zip(geoms, labels) if g is not None and not g.is_empty and lab == res]
    nres_shapes = [(g, 1) for g, lab in zip(geoms, labels) if g is not None and not g.is_empty and lab == nres]
    id_shapes = [
        (g, i + 1)
        for i, g in enumerate(geoms)
        if g is not None and not g.is_empty and labels[i] in (res, nres)
    ]

    area_res = (
        features.rasterize(
            res_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint16,
            all_touched=False,
            merge_alg=MergeAlg.add,
        ).astype(np.float64)
        if res_shapes
        else np.zeros(out_shape, dtype=np.float64)
    )
    area_nres = (
        features.rasterize(
            nres_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint16,
            all_touched=False,
            merge_alg=MergeAlg.add,
        ).astype(np.float64)
        if nres_shapes
        else np.zeros(out_shape, dtype=np.float64)
    )
    id_burn = (
        features.rasterize(
            id_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.int32,
            all_touched=False,
        )
        if id_shapes
        else np.zeros(out_shape, dtype=np.int32)
    )

    # Coverage proxy: any overlapping burn counts as full-pixel occupancy for fraction
    covered = (area_res + area_nres) > 0
    building_fraction = covered.astype(np.float32)
    # If multiple buildings overlap, fraction still 1.0 under this proxy; min_frac gate is binary.
    majority = np.zeros(out_shape, dtype=np.int16)
    use = building_fraction >= min_frac
    majority[use & (area_res > area_nres)] = 1
    majority[use & (area_nres > area_res)] = 2
    total = area_res + area_nres
    tie = use & (np.abs(area_res - area_nres) <= tie_eps * np.maximum(total, 1e-9))
    majority[tie] = -1
    best_idx = id_burn - 1  # -1 where empty
    best_idx[best_idx < 0] = -1
    return majority, building_fraction, best_idx


def window_pixel_table(
    aoi_id: str,
    majority: np.ndarray,
    building_fraction: np.ndarray,
    best_idx: np.ndarray,
    window_row_off: int,
    window_col_off: int,
    transform,
    buildings: gpd.GeoDataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    res = cfg["labels"]["residential"]
    nres = cfg["labels"]["non_residential"]
    code_to_label = {1: res, 2: nres}
    rows, cols = np.where((majority == 1) | (majority == 2))
    if len(rows) == 0:
        return pd.DataFrame()
    xs, ys = transform_xy(transform, rows, cols)
    bid = []
    for i in best_idx[rows, cols]:
        if i < 0:
            bid.append(None)
        else:
            bid.append(buildings.iloc[int(i)]["building_id"])
    return pd.DataFrame(
        {
            "pixel_id": [
                f"{aoi_id}:{int(window_row_off) + int(r)}:{int(window_col_off) + int(c)}"
                for r, c in zip(rows, cols)
            ],
            "aoi_id": aoi_id,
            "row": rows + int(window_row_off),
            "col": cols + int(window_col_off),
            "x": xs,
            "y": ys,
            "gt_label": [code_to_label[int(majority[r, c])] for r, c in zip(rows, cols)],
            "gt_building_fraction": building_fraction[rows, cols],
            "gt_building_id": bid,
        }
    )
