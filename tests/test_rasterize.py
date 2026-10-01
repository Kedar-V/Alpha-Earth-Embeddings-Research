"""Rasterize smoke tests for GHSL pixel eval paint."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from validation.rasterize import rasterize_buildings_on_window


CFG = {
    "labels": {"residential": "RES", "non_residential": "NRES", "exclude": []},
    "pixel_validation": {"minimum_building_fraction": 0.5, "ambiguous_tie_eps": 0.05},
}


def test_rasterize_majority():
    # 10 m pixels; one RES building covering top-left 2x2 block
    transform = from_origin(0, 40, 10, 10)
    gdf = gpd.GeoDataFrame(
        {
            "building_id": ["a"],
            "label": ["RES"],
            "geometry": [box(0, 20, 20, 40)],
        },
        crs="ESRI:54009",
    )
    maj, frac, idx = rasterize_buildings_on_window(gdf, "label", transform, (4, 4), CFG)
    assert (maj == 1).sum() >= 1
    assert frac.max() >= 0.5
    assert (idx >= 0).any()


def test_center_in_polygon_not_all_touched():
    """Cell is burned only if its center falls inside the footprint."""
    # 2x2 grid of 10 m cells; origin upper-left at (0,20)
    transform = from_origin(0, 20, 10, 10)
    # Footprint covers only the lower-left quadrant of cell (0,0) — misses center (5,15)
    # Cell (0,0) center is at (5, 15). Polygon in lower-left of that cell: (0,10)-(4,10)-(4,14)-(0,14)
    # should NOT burn cell (0,0).
    gdf = gpd.GeoDataFrame(
        {
            "building_id": ["edge"],
            "label": ["RES"],
            "geometry": [box(0, 10, 4, 14)],
        },
        crs="ESRI:54009",
    )
    maj, frac, idx = rasterize_buildings_on_window(gdf, "label", transform, (2, 2), CFG)
    assert int((maj == 1).sum()) == 0
    assert float(frac.max()) == 0.0


def test_center_inside_burns_cell():
    transform = from_origin(0, 20, 10, 10)
    # Polygon covers center of cell (0,0) at (5,15)
    gdf = gpd.GeoDataFrame(
        {
            "building_id": ["c"],
            "label": ["NRES"],
            "geometry": [box(2, 12, 8, 18)],
        },
        crs="ESRI:54009",
    )
    maj, frac, idx = rasterize_buildings_on_window(gdf, "label", transform, (2, 2), CFG)
    assert maj[0, 0] == 2
    assert frac[0, 0] == 1.0


def test_res_vs_nres_majority_by_burn_count():
    transform = from_origin(0, 20, 10, 10)
    # Two footprints both covering same cell center; one RES one NRES → tie / ambiguous
    gdf = gpd.GeoDataFrame(
        {
            "building_id": ["r", "n"],
            "label": ["RES", "NRES"],
            "geometry": [box(2, 12, 8, 18), box(2.5, 12.5, 7.5, 17.5)],
        },
        crs="ESRI:54009",
    )
    maj, frac, idx = rasterize_buildings_on_window(gdf, "label", transform, (2, 2), CFG)
    assert frac[0, 0] == 1.0
    # equal burn counts → ambiguous (-1) under tie_eps
    assert maj[0, 0] == -1
