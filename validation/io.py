"""Load config and tabular validation inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from shapely import wkt as shapely_wkt

from .schemas import BINARY_LABELS, GT_REQUIRED, PRED_REQUIRED


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else Path(__file__).with_name("config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against repo root
    for key in ("gt_buildings", "model_predictions", "training_buildings", "ghsl_tiles_dir", "ghsl_vrt", "output_dir"):
        val = cfg.get("paths", {}).get(key)
        if isinstance(val, str) and not Path(val).is_absolute():
            cfg["paths"][key] = str(ROOT / val)
    return cfg


def _as_geometry(series: pd.Series) -> gpd.GeoSeries:
    if isinstance(series.dtype, gpd.array.GeometryDtype):
        return gpd.GeoSeries(series)
    return gpd.GeoSeries(series.map(lambda g: shapely_wkt.loads(g) if isinstance(g, str) else g))


def load_gt_buildings(path: str | Path, cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    missing = [c for c in GT_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"GT missing columns: {missing}")
    gdf = gpd.GeoDataFrame(df.copy(), geometry=_as_geometry(df["geometry"]), crs="EPSG:4326")
    if gdf.crs is None:
        raise ValueError("GT geometries have no CRS")
    labels = set(gdf["label"].astype(str).unique())
    allowed = set(BINARY_LABELS) | set(cfg["labels"]["exclude"])
    bad = labels - allowed
    if bad:
        raise ValueError(f"Unexpected GT labels: {sorted(bad)}")
    if gdf["building_id"].isna().any():
        raise ValueError("GT has missing building_id")
    if gdf["label"].isna().any():
        raise ValueError("GT has missing labels")
    if gdf["building_id"].duplicated().any():
        raise ValueError("Duplicate building_id in GT")
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        still = ~gdf.geometry.is_valid
        if still.any():
            raise ValueError(f"{int(still.sum())} invalid geometries after fix")
    return gdf


def load_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    missing = [c for c in PRED_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    if df["building_id"].duplicated().any():
        raise ValueError("Duplicate building_id in predictions")
    bad = set(df["prediction"].astype(str).unique()) - {"RES", "NRES"}
    if bad:
        raise ValueError(f"Unexpected prediction labels: {sorted(bad)}")
    return df


def audit_gt(gdf: gpd.GeoDataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    labels = gdf["label"].astype(str)
    by_aoi = gdf.groupby("aoi_id").size().to_dict()
    counts = labels.value_counts().to_dict()
    binary = gdf[labels.isin(BINARY_LABELS)]
    n_bin = len(binary)
    audit = {
        "n_aois": int(gdf["aoi_id"].nunique()),
        "n_buildings": int(len(gdf)),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "n_binary": int(n_bin),
        "class_balance_res": float((binary["label"] == cfg["labels"]["residential"]).mean()) if n_bin else None,
        "buildings_per_aoi": {k: int(v) for k, v in by_aoi.items()},
        "n_duplicate_geometries": int(gdf.geometry.map(lambda g: g.wkb).duplicated().sum()),
        "crs": str(gdf.crs),
    }
    return audit
