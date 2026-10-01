#!/usr/bin/env python3
"""Build polygon as-is + mean embedding tables for notebook 03.

Writes:
  outputs/building_gt/building_polygons.parquet
  outputs/building_gt/building_gt_pixels.parquet
  outputs/building_gt/building_gt_mean.parquet
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import duckdb
import ee
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping as shapely_mapping

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "building_gt"
RELEASE = "2026-08-19.0"
BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
EMBED_YEAR = 2024
EE_POLY_BATCH = 20
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-kedarvaidya0504")
BANDS = [f"A{i:02d}" for i in range(64)]
REBUILD = os.environ.get("REBUILD_POLYGON_EMBEDS", "0") == "1"

US_REGIONS = {
    "Boston_NE": (-71.20, 42.28, -70.95, 42.42),
    "NYC_NE": (-74.05, 40.68, -73.90, 40.82),
    "Chicago_MW": (-87.75, 41.80, -87.55, 41.95),
    "Atlanta_SE": (-84.45, 33.70, -84.30, 33.85),
    "Houston_S": (-95.50, 29.70, -95.25, 29.85),
    "Denver_W": (-105.05, 39.68, -104.90, 39.80),
    "Phoenix_SW": (-112.15, 33.40, -111.95, 33.55),
    "Seattle_NW": (-122.40, 47.55, -122.25, 47.68),
    "LA_W": (-118.35, 34.00, -118.20, 34.12),
}


def init_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print("Earth Engine initialized · project =", project, flush=True)


def annual_embedding_image(year: int) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
    )


def fetch_polygons_for_labels(label_df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    part_dir = OUT_DIR / "_polygon_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, bbox in US_REGIONS.items():
        part = part_dir / f"{name}.parquet"
        if part.exists() and not REBUILD:
            print(f"Polygons · {name} · cached", flush=True)
            frames.append(pd.read_parquet(part))
            continue
        sub = label_df.loc[label_df["region"] == name, ["id"]].copy()
        if sub.empty:
            continue
        xmin, ymin, xmax, ymax = bbox
        print(f"Polygons · {name} · {len(sub)} ids ...", flush=True)
        con.register("want_ids", sub)
        sql = f"""
        SELECT
          CAST(b.id AS VARCHAR) AS id,
          ST_AsText(b.geometry) AS wkt,
          ST_Area(ST_Transform(b.geometry, 'EPSG:3857')) AS area_m2
        FROM read_parquet('{BUILDINGS}', hive_partitioning=1) b
        INNER JOIN want_ids w ON CAST(b.id AS VARCHAR) = w.id
        WHERE b.bbox.xmin BETWEEN {xmin} AND {xmax}
          AND b.bbox.ymin BETWEEN {ymin} AND {ymax}
          AND b.geometry IS NOT NULL
        """
        df = con.execute(sql).df()
        con.unregister("want_ids")
        df.to_parquet(part, index=False)
        frames.append(df)
        print(f"  got {len(df)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["id"].astype(str)
    return out.drop_duplicates("id")


def _ee_geom(wkt_s: str) -> ee.Geometry:
    g = shapely_wkt.loads(wkt_s)
    if not g.is_valid:
        g = g.buffer(0)
    if g.is_empty:
        raise ValueError("empty geometry")
    return ee.Geometry(shapely_mapping(g), opt_geodesic=False)


def _getinfo_retry(obj, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            return obj.getInfo()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last


def extract_pixels(
    poly_df: pd.DataFrame,
    partial_path: Path,
    already: pd.DataFrame | None = None,
) -> pd.DataFrame:
    init_ee(EE_PROJECT)
    img = annual_embedding_image(EMBED_YEAR).select(BANDS)
    rows: list[dict] = []
    n = len(poly_df)
    start = 0
    batch_size = EE_POLY_BATCH
    saved_at = 0
    prior = already if already is not None and len(already) else pd.DataFrame()
    while start < n:
        width = min(batch_size, n - start)
        batch = poly_df.iloc[start : start + width]
        feats = []
        for _, r in batch.iterrows():
            try:
                geom = _ee_geom(r["wkt"])
            except Exception:
                continue
            feats.append(ee.Feature(geom, {"row_id": str(r["id"]), "region": str(r["region"])}))
        if not feats:
            start += width
            continue
        fc = ee.FeatureCollection(feats)
        sampled = img.sampleRegions(
            collection=fc,
            scale=10,
            geometries=True,
            tileScale=4,
        )
        print(f"EE sampleRegions · {start}:{start + width} / {n} · rows={len(rows)}", flush=True)
        try:
            info = _getinfo_retry(sampled)
        except Exception as exc:
            if width == 1:
                print(f"  skip {batch.iloc[0]['id']}: {exc}", flush=True)
                start += 1
                continue
            print(f"  batch failed ({exc}); splitting", flush=True)
            batch_size = max(1, width // 2)
            continue
        for f in info.get("features", []):
            props = f.get("properties") or {}
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            row = {
                "id": props.get("row_id"),
                "region": props.get("region"),
                "pixel_lon": coords[0],
                "pixel_lat": coords[1],
                "embed_year": EMBED_YEAR,
                "sample_method": "polygon_pixels",
            }
            for b in BANDS:
                row[b] = props.get(b)
            rows.append(row)
        start += width
        if rows and (start - saved_at) >= 100:
            chk = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True) if len(prior) else pd.DataFrame(rows)
            chk.to_parquet(partial_path, index=False)
            saved_at = start
            print(f"  checkpoint {partial_path} · {len(chk)} pixels", flush=True)
        time.sleep(0.2)
    if rows:
        chk = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True) if len(prior) else pd.DataFrame(rows)
        chk.to_parquet(partial_path, index=False)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_poly = OUT_DIR / "building_polygons.parquet"
    cache_pixels = OUT_DIR / "building_gt_pixels.parquet"
    cache_mean = OUT_DIR / "building_gt_mean.parquet"
    partial_px = OUT_DIR / "building_gt_pixels.partial.parquet"
    labels = pd.read_parquet(OUT_DIR / "building_labels.parquet")
    labels["id"] = labels["id"].astype(str)
    print("labels", len(labels), flush=True)

    if cache_poly.exists() and not REBUILD:
        polygons = pd.read_parquet(cache_poly)
        print("Loaded polygons", len(polygons), flush=True)
    else:
        polygons = fetch_polygons_for_labels(labels)
        polygons = labels.merge(polygons, on="id", how="inner")
        polygons.to_parquet(cache_poly, index=False)
        print("Wrote", cache_poly, "n=", len(polygons), "missing=", len(labels) - len(polygons), flush=True)

    if cache_pixels.exists() and not REBUILD:
        pixels = pd.read_parquet(cache_pixels)
        print("Loaded pixels", len(pixels), flush=True)
    else:
        already = pd.DataFrame()
        todo = polygons
        if partial_px.exists() and not REBUILD:
            already = pd.read_parquet(partial_px)
            already["id"] = already["id"].astype(str)
            todo = polygons[~polygons["id"].isin(set(already["id"]))].copy()
            print("Resume pixels already", already["id"].nunique(), "todo", len(todo), flush=True)
        raw = extract_pixels(todo, partial_px, already=already) if len(todo) else pd.DataFrame()
        parts = [p for p in [already, raw] if len(p)]
        pixels = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        pixels = pixels.dropna(subset=BANDS).reset_index(drop=True)
        pixels["id"] = pixels["id"].astype(str)
        pixels = pixels.sort_values(["id", "pixel_lat", "pixel_lon"]).reset_index(drop=True)
        pixels["pixel_idx"] = pixels.groupby("id").cumcount()
        meta = labels[["id", "subtype", "class", "label", "lon", "lat"]].copy()
        extra = [c for c in ["subtype", "class", "label", "lon", "lat"] if c in pixels.columns]
        pixels = pixels.drop(columns=extra, errors="ignore").merge(meta, on="id", how="left")
        pixels.to_parquet(cache_pixels, index=False)
        if partial_px.exists():
            partial_px.unlink()
        print("Wrote", cache_pixels, "n_pixels=", len(pixels), "n_buildings=", pixels["id"].nunique(), flush=True)

    if cache_mean.exists() and not REBUILD:
        gt_mean = pd.read_parquet(cache_mean)
        print("Loaded mean", len(gt_mean), flush=True)
    else:
        n_pix = pixels.groupby("id").size().rename("n_pixels")
        means = pixels.groupby("id", sort=False)[BANDS].mean()
        vec = means.to_numpy(dtype=float)
        norms = np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        means_unit = pd.DataFrame(vec / norms, index=means.index, columns=BANDS)
        gt_mean = labels.merge(n_pix, on="id", how="inner").merge(means_unit, on="id", how="inner")
        gt_mean["embed_year"] = EMBED_YEAR
        gt_mean["sample_method"] = "polygon_mean"
        gt_mean = gt_mean.merge(polygons[["id", "wkt", "area_m2"]], on="id", how="left")
        gt_mean.to_parquet(cache_mean, index=False)
        print("Wrote", cache_mean, "n=", len(gt_mean), flush=True)

    print("\nPixels per building:\n", pixels.groupby("id").size().describe(), flush=True)
    print("median n_pixels by label:\n", gt_mean.groupby("label")["n_pixels"].median().sort_values(ascending=False), flush=True)


if __name__ == "__main__":
    main()
