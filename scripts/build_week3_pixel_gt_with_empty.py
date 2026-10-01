#!/usr/bin/env python3
"""Build week3 pixel GT v2: existing RES/NRES/mixed + AOI-sampled empties.

For each metro AOI:
  - keep all v1 building-overlapped pixels (+ frac_empty)
  - sample empty 10 m AE cells with no Track-A footprint overlap
  - cap: n_empty = n_residential (per region), seeded
  - fetch A00–A63 from public AlphaEarth COGs (CPU process pool)

Writes datasets/week3_pixel_gt_v2/{pixels.parquet, manifest.json, figures/}.

Usage:
  .venv/bin/python scripts/build_week3_pixel_gt_with_empty.py
  .venv/bin/python scripts/build_week3_pixel_gt_with_empty.py --regions NYC_NE --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, box
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "datasets" / "week3_pixel_gt" / "pixels.parquet"
SRC_BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT_DIR = ROOT / "datasets" / "week3_pixel_gt_v2"
FIG_DIR = OUT_DIR / "figures"
CACHE = OUT_DIR / "_cache"

BANDS = [f"A{i:02d}" for i in range(64)]
SEED = 42
LABEL_DEFINITION = "resi_def_v1_pixel_overlap_v1_with_empty"
EMBED_YEAR = 2024
INDEX_URL = (
    "https://storage.googleapis.com/alphaearth_foundations/"
    "satellite_embedding/v1/annual/aef_index.parquet"
)
INDEX_FULL = CACHE / "aef_index.parquet"
INDEX_US = CACHE / "aef_index_us_2024.parquet"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "268435456",
}

# Metro AOIs (WGS84) — same keys as week3 regions
US_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Boston_NE": (-71.20, 42.28, -70.95, 42.42),
    "NYC_NE": (-74.05, 40.68, -73.90, 40.82),
    "Chicago_MW": (-87.75, 41.80, -87.55, 41.95),
    "Atlanta_SE": (-84.45, 33.70, -84.30, 33.85),
    "Houston_S": (-95.50, 29.70, -95.25, 29.85),
    "Denver_W": (-105.05, 39.68, -104.90, 39.80),
    "Phoenix_SW": (-112.15, 33.40, -111.95, 33.55),
    "Seattle_NW": (-122.40, 47.55, -122.25, 47.68),
    "LA_W": (-118.35, 34.00, -118.20, 34.12),
    "SF_W": (-122.45, 37.74, -122.38, 37.81),
}


def pixel_id_from_wkt(pixel_wkt: str) -> str:
    return hashlib.sha1(pixel_wkt.encode("utf-8")).hexdigest()[:16]


def gs_to_https(path: str) -> str:
    return path.replace("gs://", "https://storage.googleapis.com/")


def dequantize(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32)
    return ((v / 127.5) ** 2) * np.sign(v)


def ensure_index() -> pd.DataFrame:
    import urllib.request

    CACHE.mkdir(parents=True, exist_ok=True)
    if not INDEX_FULL.exists():
        print(f"Downloading AEF index -> {INDEX_FULL} ...", flush=True)
        urllib.request.urlretrieve(INDEX_URL, INDEX_FULL)
    if not INDEX_US.exists():
        print("Filtering 2024 US tiles ...", flush=True)
        t = pd.read_parquet(INDEX_FULL)
        t = t.loc[t["year"] == EMBED_YEAR].copy()
        t = t.loc[
            (t["wgs84_west"] <= -66.0)
            & (t["wgs84_east"] >= -125.0)
            & (t["wgs84_south"] <= 50.0)
            & (t["wgs84_north"] >= 24.0)
        ].copy()
        t.to_parquet(INDEX_US, index=False)
    return pd.read_parquet(INDEX_US)


def _pixel_wkt_from_center(lon: float, lat: float, half_m: float, crs: str) -> str:
    """Approximate 10 m pixel footprint as WGS84 polygon around center."""
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ux, uy = tfm.transform(lon, lat)
    ring = [
        inv.transform(ux - half_m, uy - half_m),
        inv.transform(ux + half_m, uy - half_m),
        inv.transform(ux + half_m, uy + half_m),
        inv.transform(ux - half_m, uy + half_m),
        inv.transform(ux - half_m, uy - half_m),
    ]
    return "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"


def _sample_empty_candidates(
    region: str,
    n_target: int,
    buildings_wkt: list[str],
    occupied_ids: set[str],
    crs: str,
    seed: int,
    oversample: int = 8,
) -> list[dict]:
    """Reject-sample empty 10 m cells inside the metro AOI."""
    if region not in US_REGIONS or n_target <= 0:
        return []
    xmin, ymin, xmax, ymax = US_REGIONS[region]
    rng = np.random.default_rng(seed)

    b_geoms = []
    for w in buildings_wkt:
        g = shapely_wkt.loads(w)
        if not g.is_valid:
            g = g.buffer(0)
        b_geoms.append(g)
    bg = gpd.GeoDataFrame(geometry=b_geoms, crs="EPSG:4326")
    sindex = bg.sindex

    half_m = 5.0
    found: dict[str, dict] = {}
    attempts = 0
    max_attempts = max(n_target * oversample * 20, 5000)
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    while len(found) < n_target and attempts < max_attempts:
        attempts += 1
        batch = min(2048, (n_target - len(found)) * oversample)
        lons = rng.uniform(xmin, xmax, size=batch)
        lats = rng.uniform(ymin, ymax, size=batch)
        for lon, lat in zip(lons, lats):
            if len(found) >= n_target:
                break
            ux, uy = tfm.transform(float(lon), float(lat))
            ux = np.floor(ux / 10.0) * 10.0 + 5.0
            uy = np.floor(uy / 10.0) * 10.0 + 5.0
            lon_c, lat_c = inv.transform(ux, uy)
            pwkt = _pixel_wkt_from_center(lon_c, lat_c, half_m, crs)
            pid = pixel_id_from_wkt(pwkt)
            if pid in occupied_ids or pid in found:
                continue
            pix = shapely_wkt.loads(pwkt)
            hits = list(sindex.query(pix, predicate="intersects"))
            if hits:
                if any(pix.intersects(bg.geometry.iloc[i]) for i in hits):
                    continue
            found[pid] = {
                "pixel_id": pid,
                "pixel_wkt": pwkt,
                "pixel_lon": float(lon_c),
                "pixel_lat": float(lat_c),
                "region": region,
                "crs": crs,
            }

    return list(found.values())


def _fetch_embeddings_batch(payload: dict) -> list[dict]:
    """Open one COG tile and fill A00–A63 for candidate points that fall on it."""
    path = payload["path"]
    crs = payload["crs"]
    cands = payload["cands"]
    out: list[dict] = []
    tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(gs_to_https(path)) as src:
            a = src.transform.a
            e = src.transform.e
            for c in cands:
                ux, uy = tfm.transform(c["pixel_lon"], c["pixel_lat"])
                try:
                    rr, cc = src.index(ux, uy)
                except Exception:
                    continue
                if not (0 <= rr < src.height and 0 <= cc < src.width):
                    continue
                pix = src.read(window=((rr, rr + 1), (cc, cc + 1)))[:, 0, 0]
                if (pix == -128).any():
                    continue
                emb = dequantize(pix)
                # Re-center to exact COG pixel
                x_c = src.transform.c + a * (cc + 0.5)
                y_c = src.transform.f + e * (rr + 0.5)
                lon_c, lat_c = inv.transform(x_c, y_c)
                hx, hy = abs(a) / 2.0, abs(e) / 2.0
                ring = [
                    inv.transform(x_c - hx, y_c - hy),
                    inv.transform(x_c + hx, y_c - hy),
                    inv.transform(x_c + hx, y_c + hy),
                    inv.transform(x_c - hx, y_c + hy),
                    inv.transform(x_c - hx, y_c - hy),
                ]
                pwkt = "POLYGON ((" + ", ".join(f"{x} {y}" for x, y in ring) + "))"
                pid = pixel_id_from_wkt(pwkt)
                row = {
                    "pixel_id": pid,
                    "pixel_wkt": pwkt,
                    "pixel_lon": float(lon_c),
                    "pixel_lat": float(lat_c),
                    "region": c["region"],
                    "crs": crs,
                    "pixel_size_m": float(abs(a)),
                    "pixel_area_m2": float(abs(a) * abs(e)),
                    "area_res_m2": 0.0,
                    "area_nres_m2": 0.0,
                    "frac_res": 0.0,
                    "frac_nres": 0.0,
                    "frac_res_full": 0.0,
                    "frac_nres_full": 0.0,
                    "frac_empty": 1.0,
                    "frac_empty_full": 1.0,
                    "label": "empty",
                    "n_res_buildings": 0,
                    "n_nres_buildings": 0,
                    "building_ids": "[]",
                    "label_definition": LABEL_DEFINITION,
                }
                for k, band in enumerate(BANDS):
                    row[band] = float(emb[k])
                out.append(row)
    return out


def _assign_cands_to_tiles(cands: list[dict], index: pd.DataFrame) -> list[dict]:
    """Group candidates by covering AE tile for batched COG reads."""
    buckets: dict[str, dict] = {}
    for c in cands:
        lon, lat = c["pixel_lon"], c["pixel_lat"]
        hit = index[
            (index["wgs84_west"] <= lon)
            & (index["wgs84_east"] >= lon)
            & (index["wgs84_south"] <= lat)
            & (index["wgs84_north"] >= lat)
        ]
        if hit.empty:
            continue
        # Prefer tile whose CRS matches candidate crs when possible
        prefer = hit.loc[hit["crs"] == c["crs"]] if "crs" in hit.columns else hit
        row = prefer.iloc[0] if len(prefer) else hit.iloc[0]
        path = str(row["path"])
        if path not in buckets:
            buckets[path] = {"path": path, "crs": str(row["crs"]), "cands": []}
        buckets[path]["cands"].append(c)
    return list(buckets.values())


def _process_region_empties(payload: dict) -> pd.DataFrame:
    region = payload["region"]
    n_target = int(payload["n_target"])
    buildings_wkt = payload["buildings_wkt"]
    occupied_ids = set(payload["occupied_ids"])
    crs = payload["crs"]
    seed = int(payload["seed"])
    index = pd.DataFrame(payload["index_records"])

    cands = _sample_empty_candidates(
        region, n_target, buildings_wkt, occupied_ids, crs, seed
    )
    if not cands:
        return pd.DataFrame()

    jobs = _assign_cands_to_tiles(cands, index)
    rows: list[dict] = []
    seen: set[str] = set()
    for job in jobs:
        for r in _fetch_embeddings_batch(job):
            if r["pixel_id"] in seen or r["pixel_id"] in occupied_ids:
                continue
            seen.add(r["pixel_id"])
            rows.append(r)
            if len(rows) >= n_target:
                break
        if len(rows) >= n_target:
            break
    return pd.DataFrame(rows[:n_target])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "pixels.parquet"
    if out_path.exists() and not args.rebuild:
        print(f"Exists: {out_path} (pass --rebuild to overwrite)", flush=True)
        return

    t0 = time.time()
    print(f"Loading v1 {V1}", flush=True)
    v1 = pd.read_parquet(V1)
    v1 = v1.copy()
    v1["frac_empty_full"] = np.clip(
        1.0 - v1["frac_res_full"].to_numpy() - v1["frac_nres_full"].to_numpy(), 0.0, 1.0
    )
    v1["frac_empty"] = np.round(v1["frac_empty_full"], 2)
    v1["label_definition"] = LABEL_DEFINITION

    buildings = pd.read_parquet(SRC_BUILDINGS, columns=["region", "wkt"])
    index = ensure_index()
    # slim index for worker payloads
    index_slim = index[["path", "crs", "wgs84_west", "wgs84_east", "wgs84_south", "wgs84_north"]].copy()

    regions = sorted(v1["region"].astype(str).unique().tolist())
    if args.regions:
        regions = [r for r in regions if r in set(args.regions)]

    payloads = []
    for i, region in enumerate(regions):
        sub = v1.loc[v1["region"] == region]
        n_res = int((sub["label"] == "residential").sum())
        n_target = n_res
        crs = str(sub["crs"].iloc[0])
        b_wkt = buildings.loc[buildings["region"].astype(str) == region, "wkt"].astype(str).tolist()
        payloads.append(
            {
                "region": region,
                "n_target": n_target,
                "buildings_wkt": b_wkt,
                "occupied_ids": sub["pixel_id"].astype(str).tolist(),
                "crs": crs,
                "seed": args.seed + i,
                "index_records": index_slim.to_dict(orient="records"),
            }
        )
        print(f"  plan {region}: n_empty_target={n_target} (n_res={n_res})", flush=True)

    n_workers = args.workers or min(len(payloads), os.cpu_count() or 1)
    print(f"Sampling empties · {len(payloads)} regions · workers={n_workers}", flush=True)

    empty_parts: list[pd.DataFrame] = []
    # Regions run in parallel; each region does its own sequential tile reads
    # (COG I/O bound — more region workers = better utilization).
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process_region_empties, p): p["region"] for p in payloads}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="regions"):
            region = futs[fut]
            df = fut.result()
            print(f"  {region}: empties={len(df)}", flush=True)
            if len(df):
                empty_parts.append(df)

    empties = pd.concat(empty_parts, ignore_index=True) if empty_parts else pd.DataFrame()

    # Align columns
    for col in v1.columns:
        if col not in empties.columns and len(empties):
            if col in BANDS:
                empties[col] = np.nan
            elif col.startswith("frac") or col.endswith("_m2") or col == "pixel_size_m":
                empties[col] = 0.0
            elif col.startswith("n_"):
                empties[col] = 0
            else:
                empties[col] = None

    if len(empties):
        empties = empties[v1.columns.tolist()]
        out = pd.concat([v1, empties], ignore_index=True)
    else:
        out = v1

    # Drop accidental duplicate pixel_ids (prefer non-empty)
    out["_pri"] = (out["label"] != "empty").astype(int)
    out = out.sort_values(["pixel_id", "_pri"], ascending=[True, False]).drop_duplicates(
        "pixel_id", keep="first"
    )
    out = out.drop(columns=["_pri"])

    out.to_parquet(out_path, index=False)

    counts = out["label"].value_counts().to_dict()
    by_region = (
        out.groupby(["region", "label"]).size().unstack(fill_value=0).to_dict()
    )
    manifest = {
        "name": "week3_pixel_gt_v2_with_empty",
        "label_definition": LABEL_DEFINITION,
        "empty_rule": "no Track-A footprint overlap (frac_res==0 and frac_nres==0)",
        "empty_sampling": {
            "method": "AOI_uniform_reject_sample_snap_10m",
            "cap": "n_empty = n_residential per region",
            "seed": args.seed,
        },
        "frac_empty": "clip(1 - frac_res - frac_nres, 0, 1)",
        "sources": {
            "v1": str(V1.relative_to(ROOT)),
            "buildings": str(SRC_BUILDINGS.relative_to(ROOT)),
            "ae_cogs": "gs://alphaearth_foundations (public)",
        },
        "n_pixels": int(len(out)),
        "n_empty": int((out["label"] == "empty").sum()),
        "n_regions": int(out["region"].nunique()),
        "regions": sorted(out["region"].astype(str).unique().tolist()),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "label_counts_by_region": {
            str(k): {str(kk): int(vv) for kk, vv in d.items()} for k, d in by_region.items()
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # QA plot
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["residential", "non_residential", "mixed", "empty"]
    vals = [int(counts.get(k, 0)) for k in order]
    ax.bar(order, vals, color=["#2ca02c", "#d62728", "#9467bd", "#7f7f7f"])
    ax.set_ylabel("count")
    ax.set_title("Pixel GT v2 label counts (incl. AOI-sampled empty)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    tab = out.groupby(["region", "label"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    tab.plot(kind="bar", stacked=True, ax=ax, color=["#2ca02c", "#d62728", "#9467bd", "#7f7f7f"])
    ax.set_ylabel("count")
    ax.set_title("Label counts by region (v2)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(manifest["label_counts"], indent=2), flush=True)
    print(f"Wrote {out_path}  n={len(out)}  elapsed={manifest['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
