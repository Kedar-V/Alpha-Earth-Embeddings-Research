#!/usr/bin/env python3
"""Export ALL Track-A Overture footprints into google_maps_overture GeoJSON (no sampling).

  python scripts/export_all_overture_map_geojson.py

Writes outputs/google_maps_overture/geojson/<region>.geojson + updates regions.json.
Does not touch index.html.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import from_wkt, is_empty
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT = ROOT / "outputs" / "google_maps_overture"
GEO = OUT / "geojson"
WEEK6_GHSL = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)

REGION_PRETTY = {
    "Boston_NE": "Boston",
    "NYC_NE": "New York City",
    "Atlanta_SE": "Atlanta",
    "Houston_S": "Houston",
    "Chicago_MW": "Chicago",
    "Phoenix_SW": "Phoenix",
    "Denver_W": "Denver",
    "LA_W": "Los Angeles",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coords_round(obj: Any, nd: int = 6) -> Any:
    if isinstance(obj, (float, int)):
        return round(float(obj), nd)
    if isinstance(obj, (list, tuple)):
        return [_coords_round(x, nd) for x in obj]
    return obj


def export_region(payload: dict[str, Any]) -> dict[str, Any]:
    region = payload["region"]
    buildings_path = payload["buildings_path"]
    out_path = Path(payload["out_path"])
    ghsl_path = payload.get("ghsl_path")

    t0 = time.time()
    cols = ["id", "region", "lon", "lat", "wkt", "y_res", "label", "class"]
    df = pd.read_parquet(buildings_path, columns=cols)
    df = df[df["region"] == region].copy()
    n_all = len(df)

    ghsl = None
    if ghsl_path and Path(ghsl_path).exists():
        g = pd.read_parquet(
            ghsl_path, columns=["id", "ghsl_frac_res", "ghsl_frac_nres", "ghsl_covered"]
        )
        g["id"] = g["id"].astype(str)
        g = g.drop_duplicates("id")
        df["id"] = df["id"].astype(str)
        df = df.merge(g, on="id", how="left")
        df["ghsl_covered"] = df["ghsl_covered"].fillna(False).astype(bool)
        bu = df["ghsl_frac_res"].fillna(0.0) + df["ghsl_frac_nres"].fillna(0.0)
        share = np.where(bu > 1e-9, df["ghsl_frac_nres"].fillna(0.0) / bu, np.nan)
        df["ghsl_y"] = np.where(
            ~df["ghsl_covered"] | ~np.isfinite(share),
            -1,
            np.where(share < 0.5, 1, 0),
        ).astype(int)
    else:
        df["ghsl_covered"] = False
        df["ghsl_y"] = -1
        df["ghsl_frac_res"] = np.nan
        df["ghsl_frac_nres"] = np.nan

    geoms = from_wkt(df["wkt"].to_numpy(dtype=object), on_invalid="ignore")
    feats: list[dict[str, Any]] = []
    failed = 0
    for i, geom in enumerate(geoms):
        if geom is None or is_empty(geom):
            failed += 1
            continue
        row = df.iloc[i]
        props: dict[str, Any] = {
            "id": str(row["id"]),
            "region": region,
            "y_res": int(row["y_res"]),
            "label": str(row["label"]) if pd.notna(row["label"]) else "",
            "class": str(row["class"]) if pd.notna(row["class"]) else "",
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "ghsl_covered": bool(row["ghsl_covered"]),
            "ghsl_y": int(row["ghsl_y"]),
        }
        if pd.notna(row["ghsl_frac_res"]):
            props["ghsl_frac_res"] = round(float(row["ghsl_frac_res"]), 4)
        if pd.notna(row["ghsl_frac_nres"]):
            props["ghsl_frac_nres"] = round(float(row["ghsl_frac_nres"]), 4)
        gj = mapping(geom)
        gj["coordinates"] = _coords_round(gj["coordinates"], 6)
        feats.append({"type": "Feature", "geometry": gj, "properties": props})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # write via temp for atomic replace
    tmp = out_path.with_suffix(".geojson.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[')
        for j, feat in enumerate(feats):
            if j:
                f.write(",")
            json.dump(feat, f, separators=(",", ":"))
        f.write("]}")
    tmp.replace(out_path)

    n_res = int((df["y_res"] == 1).sum())
    n_nres = int((df["y_res"] == 0).sum())
    n_hotel = int((df["label"].astype(str) == "hotel").sum())
    n_hotel_res = int(((df["label"].astype(str) == "hotel") & (df["ghsl_y"] == 1)).sum())
    n_hotel_nres = int(((df["label"].astype(str) == "hotel") & (df["ghsl_y"] == 0)).sum())
    west, east = float(df["lon"].min()), float(df["lon"].max())
    south, north = float(df["lat"].min()), float(df["lat"].max())
    return {
        "region": region,
        "pretty": REGION_PRETTY.get(region, region),
        "n_sample": len(feats),
        "n_available": n_all,
        "n_res": n_res,
        "n_nres": n_nres,
        "n_hotels": n_hotel,
        "n_hotels_ghsl_res": n_hotel_res,
        "n_hotels_ghsl_nres": n_hotel_nres,
        "failed": failed,
        "center": {"lon": float(df["lon"].mean()), "lat": float(df["lat"].mean())},
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "geojson": f"geojson/{region}.geojson",
        "bytes": out_path.stat().st_size,
        "seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    n_cpus = os.cpu_count() or 1
    regions = list(REGION_PRETTY.keys())
    GEO.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / "_export_all_runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    log_path = run_dir / "job.log"

    def log(msg: str) -> None:
        line = f"[{utc()}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    progress: dict[str, Any] = {
        "status": "running",
        "stage": "export",
        "started_at": utc(),
        "completed": 0,
        "total": len(regions),
        "n_cpus": n_cpus,
    }
    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))

    payloads = [
        {
            "region": r,
            "buildings_path": str(BUILDINGS),
            "out_path": str(GEO / f"{r}.geojson"),
            "ghsl_path": str(WEEK6_GHSL) if WEEK6_GHSL.exists() else None,
        }
        for r in regions
    ]

    log(f"export ALL footprints · regions={len(regions)} workers={min(n_cpus, len(regions))}")
    metas: list[dict[str, Any]] = []
    workers = min(n_cpus, len(regions))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(export_region, p): p["region"] for p in payloads}
        for fut in as_completed(futs):
            reg = futs[fut]
            try:
                meta = fut.result()
            except Exception as e:
                log(f"FAIL {reg}: {e}")
                (run_dir / "failures").mkdir(exist_ok=True)
                (run_dir / "failures" / f"{reg}.txt").write_text(str(e))
                progress["completed"] = int(progress["completed"]) + 1
                (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
                continue
            metas.append(meta)
            progress["completed"] = int(progress["completed"]) + 1
            progress["last_region"] = reg
            progress["updated_at"] = utc()
            (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
            log(
                f"{reg}: n={meta['n_sample']:,} RES={meta['n_res']:,} NRES={meta['n_nres']:,} "
                f"MB={meta['bytes']/1e6:.1f} t={meta['seconds']}s"
            )

    # stable order
    order = {r: i for i, r in enumerate(regions)}
    metas.sort(key=lambda m: order.get(m["region"], 999))
    regions_path = OUT / "regions.json"
    slim = [
        {
            "region": m["region"],
            "pretty": m["pretty"],
            "n_sample": m["n_sample"],
            "n_res": m["n_res"],
            "n_nres": m["n_nres"],
            "n_hotels": m["n_hotels"],
            "n_hotels_ghsl_res": m["n_hotels_ghsl_res"],
            "n_hotels_ghsl_nres": m["n_hotels_ghsl_nres"],
            "n_available": m["n_available"],
            "center": m["center"],
            "bounds": m["bounds"],
            "geojson": m["geojson"],
        }
        for m in metas
    ]
    regions_path.write_text(json.dumps(slim, indent=2))
    summary = {
        "status": "completed",
        "n_regions": len(metas),
        "n_features": int(sum(m["n_sample"] for m in metas)),
        "total_bytes": int(sum(m["bytes"] for m in metas)),
        "regions": metas,
        "regions_json": str(regions_path),
        "finished_at": utc(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "full_export_manifest.json").write_text(json.dumps(summary, indent=2))
    progress["status"] = "completed"
    progress["stage"] = "done"
    progress["percent_complete"] = 100.0
    (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
    (run_dir / "DONE").write_text(utc() + "\n")
    log(f"DONE features={summary['n_features']:,} bytes={summary['total_bytes']/1e6:.1f}MB → {GEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
