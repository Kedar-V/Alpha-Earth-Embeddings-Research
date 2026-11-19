#!/usr/bin/env python3
"""Serve Overture footprint map UI + bbox GeoJSON API (all buildings, no sampling).

  python scripts/serve_google_maps_overture.py
  → http://127.0.0.1:8770/

Static: outputs/google_maps_overture/
API:    GET /api/buildings?region=LA_W&west=&south=&east=&north=&limit=8000
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from shapely import from_wkt, is_empty
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "outputs" / "google_maps_overture"
ENV_PATH = ROOT / ".env"
BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
WEEK6_GHSL = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)


def load_api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "GOOGLE_MAPS_API_KEY":
                return v.strip().strip('"').strip("'")
    return ""


def _coords_round(obj: Any, nd: int = 6) -> Any:
    if isinstance(obj, (float, int)):
        return round(float(obj), nd)
    if isinstance(obj, (list, tuple)):
        return [_coords_round(x, nd) for x in obj]
    return obj


class BuildingIndex:
    """In-memory per-region building store for fast bbox queries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._regions: dict[str, dict[str, Any]] = {}
        self._loading: set[str] = set()

    def ensure(self, region: str) -> dict[str, Any]:
        with self._lock:
            if region in self._regions:
                return self._regions[region]
            if region in self._loading:
                hold = True
            else:
                self._loading.add(region)
                hold = False
        if hold:
            # wait for other thread
            while True:
                with self._lock:
                    if region in self._regions:
                        return self._regions[region]
                import time

                time.sleep(0.05)

        t0 = __import__("time").time()
        cols = ["id", "region", "lon", "lat", "wkt", "y_res", "label", "class"]
        df = pd.read_parquet(BUILDINGS, columns=cols)
        df = df[df["region"] == region].copy()
        df["id"] = df["id"].astype(str)

        if WEEK6_GHSL.exists():
            g = pd.read_parquet(
                WEEK6_GHSL, columns=["id", "ghsl_frac_res", "ghsl_frac_nres", "ghsl_covered"]
            )
            g["id"] = g["id"].astype(str)
            g = g.drop_duplicates("id")
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

        store = {
            "lon": df["lon"].to_numpy(dtype=np.float64),
            "lat": df["lat"].to_numpy(dtype=np.float64),
            "wkt": df["wkt"].to_numpy(dtype=object),
            "id": df["id"].to_numpy(dtype=object),
            "y_res": df["y_res"].to_numpy(dtype=np.int8),
            "label": df["label"].astype(str).to_numpy(dtype=object),
            "class": df["class"].fillna("").astype(str).to_numpy(dtype=object),
            "ghsl_y": df["ghsl_y"].to_numpy(dtype=np.int8),
            "ghsl_covered": df["ghsl_covered"].to_numpy(dtype=bool),
            "ghsl_frac_res": df["ghsl_frac_res"].to_numpy(dtype=np.float64),
            "ghsl_frac_nres": df["ghsl_frac_nres"].to_numpy(dtype=np.float64),
            "n": len(df),
            "load_s": round(__import__("time").time() - t0, 2),
        }
        with self._lock:
            self._regions[region] = store
            self._loading.discard(region)
        print(f"[index] {region}: {store['n']:,} buildings in {store['load_s']}s", flush=True)
        return store

    def query(
        self,
        region: str,
        west: float,
        south: float,
        east: float,
        north: float,
        limit: int = 8000,
    ) -> dict[str, Any]:
        st = self.ensure(region)
        lon, lat = st["lon"], st["lat"]
        mask = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        idx = np.flatnonzero(mask)
        n_match = int(idx.size)
        truncated = False
        if idx.size > limit:
            # prefer denser center: take random sample for stability under pan
            rng = np.random.default_rng(abs(hash((region, round(west, 4), round(south, 4)))) % (2**32))
            idx = rng.choice(idx, size=limit, replace=False)
            truncated = True

        feats: list[dict[str, Any]] = []
        for i in idx:
            geom = from_wkt(st["wkt"][i], on_invalid="ignore")
            if geom is None or is_empty(geom):
                continue
            props: dict[str, Any] = {
                "id": str(st["id"][i]),
                "region": region,
                "y_res": int(st["y_res"][i]),
                "label": str(st["label"][i]),
                "class": str(st["class"][i]),
                "lon": float(st["lon"][i]),
                "lat": float(st["lat"][i]),
                "ghsl_covered": bool(st["ghsl_covered"][i]),
                "ghsl_y": int(st["ghsl_y"][i]),
            }
            fr, fn = st["ghsl_frac_res"][i], st["ghsl_frac_nres"][i]
            if np.isfinite(fr):
                props["ghsl_frac_res"] = round(float(fr), 4)
            if np.isfinite(fn):
                props["ghsl_frac_nres"] = round(float(fn), 4)
            gj = mapping(geom)
            gj["coordinates"] = _coords_round(gj["coordinates"], 6)
            feats.append({"type": "Feature", "geometry": gj, "properties": props})

        return {
            "type": "FeatureCollection",
            "features": feats,
            "meta": {
                "region": region,
                "n_match": n_match,
                "n_returned": len(feats),
                "truncated": truncated,
                "limit": limit,
                "bbox": [west, south, east, north],
                "n_region_total": int(st["n"]),
            },
        }


INDEX = BuildingIndex()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, api_key: str = "", **kwargs):
        self.api_key = api_key
        super().__init__(*args, directory=str(DIR), **kwargs)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in ("/config.js", "config.js"):
            body = f"window.GOOGLE_MAPS_API_KEY = {json.dumps(self.api_key)};\n".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path in ("/api/buildings", "api/buildings"):
            qs = parse_qs(parsed.query)
            try:
                region = qs["region"][0]
                west = float(qs["west"][0])
                south = float(qs["south"][0])
                east = float(qs["east"][0])
                north = float(qs["north"][0])
                limit = int(qs.get("limit", ["8000"])[0])
                limit = max(100, min(limit, 20000))
            except Exception:
                self._json(400, {"error": "need region,west,south,east,north"})
                return
            if east < west or north < south:
                self._json(400, {"error": "invalid bbox"})
                return
            # pad tiny boxes
            if (east - west) < 1e-6 or (north - south) < 1e-6:
                self._json(200, {"type": "FeatureCollection", "features": [], "meta": {}})
                return
            try:
                fc = INDEX.query(region, west, south, east, north, limit=limit)
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            self._json(200, fc)
            return

        return super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--preload", nargs="*", default=None, help="regions to preload (default: none)")
    args = ap.parse_args()

    if not (DIR / "index.html").exists():
        raise SystemExit(f"Missing {DIR / 'index.html'}")

    key = load_api_key()
    if args.preload:
        for r in args.preload:
            INDEX.ensure(r)

    def factory(*a, **kw):
        return Handler(*a, api_key=key, **kw)

    httpd = ThreadingHTTPServer((args.host, args.port), factory)
    print(f"Serving {DIR}")
    print(f"Open http://{args.host}:{args.port}/")
    print("API /api/buildings?region=&west=&south=&east=&north=  (viewport, all footprints)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    raise SystemExit(main())
