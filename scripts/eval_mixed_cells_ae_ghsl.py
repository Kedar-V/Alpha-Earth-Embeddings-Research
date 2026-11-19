#!/usr/bin/env python3
"""Mixed-cell experiment: AE 10 m vs GHSL 10 m grids vs Overture RES/NRES polygons.

A cell is MIXED if it geometrically overlaps ≥1 residential (y_res=1) and ≥1
non-residential (y_res=0) Track-A Overture footprint.

  AE   — week-4 soft GT on AlphaEarth 10 m cells (frac_res_full ∩ frac_nres_full)
  GHSL — burn Overture footprints onto native Mollweide 10 m BUILT-S grid

Also writes per-metro sample GeoJSON for Google Maps overlays and patches
outputs/google_maps_overture/index.html with AE/GHSL mixed-cell toggles.

Outputs:
  outputs/mixed_cells_ae_ghsl/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio import features
from rasterio.transform import xy as transform_xy
from rasterio.windows import Window, from_bounds
from shapely import wkt as shapely_wkt
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILDINGS = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
PIXEL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
S_TILE_DIR = ROOT / "data" / "ghsl" / "built_s_tiles"
GMAPS = ROOT / "outputs" / "google_maps_overture"
GHSL_CRS = "ESRI:54009"
MAP_SAMPLE = 400
SEED = 42

REGION_ORDER = [
    "Boston_NE",
    "NYC_NE",
    "Atlanta_SE",
    "Houston_S",
    "Chicago_MW",
    "Phoenix_SW",
    "Denver_W",
    "LA_W",
    "SF_W",
    "Seattle_NW",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev:
            done = int(prev["completed"])
            total = max(1, int(prev["total"]))
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def classify_soft(fr: np.ndarray, fn: np.ndarray) -> dict[str, int]:
    fr = np.asarray(fr, float)
    fn = np.asarray(fn, float)
    valid = np.isfinite(fr) & np.isfinite(fn) & ((fr + fn) > 1e-12)
    only_res = valid & (fr > 1e-12) & (fn <= 1e-12)
    only_nres = valid & (fn > 1e-12) & (fr <= 1e-12)
    mixed = valid & (fr > 1e-12) & (fn > 1e-12)
    n = int(valid.sum())
    return {
        "n_gt": n,
        "n_only_res": int(only_res.sum()),
        "n_only_nres": int(only_nres.sum()),
        "n_mixed": int(mixed.sum()),
        "pct_only_res": 100.0 * float(only_res.sum()) / n if n else float("nan"),
        "pct_only_nres": 100.0 * float(only_nres.sum()) / n if n else float("nan"),
        "pct_mixed": 100.0 * float(mixed.sum()) / n if n else float("nan"),
    }


def _tile_index(s_paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for i, p in enumerate(s_paths):
        with rasterio.open(p) as ds:
            b = ds.bounds
            out.append(
                {
                    "i": i,
                    "path": str(p),
                    "left": b.left,
                    "bottom": b.bottom,
                    "right": b.right,
                    "top": b.top,
                    "transform": ds.transform,
                    "width": ds.width,
                    "height": ds.height,
                }
            )
    return out


def _affine(tm: dict[str, Any]) -> rasterio.Affine:
    t = tm["transform"]
    if isinstance(t, rasterio.Affine):
        return t
    return rasterio.Affine(*t)


def _ghsl_region_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Burn RES/NRES footprints onto GHSL 10 m grid; return counts + sample cells."""
    region = payload["region"]
    buildings = pd.read_parquet(payload["buildings_path"], columns=["id", "y_res", "wkt"])
    tile_meta = payload["tile_meta"]
    map_sample = int(payload["map_sample"])
    seed = int(payload["seed"])
    region_seed = int(payload.get("region_seed", 0))

    to_moll = Transformer.from_crs("EPSG:4326", GHSL_CRS, always_xy=True)

    def to_m(x, y, z=None):
        return to_moll.transform(x, y)

    res_keys: set[tuple[int, int, int]] = set()
    nres_keys: set[tuple[int, int, int]] = set()
    n_res_b = 0
    n_nres_b = 0
    failed = 0

    for y_res, wkt in zip(buildings["y_res"].to_numpy(), buildings["wkt"].to_numpy()):
        try:
            geom = shapely_wkt.loads(wkt)
            if geom.is_empty:
                failed += 1
                continue
            geom_m = shapely_transform(to_m, geom)
            if geom_m.is_empty:
                failed += 1
                continue
        except Exception:
            failed += 1
            continue
        y = int(y_res)
        if y == 1:
            n_res_b += 1
            target = res_keys
        else:
            n_nres_b += 1
            target = nres_keys
        minx, miny, maxx, maxy = geom_m.bounds
        for tm in tile_meta:
            if maxx < tm["left"] or minx > tm["right"] or maxy < tm["bottom"] or miny > tm["top"]:
                continue
            transform = _affine(tm)
            window = from_bounds(minx, miny, maxx, maxy, transform=transform)
            window = Window(
                max(0, int(window.col_off) - 1),
                max(0, int(window.row_off) - 1),
                max(1, int(np.ceil(window.width)) + 2),
                max(1, int(np.ceil(window.height)) + 2),
            ).intersection(Window(0, 0, tm["width"], tm["height"]))
            if window.width <= 0 or window.height <= 0:
                continue
            win_transform = rasterio.windows.transform(window, transform)
            shape = (int(window.height), int(window.width))
            mask = features.rasterize(
                [(geom_m, 1)],
                out_shape=shape,
                transform=win_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )
            if not mask.any():
                c = geom_m.centroid
                try:
                    from rasterio.transform import rowcol

                    rr, cc = rowcol(win_transform, c.x, c.y)
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        mask = np.zeros(shape, dtype=np.uint8)
                        mask[rr, cc] = 1
                    else:
                        continue
                except Exception:
                    continue
            rows_l, cols_l = np.where(mask == 1)
            r0 = int(window.row_off)
            c0 = int(window.col_off)
            ti = int(tm["i"])
            for rr, cc in zip(rows_l.tolist(), cols_l.tolist()):
                target.add((ti, r0 + int(rr), c0 + int(cc)))

    mixed = res_keys & nres_keys
    only_res = res_keys - nres_keys
    only_nres = nres_keys - res_keys
    n_gt = len(res_keys | nres_keys)

    rng = np.random.default_rng(seed + region_seed)
    mixed_list = list(mixed)
    if len(mixed_list) > map_sample:
        idx = rng.choice(len(mixed_list), size=map_sample, replace=False)
        mixed_list = [mixed_list[i] for i in idx]
    sample_features: list[dict[str, Any]] = []
    to_wgs = Transformer.from_crs(GHSL_CRS, "EPSG:4326", always_xy=True)

    def cell_feature(ti: int, r: int, c: int, kind: str) -> dict[str, Any] | None:
        transform = _affine(tile_meta[ti])
        # corners in row/col pixel space
        corners = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
        try:
            ring = []
            for rr, cc in corners:
                x, y = transform_xy(transform, rr, cc, offset="ul")
                lon, lat = to_wgs.transform(float(x), float(y))
                ring.append([lon, lat])
            ring.append(ring[0])
            return {
                "type": "Feature",
                "properties": {
                    "region": region,
                    "grid": "ghsl",
                    "kind": kind,
                    "tile_i": ti,
                    "row": r,
                    "col": c,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        except Exception:
            return None

    for ti, r, c in mixed_list:
        feat = cell_feature(ti, r, c, "mixed")
        if feat:
            sample_features.append(feat)

    for kind, pool in (("only_res", list(only_res)), ("only_nres", list(only_nres))):
        if not pool:
            continue
        k = min(40, len(pool))
        pick = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
        for ti, r, c in pick:
            feat = cell_feature(ti, r, c, kind)
            if feat:
                sample_features.append(feat)

    return {
        "region": region,
        "grid": "ghsl",
        "n_buildings_res": n_res_b,
        "n_buildings_nres": n_nres_b,
        "n_failed_geoms": failed,
        "n_gt": n_gt,
        "n_only_res": len(only_res),
        "n_only_nres": len(only_nres),
        "n_mixed": len(mixed),
        "pct_only_res": 100.0 * len(only_res) / n_gt if n_gt else float("nan"),
        "pct_only_nres": 100.0 * len(only_nres) / n_gt if n_gt else float("nan"),
        "pct_mixed": 100.0 * len(mixed) / n_gt if n_gt else float("nan"),
        "sample_features": sample_features,
    }


def score_ae(run_dir: Path, map_sample: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    log(run_dir, "AE: loading week4 soft GT pixels")
    pix = pd.read_parquet(
        PIXEL_GT,
        columns=[
            "pixel_id",
            "region",
            "pixel_wkt",
            "pixel_lon",
            "pixel_lat",
            "frac_res_full",
            "frac_nres_full",
        ],
    )
    rows = []
    sample_features: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for region, sub in pix.groupby("region", sort=False):
        fr = sub["frac_res_full"].to_numpy()
        fn = sub["frac_nres_full"].to_numpy()
        stats = classify_soft(fr, fn)
        stats["region"] = region
        stats["grid"] = "ae"
        rows.append(stats)
        valid = np.isfinite(fr) & np.isfinite(fn) & ((fr + fn) > 1e-12)
        mixed_mask = valid & (fr > 1e-12) & (fn > 1e-12)
        mixed_idx = np.flatnonzero(mixed_mask)
        if len(mixed_idx) > map_sample:
            mixed_idx = rng.choice(mixed_idx, size=map_sample, replace=False)
        for i in mixed_idx:
            r = sub.iloc[int(i)]
            try:
                geom = shapely_wkt.loads(r["pixel_wkt"])
                sample_features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "region": region,
                            "grid": "ae",
                            "kind": "mixed",
                            "pixel_id": str(r["pixel_id"]),
                            "frac_res_full": round(float(r["frac_res_full"]), 4),
                            "frac_nres_full": round(float(r["frac_nres_full"]), 4),
                        },
                        "geometry": mapping(geom),
                    }
                )
            except Exception:
                continue
        log(
            run_dir,
            f"AE {region}: gt={stats['n_gt']:,} mixed={stats['n_mixed']:,} ({stats['pct_mixed']:.2f}%)",
        )
    df = pd.DataFrame(rows)
    # pooled
    pooled = classify_soft(pix["frac_res_full"].to_numpy(), pix["frac_nres_full"].to_numpy())
    pooled["region"] = "ALL"
    pooled["grid"] = "ae"
    return df, {"pooled": pooled, "sample_features": sample_features}


def patch_google_maps(run_dir: Path, gmaps_dir: Path) -> None:
    """Copy mixed-cell geojson into google maps output and inject layer toggles."""
    if not (gmaps_dir / "index.html").exists():
        log(run_dir, f"Google Maps UI missing at {gmaps_dir} — skip patch")
        return
    src = run_dir / "artifacts" / "geojson"
    dst = gmaps_dir / "geojson" / "mixed_cells"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.geojson"):
        (dst / p.name).write_text(p.read_text())
    # pointer file
    atomic_json(
        gmaps_dir / "mixed_cells_run.json",
        {
            "run_dir": str(run_dir),
            "geojson_dir": "geojson/mixed_cells",
            "updated_at": utc_now(),
        },
    )
    html_path = gmaps_dir / "index.html"
    html = html_path.read_text()
    # idempotent inject markers
    if "lyrAeMixed" in html:
        log(run_dir, "Google Maps UI already has mixed-cell toggles")
        return
    # add checkboxes after footprints toggle
    needle = '<div class="row"><input type="checkbox" id="lyrOverture" checked/><label for="lyrOverture">Footprints</label></div>'
    inject = (
        needle
        + '\n  <div class="row"><input type="checkbox" id="lyrAeMixed"/><label for="lyrAeMixed">AE mixed cells</label></div>'
        + '\n  <div class="row"><input type="checkbox" id="lyrGhslMixed"/><label for="lyrGhslMixed">GHSL mixed cells</label></div>'
    )
    if needle not in html:
        log(run_dir, "Could not find footprints toggle — writing companion HTML snippet instead")
        (run_dir / "artifacts" / "google_maps_layer_snippet.html").write_text(inject)
        return
    html = html.replace(needle, inject, 1)
    # add JS state + loaders — append before </script> of main block
    js = r"""
// --- mixed cell overlays (AE / GHSL) ---
let aeMixedLayer = null, ghslMixedLayer = null;
let aeMixedOn = false, ghslMixedOn = false;
let mixedCache = {};

function mixedStyle(feature) {
  const kind = feature.getProperty("kind");
  const grid = feature.getProperty("grid");
  let fill = "#9B59B6";
  if (kind === "only_res") fill = "#1F6FA8";
  else if (kind === "only_nres") fill = "#C2691A";
  else fill = (grid === "ae") ? "#8E44AD" : "#E74C3C";
  return {
    fillColor: fill,
    fillOpacity: kind === "mixed" ? 0.55 : 0.25,
    strokeColor: fill,
    strokeWeight: 1,
    strokeOpacity: 0.9,
  };
}

async function loadMixed(region, grid) {
  const key = region + "|" + grid;
  if (mixedCache[key]) return mixedCache[key];
  const path = `geojson/mixed_cells/${grid}_${region}.geojson`;
  const resp = await fetch(path);
  if (!resp.ok) throw new Error("missing " + path);
  const geo = await resp.json();
  mixedCache[key] = geo;
  return geo;
}

function ensureMixedLayer(which) {
  if (which === "ae" && !aeMixedLayer) {
    aeMixedLayer = new google.maps.Data({ map: null });
    aeMixedLayer.setStyle(mixedStyle);
    aeMixedLayer.addListener("click", (e) => {
      const p = e.feature;
      infoWindow.setContent(
        `<div style="color:#14231F"><b>AE ${p.getProperty("kind")}</b><br/>` +
        `frac_res=${p.getProperty("frac_res_full") ?? "—"} nres=${p.getProperty("frac_nres_full") ?? "—"}<br/>` +
        `${p.getProperty("pixel_id") ?? ""}</div>`
      );
      infoWindow.setPosition(e.latLng);
      infoWindow.open(map);
    });
  }
  if (which === "ghsl" && !ghslMixedLayer) {
    ghslMixedLayer = new google.maps.Data({ map: null });
    ghslMixedLayer.setStyle(mixedStyle);
    ghslMixedLayer.addListener("click", (e) => {
      const p = e.feature;
      infoWindow.setContent(
        `<div style="color:#14231F"><b>GHSL ${p.getProperty("kind")}</b><br/>` +
        `tile=${p.getProperty("tile_i")} r=${p.getProperty("row")} c=${p.getProperty("col")}</div>`
      );
      infoWindow.setPosition(e.latLng);
      infoWindow.open(map);
    });
  }
}

async function refreshMixedLayers() {
  if (!currentRegion) return;
  ensureMixedLayer("ae");
  ensureMixedLayer("ghsl");
  if (aeMixedOn) {
    try {
      const geo = await loadMixed(currentRegion, "ae");
      aeMixedLayer.forEach((f) => aeMixedLayer.remove(f));
      aeMixedLayer.addGeoJson(geo);
      aeMixedLayer.setMap(map);
    } catch (err) {
      console.warn(err);
      aeMixedLayer.setMap(null);
    }
  } else if (aeMixedLayer) aeMixedLayer.setMap(null);
  if (ghslMixedOn) {
    try {
      const geo = await loadMixed(currentRegion, "ghsl");
      ghslMixedLayer.forEach((f) => ghslMixedLayer.remove(f));
      ghslMixedLayer.addGeoJson(geo);
      ghslMixedLayer.setMap(map);
    } catch (err) {
      console.warn(err);
      ghslMixedLayer.setMap(null);
    }
  } else if (ghslMixedLayer) ghslMixedLayer.setMap(null);
}

document.getElementById("lyrAeMixed").addEventListener("change", (e) => {
  aeMixedOn = e.target.checked;
  refreshMixedLayers();
});
document.getElementById("lyrGhslMixed").addEventListener("change", (e) => {
  ghslMixedOn = e.target.checked;
  refreshMixedLayers();
});
"""
    # Hook into city change: find loadCity or similar
    if "async function loadCity" in html or "function loadCity" in html:
        html = html.replace(
            "lastGeo = geo;",
            "lastGeo = geo;\n  refreshMixedLayers();",
            1,
        )
    # insert JS before final script close - look for hotelsGhslRes listener end area
    marker = "fillLegend();"
    # insert once after first fillLegend in init
    if "// --- mixed cell overlays" not in html:
        # append before </script></body>
        html = html.replace("</script>\n</body>", js + "\n</script>\n</body>", 1)
    html_path.write_text(html)
    # update subtitle
    html2 = html_path.read_text()
    html2 = html2.replace(
        "Track-A footprints · GHSL colors = polygon-zonal BUILT-S majority (green = RES)",
        "Track-A footprints · toggle AE/GHSL mixed cells (overlap both RES+NRES polygons)",
        1,
    )
    html_path.write_text(html2)
    log(run_dir, f"Patched Google Maps UI at {html_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "mixed_cells_ae_ghsl")
    ap.add_argument("--map-sample", type=int, default=MAP_SAMPLE)
    ap.add_argument("--workers", type=int, default=0, help="0 → os.cpu_count()")
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--skip-ghsl", action="store_true")
    ap.add_argument("--skip-map-patch", action="store_true")
    args = ap.parse_args()

    n_cpus = os.cpu_count() or 1
    workers = args.workers or n_cpus
    regions = args.regions or list(REGION_ORDER)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "artifacts/geojson", "checkpoints", "figures", "logs", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_steps = 2 + len(regions)  # ae + ghsl regions + finalize
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total_steps,
        n_cpus=n_cpus,
        workers=workers,
        regions=regions,
        map_sample=args.map_sample,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(
        f"cwd={Path.cwd()}\nargv={sys.argv}\ncpus={n_cpus}\nworkers={workers}\n"
    )
    (run_dir / "environment.txt").write_text(
        f"python={sys.version}\npandas={pd.__version__}\nnumpy={np.__version__}\n"
        f"rasterio={rasterio.__version__}\ncpus={n_cpus}\nworkers={workers}\n"
    )
    log(run_dir, f"start mixed-cell AE vs GHSL · cpus={n_cpus} workers={workers} regions={len(regions)}")

    # --- AE ---
    write_progress(run_dir, stage="ae_soft_gt", completed=0)
    ae_df, ae_extra = score_ae(run_dir, args.map_sample)
    ae_df.to_csv(run_dir / "artifacts" / "ae_mixed_by_region.csv", index=False)
    atomic_json(run_dir / "artifacts" / "ae_mixed_pooled.json", ae_extra["pooled"])
    # write AE geojson per region
    by_reg: dict[str, list] = {r: [] for r in regions}
    for f in ae_extra["sample_features"]:
        reg = f["properties"]["region"]
        if reg in by_reg:
            by_reg[reg].append(f)
    for reg, feats in by_reg.items():
        fc = {"type": "FeatureCollection", "features": feats}
        (run_dir / "artifacts" / "geojson" / f"ae_{reg}.geojson").write_text(json.dumps(fc))
    write_progress(run_dir, stage="ae_done", completed=1, ae_mixed=ae_extra["pooled"]["n_mixed"])
    log(
        run_dir,
        f"AE pooled: gt={ae_extra['pooled']['n_gt']:,} mixed={ae_extra['pooled']['n_mixed']:,} "
        f"({ae_extra['pooled']['pct_mixed']:.2f}%)",
    )

    # --- GHSL ---
    ghsl_rows: list[dict[str, Any]] = []
    if args.skip_ghsl:
        log(run_dir, "skip GHSL")
        write_progress(run_dir, stage="ghsl_skipped", completed=1 + len(regions))
    else:
        write_progress(run_dir, stage="ghsl_prep", completed=1)
        s_paths = sorted(S_TILE_DIR.glob("GHS_BUILT_S_E2018_*.tif"))
        if not s_paths:
            raise FileNotFoundError(f"No GHSL tiles in {S_TILE_DIR}")
        tile_meta_raw = _tile_index(s_paths)
        # make transforms JSON-serializable for workers
        tile_meta = []
        for tm in tile_meta_raw:
            t = tm["transform"]
            tile_meta.append(
                {
                    **{k: tm[k] for k in ("i", "path", "left", "bottom", "right", "top", "width", "height")},
                    "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                }
            )
        log(run_dir, f"GHSL: splitting buildings by metro · {len(s_paths)} tiles · workers={workers}")
        bldg = pd.read_parquet(BUILDINGS, columns=["id", "region", "y_res", "wkt"])
        payloads = []
        for ri, reg in enumerate(regions):
            sub_path = run_dir / "checkpoints" / f"buildings_{reg}.parquet"
            bldg.loc[bldg["region"] == reg, ["id", "y_res", "wkt"]].to_parquet(sub_path, index=False)
            payloads.append(
                {
                    "region": reg,
                    "buildings_path": str(sub_path),
                    "tile_meta": tile_meta,
                    "map_sample": args.map_sample,
                    "seed": SEED,
                    "region_seed": ri * 17 + 3,
                }
            )
        log(run_dir, f"GHSL: launching {len(payloads)} metro jobs")
        done = 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_ghsl_region_worker, p): p["region"] for p in payloads}
            for fut in as_completed(futs):
                reg = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log(run_dir, f"GHSL FAIL {reg}: {e}")
                    (run_dir / "failures" / f"{reg}.txt").write_text(str(e))
                    done += 1
                    write_progress(
                        run_dir,
                        stage=f"ghsl_fail_{reg}",
                        completed=done,
                        failed=done - 1 - len(ghsl_rows),
                    )
                    continue
                feats = result.pop("sample_features")
                ghsl_rows.append(result)
                fc = {"type": "FeatureCollection", "features": feats}
                (run_dir / "artifacts" / "geojson" / f"ghsl_{reg}.geojson").write_text(json.dumps(fc))
                atomic_json(run_dir / "checkpoints" / f"ghsl_{reg}.json", result)
                done += 1
                write_progress(
                    run_dir,
                    stage=f"ghsl_{reg}",
                    completed=done,
                    last_region=reg,
                    n_mixed_region=result["n_mixed"],
                )
                log(
                    run_dir,
                    f"GHSL {reg}: gt={result['n_gt']:,} mixed={result['n_mixed']:,} "
                    f"({result['pct_mixed']:.2f}%)",
                )

    ghsl_df = pd.DataFrame(ghsl_rows)
    if len(ghsl_df):
        ghsl_df.to_csv(run_dir / "artifacts" / "ghsl_mixed_by_region.csv", index=False)
        pooled_g = {
            "region": "ALL",
            "grid": "ghsl",
            "n_gt": int(ghsl_df["n_gt"].sum()),
            "n_only_res": int(ghsl_df["n_only_res"].sum()),
            "n_only_nres": int(ghsl_df["n_only_nres"].sum()),
            "n_mixed": int(ghsl_df["n_mixed"].sum()),
        }
        n = pooled_g["n_gt"]
        pooled_g["pct_only_res"] = 100.0 * pooled_g["n_only_res"] / n if n else float("nan")
        pooled_g["pct_only_nres"] = 100.0 * pooled_g["n_only_nres"] / n if n else float("nan")
        pooled_g["pct_mixed"] = 100.0 * pooled_g["n_mixed"] / n if n else float("nan")
        atomic_json(run_dir / "artifacts" / "ghsl_mixed_pooled.json", pooled_g)
    else:
        pooled_g = {}

    # --- figures ---
    write_progress(run_dir, stage="figures", completed=1 + len(regions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, title in [
        (axes[0], ae_df, "AlphaEarth 10 m cells"),
        (axes[1], ghsl_df if len(ghsl_df) else ae_df.iloc[0:0], "GHSL 10 m cells"),
    ]:
        if df is None or df.empty:
            ax.set_title(title + " (none)")
            continue
        order = [r for r in REGION_ORDER if r in set(df["region"])]
        sub = df.set_index("region").loc[order]
        x = np.arange(len(order))
        ax.bar(x, sub["pct_only_res"], label="only RES", color="#1F6FA8")
        ax.bar(
            x,
            sub["pct_only_nres"],
            bottom=sub["pct_only_res"],
            label="only NRES",
            color="#C2691A",
        )
        bottom = sub["pct_only_res"].to_numpy() + sub["pct_only_nres"].to_numpy()
        ax.bar(x, sub["pct_mixed"], bottom=bottom, label="mixed", color="#8E44AD")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in order], fontsize=8)
        ax.set_ylabel("% of cells with ≥1 Overture polygon")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Cell overlap with Overture RES / NRES polygons", fontsize=13)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mixed_distribution_by_metro.png", dpi=140)
    plt.close(fig)

    # comparison table
    cmp_rows = []
    ae_map = ae_df.set_index("region") if len(ae_df) else None
    gh_map = ghsl_df.set_index("region") if len(ghsl_df) else None
    for reg in REGION_ORDER:
        if reg not in regions:
            continue
        row = {"region": reg}
        if ae_map is not None and reg in ae_map.index:
            row.update({f"ae_{k}": ae_map.loc[reg, k] for k in ("n_gt", "n_mixed", "pct_mixed")})
        if gh_map is not None and reg in gh_map.index:
            row.update({f"ghsl_{k}": gh_map.loc[reg, k] for k in ("n_gt", "n_mixed", "pct_mixed")})
        cmp_rows.append(row)
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(run_dir / "artifacts" / "comparison_by_region.csv", index=False)

    if not args.skip_map_patch:
        patch_google_maps(run_dir, GMAPS)

    summary = {
        "status": "completed",
        "ae_pooled": ae_extra["pooled"],
        "ghsl_pooled": pooled_g,
        "n_regions": len(regions),
        "map_sample_per_region": args.map_sample,
        "runtime_seconds": round(time.time() - t0, 1),
        "artifacts": {
            "ae_by_region": "artifacts/ae_mixed_by_region.csv",
            "ghsl_by_region": "artifacts/ghsl_mixed_by_region.csv",
            "comparison": "artifacts/comparison_by_region.csv",
            "figure": "figures/mixed_distribution_by_metro.png",
            "geojson": "artifacts/geojson/",
        },
        "google_maps": str(GMAPS),
        "serve": "python scripts/serve_google_maps_overture.py → http://127.0.0.1:8770/",
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(
        run_dir,
        status="completed",
        stage="done",
        completed=total_steps,
        total=total_steps,
        percent_complete=100.0,
    )
    log(
        run_dir,
        f"DONE AE mixed={ae_extra['pooled']['pct_mixed']:.2f}% "
        f"GHSL mixed={pooled_g.get('pct_mixed', float('nan')):.2f}% · run_dir={run_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
