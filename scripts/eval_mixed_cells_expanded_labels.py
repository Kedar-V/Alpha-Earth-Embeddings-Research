#!/usr/bin/env python3
"""Rerun AE mixed-cell rates with expanded Overture RES labels.

Hypothesis: Track-A allowlist undercounts RES∩NRES overlaps by excluding
generic `residential` / `terrace` (and subtype=residential) that abut shops.

Protocols (same AE 10 m pixel geometries from week4_pixel_gt):
  A track_a              — current CLASS_TO_LABEL only (baseline)
  B +residential_terrace — add class in {residential, terrace} as RES
  C +subtype_residential — also treat subtype=residential as RES when class
                           is not already an allowlisted NRES class

Does not change model training labels — diagnostics only.

Outputs: outputs/mixed_cells_expanded_labels/<timestamp>/
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

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PIXEL_GT = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
# Current public Overture releases (as of 2026-11); old 2025/2026-08 paths are gone.
RELEASE = "2026-10-21.0"
BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"

# From build_building_type_dataset.CLASS_TO_LABEL
TRACK_A_RES = {
    "house",
    "detached",
    "bungalow",
    "semi",
    "semidetached_house",
    "apartments",
    "dormitory",
}
TRACK_A_NRES = {
    "school",
    "kindergarten",
    "college",
    "university",
    "hospital",
    "clinic",
    "retail",
    "commercial",
    "supermarket",
    "office",
    "hotel",
    "industrial",
    "warehouse",
    "manufacture",
    "church",
    "cathedral",
    "mosque",
    "temple",
    "synagogue",
    "civic",
    "government",
    "public",
    "fire_station",
    "police",
}
EXPAND_RES_CLASS = {"residential", "terrace"}

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


def duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    # DuckDB 1.5+ needs an explicit empty S3 secret for anonymous public buckets.
    con.execute(
        """
        CREATE OR REPLACE SECRET overture_anon (
          TYPE s3,
          KEY_ID '',
          SECRET '',
          REGION 'us-west-2'
        );
        """
    )
    con.execute("SET enable_progress_bar=false;")
    return con


def fetch_region_buildings(region: str, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Pull Track-A classes + residential/terrace from Overture (public S3)."""
    xmin, ymin, xmax, ymax = bbox
    classes = sorted(TRACK_A_RES | TRACK_A_NRES | EXPAND_RES_CLASS)
    class_list = ", ".join(f"'{c}'" for c in classes)
    sql = f"""
    WITH b AS (
      SELECT
        id,
        subtype,
        class,
        ST_X(ST_Centroid(geometry)) AS lon,
        ST_Y(ST_Centroid(geometry)) AS lat,
        ST_AsText(geometry) AS wkt
      FROM read_parquet('{BUILDINGS}', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
        AND bbox.ymin BETWEEN {ymin} AND {ymax}
        AND class IS NOT NULL
        AND class IN ({class_list})
        AND geometry IS NOT NULL
    )
    SELECT id, '{region}' AS region, subtype, class, lon, lat, wkt
    FROM b
    WHERE lon BETWEEN {xmin} AND {xmax}
      AND lat BETWEEN {ymin} AND {ymax}
    """
    con = duck()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def assign_y(df: pd.DataFrame, protocol: str) -> pd.Series:
    cls = df["class"].astype(str)
    sub = df["subtype"].fillna("").astype(str) if "subtype" in df.columns else pd.Series("", index=df.index)
    y = pd.Series(np.nan, index=df.index, dtype=float)
    y = y.mask(cls.isin(TRACK_A_RES), 1.0)
    y = y.mask(cls.isin(TRACK_A_NRES), 0.0)
    if protocol in {"expand_residential_terrace", "expand_subtype_residential"}:
        y = y.mask(cls.isin(EXPAND_RES_CLASS) & y.isna(), 1.0)
    if protocol == "expand_subtype_residential":
        # subtype=residential as RES unless already hard NRES class
        y = y.mask((sub == "residential") & ~cls.isin(TRACK_A_NRES) & y.isna(), 1.0)
    if protocol == "track_a":
        y = y.mask(~cls.isin(TRACK_A_RES | TRACK_A_NRES), np.nan)
    return y


def _empty_stats(protocol: str, n_pixels: int, n_buildings: int = 0) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "n_buildings": n_buildings,
        "n_buildings_res": 0,
        "n_buildings_nres": 0,
        "n_pixels": n_pixels,
        "n_gt": 0,
        "n_mixed": 0,
        "n_only_res": 0,
        "n_only_nres": 0,
        "pct_mixed": float("nan"),
        "pct_only_res": float("nan"),
        "pct_only_nres": float("nan"),
    }


def score_all_protocols(
    pixels: pd.DataFrame,
    buildings: pd.DataFrame,
    protocols: list[str],
    crs: str,
) -> dict[str, dict[str, Any]]:
    """One spatial join, then label/aggregate per protocol (vectorized)."""
    n_pixels = int(len(pixels))
    if buildings.empty or pixels.empty:
        return {p: _empty_stats(p, n_pixels) for p in protocols}

    b_geoms = []
    for w in buildings["wkt"].tolist():
        g = shapely_wkt.loads(w)
        if not g.is_valid:
            g = g.buffer(0)
        b_geoms.append(g)
    bg = gpd.GeoDataFrame(
        {
            "class": buildings["class"].astype(str).to_numpy(),
            "subtype": (
                buildings["subtype"].fillna("").astype(str).to_numpy()
                if "subtype" in buildings.columns
                else np.array([""] * len(buildings))
            ),
        },
        geometry=b_geoms,
        crs="EPSG:4326",
    ).to_crs(crs)

    p_geoms = [shapely_wkt.loads(w) for w in pixels["pixel_wkt"].tolist()]
    pg = gpd.GeoDataFrame(
        {"pixel_id": pixels["pixel_id"].astype(str).to_numpy()},
        geometry=p_geoms,
        crs="EPSG:4326",
    ).to_crs(crs)

    # Restrict to buildings that are labeled under at least one protocol
    any_label = bg["class"].isin(TRACK_A_RES | TRACK_A_NRES | EXPAND_RES_CLASS) | (
        bg["subtype"] == "residential"
    )
    bg = bg[any_label].copy()
    if bg.empty:
        return {p: _empty_stats(p, n_pixels) for p in protocols}

    joined = gpd.sjoin(pg, bg, how="inner", predicate="intersects")
    if joined.empty:
        return {p: _empty_stats(p, n_pixels, n_buildings=int(len(bg))) for p in protocols}

    hit = pd.DataFrame(
        {
            "pixel_id": joined["pixel_id"].to_numpy(),
            "class": joined["class"].to_numpy(),
            "subtype": joined["subtype"].to_numpy(),
        }
    )

    out: dict[str, dict[str, Any]] = {}
    for protocol in protocols:
        y = assign_y(hit, protocol)
        labeled = hit.assign(y_res=y)
        labeled = labeled[labeled["y_res"].isin([0.0, 1.0])]
        # building inventory under this protocol
        by = assign_y(bg[["class", "subtype"]], protocol)
        n_b = int(by.isin([0.0, 1.0]).sum())
        n_b_res = int((by == 1.0).sum())
        n_b_nres = int((by == 0.0).sum())
        if labeled.empty:
            out[protocol] = _empty_stats(protocol, n_pixels, n_buildings=n_b)
            out[protocol]["n_buildings_res"] = n_b_res
            out[protocol]["n_buildings_nres"] = n_b_nres
            continue
        g = labeled.groupby("pixel_id")["y_res"].agg(["min", "max"])
        has_res = g["max"] >= 0.5
        has_nres = g["min"] < 0.5
        n_mixed = int((has_res & has_nres).sum())
        n_only_res = int((has_res & ~has_nres).sum())
        n_only_nres = int((~has_res & has_nres).sum())
        n_gt = n_mixed + n_only_res + n_only_nres
        out[protocol] = {
            "protocol": protocol,
            "n_buildings": n_b,
            "n_buildings_res": n_b_res,
            "n_buildings_nres": n_b_nres,
            "n_pixels": n_pixels,
            "n_gt": n_gt,
            "n_mixed": n_mixed,
            "n_only_res": n_only_res,
            "n_only_nres": n_only_nres,
            "pct_mixed": 100.0 * n_mixed / n_gt if n_gt else float("nan"),
            "pct_only_res": 100.0 * n_only_res / n_gt if n_gt else float("nan"),
            "pct_only_nres": 100.0 * n_only_nres / n_gt if n_gt else float("nan"),
        }
    return out


def _fetch_one(payload: dict[str, Any]) -> tuple[str, int, int, int, str | None]:
    region = payload["region"]
    bpath = Path(payload["buildings_path"])
    if payload.get("skip_fetch") and bpath.exists():
        raw = pd.read_parquet(bpath, columns=["class"])
        return (
            region,
            int(len(raw)),
            int((raw["class"] == "residential").sum()),
            int((raw["class"] == "terrace").sum()),
            None,
        )
    try:
        raw = fetch_region_buildings(region, tuple(payload["bbox"]))
        raw["id"] = raw["id"].astype(str)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(bpath, index=False)
        return (
            region,
            int(len(raw)),
            int((raw["class"] == "residential").sum()),
            int((raw["class"] == "terrace").sum()),
            None,
        )
    except Exception as e:
        fail = Path(payload["fail_path"])
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(str(e))
        return region, 0, 0, 0, str(e)


def _region_worker(payload: dict[str, Any]) -> dict[str, Any]:
    region = payload["region"]
    buildings = pd.read_parquet(payload["buildings_path"])
    pixels = pd.read_parquet(payload["pixels_path"])
    crs = payload["crs"]
    protocols = payload["protocols"]
    scored = score_all_protocols(pixels, buildings, protocols, crs)
    cls = buildings["class"].astype(str)
    return {
        "region": region,
        "crs": crs,
        "protocols": scored,
        "n_fetched": int(len(buildings)),
        "n_class_residential": int((cls == "residential").sum()),
        "n_class_terrace": int((cls == "terrace").sum()),
        "n_track_a": int(cls.isin(TRACK_A_RES | TRACK_A_NRES).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "mixed_cells_expanded_labels")
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip-fetch", action="store_true", help="Reuse checkpoints/buildings_*.parquet")
    args = ap.parse_args()

    regions = args.regions or list(US_REGIONS.keys())
    # Cap scoring workers — each region holds a large GeoDataFrame (LA ~800k pixels)
    workers = args.workers or min(len(regions), max(1, (os.cpu_count() or 4) // 4), 6)
    protocols = ["track_a", "expand_residential_terrace", "expand_subtype_residential"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = len(regions) + len(regions) + 1  # fetch + score + finalize
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=total,
        n_cpus=os.cpu_count(),
        workers=workers,
        regions=regions,
        protocols=protocols,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"argv={sys.argv}\nworkers={workers}\nregions={regions}\n")
    log(run_dir, f"start expanded-label mixed-cell rerun · regions={len(regions)} workers={workers}")

    # --- fetch buildings (parallel S3 pulls) ---
    write_progress(run_dir, stage="fetch_overture", completed=0)
    pix_all = pd.read_parquet(PIXEL_GT, columns=["pixel_id", "pixel_wkt", "region", "crs"])
    for region in regions:
        ppath = run_dir / "checkpoints" / f"pixels_{region}.parquet"
        sub_pix = pix_all[pix_all["region"] == region][["pixel_id", "pixel_wkt", "crs"]].copy()
        sub_pix.to_parquet(ppath, index=False)

    fetch_workers = min(len(regions), os.cpu_count() or 1)
    fetch_payloads = [
        {
            "region": region,
            "bbox": US_REGIONS[region],
            "buildings_path": str(run_dir / "checkpoints" / f"buildings_{region}.parquet"),
            "fail_path": str(run_dir / "failures" / f"fetch_{region}.txt"),
            "skip_fetch": bool(args.skip_fetch),
        }
        for region in regions
    ]
    fetch_done = 0
    with ProcessPoolExecutor(max_workers=fetch_workers) as pool:
        futs = {pool.submit(_fetch_one, p): p["region"] for p in fetch_payloads}
        for fut in as_completed(futs):
            region, n, n_res, n_ter, err = fut.result()
            fetch_done += 1
            if err:
                log(run_dir, f"FETCH FAIL {region}: {err}")
            else:
                log(run_dir, f"fetched {region}: n={n:,} residential={n_res:,} terrace={n_ter:,}")
            write_progress(
                run_dir,
                stage=f"fetched_{region}",
                completed=fetch_done,
                fetch_done=fetch_done,
                fetch_total=len(regions),
            )

    write_progress(run_dir, stage="score", completed=len(regions))

    payloads = []
    for region in regions:
        bpath = run_dir / "checkpoints" / f"buildings_{region}.parquet"
        ppath = run_dir / "checkpoints" / f"pixels_{region}.parquet"
        if not bpath.exists() or not ppath.exists():
            continue
        # CRS from pixel file mode
        crs = str(pd.read_parquet(ppath, columns=["crs"])["crs"].mode().iloc[0])
        payloads.append(
            {
                "region": region,
                "buildings_path": str(bpath),
                "pixels_path": str(ppath),
                "crs": crs,
                "protocols": protocols,
            }
        )

    results = []
    done = 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_region_worker, p): p["region"] for p in payloads}
        for fut in as_completed(futs):
            region = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                log(run_dir, f"SCORE FAIL {region}: {e}")
                (run_dir / "failures" / f"score_{region}.txt").write_text(str(e))
                done += 1
                write_progress(run_dir, stage=f"fail_{region}", completed=done)
                continue
            results.append(res)
            atomic_json(run_dir / "checkpoints" / f"scores_{region}.json", res)
            for proto, stats in res["protocols"].items():
                log(
                    run_dir,
                    f"{region} {proto}: buildings={stats['n_buildings']:,} "
                    f"mixed={stats['n_mixed']:,}/{stats['n_gt']:,} ({stats['pct_mixed']:.2f}%)",
                )
            done += 1
            write_progress(run_dir, stage=f"scored_{region}", completed=done, last_region=region)

    # flatten table
    rows = []
    for res in results:
        for proto, stats in res["protocols"].items():
            rows.append({"region": res["region"], **stats})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "artifacts" / "mixed_by_region_protocol.csv", index=False)

    # pooled
    pooled_rows = []
    for proto in protocols:
        sub = df[df["protocol"] == proto]
        n_gt = int(sub["n_gt"].sum())
        n_mixed = int(sub["n_mixed"].sum())
        pooled_rows.append(
            {
                "protocol": proto,
                "n_gt": n_gt,
                "n_mixed": n_mixed,
                "pct_mixed": 100.0 * n_mixed / n_gt if n_gt else float("nan"),
                "n_buildings": int(sub["n_buildings"].sum()),
                "n_buildings_res": int(sub["n_buildings_res"].sum()),
                "n_buildings_nres": int(sub["n_buildings_nres"].sum()),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(run_dir / "artifacts" / "mixed_pooled_by_protocol.csv", index=False)
    atomic_json(run_dir / "artifacts" / "mixed_pooled_by_protocol.json", pooled.to_dict(orient="records"))

    # NYC focus
    nyc = df[df["region"] == "NYC_NE"].copy()
    if len(nyc):
        nyc.to_csv(run_dir / "artifacts" / "nyc_focus.csv", index=False)
        log(run_dir, "NYC focus:")
        for _, r in nyc.iterrows():
            log(
                run_dir,
                f"  {r.protocol}: mixed={r.pct_mixed:.2f}% "
                f"({int(r.n_mixed):,}/{int(r.n_gt):,}) buildings_res/nres="
                f"{int(r.n_buildings_res):,}/{int(r.n_buildings_nres):,}",
            )

    # figure: NYC + pooled bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(nyc):
        axes[0].bar(nyc["protocol"], nyc["pct_mixed"], color=["#1F6FA8", "#8E44AD", "#C2691A"])
        axes[0].axhline(10, color="#333", ls="--", lw=1, label="10% expectation")
        axes[0].set_ylabel("% mixed AE cells")
        axes[0].set_title("NYC_NE")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].legend(fontsize=8)
    axes[1].bar(pooled["protocol"], pooled["pct_mixed"], color=["#1F6FA8", "#8E44AD", "#C2691A"])
    axes[1].axhline(10, color="#333", ls="--", lw=1)
    axes[1].set_title("All metros pooled")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("AE 10 m mixed rate vs expanded Overture RES labels")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "mixed_pct_by_protocol.png", dpi=140)
    plt.close(fig)

    summary = {
        "status": "completed",
        "protocols": protocols,
        "pooled": pooled.to_dict(orient="records"),
        "nyc": nyc.to_dict(orient="records") if len(nyc) else [],
        "runtime_seconds": round(time.time() - t0, 1),
        "note": "Adds class=residential/terrace (and subtype=residential) as RES; NRES unchanged.",
        "artifacts": {
            "by_region": "artifacts/mixed_by_region_protocol.csv",
            "pooled": "artifacts/mixed_pooled_by_protocol.csv",
            "nyc": "artifacts/nyc_focus.csv",
            "figure": "figures/mixed_pct_by_protocol.png",
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(run_dir, status="completed", stage="done", completed=total, total=total, percent_complete=100.0)
    log(run_dir, f"DONE · run_dir={run_dir}")
    for r in pooled.itertuples():
        log(run_dir, f"pooled {r.protocol}: mixed={r.pct_mixed:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
