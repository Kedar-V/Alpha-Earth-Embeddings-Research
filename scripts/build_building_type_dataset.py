#!/usr/bin/env python3
"""Build the canonical labeled building-type + AlphaEarth dataset.

Saves under outputs/building_type_dataset/:
  - dataset.parquet          full table (meta + WKT + 64-D bands)
  - dataset_browse.csv       lightweight view (no WKT / embeddings)
  - labels_only.parquet      footprints + labels before embeddings
  - manifest.json            schema, taxonomy, provenance
  - gaps.json / GAPS.md      what is missing vs the research brief
  - coverage_by_region.csv   counts / fill summaries
  - label_counts.csv

Usage (from repo root):
  python scripts/build_building_type_dataset.py
  python scripts/build_building_type_dataset.py --rebuild-labels
  python scripts/build_building_type_dataset.py --rebuild-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "building_type_dataset"
LEGACY = ROOT / "outputs" / "sota_building_type"

RELEASE = "2026-08-19.0"
BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
EMBED_YEAR = 2024
# Natural Track-A mix (no per-class polygon cap). Optional random cap via --max-per-region.
MAX_PER_REGION: int | None = None
EE_BATCH = 400
RANDOM_STATE = 42
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-kedarvaidya0504")
BANDS = [f"A{i:02d}" for i in range(64)]
# RES = single_family + multi_family (resi_def_v1); balance RES-heavy pixels later by undersampling.

# Research-brief categories → Overture class members (high-precision only)
CLASS_TO_LABEL: dict[str, str] = {
    # Single-family residential
    "house": "single_family",
    "detached": "single_family",
    "bungalow": "single_family",
    "semi": "single_family",
    "semidetached_house": "single_family",
    # Multi-family residential
    "apartments": "multi_family",
    "dormitory": "multi_family",
    # Schools
    "school": "education",
    "kindergarten": "education",
    "college": "education",
    "university": "education",
    # Hospitals
    "hospital": "medical",
    "clinic": "medical",
    # Commercial or office
    "retail": "retail",
    "commercial": "retail",
    "supermarket": "retail",
    "office": "office",
    # Additional (kept when label quality is high)
    "hotel": "hotel",
    "industrial": "industrial",
    "warehouse": "industrial",
    "manufacture": "industrial",
    "church": "religious",
    "cathedral": "religious",
    "mosque": "religious",
    "temple": "religious",
    "synagogue": "religious",
    "civic": "civic",
    "government": "civic",
    "public": "civic",
    "fire_station": "civic",
    "police": "civic",
}

LABEL_ORDER = [
    "single_family",
    "multi_family",
    "education",
    "medical",
    "retail",
    "office",
    "hotel",
    "industrial",
    "religious",
    "civic",
]

# Brief-facing rollup (commercial ∪ office)
BRIEF_ROLLUP = {
    "single_family": "Single-family residential",
    "multi_family": "Multi-family residential",
    "education": "Schools",
    "medical": "Hospitals",
    "retail": "Commercial or office",
    "office": "Commercial or office",
    "hotel": "Additional — hotel",
    "industrial": "Additional — industrial",
    "religious": "Additional — religious",
    "civic": "Additional — civic",
}

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


def duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET enable_progress_bar=false;")
    return con


def sample_region(
    con: duckdb.DuckDBPyConnection,
    name: str,
    bbox: tuple[float, float, float, float],
    max_n: int | None = None,
) -> pd.DataFrame:
    """All Track-A typed buildings in AOI (natural class mix). Optional random cap.

    Does **not** stratify by class / y_res. RES-majority is expected (~90% count in
    these AOIs). Downstream pixel balance undersamples RES, not NRES.
    """
    xmin, ymin, xmax, ymax = bbox
    class_list = ", ".join(f"'{c}'" for c in sorted(CLASS_TO_LABEL))
    limit_clause = f"WHERE rn <= {int(max_n)}" if max_n is not None else ""
    sql = f"""
    WITH b AS (
      SELECT
        id,
        subtype,
        class,
        sources,
        height,
        num_floors,
        ST_X(ST_Centroid(geometry)) AS lon,
        ST_Y(ST_Centroid(geometry)) AS lat,
        ST_AsText(geometry) AS wkt,
        ST_Area(ST_Transform(geometry, 'EPSG:3857')) AS area_m2_approx
      FROM read_parquet('{BUILDINGS}', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
        AND bbox.ymin BETWEEN {ymin} AND {ymax}
        AND class IS NOT NULL
        AND class IN ({class_list})
        AND geometry IS NOT NULL
    ),
    filtered AS (
      SELECT * FROM b
      WHERE lon BETWEEN {xmin} AND {xmax}
        AND lat BETWEEN {ymin} AND {ymax}
    ),
    ranked AS (
      SELECT *, row_number() OVER (ORDER BY hash(id)) AS rn
      FROM filtered
    )
    SELECT
      id, '{name}' AS region, subtype, class, sources,
      height, num_floors, lon, lat, wkt, area_m2_approx
    FROM ranked
    {limit_clause}
    """
    return con.execute(sql).df()


def primary_source(sources) -> str | None:
    try:
        if sources is None:
            return None
        if isinstance(sources, (list, np.ndarray)) and len(sources):
            s0 = sources[0]
            if isinstance(s0, dict):
                return s0.get("dataset")
            if hasattr(s0, "get"):
                return s0.get("dataset")
        return None
    except Exception:
        return None


def build_labels(rebuild: bool, max_per_region: int | None = MAX_PER_REGION) -> pd.DataFrame:
    cache = OUT_DIR / "labels_only.parquet"
    legacy = LEGACY / "buildings_labeled.parquet"
    if cache.exists() and not rebuild:
        print("Loaded labels:", cache)
        return pd.read_parquet(cache)
    if legacy.exists() and not rebuild:
        print("Promoting labels from", legacy)
        raw = pd.read_parquet(legacy)
    else:
        con = duck()
        frames = []
        for name, bbox in US_REGIONS.items():
            cap = f"max={max_per_region}" if max_per_region else "all typed"
            print(f"Sampling · {name} ({cap}, unstratified) ...", flush=True)
            part = sample_region(con, name, bbox, max_per_region)
            n_res = int(part["class"].isin(["house", "detached", "bungalow", "semi", "semidetached_house", "apartments", "dormitory"]).sum())
            print(f"  → {len(part):,} buildings · ~{100 * n_res / max(len(part), 1):.0f}% RES classes", flush=True)
            frames.append(part)
        raw = pd.concat(frames, ignore_index=True)

    labels = raw.copy()
    labels["id"] = labels["id"].astype(str)
    labels["label"] = labels["class"].map(CLASS_TO_LABEL)
    labels = labels[labels["label"].notna()].copy()
    labels["brief_category"] = labels["label"].map(BRIEF_ROLLUP)
    labels["y_res"] = labels["label"].isin(["single_family", "multi_family"]).astype(int)
    labels["label_definition"] = "resi_def_v1"
    labels["source_primary"] = labels["sources"].map(primary_source) if "sources" in labels.columns else None
    labels["embed_year"] = EMBED_YEAR
    labels["label_source"] = "overture_buildings.class"
    labels.to_parquet(cache, index=False)
    print("Wrote", cache, "· n =", len(labels))
    return labels


def init_ee(project: str) -> None:
    import ee

    if not project or project == "YOUR_EE_PROJECT_ID":
        raise ValueError("Set EE_PROJECT to your Earth Engine Cloud project id")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def extract_embeddings(df: pd.DataFrame, year: int = EMBED_YEAR, batch_size: int = EE_BATCH) -> pd.DataFrame:
    import ee

    init_ee(EE_PROJECT)
    img = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
        .select(BANDS)
    )
    rows: list[dict] = []
    n = len(df)
    for start in range(0, n, batch_size):
        batch = df.iloc[start : start + batch_size]
        feats = []
        for _, r in batch.iterrows():
            feats.append(
                ee.Feature(
                    ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                    {"id": str(r["id"])},
                )
            )
        fc = ee.FeatureCollection(feats)
        sampled = img.sampleRegions(collection=fc, scale=10, geometries=False)
        for f in sampled.getInfo()["features"]:
            props = f["properties"]
            row = {"id": str(props["id"])}
            for b in BANDS:
                row[b] = props.get(b)
            rows.append(row)
        print(f"  embeddings {min(start + batch_size, n)}/{n}", flush=True)
        time.sleep(0.2)
    emb = pd.DataFrame(rows).dropna(subset=BANDS).reset_index(drop=True)
    return emb


def attach_embeddings(labels: pd.DataFrame, rebuild: bool) -> pd.DataFrame:
    cache = OUT_DIR / "dataset.parquet"
    legacy = LEGACY / "dataset.parquet"
    if cache.exists() and not rebuild:
        print("Loaded dataset:", cache)
        return pd.read_parquet(cache)
    if legacy.exists() and not rebuild:
        print("Promoting embeddings from", legacy)
        ds = pd.read_parquet(legacy)
        # Re-derive brief fields for canonical schema
        ds = ds.copy()
        ds["id"] = ds["id"].astype(str)
        if "brief_category" not in ds.columns:
            ds["brief_category"] = ds["label"].map(BRIEF_ROLLUP)
        if "embed_year" not in ds.columns:
            ds["embed_year"] = EMBED_YEAR
        if "label_source" not in ds.columns:
            ds["label_source"] = "overture_buildings.class"
        return ds

    print("Extracting AlphaEarth embeddings via Earth Engine ...")
    emb = extract_embeddings(labels)
    ds = labels.merge(emb, on="id", how="inner")
    print("Joined n =", len(ds), "(dropped", len(labels) - len(ds), "without embeddings)")
    return ds


def write_gaps(ds: pd.DataFrame) -> dict:
    label_counts = ds["label"].value_counts().to_dict()
    brief_counts = ds["brief_category"].value_counts().to_dict() if "brief_category" in ds.columns else {}

    gaps = {
        "dataset_version": "building_type_v1",
        "present": {
            "labels": {
                "source": "Overture Maps buildings.class (OSM-derived high-precision tags)",
                "release": RELEASE,
                "categories_research_brief": [
                    "Single-family residential",
                    "Multi-family residential",
                    "Schools",
                    "Hospitals",
                    "Commercial or office buildings",
                    "Additional: hotel, industrial, religious, civic",
                ],
                "label_counts": label_counts,
                "brief_category_counts": brief_counts,
            },
            "geometry": {
                "footprints": "Overture building polygons (WKT)",
                "centroid": "lon/lat from footprint centroid",
            },
            "embeddings": {
                "product": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                "year": EMBED_YEAR,
                "dim": 64,
                "sample": "centroid Reducer / sampleRegions @ 10 m",
            },
            "weak_3d_priors": {
                "height": float(ds["height"].notna().mean()) if "height" in ds.columns else None,
                "num_floors": float(ds["num_floors"].notna().mean()) if "num_floors" in ds.columns else None,
            },
            "geography": {
                "scope": "10 US metro compact AOIs (not national / not global)",
                "regions": sorted(ds["region"].unique().tolist()),
            },
        },
        "missing": [
            {
                "id": "cadastral_parcels",
                "item": "Cadastral / parcel land-use attributes",
                "status": "missing",
                "why": "Research brief allows cadastral data; not joined. Would improve commercial vs residential in US cities with open assessor data.",
                "next": "Join city/county parcel LU codes (e.g. NYC PLUTO, LA County parcels) by spatial intersection.",
            },
            {
                "id": "zoning_landuse",
                "item": "Zoning / land-use layers (non-OSM)",
                "status": "missing",
                "why": "No municipal zoning or NLCD/local LU attached; labels are building-use tags only.",
                "next": "Attach zoning district or land-use raster/polygon as weak labels outside OSM-dense cores.",
            },
            {
                "id": "places_poi_join",
                "item": "Overture Places POI spatial join",
                "status": "missing",
                "why": "Schools/hospitals are sparse as building.class; Places often denser (noisy GT).",
                "next": "Point-in-polygon / nearest-building join for education & health_care taxonomy.",
            },
            {
                "id": "ghs_obat",
                "item": "GHS-OBAT residential / non-residential weak labels",
                "status": "missing",
                "why": "Useful for global binary expansion; not in this multi-class gold set.",
                "next": "Optional weak-label track; keep separate from high-precision class GT.",
            },
            {
                "id": "street_view_vhr",
                "item": "Street-view / VHR image features",
                "status": "missing",
                "why": "Dataset is multi-modal-ready (WKT kept) but only AlphaEarth features are filled.",
                "next": "Join GSV / satellite chips keyed by id for Protocol A SOTA comparison.",
            },
            {
                "id": "neighbor_graph",
                "item": "Building-neighbor / street graph features",
                "status": "missing",
                "why": "SOTA multi-modal papers use graphs; not computed here.",
                "next": "kNN footprint graph or OSM street network features.",
            },
            {
                "id": "footprint_mean_pooling",
                "item": "Footprint-mean AlphaEarth pooling",
                "status": "missing",
                "why": "Current vectors are centroid-only; small buildings get 1–4 pixels of context.",
                "next": "Mean-pool embedding image over polygon (+ optional 50/200 m buffers).",
            },
            {
                "id": "global_south_coverage",
                "item": "Global / Global-South labeled coverage",
                "status": "missing",
                "why": "US metros only. Overture class fill collapses in many African extracts.",
                "next": "Europe OSM-rich cities for transfer; Places/GHS for weak labels elsewhere.",
            },
            {
                "id": "ambiguous_osm_classes",
                "item": "Ambiguous OSM classes as gold labels",
                "status": "intentionally_excluded",
                "why": "class=residential, terrace, bare building=yes are ambiguous for SF vs MF / use.",
                "next": "Keep excluded from gold multi-class; optional weak-label bucket later.",
            },
            {
                "id": "full_metro_extents",
                "item": "Full metro / state extents",
                "status": "missing",
                "why": "AOIs are compact downtown boxes for cheap queries + denser tags.",
                "next": "Expand downtown AOIs to full metro extents once pipeline is stable.",
            },
            {
                "id": "temporal_embeddings",
                "item": "Multi-year AlphaEarth stack (2017–2024)",
                "status": "missing",
                "why": "Only calendar year 2024 attached.",
                "next": "Add prior years for change / stability features.",
            },
            {
                "id": "building_parts_3d",
                "item": "building_part geometry / detailed 3D",
                "status": "missing",
                "why": "has_parts not joined; height/floors sparse.",
                "next": "Optional join to theme=buildings/type=building_part.",
            },
        ],
        "imbalance_notes": {
            "smallest_classes": sorted(label_counts, key=label_counts.get)[:3],
            "largest_classes": sorted(label_counts, key=label_counts.get, reverse=True)[:3],
            "note": (
                "Unstratified Track-A sample — RES dominates by count (~90% in these AOIs). "
                "Do not rebalance at polygon level; undersample RES pixels later for 1:1 training."
            ),
        },
    }
    return gaps


def gaps_markdown(gaps: dict) -> str:
    lines = [
        "# Building-type dataset — what's present vs missing",
        "",
        f"**Version:** `{gaps['dataset_version']}`  ",
        f"**Overture release:** `{gaps['present']['labels']['release']}`  ",
        f"**Embeddings:** AlphaEarth annual {gaps['present']['embeddings']['year']} (64-D centroid)",
        "",
        "## Present",
        "",
        "- High-precision Overture `class` → study labels (SF, MF, schools, hospitals, commercial/office, + extras)",
        "- Building footprints (WKT) + centroids",
        "- AlphaEarth Satellite Embedding V1 associated per building",
        f"- Geography: {gaps['present']['geography']['scope']}",
        "",
        "### Label counts",
        "",
        "| label | n |",
        "|---|---:|",
    ]
    for k, v in sorted(gaps["present"]["labels"]["label_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v} |")
    lines += [
        "",
        "## Missing (tracked gaps)",
        "",
        "| ID | Item | Status | Why / next |",
        "|---|---|---|---|",
    ]
    for g in gaps["missing"]:
        lines.append(
            f"| `{g['id']}` | {g['item']} | {g['status']} | {g['why']} → {g['next']} |"
        )
    lines += [
        "",
        "## How to use",
        "",
        "```python",
        "import pandas as pd",
        "df = pd.read_parquet('outputs/building_type_dataset/dataset.parquet')",
        "X = df[[f'A{i:02d}' for i in range(64)]].to_numpy()",
        "y = df['label']",
        "```",
        "",
    ]
    return "\n".join(lines)


def finalize(ds: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    if "brief_category" not in ds.columns:
        ds["brief_category"] = ds["label"].map(BRIEF_ROLLUP)
    if "split_seed" not in ds.columns:
        ds["split_seed"] = rng.integers(0, 10_000, size=len(ds))

    meta_cols = [
        "id",
        "region",
        "lon",
        "lat",
        "wkt",
        "subtype",
        "class",
        "label",
        "brief_category",
        "y_res",
        "height",
        "num_floors",
        "area_m2_approx",
        "source_primary",
        "embed_year",
        "label_source",
        "split_seed",
    ]
    meta_cols = [c for c in meta_cols if c in ds.columns]
    export = ds[meta_cols + [b for b in BANDS if b in ds.columns]].copy()
    assert export[BANDS].notna().all().all(), "Null embedding bands present"

    dataset_path = OUT_DIR / "dataset.parquet"
    export.to_parquet(dataset_path, index=False)
    browse = export.drop(columns=["wkt"] + BANDS, errors="ignore")
    browse.to_csv(OUT_DIR / "dataset_browse.csv", index=False)

    label_counts = (
        export["label"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int).rename("n").to_frame()
    )
    label_counts.to_csv(OUT_DIR / "label_counts.csv")

    coverage = (
        export.groupby("region")
        .agg(
            n=("id", "size"),
            n_labels=("label", "nunique"),
            pct_residential=("y_res", "mean"),
            pct_has_height=("height", lambda s: float(s.notna().mean()) if s.notna().any() or True else 0.0),
        )
        .reset_index()
    )
    coverage["pct_residential"] = 100 * coverage["pct_residential"]
    coverage["pct_has_height"] = 100 * export.groupby("region")["height"].apply(
        lambda s: s.notna().mean()
    ).to_numpy()
    coverage.to_csv(OUT_DIR / "coverage_by_region.csv", index=False)

    gaps = write_gaps(export)
    with open(OUT_DIR / "gaps.json", "w") as f:
        json.dump(gaps, f, indent=2)
    (OUT_DIR / "GAPS.md").write_text(gaps_markdown(gaps))

    manifest = {
        "name": "building_type_v1",
        "label_definition": "resi_def_v1",
        "label_definition_doc": "docs/residential_definition_v1.md",
        "purpose": (
            "Labeled functional building-type dataset (research brief) with AlphaEarth "
            "64-D annual embeddings for downstream classification / carbon archetypes."
        ),
        "overture_release": RELEASE,
        "label_source": "Overture buildings.class (OSM-derived); ambiguous classes excluded",
        "sampling": (
            "Unstratified Track-A class in AOI (natural mix). No PARTITION BY class / y_res. "
            "Optional --max-per-region is a random hash cap, still unstratified."
        ),
        "y_res_rule": "resi_def_v1 Building GT: 1 iff label in {single_family, multi_family}",
        "taxonomy": LABEL_ORDER,
        "brief_rollup": BRIEF_ROLLUP,
        "class_to_label": CLASS_TO_LABEL,
        "embedding": {
            "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "year": EMBED_YEAR,
            "bands": BANDS,
            "sample": "centroid sampleRegions @ 10m",
        },
        "n_buildings": int(len(export)),
        "n_regions": int(export["region"].nunique()),
        "regions": sorted(export["region"].unique().tolist()),
        "label_counts": export["label"].value_counts().to_dict(),
        "brief_category_counts": export["brief_category"].value_counts().to_dict(),
        "paths": {
            "dataset": str(dataset_path.relative_to(ROOT)),
            "browse": "outputs/building_type_dataset/dataset_browse.csv",
            "gaps": "outputs/building_type_dataset/GAPS.md",
        },
        "not_in_scope_see_gaps": [g["id"] for g in gaps["missing"]],
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Building-type dataset ready ===")
    print("n =", len(export))
    print(export["label"].value_counts().reindex(LABEL_ORDER).to_string())
    print("\nWrote:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name:30s} {p.stat().st_size:10d} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument(
        "--max-per-region",
        type=int,
        default=None,
        help="Optional random cap per metro (unstratified). Default: all Track-A typed buildings.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Stop after writing labels_only.parquet (skip centroid EE embeddings).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = build_labels(rebuild=args.rebuild_labels, max_per_region=args.max_per_region)
    pct_res = 100.0 * float(labels["y_res"].mean()) if "y_res" in labels.columns else float("nan")
    print(f"labels_only n={len(labels):,}  pct_res={pct_res:.1f}%", flush=True)
    if args.labels_only and not args.rebuild_embeddings:
        print("Stopped after labels (--labels-only). Bbox-grid / pixel GT can use labels_only.parquet.")
        return
    ds = attach_embeddings(labels, rebuild=args.rebuild_embeddings)
    finalize(ds)


if __name__ == "__main__":
    main()
