#!/usr/bin/env python3
"""Generate notebooks/04_sota_comparable_building_type_dataset.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "04_sota_comparable_building_type_dataset.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


SAMPLE_SQL = '''
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("SET s3_region='us-west-2';")
con.execute("SET s3_access_key_id='';")
con.execute("SET s3_secret_access_key='';")
print("DuckDB ready ·", RELEASE)


def sample_region(name: str, bbox: tuple[float, float, float, float], max_n: int | None = None) -> pd.DataFrame:
    """Pull high-precision Overture buildings whose class maps into STUDY_TAXONOMY.

    Unstratified natural mix (no PARTITION BY class). Optional random cap via max_n.
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
      id,
      '{name}' AS region,
      subtype,
      class,
      sources,
      height,
      num_floors,
      lon,
      lat,
      wkt,
      area_m2_approx
    FROM ranked
    {limit_clause}
    """
    return con.execute(sql).df()


cache_labels = OUT_DIR / "buildings_labeled.parquet"
if cache_labels.exists():
    raw = pd.read_parquet(cache_labels)
    print("Loaded cached labels:", cache_labels)
else:
    frames = []
    for name, bbox in US_REGIONS.items():
        print(f"Sampling · {name} (unstratified) ...", flush=True)
        try:
            frames.append(sample_region(name, bbox, MAX_PER_REGION))
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(cache_labels, index=False)
    print("Wrote", cache_labels)

print("raw n =", len(raw))
raw.head()
'''

MAP_CELL = r'''
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(-128, -66)
ax.set_ylim(23, 50)
ax.set_aspect(1.3)
ax.set_facecolor("#f4f7fb")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_title("SOTA-comparable dataset · buildings per region")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

sizes = counts_by_region.reindex(region_meta["region"]).fillna(0)["n"]
ax.scatter(
    region_meta["lon"],
    region_meta["lat"],
    s=sizes.to_numpy() * 0.8,
    c="#4C78A8",
    edgecolors="k",
    linewidths=0.5,
    zorder=3,
)
for _, r in region_meta.iterrows():
    n = int(counts_by_region.loc[r["region"], "n"]) if r["region"] in counts_by_region.index else 0
    ax.annotate(f"{r['region']}\nn={n}", (r["lon"], r["lat"]), textcoords="offset points", xytext=(6, 6), fontsize=8)

fig.tight_layout()
map_path = OUT_DIR / "dataset_coverage_map.png"
fig.savefig(map_path, dpi=150)
print("Saved", map_path)
plt.show()
'''


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# 04 — SOTA-comparable building-type dataset

**Why this notebook.** True building-type SOTA (often **86–96%** accuracy on 10–12 classes) uses
**street-view / VHR imagery + multi-modal graphs**, not AlphaEarth alone. Notebook 02’s binary
residential probe (~83% pooled / ~77% leave-one-region-out) sits below that for expected reasons
(64-D @ 10 m + logistic regression).

This notebook builds a **fairer comparison dataset**:

1. **Fine-grained building-type labels** from Overture `class` (high-precision OSM tags only).
2. **AlphaEarth Satellite Embedding V1** (2024) at each footprint centroid.
3. A **multi-modal-ready schema** (WKT footprint, lon/lat, height/floors) so street-view / VHR /
   neighbor-graph features can be joined later without resampling labels.

**Next step (analysis):** train a simple classifier on AlphaEarth-only features and report
multi-class metrics against the SOTA ballpark — not claiming the same modalities.

**Requires.** EE auth (`earthengine authenticate --auth_mode=notebook`) + `EE_PROJECT`; network for Overture S3.
"""
        ),
        md(
            """
## Setup
"""
        ),
        code(
            """
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import duckdb
import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent

OUT_DIR = ROOT / "outputs" / "sota_building_type"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RELEASE = "2026-08-19.0"
BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
EMBED_YEAR = 2024
N_PER_CLASS = None  # unused — sampling is unstratified
MAX_PER_REGION = None  # None = all Track-A class buildings in AOI (natural RES mix)
EE_BATCH = 400
RANDOM_STATE = 42
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-kedarvaidya0504")

print("ROOT:", ROOT)
print("OUT_DIR:", OUT_DIR)
print("EE_PROJECT:", EE_PROJECT)
"""
        ),
        md(
            """
## Study taxonomy (SOTA-style multi-class)

Maps high-precision Overture `class` values onto a compact **8-way** label set used in many
building-type papers. Ambiguous tags (`residential`, `terrace`, bare `building=yes`) are **excluded**
so GT stays high-precision / lower-recall — the right setup for SOTA-style comparison.
"""
        ),
        code(
            """
# Overture class → study label (high-precision only)
CLASS_TO_LABEL: dict[str, str] = {
    # single-family
    "house": "single_family",
    "detached": "single_family",
    "bungalow": "single_family",
    "semi": "single_family",
    "semidetached_house": "single_family",
    # multi-family
    "apartments": "multi_family",
    "dormitory": "multi_family",
    # commercial / lodging
    "retail": "retail",
    "commercial": "retail",
    "supermarket": "retail",
    "hotel": "hotel",
    # office
    "office": "office",
    # industrial
    "industrial": "industrial",
    "warehouse": "industrial",
    "manufacture": "industrial",
    # education
    "school": "education",
    "kindergarten": "education",
    "college": "education",
    "university": "education",
    # medical
    "hospital": "medical",
    "clinic": "medical",
    # civic / religious
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
    "retail",
    "hotel",
    "office",
    "industrial",
    "education",
    "medical",
    "religious",
    "civic",
]

# Binary rollup for continuity with notebook 02
RESIDENTIAL_LABELS = {"single_family", "multi_family"}

print("Mapped Overture classes:", len(CLASS_TO_LABEL))
print("Study labels:", LABEL_ORDER)
"""
        ),
        md(
            """
## US metros (street-view dense → future multi-modal)

AOIs are compact downtown/near-downtown boxes where OSM class tags and Google Street View
coverage are both relatively dense — useful when we later attach GSV / VHR / graph neighbors.
"""
        ),
        code(
            """
US_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # (xmin, ymin, xmax, ymax)
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


def region_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = bbox
    return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))


region_meta = pd.DataFrame(
    [
        {
            "region": name,
            "lon": region_centroid(b)[0],
            "lat": region_centroid(b)[1],
            "xmin": b[0],
            "ymin": b[1],
            "xmax": b[2],
            "ymax": b[3],
        }
        for name, b in US_REGIONS.items()
    ]
)
region_meta
"""
        ),
        md(
            """
## 1. Sample Overture buildings (high-precision classes only)
"""
        ),
        code(SAMPLE_SQL),
        code(
            """
# Attach study labels + binary residential flag
labels = raw.copy()
labels["label"] = labels["class"].map(CLASS_TO_LABEL)
labels = labels.dropna(subset=["label"]).reset_index(drop=True)
labels["y_res"] = labels["label"].isin(RESIDENTIAL_LABELS).astype(int)

# Keep a compact source flag if present
def primary_source(sources) -> str | None:
    try:
        if sources is None or (isinstance(sources, float) and np.isnan(sources)):
            return None
        if isinstance(sources, str):
            s = json.loads(sources)
        else:
            s = sources
        if isinstance(s, list) and s:
            return str(s[0].get("dataset") or s[0].get("record_id") or s[0])[:80]
        return str(s)[:80]
    except Exception:
        return None

if "sources" in labels.columns:
    labels["source_primary"] = labels["sources"].map(primary_source)

counts = (
    labels.groupby(["region", "label"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[c for c in LABEL_ORDER if c in labels["label"].unique()], fill_value=0)
)
counts_by_region = labels.groupby("region").size().rename("n").to_frame()
counts_path = OUT_DIR / "label_counts_by_region.csv"
counts.to_csv(counts_path)
print("Wrote", counts_path)
print("Total labeled buildings:", len(labels))
print("Label distribution:")
print(labels["label"].value_counts())
counts
"""
        ),
        code(MAP_CELL),
        md(
            """
## 2. Extract AlphaEarth embeddings (64-D @ centroid)

Same protocol as notebook 02. Cached under `outputs/sota_building_type/`.
"""
        ),
        code(
            """
BANDS = [f"A{i:02d}" for i in range(64)]
CACHE_EMB = OUT_DIR / "buildings_with_embeddings.parquet"


def init_ee(project: str) -> None:
    if not project or project == "YOUR_EE_PROJECT_ID":
        raise ValueError("Set EE_PROJECT to your Earth Engine Cloud project id.")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project)
    print("Earth Engine initialized ·", project)


def annual_embedding_image(year: int) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
    )


def extract_embeddings(df: pd.DataFrame, year: int = EMBED_YEAR, batch_size: int = EE_BATCH) -> pd.DataFrame:
    init_ee(EE_PROJECT)
    img = annual_embedding_image(year)
    rows: list[dict] = []
    n = len(df)
    for start in range(0, n, batch_size):
        batch = df.iloc[start : start + batch_size]
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]),
                {"row_id": str(r.id)},
            )
            for r in batch.itertuples()
        ]
        fc = ee.FeatureCollection(feats)
        sampled = img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=10,
            tileScale=4,
        )
        print(f"EE reduceRegions · {start}:{start + len(batch)} / {n}", flush=True)
        info = sampled.getInfo()
        for f in info["features"]:
            props = f["properties"]
            row = {"id": props.get("row_id")}
            for b in BANDS:
                row[b] = props.get(b)
            rows.append(row)
        time.sleep(0.4)
    emb = pd.DataFrame(rows).dropna(subset=BANDS).reset_index(drop=True)
    return emb


if CACHE_EMB.exists():
    dataset = pd.read_parquet(CACHE_EMB)
    print("Loaded", CACHE_EMB, "· n =", len(dataset))
else:
    emb = extract_embeddings(labels)
    dataset = labels.merge(emb, on="id", how="inner")
    dataset.to_parquet(CACHE_EMB, index=False)
    print("Wrote", CACHE_EMB, "· n =", len(dataset))

# sanity: unit-ish norms
X = dataset[BANDS].to_numpy(dtype=float)
norms = np.linalg.norm(X, axis=1)
print(f"Embedding L2 norms · mean={norms.mean():.3f} std={norms.std():.3f}")
print(dataset.groupby("label").size().reindex(LABEL_ORDER).dropna())
"""
        ),
        md(
            """
## 3. Multi-modal-ready export + splits

Columns for future modalities (not filled here):

| Future field | Role |
|---|---|
| `wkt` | footprint for VHR / GSV frustum / polygon-mean pooling |
| `lon`,`lat` | GSV lookup / neighbor graph nodes |
| `height`,`num_floors` | weak 3D priors (often sparse) |
| `A00`…`A63` | AlphaEarth (available now) |

SOTA papers typically add street-level crops + satellite VHR patches + street/parcel graphs on top.
"""
        ),
        code(
            """
# Canonical column order
meta_cols = [
    "id",
    "region",
    "lon",
    "lat",
    "wkt",
    "subtype",
    "class",
    "label",
    "y_res",
    "height",
    "num_floors",
    "area_m2_approx",
    "source_primary",
]
meta_cols = [c for c in meta_cols if c in dataset.columns]
export = dataset[meta_cols + BANDS].copy()

# Region-holdout friendly split tags (analysis notebooks can re-split)
rng = np.random.default_rng(RANDOM_STATE)
export["split_seed"] = rng.integers(0, 10_000, size=len(export))

dataset_path = OUT_DIR / "dataset.parquet"
export.to_parquet(dataset_path, index=False)

# Also write a lightweight CSV without WKT / embeddings for quick browsing
browse = export.drop(columns=["wkt"] + BANDS, errors="ignore")
browse_path = OUT_DIR / "dataset_browse.csv"
browse.to_csv(browse_path, index=False)

manifest = {
    "name": "sota_comparable_building_type_v1",
    "purpose": (
        "Multi-class building-type dataset for comparing AlphaEarth+simple-classifier "
        "against SOTA that uses street-view / VHR / multi-modal graphs (typically 86-96% "
        "on 10-12 classes)."
    ),
    "overture_release": RELEASE,
    "embedding": {
        "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "year": EMBED_YEAR,
        "bands": BANDS,
        "sample": "centroid Reducer.first @ 10m",
    },
    "label_source": "Overture buildings.class (OSM-derived); ambiguous classes excluded",
    "taxonomy": LABEL_ORDER,
    "class_to_label": CLASS_TO_LABEL,
    "n_buildings": int(len(export)),
    "n_regions": int(export["region"].nunique()),
    "regions": sorted(export["region"].unique().tolist()),
    "label_counts": export["label"].value_counts().to_dict(),
    "files": {
        "dataset_parquet": str(dataset_path.relative_to(ROOT)),
        "browse_csv": str(browse_path.relative_to(ROOT)),
        "label_counts": str(counts_path.relative_to(ROOT)),
    },
    "sota_reference_notes": [
        "HeteroGNN + hierarchical urban features + cross-view imagery: ~86-96% acc on 10-12 types",
        "GHS-OBAT: global res/non-res attributes on Overture footprints (different GT/modality)",
        "This dataset provides AlphaEarth-only features now; GSV/VHR/graph columns to be joined later",
    ],
    "ee_project_used": EE_PROJECT,
}
manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print("Wrote", dataset_path)
print("Wrote", browse_path)
print("Wrote", manifest_path)
print(json.dumps({k: manifest[k] for k in ["n_buildings", "n_regions", "label_counts"]}, indent=2))
"""
        ),
        md(
            """
## 4. Dataset QA summary

Analysis of AlphaEarth + simple classifier performance will use `dataset.parquet` in a follow-up
notebook / cells. Quick QA below.
"""
        ),
        code(
            """
qa = pd.DataFrame(
    {
        "n": export.groupby("region").size(),
        "n_labels": export.groupby("region")["label"].nunique(),
        "pct_residential": export.groupby("region")["y_res"].mean() * 100,
        "pct_has_height": export.groupby("region")["height"].apply(lambda s: 100 * s.notna().mean())
        if "height" in export.columns
        else np.nan,
    }
).sort_values("n", ascending=False)

qa_path = OUT_DIR / "qa_by_region.csv"
qa.to_csv(qa_path)
print("Wrote", qa_path)
qa
"""
        ),
        md(
            """
## Done

Artifacts in `outputs/sota_building_type/`:

- `dataset.parquet` — full multi-modal-ready table + AlphaEarth bands
- `dataset_browse.csv` — labels/coords only
- `manifest.json` — taxonomy, SOTA comparison notes, file index
- `label_counts_by_region.csv`, `qa_by_region.csv`, `dataset_coverage_map.png`

**Next:** run a simple classifier (LR / HGB) on `A00…A63` → `label`, report accuracy / macro-F1
vs the SOTA 86–96% ballpark, with leave-one-region-out transfer.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "AlphaEarth (.venv)",
                "language": "python",
                "name": "alphaearth",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
