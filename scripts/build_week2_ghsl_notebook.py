#!/usr/bin/env python3
"""Generate notebooks/09_week2_ghsl_degree_of_urbanization.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "09_week2_ghsl_degree_of_urbanization.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# Week 2 — GHSL Degree of Urbanization

Kyle’s question, in plain language: **does the residential vs non-residential work behave the same in the city, the suburbs, and the countryside?**

This notebook answers that with one extra label on each building — **degree of urbanization (DoU)** from the JRC Global Human Settlement Layer (GHS-SMOD).

**Three things to look at (in order)**

1. **Where are our current buildings?** Downtown boxes from notebook 02 are almost entirely urban.
2. **Do Overture labels exist outside the city?** Fill rate = share of footprints that have a building-use tag.
3. **Does the classifier hold up?** Same AlphaEarth + logistic regression + leave-one-region-out, sliced by urban / peri-urban / rural.

**Not this notebook.** Comparing against GHSL *building area / volume* (residential vs non-residential) is a later paper experiment.

**Requires.** Cached outputs from this notebook’s first run, or Earth Engine + Overture S3. Re-running with caches is fast.
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

# so figures render inside the notebook, not only as files
%matplotlib inline

import duckdb
import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_rows", 60)
pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcfe",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "font.size": 10,
})

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent

OUT_DIR = ROOT / "outputs" / "week2_ghsl_dou"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIAL = ROOT / "outputs" / "trial_residential"

RELEASE = "2026-08-19.0"
BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
EMBED_YEAR = 2024
SMOD_YEAR = 2025
PAD_DEG = 0.35
N_FILL = 1200
N_RING_PER_CLASS = None  # legacy; unused
MAX_RING_PER_REGION = None  # None = all ring subtype-labeled buildings (unstratified)
EE_BATCH_SMOD = 800
EE_BATCH_EMB = 400
RANDOM_STATE = 42
BANDS = [f"A{i:02d}" for i in range(64)]
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-kedarvaidya0504")

print("ROOT:", ROOT)
print("OUT:", OUT_DIR)
print("EE_PROJECT:", EE_PROJECT)
print("trial embeddings:", (TRIAL / "building_embeddings.parquet").exists())
"""
        ),
        md(
            """
## 1. What is Degree of Urbanization?

GHS-SMOD is a **1 km** map of how urban a place is (UN Degree of Urbanisation). Kyle asked for three buckets:

| Color | Kyle class | Meaning | GHS-SMOD codes |
|---|---|---|---|
| red | **Urban** | City / dense town | 30, 23, 22 |
| gold | **Peri-urban** | Suburb / fringe | 21 |
| green | **Rural** | Countryside / village | 13, 12, 11 |

Water and no-data cells are dropped. DoU describes the *neighborhood*, not the individual building.
"""
        ),
        code(
            """
SMOD_L2 = {
    10: "water",
    11: "very_low_density_rural",
    12: "low_density_rural",
    13: "rural_cluster",
    21: "suburban_peri_urban",
    22: "semi_dense_urban_cluster",
    23: "dense_urban_cluster",
    30: "urban_centre",
}
DOU_ORDER = ["urban", "peri_urban", "rural"]
DOU_LABELS = {"urban": "Urban", "peri_urban": "Peri-urban", "rural": "Rural"}
DOU_COLORS = {"urban": "#c0392b", "peri_urban": "#f1c40f", "rural": "#1e8449"}
DOU_TICKS = [DOU_LABELS[k] for k in DOU_ORDER]
REGION_LABELS = {
    "Boston_NE": "Boston", "NYC_NE": "New York", "Chicago_MW": "Chicago",
    "Atlanta_SE": "Atlanta", "Houston_S": "Houston", "Denver_W": "Denver",
    "Phoenix_SW": "Phoenix", "Seattle_NW": "Seattle", "LA_W": "Los Angeles",
}


def smod_l2(code) -> str | None:
    if pd.isna(code):
        return None
    return SMOD_L2.get(int(code))


def kyle_dou(code) -> str | None:
    if pd.isna(code):
        return None
    c = int(code)
    if c in (30, 23, 22):
        return "urban"
    if c == 21:
        return "peri_urban"
    if c in (13, 12, 11):
        return "rural"
    return None


def attach_dou(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["smod_code"] = pd.to_numeric(out["smod_code"], errors="coerce")
    out["smod_l2"] = out["smod_code"].map(smod_l2)
    out["dou"] = out["smod_code"].map(kyle_dou)
    return out


def pretty_region(s: pd.Series | str):
    if isinstance(s, str):
        return REGION_LABELS.get(s, s)
    return s.map(lambda x: REGION_LABELS.get(x, x))


def savefig(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("Saved", path)


def annotate_bars(ax, values, ns=None, fmt="{:.1f}", dy=0.015, yabs=False):
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        label = fmt.format(v)
        if ns is not None and pd.notna(ns[i]):
            label = f"{label}\\nn={int(ns[i]):,}"
        y = (v + dy) if not yabs else dy
        ax.text(i, (v + dy) if not yabs else v + dy, label, ha="center", va="bottom", fontsize=8)


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
}


def expand_bbox(bbox: tuple[float, float, float, float], pad: float = PAD_DEG):
    xmin, ymin, xmax, ymax = bbox
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def in_bbox(lon, lat, bbox) -> pd.Series:
    xmin, ymin, xmax, ymax = bbox
    return (lon.between(xmin, xmax)) & (lat.between(ymin, ymax))


region_meta = pd.DataFrame(
    [
        {
            "region": name,
            "label": REGION_LABELS[name],
            "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3],
            "hxmin": expand_bbox(b)[0], "hymin": expand_bbox(b)[1],
            "hxmax": expand_bbox(b)[2], "hymax": expand_bbox(b)[3],
            "clon": 0.5 * (b[0] + b[2]),
            "clat": 0.5 * (b[1] + b[3]),
        }
        for name, b in US_REGIONS.items()
    ]
)

# Color key — keep this in your head for every later plot
fig, ax = plt.subplots(figsize=(10.5, 2.6))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 3.2)
ax.axis("off")
ax.set_title("How GHS-SMOD codes roll up to Kyle’s three classes", loc="left", pad=8)
left = [
    (30, "Urban centre", "urban"),
    (23, "Dense urban cluster", "urban"),
    (22, "Semi-dense urban cluster", "urban"),
    (21, "Suburban / peri-urban", "peri_urban"),
    (13, "Rural cluster", "rural"),
    (12, "Low density rural", "rural"),
    (11, "Very low density rural", "rural"),
]
for i, (code, name, dou) in enumerate(left):
    y = 2.7 - i * 0.35
    ax.add_patch(FancyBboxPatch((0.15, y - 0.12), 4.6, 0.3, boxstyle="round,pad=0.02",
                                facecolor=DOU_COLORS[dou], edgecolor="k", linewidth=0.4, alpha=0.85))
    ax.text(0.25, y + 0.03, f"{code}   {name}", va="center", fontsize=8, color="white" if dou != "peri_urban" else "#222")
for i, dou in enumerate(DOU_ORDER):
    x = 6.4
    y = 2.35 - i * 0.75
    ax.add_patch(FancyBboxPatch((x, y), 3.6, 0.58, boxstyle="round,pad=0.04",
                                facecolor=DOU_COLORS[dou], edgecolor="k", linewidth=0.6, alpha=0.9))
    ax.text(x + 1.8, y + 0.29, DOU_LABELS[dou], ha="center", va="center", fontsize=13,
            color="white" if dou != "peri_urban" else "#222", fontweight="semibold")
ax.annotate("", xy=(6.25, 1.9), xytext=(4.9, 1.9),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
ax.text(5.55, 2.15, "rollup", ha="center", fontsize=8, color="#444")
savefig(fig, "taxonomy_rollup.png")
plt.show()
region_meta[["region", "label", "xmin", "ymin", "xmax", "ymax"]]
"""
        ),
        md(
            """
## 2. Earth Engine helpers

SMOD is read at the building point (1 km). AlphaEarth uses the same centroid at 10 m as notebook 02. Cached parquets skip these calls on re-run.
"""
        ),
        code(
            """
def init_ee(project: str) -> None:
    if not project or project == "YOUR_EE_PROJECT_ID":
        raise ValueError("Set EE_PROJECT to your Earth Engine Cloud project id.")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project)
    print("Earth Engine initialized ·", project)


def smod_image(year: int = SMOD_YEAR) -> ee.Image:
    return ee.Image(f"JRC/GHSL/P2023A/GHS_SMOD_V2-0/{year}").select("smod_code")


def annual_embedding_image(year: int) -> ee.Image:
    return (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .mosaic()
    )


def reduce_points(img: ee.Image, df: pd.DataFrame, scale: int, batch_size: int, extra_props=()):
    rows = []
    n = len(df)
    for start in range(0, n, batch_size):
        batch = df.iloc[start : start + batch_size]
        feats = []
        for i, r in batch.iterrows():
            props = {"row_ix": int(i)}
            for k in extra_props:
                props[k] = r[k] if pd.notna(r[k]) else None
            feats.append(ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]), props))
        fc = ee.FeatureCollection(feats)
        sampled = img.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=scale, tileScale=4
        )
        print(f"  EE {start}:{start + len(batch)} / {n}", flush=True)
        info = sampled.getInfo()
        for f in info["features"]:
            rows.append(f["properties"])
        time.sleep(0.25)
    return pd.DataFrame(rows)


def sample_smod(df: pd.DataFrame) -> pd.DataFrame:
    init_ee(EE_PROJECT)
    out = df.reset_index(drop=True).copy()
    raw = reduce_points(smod_image(), out, scale=1000, batch_size=EE_BATCH_SMOD)
    if "smod_code" not in raw.columns and "first" in raw.columns:
        raw["smod_code"] = raw["first"]
    out["smod_code"] = np.nan
    if len(raw) and "smod_code" in raw.columns:
        mapped = raw.set_index("row_ix")["smod_code"]
        idx = mapped.index.intersection(out.index)
        out.loc[idx, "smod_code"] = mapped.loc[idx].to_numpy()
    return attach_dou(out)
"""
        ),
        md(
            """
## 3. Question 1 — is the current sample only downtown?

Notebook 02 used tight metro cores. If those points are almost all **urban**, then last week’s LORO F1 is an **urban-core** number and cannot answer Kyle.
"""
        ),
        code(
            """
core_path = TRIAL / "building_embeddings.parquet"
if not core_path.exists():
    raise FileNotFoundError(
        f"Missing {core_path}. Run notebooks/02_trial_residential_classifier.ipynb first."
    )

core = pd.read_parquet(core_path)
print("Existing trial buildings:", len(core), "· regions:", sorted(core["region"].unique()))

core_smod_path = OUT_DIR / "trial_core_with_dou.parquet"
if core_smod_path.exists():
    core = pd.read_parquet(core_smod_path)
    print("Loaded cached core + DoU")
else:
    core = sample_smod(core)
    core.to_parquet(core_smod_path, index=False)
    print("Wrote", core_smod_path)

core_counts = (
    core.groupby(["region", "dou"], dropna=False)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=DOU_ORDER, fill_value=0)
)
core_counts.index = pretty_region(core_counts.index.to_series())
print(f"Downtown sample is {100 * (core['dou'] == 'urban').mean():.1f}% urban")
core_counts
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
shares = 100 * core["dou"].value_counts(dropna=False).reindex(DOU_ORDER).fillna(0) / len(core)
bars = ax.bar(DOU_TICKS, shares, color=[DOU_COLORS[k] for k in DOU_ORDER], edgecolor="k", linewidth=0.5)
ax.set_ylim(0, 115)
ax.set_ylabel("Share of notebook-02 buildings (%)")
ax.set_title("Downtown boxes are almost entirely urban")
for b, k in zip(bars, DOU_ORDER):
    n = int((core["dou"] == k).sum())
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{shares[k]:.1f}%\\nn={n:,}",
            ha="center", fontsize=8)
ax.text(0.5, 0.92, "Takeaway: last week’s F1 ≈ urban-core F1", transform=ax.transAxes,
        ha="center", fontsize=9, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff6d6", edgecolor="#ddd"))

ax = axes[1]
core_pct = 100 * core_counts.div(core_counts.sum(axis=1), axis=0)
core_pct.plot(kind="barh", stacked=True, ax=ax, color=[DOU_COLORS[k] for k in DOU_ORDER],
              edgecolor="k", linewidth=0.2, legend=False)
ax.set_xlabel("Share of buildings (%)")
ax.set_ylabel("")
ax.set_title("Every metro core looks the same")
ax.set_xlim(0, 100)
ax.invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=DOU_COLORS[k]) for k in DOU_ORDER]
ax.legend(handles, DOU_TICKS, loc="lower right", frameon=True, fontsize=8)

fig.tight_layout()
savefig(fig, "core_is_urban.png")
plt.show()
"""
        ),
        md(
            """
## 4. Expand each metro so suburbs and countryside exist

We keep the notebook-02 downtown box and add a **~39 km hinterland**. Random buildings (tagged or not) from this larger box are used for fill-rate. Labeled buildings from the ring only are added to the classifier.
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(-128, -66)
ax.set_ylim(24, 50)
ax.set_aspect(1.25)
ax.set_title("Nine US metros — small black box is last week’s downtown; large box is this week’s hinterland")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
for _, r in region_meta.iterrows():
    ax.add_patch(Rectangle((r["hxmin"], r["hymin"]), r["hxmax"] - r["hxmin"], r["hymax"] - r["hymin"],
                           fill=False, edgecolor="#4c72b0", lw=1.2, linestyle="--"))
    ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"] - r["xmin"], r["ymax"] - r["ymin"],
                           fill=False, edgecolor="k", lw=1.4))
    ax.scatter(r["clon"], r["clat"], s=18, c="k", zorder=3)
    ax.annotate(r["label"], (r["clon"], r["clat"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
ax.plot([], [], "k-", lw=1.4, label="Downtown core (notebook 02)")
ax.plot([], [], color="#4c72b0", ls="--", lw=1.2, label="Hinterland used this week")
ax.legend(loc="lower left", frameon=True)
fig.tight_layout()
savefig(fig, "us_core_vs_hinterland.png")
plt.show()
"""
        ),
        md(
            """
## 5. Question 2 — do building-use tags disappear outside the city?

**Fill rate** = percent of footprints with a non-null Overture `subtype`. Unlabeled buildings are *not* treated as non-residential. If fill rate falls in peri-urban / rural cells, rural ground truth is thinner and maybe more biased.
"""
        ),
        code(
            r'''
def duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")
    con.execute("SET enable_progress_bar=false;")
    return con


def sample_fill_region(con, name: str, bbox, n: int) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bbox
    sql = f"""
    WITH b AS (
      SELECT
        id,
        subtype,
        class,
        (bbox.xmin + bbox.xmax) / 2.0 AS lon,
        (bbox.ymin + bbox.ymax) / 2.0 AS lat
      FROM read_parquet('{BUILDINGS}', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
        AND bbox.ymin BETWEEN {ymin} AND {ymax}
        AND geometry IS NOT NULL
    )
    SELECT id, '{name}' AS region, subtype, class, lon, lat
    FROM b
    WHERE lon BETWEEN {xmin} AND {xmax}
      AND lat BETWEEN {ymin} AND {ymax}
    ORDER BY hash(id)
    LIMIT {n}
    """
    return con.execute(sql).df()


fill_path = OUT_DIR / "fill_sample_with_dou.parquet"
if fill_path.exists():
    fill = pd.read_parquet(fill_path)
    print("Loaded cached fill sample:", len(fill))
else:
    con = duck()
    frames = []
    for name, bbox in US_REGIONS.items():
        hb = expand_bbox(bbox)
        print(f"Overture fill sample · {name} {hb}", flush=True)
        frames.append(sample_fill_region(con, name, hb, N_FILL))
    fill = pd.concat(frames, ignore_index=True)
    fill = sample_smod(fill)
    fill.to_parquet(fill_path, index=False)
    print("Wrote", fill_path, "· n =", len(fill))

fill = fill[fill["dou"].isin(DOU_ORDER)].copy()
fill["has_subtype"] = fill["subtype"].notna()
fill["has_class"] = fill["class"].notna()
fill["is_res"] = fill["subtype"] == "residential"
fill["city"] = pretty_region(fill["region"])
fill["in_core"] = False
for name, bbox in US_REGIONS.items():
    m = fill["region"] == name
    fill.loc[m, "in_core"] = in_bbox(fill.loc[m, "lon"], fill.loc[m, "lat"], bbox).to_numpy()

fill_by = (
    fill.groupby("dou")
    .agg(
        n=("id", "size"),
        pct_subtype=("has_subtype", "mean"),
        pct_class=("has_class", "mean"),
        n_labeled=("has_subtype", "sum"),
        n_res=("is_res", "sum"),
    )
    .reindex(DOU_ORDER)
)
fill_by["pct_subtype"] *= 100
fill_by["pct_class"] *= 100
fill_by["pct_res_among_labeled"] = 100 * fill_by["n_res"] / fill_by["n_labeled"].replace(0, np.nan)
fill_by.to_csv(OUT_DIR / "fill_rate_by_dou.csv")
fill_by
'''
        ),
        code(
            """
fill_region = (
    fill.groupby(["region", "city", "dou"])
    .agg(n=("id", "size"), pct_subtype=("has_subtype", "mean"))
    .reset_index()
)
fill_region["pct_subtype"] *= 100
fill_region.to_csv(OUT_DIR / "fill_rate_by_region_dou.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

ax = axes[0]
vals = fill_by["pct_subtype"].to_numpy()
ax.bar(DOU_TICKS, vals, color=[DOU_COLORS[k] for k in DOU_ORDER], edgecolor="k", linewidth=0.5)
ax.set_ylim(0, max(50, np.nanmax(vals) * 1.35))
ax.set_ylabel("Footprints with a building-use tag (%)")
ax.set_title("Tags get rarer as you leave the city")
annotate_bars(ax, vals, fill_by["n"].to_numpy(), fmt="{:.1f}%", dy=1.2)

ax = axes[1]
mix_n = fill.groupby("dou").size().reindex(DOU_ORDER)
ax.bar(DOU_TICKS, mix_n, color=[DOU_COLORS[k] for k in DOU_ORDER], edgecolor="k", linewidth=0.5)
ax.set_ylabel("Buildings in the hinterland sample")
ax.set_title("Expanded boxes do contain suburbs & rural")
annotate_bars(ax, mix_n.to_numpy(), fmt="{:,.0f}", dy=80)

ax = axes[2]
lab = fill[fill["has_subtype"]]
res_share = 100 * lab.groupby("dou")["is_res"].mean().reindex(DOU_ORDER)
ax.bar(DOU_TICKS, res_share, color=[DOU_COLORS[k] for k in DOU_ORDER], edgecolor="k", linewidth=0.5)
ax.set_ylim(0, 110)
ax.set_ylabel("Residential share of *tagged* buildings (%)")
ax.set_title("Among tags, still mostly houses")
annotate_bars(ax, res_share.to_numpy(), lab.groupby("dou").size().reindex(DOU_ORDER).to_numpy(),
              fmt="{:.0f}%", dy=2)

fig.tight_layout()
savefig(fig, "fill_rate_story.png")
plt.show()
"""
        ),
        code(
            """
heat = (
    fill_region.pivot(index="city", columns="dou", values="pct_subtype")
    .reindex(columns=DOU_ORDER)
    .rename(columns=DOU_LABELS)
)
heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).index]

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1.15]})

ax = axes[0]
im = ax.imshow(heat.to_numpy(), cmap="YlOrRd", vmin=0, vmax=max(40, heat.to_numpy().max()), aspect="auto")
ax.set_xticks(range(3), list(heat.columns))
ax.set_yticks(range(len(heat)), list(heat.index))
ax.set_title("Fill rate (%) by city × DoU")
ax.grid(False)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.to_numpy()[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.55 * heat.to_numpy().max() else "#222")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% tagged")

ax = axes[1]
city_mix = (
    fill.groupby(["city", "dou"]).size().unstack(fill_value=0).reindex(columns=DOU_ORDER)
)
city_mix = 100 * city_mix.div(city_mix.sum(axis=1), axis=0)
city_mix = city_mix.loc[heat.index]
city_mix.plot(kind="barh", stacked=True, ax=ax, color=[DOU_COLORS[k] for k in DOU_ORDER],
              edgecolor="k", linewidth=0.2)
ax.set_xlabel("Share of hinterland sample (%)")
ax.set_ylabel("")
ax.set_title("DoU mix inside each expanded metro")
ax.set_xlim(0, 100)
ax.invert_yaxis()
ax.legend(DOU_TICKS, loc="lower right", fontsize=8, frameon=True)

fig.tight_layout()
savefig(fig, "fill_rate_heatmap_and_mix.png")
plt.show()
"""
        ),
        code(
            """
# 3×3 maps: random hinterland buildings, colored by DoU
fig, axes = plt.subplots(3, 3, figsize=(12.4, 11.2))
for ax, name in zip(axes.ravel(), US_REGIONS):
    bbox = US_REGIONS[name]
    hb = expand_bbox(bbox)
    g = fill[fill["region"] == name]
    for dou in DOU_ORDER:
        s = g[g["dou"] == dou]
        ax.scatter(s["lon"], s["lat"], s=6, c=DOU_COLORS[dou], alpha=0.7, edgecolors="none", zorder=2)
    x0, y0, x1, y1 = bbox
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=0.9)
    ax.set_xlim(hb[0], hb[2])
    ax.set_ylim(hb[1], hb[3])
    ax.set_aspect(1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(REGION_LABELS[name], fontsize=10)
    ax.grid(False)
    ax.set_facecolor("#f4f6f8")
handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=DOU_COLORS[k], markersize=7)
           for k in DOU_ORDER]
fig.legend(handles, DOU_TICKS + ["black box = downtown"], loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Where the hinterland sample sits — red = urban, gold = suburb, green = rural", y=0.995, fontsize=12)
fig.tight_layout()
savefig(fig, "hinterland_maps_by_dou.png")
plt.show()
"""
        ),
        md(
            """
**How to read the maps.** Each panel is one metro. The small black rectangle is last week’s downtown. Gold and green dots outside that box are why we expanded. Cities differ: Atlanta has a wide suburban ring; LA’s expanded box is still mostly urban.
"""
        ),
        md(
            """
## 6. Question 3 — does the classifier get worse outside the city?

Add **labeled** buildings from the ring (outside the downtown box) so peri-urban and rural cells can enter training and test. Same model as notebook 02: logistic regression on 64-D AlphaEarth, leave-one-region-out.
"""
        ),
        code(
            r'''
def sample_ring_labeled(con, name: str, core_bbox, ring_bbox, max_n: int | None = None) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = ring_bbox
    cx0, cy0, cx1, cy1 = core_bbox
    limit_clause = f"WHERE rn <= {int(max_n)}" if max_n is not None else ""
    sql = f"""
    WITH b AS (
      SELECT
        id,
        subtype,
        class,
        ST_X(ST_Centroid(geometry)) AS lon,
        ST_Y(ST_Centroid(geometry)) AS lat
      FROM read_parquet('{BUILDINGS}', hive_partitioning=1)
      WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
        AND bbox.ymin BETWEEN {ymin} AND {ymax}
        AND subtype IS NOT NULL
        AND geometry IS NOT NULL
    ),
    labeled AS (
      SELECT
        id, subtype, class, lon, lat,
        CASE WHEN subtype = 'residential' THEN 1 ELSE 0 END AS y
      FROM b
      WHERE lon BETWEEN {xmin} AND {xmax}
        AND lat BETWEEN {ymin} AND {ymax}
        AND NOT (lon BETWEEN {cx0} AND {cx1} AND lat BETWEEN {cy0} AND {cy1})
    ),
    ranked AS (
      SELECT *, row_number() OVER (ORDER BY hash(id)) AS rn
      FROM labeled
    )
    SELECT id, '{name}' AS region, subtype, class, lon, lat, y
    FROM ranked
    {limit_clause}
    """
    return con.execute(sql).df()


ring_labels_path = OUT_DIR / "ring_labels.parquet"
if ring_labels_path.exists():
    ring_labels = pd.read_parquet(ring_labels_path)
    print("Loaded cached ring labels:", len(ring_labels))
else:
    con = duck()
    frames = []
    for name, bbox in US_REGIONS.items():
        print(f"Overture ring labels · {name}", flush=True)
        frames.append(sample_ring_labeled(con, name, bbox, expand_bbox(bbox), MAX_RING_PER_REGION))
    ring_labels = pd.concat(frames, ignore_index=True)
    ring_labels.to_parquet(ring_labels_path, index=False)
    print("Wrote", ring_labels_path)
'''
        ),
        code(
            """
def extract_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    init_ee(EE_PROJECT)
    img = annual_embedding_image(EMBED_YEAR)
    out = df.reset_index(drop=True).copy()
    raw = reduce_points(img, out, scale=10, batch_size=EE_BATCH_EMB)
    raw = raw.set_index("row_ix")
    for b in BANDS:
        out[b] = np.nan
        if b in raw.columns:
            out.loc[raw.index.intersection(out.index), b] = raw[b]
    return out.dropna(subset=BANDS).reset_index(drop=True)


ring_emb_path = OUT_DIR / "ring_embeddings_with_dou.parquet"
if ring_emb_path.exists():
    ring = pd.read_parquet(ring_emb_path)
    print("Loaded cached ring embeddings:", len(ring))
else:
    ring = extract_embeddings(ring_labels)
    ring = sample_smod(ring)
    ring.to_parquet(ring_emb_path, index=False)
    print("Wrote", ring_emb_path, "· n =", len(ring))

print(ring.groupby(["dou", "y"], dropna=False).size().unstack(fill_value=0))
"""
        ),
        code(
            """
core_m = core.dropna(subset=BANDS + ["y"]).copy()
core_m["zone"] = "core"
ring_m = ring.dropna(subset=BANDS + ["y"]).copy()
ring_m["zone"] = "ring"
keep = ["id", "region", "y", "lon", "lat", "subtype", "class", "smod_code", "smod_l2", "dou", "zone"] + BANDS
data = pd.concat([core_m[keep], ring_m[keep]], ignore_index=True)
data = data[data["dou"].isin(DOU_ORDER)].drop_duplicates("id").reset_index(drop=True)
data["city"] = pretty_region(data["region"])
data["label"] = np.where(data["y"] == 1, "Residential", "Non-residential")
data.to_parquet(OUT_DIR / "classifier_table.parquet", index=False)

X = data[BANDS].to_numpy(dtype=np.float64)
y = data["y"].to_numpy(dtype=int)
clf_fn = lambda: Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ]
)

regions = sorted(data["region"].unique())
oof_pred = np.full(len(data), -1)
oof_urban_pred = np.full(len(data), -1)
loro_rows = []

for holdout in regions:
    te = data["region"] == holdout
    tr = ~te
    if te.sum() < 20 or tr.sum() < 50:
        continue
    model = clf_fn()
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    oof_pred[te.to_numpy()] = pred
    loro_rows.append(
        {
            "region": holdout,
            "city": REGION_LABELS[holdout],
            "n_test": int(te.sum()),
            "f1_macro": float(f1_score(y[te], pred, average="macro")),
            "f1_residential": float(f1_score(y[te], pred, pos_label=1)),
        }
    )
    urb = tr & (data["dou"] == "urban")
    if urb.sum() >= 50:
        umodel = clf_fn()
        umodel.fit(X[urb], y[urb])
        oof_urban_pred[te.to_numpy()] = umodel.predict(X[te])

data["oof_pred"] = oof_pred
data["oof_urban_pred"] = oof_urban_pred
data["correct"] = data["oof_pred"] == data["y"]
loro = pd.DataFrame(loro_rows)
loro.to_csv(OUT_DIR / "loro_by_region.csv", index=False)

valid = data["oof_pred"] >= 0
dou_rows = []
for dou in DOU_ORDER:
    m = valid & (data["dou"] == dou)
    if m.sum() < 20:
        continue
    dou_rows.append({
        "train": "all_other_regions", "dou": dou, "n": int(m.sum()),
        "n_res": int(data.loc[m, "y"].sum()),
        "f1_macro": float(f1_score(data.loc[m, "y"], data.loc[m, "oof_pred"], average="macro")),
        "f1_residential": float(f1_score(data.loc[m, "y"], data.loc[m, "oof_pred"], pos_label=1)),
        "accuracy": float((data.loc[m, "y"] == data.loc[m, "oof_pred"]).mean()),
    })
    um = m & (data["oof_urban_pred"] >= 0)
    if um.sum() >= 20:
        dou_rows.append({
            "train": "urban_other_regions", "dou": dou, "n": int(um.sum()),
            "n_res": int(data.loc[um, "y"].sum()),
            "f1_macro": float(f1_score(data.loc[um, "y"], data.loc[um, "oof_urban_pred"], average="macro")),
            "f1_residential": float(f1_score(data.loc[um, "y"], data.loc[um, "oof_urban_pred"], pos_label=1)),
            "accuracy": float((data.loc[um, "y"] == data.loc[um, "oof_urban_pred"]).mean()),
        })
dou_metrics = pd.DataFrame(dou_rows)
dou_metrics.to_csv(OUT_DIR / "f1_by_dou.csv", index=False)
print("Classifier n =", len(data))
print(classification_report(data.loc[valid, "y"], data.loc[valid, "oof_pred"],
                            target_names=["non_residential", "residential"]))
"""
        ),
        md(
            """
### What we trained on

The model still sees mostly urban buildings. Rural is the smallest slice — treat rural F1 as directional, not a locked number.
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

ax = axes[0]
comp = pd.crosstab(data["dou"], data["label"]).reindex(DOU_ORDER)
x = np.arange(len(DOU_ORDER))
w = 0.38
ax.bar(x - w / 2, comp["Residential"], width=w, color="#4c72b0", edgecolor="k", linewidth=0.4, label="Residential")
ax.bar(x + w / 2, comp["Non-residential"], width=w, color="#dd8452", edgecolor="k", linewidth=0.4, label="Non-residential")
ax.set_xticks(x, DOU_TICKS)
ax.set_ylabel("Labeled buildings in the classifier")
ax.set_title("Training/test mass is still urban")
ax.legend(frameon=False)
for i, dou in enumerate(DOU_ORDER):
    ax.text(i - w / 2, comp.loc[dou, "Residential"] + 40, f"{int(comp.loc[dou, 'Residential']):,}", ha="center", fontsize=7)
    ax.text(i + w / 2, comp.loc[dou, "Non-residential"] + 40, f"{int(comp.loc[dou, 'Non-residential']):,}", ha="center", fontsize=7)

ax = axes[1]
zone = pd.crosstab(data["dou"], data["zone"]).reindex(DOU_ORDER)[["core", "ring"]]
zone.plot(kind="bar", stacked=True, ax=ax, color=["#7f8c8d", "#3498db"], edgecolor="k", linewidth=0.3)
ax.set_xticklabels(DOU_TICKS, rotation=0)
ax.set_ylabel("Buildings")
ax.set_title("Rural / peri-urban come from the hinterland ring")
ax.legend(["Downtown core", "Hinterland ring"], frameon=False)

fig.tight_layout()
savefig(fig, "classifier_sample_mix.png")
plt.show()
"""
        ),
        md(
            """
### Does F1 drop from urban → rural?

**Left:** one F1 per DoU (predictions from models that never saw that metro).  
**Middle:** same, but the model was trained only on urban buildings — Kyle’s “works in cities, fails in rural?” test.  
**Right:** ordinary LORO by city, for context.
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

sub = dou_metrics[dou_metrics["train"] == "all_other_regions"].set_index("dou").reindex(DOU_ORDER)
ax = axes[0]
ax.bar(DOU_TICKS, sub["f1_macro"], color=[DOU_COLORS[k] for k in DOU_ORDER], edgecolor="k", linewidth=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Leave-one-region-out macro F1")
ax.set_title("F1 eases off outside the city")
annotate_bars(ax, sub["f1_macro"].to_numpy(), sub["n"].to_numpy(), fmt="{:.2f}", dy=0.03)

ax = axes[1]
wide = dou_metrics.pivot(index="dou", columns="train", values="f1_macro").reindex(DOU_ORDER)
x = np.arange(len(DOU_ORDER))
w = 0.38
ax.bar(x - w / 2, wide["all_other_regions"], width=w, color="#4c72b0", edgecolor="k", linewidth=0.4,
       label="Train on all DoU")
ax.bar(x + w / 2, wide["urban_other_regions"], width=w, color="#dd8452", edgecolor="k", linewidth=0.4,
       label="Train on urban only")
ax.set_xticks(x, DOU_TICKS)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Macro F1")
ax.set_title("Urban-only training hurts rural most")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
loro_s = loro.sort_values("f1_macro")
colors = plt.cm.RdYlGn(np.clip((loro_s["f1_macro"] - 0.55) / 0.35, 0, 1))
ax.barh(loro_s["city"], loro_s["f1_macro"], color=colors, edgecolor="k", linewidth=0.3)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("LORO macro F1")
ax.set_title("City holdout (core + hinterland)")
for _, r in loro_s.iterrows():
    ax.text(r["f1_macro"] + 0.008, r["city"], f"{r['f1_macro']:.2f}", va="center", fontsize=8)

fig.tight_layout()
savefig(fig, "f1_by_dou_and_city.png")
plt.show()
"""
        ),
        code(
            """
# Confusion matrices — rows = true label, columns = predicted
f1_all = dou_metrics[dou_metrics["train"] == "all_other_regions"].set_index("dou")
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
names = ["Non-res", "Res"]
for ax, dou in zip(axes, DOU_ORDER):
    m = (data["dou"] == dou) & valid
    cm = confusion_matrix(data.loc[m, "y"], data.loc[m, "oof_pred"], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{DOU_LABELS[dou]}  ·  n={int(m.sum())}\\nmacro-F1={f1_all.loc[dou, 'f1_macro']:.2f}")
    ax.grid(False)
fig.suptitle("Where the mistakes are (out-of-metro predictions)", y=1.02)
fig.tight_layout()
savefig(fig, "confusion_by_dou.png")
plt.show()

# Per-class recall so residential cannot hide a bad non-res number
fig, ax = plt.subplots(figsize=(7.2, 4.2))
rec_rows = []
for dou in DOU_ORDER:
    m = (data["dou"] == dou) & valid
    yt, yp = data.loc[m, "y"].to_numpy(), data.loc[m, "oof_pred"].to_numpy()
    rec_rows.append({
        "dou": dou,
        "Residential recall": float(((yt == 1) & (yp == 1)).sum() / max(1, (yt == 1).sum())),
        "Non-residential recall": float(((yt == 0) & (yp == 0)).sum() / max(1, (yt == 0).sum())),
    })
rec = pd.DataFrame(rec_rows).set_index("dou")
x = np.arange(3)
w = 0.38
ax.bar(x - w / 2, rec["Residential recall"], width=w, color="#4c72b0", edgecolor="k", linewidth=0.4, label="Finds houses")
ax.bar(x + w / 2, rec["Non-residential recall"], width=w, color="#dd8452", edgecolor="k", linewidth=0.4, label="Finds non-res")
ax.set_xticks(x, DOU_TICKS)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Recall")
ax.set_title("Rural miss: non-residential is the hard class")
ax.legend(frameon=False)
for i, dou in enumerate(DOU_ORDER):
    ax.text(i - w / 2, rec.loc[dou, "Residential recall"] + 0.03, f"{rec.loc[dou, 'Residential recall']:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, rec.loc[dou, "Non-residential recall"] + 0.03, f"{rec.loc[dou, 'Non-residential recall']:.2f}", ha="center", fontsize=8)
fig.tight_layout()
savefig(fig, "recall_by_dou.png")
plt.show()
"""
        ),
        md(
            """
## 7. Two cities up close

**Boston** — downtown is a solid urban block; hinterland adds gold/green.  
**Atlanta** — more peri-urban in the classifier sample, so it is a better picture of Kyle’s suburb question.
"""
        ),
        code(
            """
def city_panel(ax, region: str, title: str):
    bbox = US_REGIONS[region]
    hb = expand_bbox(bbox)
    g = data[data["region"] == region]
    for dou in DOU_ORDER:
        s = g[g["dou"] == dou]
        ax.scatter(s["lon"], s["lat"], s=10, c=DOU_COLORS[dou], alpha=0.8, edgecolors="none")
    x0, y0, x1, y1 = bbox
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
    ax.set_xlim(hb[0], hb[2])
    ax.set_ylim(hb[1], hb[3])
    ax.set_aspect(1.25)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    counts = g["dou"].value_counts().reindex(DOU_ORDER).fillna(0).astype(int)
    ax.text(0.02, 0.02, "  ".join(f"{DOU_LABELS[k]} {counts[k]}" for k in DOU_ORDER),
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ccc", alpha=0.9))


fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
city_panel(axes[0], "Boston_NE", "Boston — downtown is 100% urban")
city_panel(axes[1], "Atlanta_SE", "Atlanta — hinterland actually adds suburbs")
handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=DOU_COLORS[k], markersize=8)
           for k in DOU_ORDER]
axes[0].legend(handles, DOU_TICKS, loc="upper right", frameon=True, fontsize=8)
fig.tight_layout()
savefig(fig, "boston_atlanta_classifier_maps.png")
plt.show()
"""
        ),
        code(
            """
# Same two cities, but from the *random* fill sample: tagged vs untagged
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
for ax, region, title in (
    (axes[0], "Boston_NE", "Boston hinterland — most footprints have no use tag"),
    (axes[1], "Houston_S", "Houston hinterland — tags are scarce in every DoU"),
):
    bbox = US_REGIONS[region]
    hb = expand_bbox(bbox)
    g = fill[fill["region"] == region]
    ax.scatter(g.loc[~g["has_subtype"], "lon"], g.loc[~g["has_subtype"], "lat"],
               s=6, c="#bdc3c7", alpha=0.55, label="No building-use tag", edgecolors="none")
    tagged = g[g["has_subtype"]]
    for dou in DOU_ORDER:
        s = tagged[tagged["dou"] == dou]
        ax.scatter(s["lon"], s["lat"], s=10, c=DOU_COLORS[dou], alpha=0.85, label=DOU_LABELS[dou], edgecolors="none")
    x0, y0, x1, y1 = bbox
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
    ax.set_xlim(hb[0], hb[2])
    ax.set_ylim(hb[1], hb[3])
    ax.set_aspect(1.25)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
axes[1].legend(loc="lower right", fontsize=8, frameon=True, markerscale=1.4)
fig.tight_layout()
savefig(fig, "tagged_vs_untagged_maps.png")
plt.show()
"""
        ),
        md(
            """
## 8. Takeaways to remember

Read the three cards left to right — they are the meeting answer.
"""
        ),
        code(
            """
f1s = dou_metrics[dou_metrics["train"] == "all_other_regions"].set_index("dou")["f1_macro"]
cards = [
    (DOU_COLORS["urban"], "Downtown sample",
     f"{100 * (core['dou'] == 'urban').mean():.0f}% urban",
     "Notebook 02 LORO is an urban-core result."),
    (DOU_COLORS["peri_urban"], "Overture fill rate",
     f"{fill_by.loc['urban', 'pct_subtype']:.0f}% → {fill_by.loc['peri_urban', 'pct_subtype']:.0f}% → {fill_by.loc['rural', 'pct_subtype']:.0f}%",
     "Urban → peri-urban → rural. Rural GT is thinner."),
    (DOU_COLORS["rural"], "LORO macro-F1",
     f"{f1s['urban']:.2f} → {f1s['peri_urban']:.2f} → {f1s['rural']:.2f}",
     "Mild drop. Rural n is small; non-res recall is the weak spot."),
]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
for ax, (color, title, big, note) in zip(axes, cards):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor="k", linewidth=0.6, alpha=0.88))
    tc = "#222" if color == DOU_COLORS["peri_urban"] else "white"
    ax.text(0.5, 0.78, title, ha="center", fontsize=11, color=tc, fontweight="semibold")
    ax.text(0.5, 0.48, big, ha="center", fontsize=16, color=tc, fontweight="bold")
    ax.text(0.5, 0.22, note, ha="center", fontsize=8, color=tc, wrap=True)
fig.tight_layout()
savefig(fig, "takeaway_cards.png")
plt.show()

print("Going forward")
print("1. Always report LORO F1 overall and by urban / peri-urban / rural.")
print("2. Do not sample only downtown if Kyle’s claim is about rural transfer.")
print("3. Never treat untagged rural footprints as non-residential.")
print("4. GHSL building area/volume comparison stays for the paper, not this week.")
"""
        ),
        md(
            """
## 9. How this enters LORO going forward

Kyle already has **region** holdout. Add **degree of urbanization** as a second reporting axis, not a replacement.

1. Report LORO macro-F1 overall **and** inside urban / peri-urban / rural.
2. Keep hinterland (or another rural draw) in the sample. Downtown-only boxes will not test his claim.
3. Do not treat unlabeled rural footprints as negatives.
4. Optional later lock: stratify the training draw by DoU, or hold out a rural slice the same way we hold out a region.
"""
        ),
        code(
            """
snap = {
    "smod": "JRC/GHSL/P2023A/GHS_SMOD_V2-0",
    "smod_year": SMOD_YEAR,
    "alphaearth_year": EMBED_YEAR,
    "kyle_dou": {"urban": [30, 23, 22], "peri_urban": [21], "rural": [13, 12, 11]},
    "n_core": int(len(core)),
    "n_fill": int(len(fill)),
    "n_classifier": int(len(data)),
    "core_share_urban": float((core["dou"] == "urban").mean()),
    "fill_rate_by_dou": fill_by["pct_subtype"].to_dict(),
    "loro_f1_by_dou": dou_metrics[dou_metrics["train"] == "all_other_regions"]
    .set_index("dou")["f1_macro"]
    .to_dict(),
    "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.suffix in {".png", ".csv", ".json", ".parquet"}),
}
snap_path = OUT_DIR / "week2_snapshot.json"
snap_path.write_text(json.dumps(snap, indent=2))
print("Wrote", snap_path)
print("Figures:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(" ", p.name)
"""
        ),
    ]

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    OUT.write_text(nbf.writes(nb))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
