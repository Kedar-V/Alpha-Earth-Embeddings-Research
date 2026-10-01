#!/usr/bin/env python3
"""Assign Overture buildings to every Census place (city / town / village / CDP).

The 15-downtown EDA uses compact bbox extracts. This scan uses Census cartographic
place polygons for the 50 states + DC, then rolls those places up to state and
Census region.

Overture `locality` is not used: it misses consolidated city-counties
(San Francisco, Denver). Census places are the official US city list.

Buildings whose centroid falls in no place (unincorporated) are excluded.
Overlapping places: keep the smallest polygon.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

CENSUS_REGION = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NY": "Northeast",
    "NJ": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "DC": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}

FIPS_TO_STATE = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

US_STATES = tuple(sorted(CENSUS_REGION))
STATE_TO_FIPS = {v: k for k, v in FIPS_TO_STATE.items()}

PLACE_ZIP_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_place_500k.zip"

DEFAULT_STUDY_CLASSES = (
    "house",
    "detached",
    "bungalow",
    "semi",
    "semidetached_house",
    "apartments",
    "dormitory",
    "retail",
    "commercial",
    "supermarket",
    "hotel",
    "office",
    "industrial",
    "warehouse",
    "manufacture",
    "school",
    "kindergarten",
    "college",
    "university",
    "hospital",
    "clinic",
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
)

LAT_BAND = {
    "CA": 1.0,
    "TX": 1.5,
    "FL": 1.0,
    "NY": 1.0,
    "PA": 1.5,
    "OH": 1.5,
    "IL": 1.5,
    "MI": 1.5,
    "NC": 1.5,
    "GA": 1.5,
    "WA": 1.5,
    "OR": 1.5,
    "AZ": 1.5,
    "CO": 2.0,
    "AK": 4.0,
}


def _quoted(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def ensure_spatial(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET enable_progress_bar=false;")


def download_census_places(out: Path, force: bool = False) -> Path:
    shp_dir = out / "census_places_cb2024"
    shp = shp_dir / "cb_2024_us_place_500k.shp"
    if shp.exists() and not force:
        return shp
    shp_dir.mkdir(parents=True, exist_ok=True)
    zpath = shp_dir / "cb_2024_us_place_500k.zip"
    print(f"Downloading Census places {PLACE_ZIP_URL} ...", flush=True)
    urllib.request.urlretrieve(PLACE_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(shp_dir)
    if not shp.exists():
        raise FileNotFoundError(shp)
    print(f"  extracted {shp}", flush=True)
    return shp


def cache_places(
    con: duckdb.DuckDBPyConnection,
    out: Path,
    force: bool = False,
) -> Path:
    """Census place polygons → local parquet with state/region attached."""
    dest = out / "us_census_places.parquet"
    if dest.exists() and not force:
        return dest
    shp = download_census_places(out, force=force)
    fips_sql = " ".join(
        f"WHEN '{fips}' THEN '{st}'" for fips, st in FIPS_TO_STATE.items()
    )
    region_sql = " ".join(
        f"WHEN '{st}' THEN '{reg}'" for st, reg in CENSUS_REGION.items()
    )
    keep = _quoted(list(FIPS_TO_STATE))
    print("Writing US Census place polygons to parquet ...", flush=True)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
          SELECT
            GEOID AS geoid,
            NAME AS city,
            NAMELSAD AS namelsad,
            LSAD AS lsad,
            CASE LSAD WHEN '57' THEN TRUE ELSE FALSE END AS is_cdp,
            CASE STATEFP {fips_sql} END AS state,
            CASE (CASE STATEFP {fips_sql} END) {region_sql} END AS region,
            ST_XMin(geom) AS xmin,
            ST_XMax(geom) AS xmax,
            ST_YMin(geom) AS ymin,
            ST_YMax(geom) AS ymax,
            ST_Area_Spheroid(geom) AS area_m2,
            geom AS geometry
          FROM ST_Read('{shp.as_posix()}')
          WHERE STATEFP IN ({keep})
        ) TO '{dest.as_posix()}' (FORMAT PARQUET)
        """
    )
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dest.as_posix()}')").fetchone()[0]
    print(f"  {n:,} Census places in {time.time() - t0:.1f}s → {dest}", flush=True)
    return dest


def _state_bbox(con: duckdb.DuckDBPyConnection, loc_path: str, state: str) -> tuple[float, float, float, float]:
    row = con.execute(
        f"""
        SELECT MIN(xmin), MIN(ymin), MAX(xmax), MAX(ymax)
        FROM read_parquet('{loc_path}')
        WHERE state = '{state}'
        """
    ).fetchone()
    pad = 0.05
    return row[0] - pad, row[1] - pad, row[2] + pad, row[3] + pad


def _lat_slices(ymin: float, ymax: float, step: float) -> list[tuple[float, float]]:
    out = []
    y = ymin
    while y < ymax - 1e-9:
        y2 = min(y + step, ymax)
        out.append((y, y2))
        y = y2
    return out or [(ymin, ymax)]


def extract_state(
    con: duckdb.DuckDBPyConnection,
    *,
    buildings: str,
    loc_path: Path,
    state: str,
    study_classes: tuple[str, ...],
) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = _state_bbox(con, loc_path.as_posix(), state)
    n_loc = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{loc_path.as_posix()}') WHERE state = '{state}'"
    ).fetchone()[0]
    if n_loc == 0:
        return pd.DataFrame()

    classes = _quoted(study_classes)
    slices = _lat_slices(ymin, ymax, LAT_BAND.get(state, 3.0))
    frames = []
    loc = loc_path.as_posix()
    last = len(slices)

    for i, (y0, y1) in enumerate(slices, 1):
        t0 = time.time()
        lat_hi = "<=" if i == last else "<"
        df = con.execute(
            f"""
            WITH loc AS (
              SELECT geoid, city, lsad, is_cdp, area_m2, geometry
              FROM read_parquet('{loc}')
              WHERE state = '{state}'
                AND ymax >= {y0} AND ymin <= {y1}
            ),
            b AS (
              SELECT
                id,
                subtype,
                class,
                height,
                num_floors,
                (bbox.xmin + bbox.xmax) / 2.0 AS lon,
                (bbox.ymin + bbox.ymax) / 2.0 AS lat
              FROM read_parquet('{buildings}', hive_partitioning=1)
              WHERE bbox.xmax > {xmin} AND bbox.xmin < {xmax}
                AND bbox.ymax > {y0} AND bbox.ymin < {y1}
                AND ((bbox.ymin + bbox.ymax) / 2.0) >= {y0}
                AND ((bbox.ymin + bbox.ymax) / 2.0) {lat_hi} {y1}
            ),
            hit AS (
              SELECT
                b.id,
                loc.city,
                loc.geoid,
                loc.lsad,
                loc.is_cdp,
                b.subtype,
                b.class,
                b.height,
                b.num_floors
              FROM b
              JOIN loc ON ST_Contains(loc.geometry, ST_Point(b.lon, b.lat))
              QUALIFY ROW_NUMBER() OVER (PARTITION BY b.id ORDER BY loc.area_m2 ASC) = 1
            )
            SELECT
              city,
              geoid,
              ANY_VALUE(lsad) AS lsad,
              ANY_VALUE(is_cdp) AS is_cdp,
              COUNT(*)::BIGINT AS n,
              COUNT(subtype)::BIGINT AS n_subtype,
              COUNT(class)::BIGINT AS n_class,
              COUNT(height)::BIGINT AS n_height,
              COUNT(num_floors)::BIGINT AS n_floors,
              SUM(CASE WHEN subtype = 'residential' THEN 1 ELSE 0 END)::BIGINT AS n_res,
              SUM(CASE WHEN subtype IS NOT NULL AND subtype != 'residential' THEN 1 ELSE 0 END)::BIGINT AS n_nonres,
              SUM(CASE WHEN class IN ({classes}) THEN 1 ELSE 0 END)::BIGINT AS n_study
            FROM hit
            GROUP BY 1, 2
            """
        ).df()
        if len(df):
            frames.append(df)
        print(
            f"    {state} band {i}/{len(slices)} lat {y0:.2f}–{y1:.2f}: "
            f"{int(df['n'].sum()) if len(df) else 0:,} buildings in {time.time() - t0:.1f}s",
            flush=True,
        )

    if not frames:
        return pd.DataFrame()

    g = pd.concat(frames, ignore_index=True)
    num_cols = ["n", "n_subtype", "n_class", "n_height", "n_floors", "n_res", "n_nonres", "n_study"]
    meta = g.groupby("geoid", as_index=False).agg(
        city=("city", "first"),
        lsad=("lsad", "first"),
        is_cdp=("is_cdp", "first"),
        **{c: (c, "sum") for c in num_cols},
    )
    meta["state"] = state
    meta["region"] = CENSUS_REGION[state]
    return meta


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, base in [
        ("pct_subtype", "n_subtype"),
        ("pct_class", "n_class"),
        ("pct_height", "n_height"),
        ("pct_floors", "n_floors"),
        ("pct_study", "n_study"),
    ]:
        out[col] = 100.0 * out[base] / out["n"].clip(lower=1)
    labeled = (out["n_res"] + out["n_nonres"]).clip(lower=1)
    out["pct_res_of_labeled"] = 100.0 * out["n_res"] / labeled
    return out


def rollup_geo(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, as_index=False).agg(
        n_cities=("geoid", "nunique"),
        n=("n", "sum"),
        n_subtype=("n_subtype", "sum"),
        n_class=("n_class", "sum"),
        n_height=("n_height", "sum"),
        n_floors=("n_floors", "sum"),
        n_res=("n_res", "sum"),
        n_nonres=("n_nonres", "sum"),
        n_study=("n_study", "sum"),
    )
    return add_rates(g)


def run_us_all_cities(
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    buildings: str,
    out: Path,
    study_classes: tuple[str, ...] | None = None,
    force: bool = False,
    states: list[str] | None = None,
    **_ignored,
) -> pd.DataFrame:
    own_con = con is None
    if own_con:
        con = duckdb.connect()
    ensure_spatial(con)
    study_classes = tuple(study_classes) if study_classes is not None else DEFAULT_STUDY_CLASSES

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state_dir = out / "us_all_cities_by_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    loc_path = cache_places(con, out, force=force)

    if states:
        want = [s.upper() for s in states]
    else:
        late = {"AK", "HI"}
        want = [s for s in US_STATES if s not in late] + sorted(late)
    for st in want:
        if st not in CENSUS_REGION:
            raise ValueError(f"Unknown state {st}")
        dest = state_dir / f"{st}.csv"
        if dest.exists() and not force:
            print(f"  skip {st} (cached {dest.name})", flush=True)
            continue
        print(f"US all-cities · {st} / {CENSUS_REGION[st]} ...", flush=True)
        t0 = time.time()
        df = extract_state(
            con,
            buildings=buildings,
            loc_path=loc_path,
            state=st,
            study_classes=study_classes,
        )
        df.to_csv(dest, index=False)
        n_b = int(df["n"].sum()) if len(df) else 0
        print(f"  wrote {dest.name}: {len(df):,} places, {n_b:,} buildings ({time.time() - t0:.1f}s)", flush=True)

    frames = []
    for st in want:
        p = state_dir / f"{st}.csv"
        if p.exists() and p.stat().st_size > 0:
            part = pd.read_csv(p)
            if len(part) and "geoid" in part.columns:
                frames.append(part)
    if not frames:
        raise RuntimeError("No state extracts found. Run the S3 scan first.")

    city = add_rates(pd.concat(frames, ignore_index=True))
    city_path = out / "us_all_cities_fill_rates.csv"
    city.to_csv(city_path, index=False)

    st_roll = rollup_geo(city, ["region", "state"])
    rg_roll = rollup_geo(city, ["region"])
    inc = city.loc[~city["is_cdp"].astype(bool)].copy()
    st_inc = rollup_geo(inc, ["region", "state"]) if len(inc) else st_roll.iloc[0:0]
    rg_inc = rollup_geo(inc, ["region"]) if len(inc) else rg_roll.iloc[0:0]

    st_roll.to_csv(out / "us_all_state_fill_rates.csv", index=False)
    rg_roll.to_csv(out / "us_all_region_fill_rates.csv", index=False)
    st_inc.to_csv(out / "us_all_state_fill_rates_incorporated.csv", index=False)
    rg_inc.to_csv(out / "us_all_region_fill_rates_incorporated.csv", index=False)

    print(
        f"All Census places: {len(city):,} · {int(city['n'].sum()):,} buildings → {city_path}",
        flush=True,
    )
    if own_con:
        con.close()
    return city


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/overture_eda"))
    p.add_argument("--release", default="2026-08-19.0")
    p.add_argument("--force", action="store_true")
    p.add_argument("--states", default="", help="Comma-separated state codes, default all 50+DC")
    args = p.parse_args()
    buildings = f"s3://overturemaps-us-west-2/release/{args.release}/theme=buildings/type=building/*"
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()] or None
    run_us_all_cities(
        buildings=buildings,
        out=args.out,
        force=args.force,
        states=states,
    )


if __name__ == "__main__":
    main()
