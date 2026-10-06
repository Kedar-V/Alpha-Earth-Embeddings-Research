"""Batched / parallel Postgres writers for international AlphaEarth ingest."""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from tqdm.auto import tqdm

LABEL_VERSION = "resi_def_v1_pixel_frac_v1"
EMBED_YEAR = 2024
BANDS = [f"A{i:02d}" for i in range(64)]
BAND_COLS = [f"a{i:02d}" for i in range(64)]


def constrained_label_fracs(frac_res_full: float, frac_nres_full: float) -> tuple[float, float, float]:
    """Round coverage fracs for ``cell_labels`` so res+nres+empty ≤ 1.

    RES and NRES unions can both cover most of a cell (overlap) so the raw
    fractions sum to > 1. The DB check rejects that; scale onto the simplex,
    round to 2 decimals, then repair rounding drift.
    """
    r = max(0.0, float(frac_res_full))
    n = max(0.0, float(frac_nres_full))
    if r + n > 1.0:
        s = r + n
        r, n = r / s, n / s
    frac_res = round(r, 2)
    frac_nres = round(n, 2)
    if frac_res + frac_nres > 1.0:
        if frac_res >= frac_nres:
            frac_res = round(1.0 - frac_nres, 2)
        else:
            frac_nres = round(1.0 - frac_res, 2)
    frac_empty = round(max(0.0, 1.0 - frac_res - frac_nres), 2)
    if frac_res + frac_nres + frac_empty > 1.000001:
        frac_empty = max(0.0, round(1.0 - frac_res - frac_nres, 2))
    return frac_res, frac_nres, frac_empty


def _dsn_from_env() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} "
        f"port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def _connect():
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(_dsn_from_env(), row_factory=dict_row)


def _prepare_cells_df(cells_df: pd.DataFrame, area_crs: str) -> pd.DataFrame:
    df = cells_df.drop_duplicates("pixel_wkt", keep="first").copy()
    df["cell_key"] = [
        hashlib.sha1(str(w).encode()).hexdigest()[:16] for w in df["pixel_wkt"].tolist()
    ]
    for i in range(64):
        col = f"A{i:02d}"
        if col not in df.columns:
            raise KeyError(col)
    if "pixel_size_m" not in df.columns:
        df["pixel_size_m"] = 10.0
    if "utm_x" not in df.columns:
        df["utm_x"] = np.nan
    if "utm_y" not in df.columns:
        df["utm_y"] = np.nan
    if "crs" not in df.columns:
        df["crs"] = area_crs
    else:
        df["crs"] = df["crs"].fillna(area_crs)
    return df


def _chunk_rows_as_lists(df: pd.DataFrame) -> list[list]:
    """Build COPY row lists without DataFrame.iterrows."""
    keys = df["cell_key"].tolist()
    wkts = df["pixel_wkt"].astype(str).tolist()
    lons = df["pixel_lon"].astype(float).to_numpy()
    lats = df["pixel_lat"].astype(float).to_numpy()
    sizes = df["pixel_size_m"].fillna(10).astype(float).to_numpy()
    uxs = df["utm_x"].to_numpy()
    uys = df["utm_y"].to_numpy()
    crss = df["crs"].astype(str).tolist()
    emb = np.column_stack([df[f"A{i:02d}"].astype(float).to_numpy() for i in range(64)])
    rows: list[list] = []
    for i in range(len(df)):
        ux, uy = uxs[i], uys[i]
        rows.append(
            [
                keys[i],
                wkts[i],
                float(lons[i]),
                float(lats[i]),
                float(sizes[i]),
                None if ux != ux else float(ux),
                None if uy != uy else float(uy),
                crss[i],
                *[float(emb[i, b]) for b in range(64)],
            ]
        )
    return rows


def _copy_upsert_cell_chunk(payload: dict) -> int:
    """One process: COPY a cell chunk into Postgres and upsert (single commit)."""
    city_id = payload["city_id"]
    aef_source_id = payload["aef_source_id"]
    rows = payload["rows"]
    if not rows:
        return 0

    import psycopg
    from psycopg import errors as pg_errors

    attempts = int(payload.get("attempts", 6))
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with _connect() as conn:
                with conn.cursor() as cur:
                    # Do NOT use ON COMMIT DROP — we commit once at the end of this txn.
                    cur.execute(
                        f"""
                        CREATE TEMP TABLE stage_cells (
                          cell_key TEXT,
                          wkt TEXT,
                          pixel_lon DOUBLE PRECISION,
                          pixel_lat DOUBLE PRECISION,
                          pixel_size_m DOUBLE PRECISION,
                          utm_x DOUBLE PRECISION,
                          utm_y DOUBLE PRECISION,
                          area_crs TEXT,
                          {", ".join(f"{c} DOUBLE PRECISION" for c in BAND_COLS)}
                        )
                        """
                    )
                    with cur.copy(
                        "COPY stage_cells ("
                        "cell_key, wkt, pixel_lon, pixel_lat, pixel_size_m, utm_x, utm_y, area_crs, "
                        + ", ".join(BAND_COLS)
                        + ") FROM STDIN"
                    ) as cp:
                        for row in rows:
                            cp.write_row(row)

                    cur.execute(
                        f"""
                        INSERT INTO alphaearth_cells (
                          city_id, geom, year, source, source_version, source_id, cell_key,
                          {", ".join(BAND_COLS)}, metadata
                        )
                        SELECT
                          %s,
                          ST_SetSRID(ST_GeomFromText(s.wkt), 4326),
                          %s, 'alphaearth', 'v1', %s, s.cell_key,
                          {", ".join("s." + c for c in BAND_COLS)},
                          jsonb_build_object('pixel_wkt_sha1', s.cell_key)
                        FROM stage_cells s
                        ON CONFLICT (city_id, year, cell_key) WHERE cell_key IS NOT NULL
                        DO UPDATE SET updated_at = now()
                        """,
                        (city_id, EMBED_YEAR, aef_source_id),
                    )
                    cur.execute(
                        """
                        INSERT INTO cell_spatial_metrics (
                          cell_id, pixel_lon, pixel_lat, pixel_size_m, pixel_area_m2,
                          area_crs, utm_x, utm_y
                        )
                        SELECT
                          a.cell_id, s.pixel_lon, s.pixel_lat, s.pixel_size_m, NULL,
                          s.area_crs, s.utm_x, s.utm_y
                        FROM stage_cells s
                        JOIN alphaearth_cells a
                          ON a.city_id = %s AND a.year = %s AND a.cell_key = s.cell_key
                        ON CONFLICT (cell_id) DO UPDATE SET
                          pixel_lon = EXCLUDED.pixel_lon,
                          pixel_lat = EXCLUDED.pixel_lat,
                          area_crs = EXCLUDED.area_crs,
                          utm_x = EXCLUDED.utm_x,
                          utm_y = EXCLUDED.utm_y,
                          updated_at = now()
                        """,
                        (city_id, EMBED_YEAR),
                    )
                conn.commit()
            return len(rows)
        except pg_errors.DeadlockDetected as exc:
            last_exc = exc
            time.sleep(0.4 * (attempt + 1) + 0.1 * (os.getpid() % 7))
            continue
        except psycopg.Error:
            raise
    assert last_exc is not None
    raise last_exc


def copy_upsert_cells(
    city_id: str,
    cells_df: pd.DataFrame,
    aef_source_id: int,
    area_crs: str,
    chunk_size: int = 10_000,
    writers: int = 8,
) -> dict[str, int]:
    """Parallel COPY+upsert of cells. Returns pixel_wkt → cell_id map."""
    if cells_df.empty:
        return {}

    df = _prepare_cells_df(cells_df, area_crs)
    n = len(df)
    writers = max(1, min(writers, max(1, (n + chunk_size - 1) // chunk_size)))

    payloads = []
    for start in range(0, n, chunk_size):
        part = df.iloc[start : start + chunk_size]
        payloads.append(
            {
                "city_id": city_id,
                "aef_source_id": aef_source_id,
                "rows": _chunk_rows_as_lists(part),
            }
        )

    with ProcessPoolExecutor(max_workers=writers) as pool:
        futs = [pool.submit(_copy_upsert_cell_chunk, p) for p in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"copy_cells:{city_id}"):
            fut.result()

    cell_id_by_wkt: dict[str, int] = {}
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.cell_id, a.cell_key, ST_AsText(a.geom) AS wkt
                FROM alphaearth_cells a
                WHERE a.city_id = %s AND a.year = %s
                """,
                (city_id, EMBED_YEAR),
            )
            key_to_id = {r["cell_key"]: int(r["cell_id"]) for r in cur.fetchall()}
    for key, wkt in zip(df["cell_key"].tolist(), df["pixel_wkt"].astype(str).tolist()):
        cid = key_to_id.get(key)
        if cid is not None:
            cell_id_by_wkt[wkt] = cid
    return cell_id_by_wkt


def _label_chunk_worker(payload: dict) -> dict[str, list]:
    """CPU worker: compute labels + intersections for a cell chunk (no DB)."""
    area_crs = payload["area_crs"]
    id_map = payload["id_map"]
    b = pd.DataFrame(payload["buildings"])
    cells = pd.DataFrame(payload["cells"])

    b_geoms = []
    for w in b["wkt"].tolist():
        g = shapely_wkt.loads(w)
        if not g.is_valid:
            g = g.buffer(0)
        b_geoms.append(g)
    bg = gpd.GeoDataFrame(
        {"id": b["id"].astype(str).tolist(), "y_res": b["y_res"].astype(int).tolist()},
        geometry=b_geoms,
        crs="EPSG:4326",
    ).to_crs(area_crs)

    p_geoms = [shapely_wkt.loads(w) for w in cells["pixel_wkt"].tolist()]
    pg = gpd.GeoDataFrame(cells, geometry=p_geoms, crs="EPSG:4326").to_crs(area_crs)
    sindex = bg.sindex

    labels: list[tuple] = []
    metrics: list[tuple] = []
    intersections: list[tuple] = []
    spatial: list[tuple] = []

    for _, prow in pg.iterrows():
        cid = int(prow["cell_id"])
        pixel = prow.geometry
        if pixel is None or pixel.is_empty:
            continue
        pixel_area = float(pixel.area)
        if pixel_area <= 0:
            continue
        cand_idx = list(sindex.query(pixel, predicate="intersects"))
        if not cand_idx:
            continue
        hits = bg.iloc[cand_idx]
        res_parts, nres_parts = [], []
        res_ids, nres_ids = [], []
        for _, h in hits.iterrows():
            if not pixel.intersects(h.geometry):
                continue
            inter = pixel.intersection(h.geometry)
            ia = float(inter.area)
            if ia <= 0:
                continue
            bid = id_map.get(str(h["id"]))
            if bid is not None:
                y = int(h["y_res"])
                intersections.append(
                    (
                        cid,
                        int(bid),
                        ia,
                        ia / pixel_area,
                        (ia / pixel_area) if y == 1 else 0.0,
                        (ia / pixel_area) if y == 0 else 0.0,
                    )
                )
            if int(h["y_res"]) == 1:
                res_parts.append(h.geometry)
                res_ids.append(str(h["id"]))
            else:
                nres_parts.append(h.geometry)
                nres_ids.append(str(h["id"]))
        if not res_parts and not nres_parts:
            continue
        area_res = float(pixel.intersection(unary_union(res_parts)).area) if res_parts else 0.0
        area_nres = float(pixel.intersection(unary_union(nres_parts)).area) if nres_parts else 0.0
        frac_res_full = float(np.clip(area_res / pixel_area, 0.0, 1.0))
        frac_nres_full = float(np.clip(area_nres / pixel_area, 0.0, 1.0))
        frac_empty_full = float(np.clip(1.0 - frac_res_full - frac_nres_full, 0.0, 1.0))
        frac_res, frac_nres, frac_empty = constrained_label_fracs(frac_res_full, frac_nres_full)
        pure_res = frac_res_full > 0 and frac_nres_full == 0
        pure_nres = frac_nres_full > 0 and frac_res_full == 0
        if pure_res:
            binary_label, purity = 1, "pure_residential"
        elif pure_nres:
            binary_label, purity = 0, "pure_non_residential"
        else:
            binary_label, purity = None, "mixed_or_empty"
        labels.append(
            (
                cid,
                LABEL_VERSION,
                frac_res,
                frac_nres,
                frac_empty,
                binary_label,
                purity,
                "ambiguous" if purity == "mixed_or_empty" else "clear",
            )
        )
        metrics.append(
            (
                cid,
                LABEL_VERSION,
                area_res,
                area_nres,
                frac_res_full,
                frac_nres_full,
                frac_empty_full,
                len(res_ids),
                len(nres_ids),
                len(set(res_ids + nres_ids)),
                area_crs,
            )
        )
        spatial.append((pixel_area, area_crs, cid))

    return {
        "labels": labels,
        "metrics": metrics,
        "intersections": intersections,
        "spatial": spatial,
    }


def _copy_upsert_labels_bundle(bundle: dict[str, list]) -> dict[str, int]:
    """One DB connection: COPY staging + upsert fragments from one worker."""
    n = {"labels": 0, "metrics": 0, "intersections": 0, "spatial": 0}
    with _connect() as conn:
        with conn.cursor() as cur:
            if bundle["intersections"]:
                cur.execute(
                    """
                    CREATE TEMP TABLE stage_ix (
                      cell_id BIGINT, building_id BIGINT,
                      intersection_area DOUBLE PRECISION,
                      cell_fraction DOUBLE PRECISION,
                      frac_res_contrib DOUBLE PRECISION,
                      frac_nres_contrib DOUBLE PRECISION
                    )
                    """
                )
                with cur.copy(
                    "COPY stage_ix (cell_id, building_id, intersection_area, cell_fraction, "
                    "frac_res_contrib, frac_nres_contrib) FROM STDIN"
                ) as cp:
                    for row in bundle["intersections"]:
                        cp.write_row(row)
                cur.execute(
                    """
                    INSERT INTO cell_building_intersections (
                      cell_id, building_id, intersection_area, cell_fraction,
                      frac_res_contrib, frac_nres_contrib
                    )
                    SELECT * FROM stage_ix
                    ON CONFLICT (cell_id, building_id) DO UPDATE SET
                      intersection_area = EXCLUDED.intersection_area,
                      cell_fraction = EXCLUDED.cell_fraction,
                      frac_res_contrib = EXCLUDED.frac_res_contrib,
                      frac_nres_contrib = EXCLUDED.frac_nres_contrib
                    """
                )
                n["intersections"] = len(bundle["intersections"])

            if bundle["labels"]:
                cur.execute(
                    """
                    CREATE TEMP TABLE stage_labels (
                      cell_id BIGINT, label_version TEXT,
                      frac_res DOUBLE PRECISION, frac_nres DOUBLE PRECISION, frac_empty DOUBLE PRECISION,
                      binary_label SMALLINT, purity_status TEXT, ambiguity_status TEXT
                    )
                    """
                )
                with cur.copy(
                    "COPY stage_labels (cell_id, label_version, frac_res, frac_nres, frac_empty, "
                    "binary_label, purity_status, ambiguity_status) FROM STDIN"
                ) as cp:
                    for row in bundle["labels"]:
                        cp.write_row(row)
                cur.execute(
                    """
                    INSERT INTO cell_labels (
                      cell_id, label_version, frac_res, frac_nres, frac_empty,
                      binary_label, purity_status, ambiguity_status
                    )
                    SELECT * FROM stage_labels
                    ON CONFLICT (cell_id, label_version) DO UPDATE SET
                      frac_res = EXCLUDED.frac_res,
                      frac_nres = EXCLUDED.frac_nres,
                      frac_empty = EXCLUDED.frac_empty,
                      binary_label = EXCLUDED.binary_label,
                      purity_status = EXCLUDED.purity_status,
                      ambiguity_status = EXCLUDED.ambiguity_status
                    """
                )
                n["labels"] = len(bundle["labels"])

            if bundle["metrics"]:
                cur.execute(
                    """
                    CREATE TEMP TABLE stage_metrics (
                      cell_id BIGINT, label_version TEXT,
                      area_res_m2 DOUBLE PRECISION, area_nres_m2 DOUBLE PRECISION,
                      frac_res_full DOUBLE PRECISION, frac_nres_full DOUBLE PRECISION, frac_empty_full DOUBLE PRECISION,
                      n_res_buildings INT, n_nres_buildings INT, n_buildings INT, area_crs TEXT
                    )
                    """
                )
                with cur.copy(
                    "COPY stage_metrics (cell_id, label_version, area_res_m2, area_nres_m2, "
                    "frac_res_full, frac_nres_full, frac_empty_full, n_res_buildings, "
                    "n_nres_buildings, n_buildings, area_crs) FROM STDIN"
                ) as cp:
                    for row in bundle["metrics"]:
                        cp.write_row(row)
                cur.execute(
                    """
                    INSERT INTO cell_label_metrics (
                      cell_id, label_version, area_res_m2, area_nres_m2,
                      frac_res_full, frac_nres_full, frac_empty_full,
                      n_res_buildings, n_nres_buildings, n_buildings, area_crs
                    )
                    SELECT * FROM stage_metrics
                    ON CONFLICT (cell_id, label_version) DO UPDATE SET
                      area_res_m2 = EXCLUDED.area_res_m2,
                      area_nres_m2 = EXCLUDED.area_nres_m2,
                      frac_res_full = EXCLUDED.frac_res_full,
                      frac_nres_full = EXCLUDED.frac_nres_full,
                      frac_empty_full = EXCLUDED.frac_empty_full,
                      n_res_buildings = EXCLUDED.n_res_buildings,
                      n_nres_buildings = EXCLUDED.n_nres_buildings,
                      n_buildings = EXCLUDED.n_buildings,
                      area_crs = EXCLUDED.area_crs
                    """
                )
                n["metrics"] = len(bundle["metrics"])

            if bundle["spatial"]:
                cur.execute(
                    """
                    CREATE TEMP TABLE stage_spatial (
                      pixel_area_m2 DOUBLE PRECISION, area_crs TEXT, cell_id BIGINT
                    )
                    """
                )
                with cur.copy(
                    "COPY stage_spatial (pixel_area_m2, area_crs, cell_id) FROM STDIN"
                ) as cp:
                    for row in bundle["spatial"]:
                        cp.write_row(row)
                cur.execute(
                    """
                    UPDATE cell_spatial_metrics m SET
                      pixel_area_m2 = s.pixel_area_m2,
                      area_crs = s.area_crs,
                      updated_at = now()
                    FROM stage_spatial s
                    WHERE m.cell_id = s.cell_id
                    """
                )
                n["spatial"] = len(bundle["spatial"])
        conn.commit()
    return n


def _compute_and_write_chunk(payload: dict) -> dict[str, int]:
    bundle = _label_chunk_worker(payload)
    return _copy_upsert_labels_bundle(bundle)


def insert_cells_and_labels_parallel(
    city_id: str,
    buildings_df: pd.DataFrame,
    id_map: dict[str, int],
    cells_df: pd.DataFrame,
    aef_source_id: int,
    workers: int = 32,
) -> dict[str, Any]:
    if cells_df.empty:
        return {"n_cells": 0, "n_labels": 0, "n_intersections": 0}

    area_crs = str(cells_df["crs"].mode().iloc[0]) if "crs" in cells_df.columns else "EPSG:4326"
    # Cap concurrent DB writers below max_connections (default 100); leave headroom.
    db_writers = max(4, min(workers, 32))
    cell_id_by_wkt = copy_upsert_cells(
        city_id,
        cells_df,
        aef_source_id,
        area_crs,
        chunk_size=10_000,
        writers=min(db_writers, 8),
    )
    if not cell_id_by_wkt:
        return {"n_cells": 0, "n_labels": 0, "n_intersections": 0, "area_crs": area_crs}

    pix = cells_df.drop_duplicates("pixel_wkt").copy()
    pix["cell_id"] = pix["pixel_wkt"].map(cell_id_by_wkt)
    pix = pix.dropna(subset=["cell_id"]).copy()
    pix["cell_id"] = pix["cell_id"].astype(int)

    b = buildings_df.copy()
    b["id"] = b["id"].astype(str)
    buildings_payload = b[["id", "wkt", "y_res"]].to_dict(orient="list")

    n_chunks = max(db_writers * 2, 8)
    chunks = np.array_split(pix, min(n_chunks, max(1, len(pix))))
    payloads = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        payloads.append(
            {
                "area_crs": area_crs,
                "id_map": id_map,
                "buildings": buildings_payload,
                "cells": ch[["cell_id", "pixel_wkt"]].to_dict(orient="list"),
            }
        )

    totals = {
        "n_cells": len(cell_id_by_wkt),
        "n_labels": 0,
        "n_intersections": 0,
        "area_crs": area_crs,
        "db_writers": db_writers,
    }
    with ProcessPoolExecutor(max_workers=db_writers) as pool:
        futs = [pool.submit(_compute_and_write_chunk, p) for p in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"labels_parallel:{city_id}"):
            n = fut.result()
            totals["n_labels"] += n.get("labels", 0)
            totals["n_intersections"] += n.get("intersections", 0)
    return totals



def _nullable_str_list(series, n: int) -> list:
    if series is None:
        return [None] * n
    out = []
    for v in series.tolist():
        if v is None:
            out.append(None)
        elif isinstance(v, float) and v != v:
            out.append(None)
        else:
            out.append(str(v))
    return out


def copy_upsert_buildings(
    city_id: str,
    buildings_df: pd.DataFrame,
    overture_source_id: int,
    release: str,
    chunk_size: int = 10_000,
) -> dict[str, int]:
    """COPY buildings + audit sidecars; return source_feature_id -> building_id."""
    if buildings_df.empty:
        return {}

    df = buildings_df.copy()
    df["id"] = df["id"].astype(str)
    n = len(df)
    keys = df["id"].tolist()
    wkts = df["wkt"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    subtypes = _nullable_str_list(df["subtype"] if "subtype" in df.columns else None, n)
    classes = df["class"].astype(str).tolist()
    harm = df["harmonized_label"].astype(str).tolist()
    briefs = _nullable_str_list(df["brief_category"] if "brief_category" in df.columns else None, n)
    yres = df["y_res"].astype(int).tolist()
    heights = df["height"].to_numpy(dtype=float) if "height" in df.columns else np.full(n, np.nan)
    floors = (
        df["num_floors"].to_numpy(dtype=float) if "num_floors" in df.columns else np.full(n, np.nan)
    )
    areas = df["area_m2_approx"].to_numpy(dtype=float)
    srcs = _nullable_str_list(df["source_primary"] if "source_primary" in df.columns else None, n)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TEMP TABLE stage_buildings (
                  source_feature_id TEXT,
                  wkt TEXT,
                  building_type TEXT,
                  subtype TEXT,
                  original_source_label TEXT,
                  harmonized_label TEXT,
                  brief_category TEXT,
                  study_label TEXT,
                  y_res SMALLINT,
                  height_m DOUBLE PRECISION,
                  num_floors INT,
                  area_m2_approx DOUBLE PRECISION,
                  source_primary TEXT
                )
                """
            )
            for start in tqdm(range(0, n, chunk_size), desc=f"copy_buildings:{city_id}"):
                end = min(start + chunk_size, n)
                cur.execute("TRUNCATE stage_buildings")
                with cur.copy(
                    "COPY stage_buildings ("
                    "source_feature_id, wkt, building_type, subtype, original_source_label, "
                    "harmonized_label, brief_category, study_label, y_res, height_m, "
                    "num_floors, area_m2_approx, source_primary"
                    ") FROM STDIN"
                ) as cp:
                    for i in range(start, end):
                        h, f, a = heights[i], floors[i], areas[i]
                        cp.write_row(
                            [
                                keys[i],
                                wkts[i],
                                labels[i],
                                subtypes[i],
                                classes[i],
                                harm[i],
                                briefs[i],
                                labels[i],
                                yres[i],
                                None if h != h else float(h),
                                None if f != f else int(f),
                                None if a != a else float(a),
                                srcs[i],
                            ]
                        )
                cur.execute(
                    """
                    INSERT INTO buildings (
                      city_id, source, source_feature_id, geom,
                      building_type, subtype, original_source_label, harmonized_label,
                      source_version, source_id, metadata
                    )
                    SELECT
                      %s, 'overture', s.source_feature_id,
                      ST_SetSRID(ST_GeomFromText(s.wkt), 4326),
                      s.building_type, s.subtype, s.original_source_label, s.harmonized_label,
                      %s, %s, '{}'::jsonb
                    FROM stage_buildings s
                    ON CONFLICT (source, source_feature_id, city_id) DO UPDATE SET
                      geom = EXCLUDED.geom,
                      building_type = EXCLUDED.building_type,
                      subtype = EXCLUDED.subtype,
                      original_source_label = EXCLUDED.original_source_label,
                      harmonized_label = EXCLUDED.harmonized_label,
                      updated_at = now()
                    """,
                    (city_id, release, overture_source_id),
                )
                cur.execute(
                    """
                    INSERT INTO building_label_audit (
                      building_id, label_definition, overture_class, overture_subtype,
                      brief_category, study_label, y_res, height_m, num_floors,
                      area_m2_approx, source_primary
                    )
                    SELECT
                      b.building_id, 'resi_def_v1', s.original_source_label, s.subtype,
                      s.brief_category, s.study_label, s.y_res, s.height_m, s.num_floors,
                      s.area_m2_approx, s.source_primary
                    FROM stage_buildings s
                    JOIN buildings b
                      ON b.city_id = %s AND b.source = 'overture'
                     AND b.source_feature_id = s.source_feature_id
                    ON CONFLICT (building_id) DO UPDATE SET
                      overture_class = EXCLUDED.overture_class,
                      study_label = EXCLUDED.study_label,
                      y_res = EXCLUDED.y_res,
                      height_m = EXCLUDED.height_m,
                      num_floors = EXCLUDED.num_floors,
                      area_m2_approx = EXCLUDED.area_m2_approx,
                      updated_at = now()
                    """,
                    (city_id,),
                )
                conn.commit()

            cur.execute(
                """
                SELECT source_feature_id, building_id
                FROM buildings
                WHERE city_id = %s AND source = 'overture'
                """,
                (city_id,),
            )
            return {str(r["source_feature_id"]): int(r["building_id"]) for r in cur.fetchall()}
