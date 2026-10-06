"""Stratified footprint-area balancing for AlphaEarth download gating."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Week4 Track-A downtown footprints: sum area_m2_approx over 10 metros.
# Equal per-city AEF budget so every city downloads the same footprint area.
WEEK4_TOTAL_FOOTPRINT_M2 = 246_042_269.8057363
WEEK4_N_CITIES = 10
WEEK4_EQUAL_CITY_AREA_M2 = WEEK4_TOTAL_FOOTPRINT_M2 / WEEK4_N_CITIES  # ≈ 24.60 km²


def haversine_m(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: float,
    lat2: float,
) -> np.ndarray:
    r = 6_371_000.0
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1
    dlat = lat2r - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def take_area_budget(df: pd.DataFrame, budget: float, seed: int) -> pd.DataFrame:
    """Keep a random permutation of rows until cumulative footprint ≈ budget."""
    if df.empty or budget <= 0:
        return df.iloc[0:0].copy()
    tot = float(df["area_m2_approx"].fillna(0).sum())
    if tot <= budget:
        return df
    sub = df.sample(frac=1.0, random_state=seed)
    areas = sub["area_m2_approx"].fillna(0).to_numpy(dtype=float)
    keep_idx: list = []
    acc = 0.0
    for i, a in zip(sub.index.tolist(), areas):
        if acc >= budget:
            break
        keep_idx.append(i)
        acc += float(a)
    return sub.loc[keep_idx]


def area_balance_ring(df: pd.DataFrame, headroom: float = 1.10, seed: int = 42) -> pd.DataFrame:
    """Keep minority class by footprint area; thin majority to minority_area * headroom."""
    if df.empty:
        return df.copy()
    if "area_m2_approx" not in df.columns:
        raise KeyError("area_m2_approx required for area balance")
    if "y_res" not in df.columns:
        raise KeyError("y_res required for area balance")

    res = df.loc[df["y_res"] == 1].copy()
    nres = df.loc[df["y_res"] == 0].copy()
    ar = float(res["area_m2_approx"].fillna(0).sum())
    an = float(nres["area_m2_approx"].fillna(0).sum())
    if ar <= 0 and an <= 0:
        return df.copy()
    if ar <= 0:
        return nres.reset_index(drop=True)
    if an <= 0:
        return res.reset_index(drop=True)

    target = min(ar, an) * float(headroom)
    if ar <= an:
        out = pd.concat([res, take_area_budget(nres, target, seed)], ignore_index=True)
    else:
        out = pd.concat([take_area_budget(res, target, seed), nres], ignore_index=True)
    return out.reset_index(drop=True)


def cap_to_target_area(
    df: pd.DataFrame,
    target_area_m2: float,
    *,
    seed: int = 42,
    ring_col: str = "sample_ring",
) -> pd.DataFrame:
    """Thin to a city-wide footprint budget, preserving class mix and ring shares.

    Allocates target to RES/NRES proportional to their current areas (keeps the
    post ring-balance ratio), then within each class allocates across rings
    proportional to that class's area in the ring.
    """
    if df.empty or target_area_m2 is None or target_area_m2 <= 0:
        return df.copy()
    tot = float(df["area_m2_approx"].fillna(0).sum())
    if tot <= float(target_area_m2):
        return df.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for y, class_seed in ((1, seed + 17), (0, seed + 29)):
        sub = df.loc[df["y_res"] == y]
        if sub.empty:
            continue
        class_area = float(sub["area_m2_approx"].fillna(0).sum())
        class_budget = float(target_area_m2) * (class_area / tot)
        if ring_col not in sub.columns:
            parts.append(take_area_budget(sub, class_budget, class_seed))
            continue
        rings = sorted(sub[ring_col].dropna().unique().tolist())
        ring_areas = {
            r: float(sub.loc[sub[ring_col] == r, "area_m2_approx"].fillna(0).sum()) for r in rings
        }
        ring_tot = sum(ring_areas.values()) or 1.0
        for j, r in enumerate(rings):
            ring_df = sub.loc[sub[ring_col] == r]
            ring_budget = class_budget * (ring_areas[r] / ring_tot)
            parts.append(take_area_budget(ring_df, ring_budget, class_seed + j))

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def stratified_area_balance(
    df: pd.DataFrame,
    center_lon: float,
    center_lat: float,
    *,
    ring_km: float = 5.0,
    max_radius_m: float | None = None,
    headroom: float = 1.10,
    target_area_m2: float | None = WEEK4_EQUAL_CITY_AREA_M2,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Area-balance RES/NRES within distance rings, then cap to a city footprint budget.

    Default ``target_area_m2`` is week4 total Track-A footprint / 10 metros
    (~24.60 km²) so every city downloads the same building area.

    Returns (sampled_df, manifest).
    """
    if df.empty:
        return df.copy(), {"n_in": 0, "n_out": 0, "rings": []}

    out = df.copy()
    dist_m = haversine_m(out["lon"].to_numpy(), out["lat"].to_numpy(), center_lon, center_lat)
    out = out.assign(dist_to_center_m=dist_m)
    if max_radius_m is not None:
        out = out.loc[out["dist_to_center_m"] <= float(max_radius_m)].copy()

    ring_m = float(ring_km) * 1000.0
    max_d = float(out["dist_to_center_m"].max()) if len(out) else 0.0
    n_rings = int(np.ceil(max_d / ring_m)) if max_d > 0 else 1
    chunks: list[pd.DataFrame] = []
    ring_stats: list[dict] = []
    for i in range(max(n_rings, 1)):
        lo, hi = i * ring_m, (i + 1) * ring_m
        ring = out.loc[(out["dist_to_center_m"] >= lo) & (out["dist_to_center_m"] < hi)]
        if ring.empty:
            continue
        bal = area_balance_ring(ring, headroom=headroom, seed=seed + i)
        bal = bal.assign(sample_ring=i)
        chunks.append(bal)
        ring_stats.append(
            {
                "ring_km": [lo / 1000.0, hi / 1000.0],
                "n_in": int(len(ring)),
                "n_out_balanced": int(len(bal)),
                "area_res_in": float(ring.loc[ring["y_res"] == 1, "area_m2_approx"].fillna(0).sum()),
                "area_nres_in": float(ring.loc[ring["y_res"] == 0, "area_m2_approx"].fillna(0).sum()),
                "area_res_balanced": float(bal.loc[bal["y_res"] == 1, "area_m2_approx"].fillna(0).sum()),
                "area_nres_balanced": float(bal.loc[bal["y_res"] == 0, "area_m2_approx"].fillna(0).sum()),
            }
        )

    balanced = pd.concat(chunks, ignore_index=True) if chunks else out.iloc[0:0].copy()
    area_balanced = float(balanced["area_m2_approx"].fillna(0).sum()) if len(balanced) else 0.0
    sampled = cap_to_target_area(
        balanced,
        float(target_area_m2) if target_area_m2 is not None else area_balanced,
        seed=seed + 100,
    )
    for i, st in enumerate(ring_stats):
        if "sample_ring" in sampled.columns:
            ring_s = sampled.loc[sampled["sample_ring"] == i]
        else:
            ring_s = sampled.iloc[0:0]
        st["n_out"] = int(len(ring_s))
        st["area_res_out"] = float(ring_s.loc[ring_s["y_res"] == 1, "area_m2_approx"].fillna(0).sum())
        st["area_nres_out"] = float(ring_s.loc[ring_s["y_res"] == 0, "area_m2_approx"].fillna(0).sum())
    if "sample_ring" in sampled.columns:
        sampled = sampled.drop(columns=["sample_ring"])

    area_out = float(sampled["area_m2_approx"].fillna(0).sum()) if len(sampled) else 0.0
    manifest = {
        "n_in": int(len(df)),
        "n_out": int(len(sampled)),
        "keep_frac": float(len(sampled) / max(len(df), 1)),
        "ring_km": float(ring_km),
        "headroom": float(headroom),
        "target_area_m2": None if target_area_m2 is None else float(target_area_m2),
        "target_area_km2": None if target_area_m2 is None else float(target_area_m2) / 1e6,
        "area_in_m2": float(out["area_m2_approx"].fillna(0).sum()) if len(out) else 0.0,
        "area_balanced_m2": area_balanced,
        "area_out_m2": area_out,
        "area_out_km2": area_out / 1e6,
        "seed": int(seed),
        "center_lon": float(center_lon),
        "center_lat": float(center_lat),
        "max_radius_m": None if max_radius_m is None else float(max_radius_m),
        "rings": ring_stats,
    }
    return sampled, manifest
