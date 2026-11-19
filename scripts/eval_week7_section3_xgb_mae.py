#!/usr/bin/env python3
"""Week 7 Section 3 MAE: XGB vs Ridge/PLS/BUILT-S on the *same* Ridge eval sets.

Building cohort: exact IDs from week6 soft∩AE∩GHSL∩PLS on y_res-balanced
buildings (n=71,497) — `outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet`.

100 m cohort: exact tile_ids from week6 cell+tile soft-argmax board (n=25,220).

Only AE predictions are recomputed from Week 7 XGB OOF; Ridge/PLS/GHSL/soft GT
are taken from the frozen week6 artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
BUILDINGS_META = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
BUILDING_CELLS = ROOT / "datasets" / "building_bbox_grid" / "building_cells.parquet"
W6_BUILD = (
    ROOT
    / "outputs"
    / "week6_overture_polygon_mae"
    / "20261110_041557"
    / "artifacts"
    / "per_building_with_pls.parquet"
)
W6_100M = ROOT / "outputs" / "week6_overture_polygon_mae_100m" / "20261110_062739"
COHORT_BUILD = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_building_ids.parquet"
COHORT_TILE = ROOT / "outputs" / "week7_section3_xgb_mae" / "ridge_section3_tile_ids.parquet"
LONLAT_DECIMALS = 7


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
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
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
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


def lonlat_key(lon, lat, nd: int = LONLAT_DECIMALS) -> np.ndarray:
    """Return aligned lon_lat keys as a plain ndarray (never a Series).

    Important: do not wrap in ``pd.Series`` — assigning a 0..n-1 Series into a
    filtered DataFrame aligns on index and silently mismatches ~most rows.
    """
    lon_s = np.round(np.asarray(lon, dtype=np.float64), nd).astype(str)
    lat_s = np.round(np.asarray(lat, dtype=np.float64), nd).astype(str)
    return np.char.add(lon_s.astype(object), np.char.add("_", lat_s.astype(object)))


def mae_pair(y_res, y_nres, p_res, p_nres) -> dict[str, float]:
    m = np.isfinite(y_res) & np.isfinite(y_nres) & np.isfinite(p_res) & np.isfinite(p_nres)
    if m.sum() == 0:
        return {"n": 0, "res": float("nan"), "nres": float("nan"), "mean": float("nan")}
    r = float(mean_absolute_error(y_res[m], p_res[m]))
    n = float(mean_absolute_error(y_nres[m], p_nres[m]))
    return {"n": int(m.sum()), "res": r, "nres": n, "mean": 0.5 * (r + n)}


# Match Section 3 metro order / row colours.
METRO_META = [
    ("Boston_NE", "Boston NE", "#ECE8F4"),
    ("NYC_NE", "NYC NE", "#ECE8F4"),
    ("Atlanta_SE", "Atlanta SE", "#E2EFE6"),
    ("Houston_S", "Houston S", "#F3EDD5"),
    ("Chicago_MW", "Chicago MW", "#F3E3E7"),
    ("Phoenix_SW", "Phoenix SW", "#EFE4D9"),
    ("Denver_W", "Denver W", "#E1ECEE"),
    ("LA_W", "LA W", "#E1ECEE"),
    ("SF_W", "SF W", "#E1ECEE"),
    ("Seattle_NW", "Seattle NW", "#E9EDDC"),
]


def _decorate_metro(metro: list[dict]) -> list[dict]:
    by = {r["region"]: r for r in metro}
    out = []
    for region, label, bg in METRO_META:
        if region not in by:
            continue
        r = dict(by[region])
        r["label"] = label
        r["bg"] = bg
        # Section-3-compatible aliases + XGB vs Ridge margin
        r["bs"] = r["ghsl_built_s"]
        r["rg"] = r["ae_ridge"]
        r["pl"] = r["ae_pls"]
        r["xg"] = r["ae_xgb"]
        r["margin"] = float(r["ae_xgb"]["mean"] - r["ae_ridge"]["mean"])
        out.append(r)
    return out


def resolve_xgb_oof(explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    roots = [
        ROOT / "outputs" / "week7_xgb_loro_gpu",
        ROOT / "outputs" / "week7_xgb_loro",
    ]
    for root in roots:
        for link_name in ("latest", "latest_run"):
            latest = root / link_name / "artifacts" / "oof_predictions_full_gt.parquet"
            if latest.exists():
                return latest.resolve()
    cands: list[Path] = []
    for root in roots:
        cands.extend(root.glob("*/artifacts/oof_predictions_full_gt.parquet"))
    cands = sorted(cands, key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError("No week7 XGB full-GT OOF found; train first")
    return cands[-1]


def aggregate_xgb_to_buildings(xgb_oof: Path, run_dir: Path) -> pd.DataFrame:
    """Mean XGB preds over inside_polygon cells per building (Ridge-compatible join)."""
    log(run_dir, f"Aggregating XGB OOF → buildings from {xgb_oof}")
    oof = pd.read_parquet(
        xgb_oof,
        columns=["pixel_lon", "pixel_lat", "pred_frac_res_full", "pred_frac_nres_full"],
    )
    oof["k"] = lonlat_key(oof["pixel_lon"].to_numpy(), oof["pixel_lat"].to_numpy())
    oof = oof.drop_duplicates("k")

    cells = pd.read_parquet(
        BUILDING_CELLS,
        columns=["id", "pixel_lon", "pixel_lat", "inside_polygon"],
    )
    cells = cells[cells["inside_polygon"]].copy()
    cells["id"] = cells["id"].astype(str)
    cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    j = cells.merge(
        oof[["k", "pred_frac_res_full", "pred_frac_nres_full"]], on="k", how="left"
    )
    agg = (
        j.groupby("id", sort=False)
        .agg(
            xgb_frac_res=("pred_frac_res_full", "mean"),
            xgb_frac_nres=("pred_frac_nres_full", "mean"),
            xgb_n_matched=("pred_frac_res_full", "count"),
        )
        .reset_index()
    )
    agg["xgb_covered"] = agg["xgb_n_matched"] > 0
    out = run_dir / "checkpoints" / "xgb_by_building.parquet"
    agg.to_parquet(out, index=False)
    log(run_dir, f"XGB buildings covered {int(agg.xgb_covered.sum()):,}/{len(agg):,}")
    return agg


def score_buildings(run_dir: Path, xgb_b: pd.DataFrame) -> dict[str, Any]:
    cohort = pd.read_parquet(COHORT_BUILD)
    cohort["id"] = cohort["id"].astype(str)
    base = pd.read_parquet(W6_BUILD)
    base["id"] = base["id"].astype(str)
    # freeze to Ridge Section 3 IDs
    m = cohort.merge(base, on=["id", "region"], how="left", suffixes=("", "_b"))
    if "y_res_b" in m.columns and m["y_res"].isna().any():
        m["y_res"] = m["y_res"].fillna(m["y_res_b"])
    m = m.merge(xgb_b, on="id", how="left")
    assert len(m) == len(cohort), f"cohort size drift {len(m)} vs {len(cohort)}"
    n_xgb = int(m["xgb_covered"].fillna(False).sum())
    log(run_dir, f"Building eval set n={len(m):,} · XGB covered {n_xgb:,}")

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ae_frac_res", "ae_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
    }
    soft_rows = []
    hard_rows = []
    for name, (pr, pn) in models.items():
        soft_rows.append(
            {
                "model": name,
                "gt": "soft",
                **mae_pair(
                    m["soft_gt_frac_res"].to_numpy(float),
                    m["soft_gt_frac_nres"].to_numpy(float),
                    m[pr].to_numpy(float),
                    m[pn].to_numpy(float),
                ),
            }
        )
        hard_rows.append(
            {
                "model": name,
                "gt": "hard",
                **mae_pair(
                    m["hard_gt_frac_res"].to_numpy(float),
                    m["hard_gt_frac_nres"].to_numpy(float),
                    m[pr].to_numpy(float),
                    m[pn].to_numpy(float),
                ),
            }
        )

    metro = []
    for region, sub in m.groupby("region", sort=False):
        row = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    metro = _decorate_metro(metro)

    m.to_parquet(run_dir / "artifacts" / "buildings_section3_with_xgb.parquet", index=False)
    pd.DataFrame(soft_rows + hard_rows).to_csv(
        run_dir / "artifacts" / "mae_buildings.csv", index=False
    )
    atomic_json(run_dir / "artifacts" / "mae_buildings_metro.json", metro)
    soft_pooled = {r["model"]: r for r in soft_rows}
    log(run_dir, "Building soft MAE (same 71,497 IDs as Ridge Section 3):")
    for name in ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb"]:
        r = soft_pooled[name]
        log(run_dir, f"  {name:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}")
    return {
        "n": len(m),
        "n_xgb_covered": n_xgb,
        "soft_pooled": soft_pooled,
        "hard_pooled": {r["model"]: r for r in hard_rows},
        "metro_soft": metro,
    }


def score_tiles(run_dir: Path, xgb_oof: Path) -> dict[str, Any]:
    """Attach XGB means onto locked Section 3 tiles via week6 cells_balanced."""
    tiles = pd.read_parquet(W6_100M / "artifacts" / "tiles_100m.parquet")
    cohort = pd.read_parquet(COHORT_TILE)
    assert set(tiles["tile_id"]) == set(cohort["tile_id"]), "tile cohort drift vs week6 board"

    cells_path = W6_100M / "checkpoints" / "cells_balanced.parquet"
    cells = pd.read_parquet(cells_path)
    oof = pd.read_parquet(
        xgb_oof,
        columns=["pixel_lon", "pixel_lat", "pred_frac_res_full", "pred_frac_nres_full"],
    )
    oof["k"] = lonlat_key(oof["pixel_lon"].to_numpy(), oof["pixel_lat"].to_numpy())
    oof = oof.drop_duplicates("k")
    if "k" not in cells.columns:
        cells["k"] = lonlat_key(cells["pixel_lon"].to_numpy(), cells["pixel_lat"].to_numpy())
    j = cells.merge(oof[["k", "pred_frac_res_full", "pred_frac_nres_full"]], on="k", how="left")
    # tile assignment: reuse week6 tile map if present on cells, else join tiles via building path
    if "tile_id" not in j.columns:
        tile_cells = pd.read_parquet(
            ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet",
            columns=["pixel_id", "region", "tile_id"],
        )
        if "pixel_id" not in j.columns:
            # rebuild from lonlat against ridge oof keys — cells_balanced should have pixel_id
            raise RuntimeError("cells_balanced missing tile_id and pixel_id")
        j = j.merge(tile_cells, on=["pixel_id", "region"], how="left")

    xgb_tiles = (
        j.groupby("tile_id", sort=False)
        .agg(
            xgb_frac_res=("pred_frac_res_full", "mean"),
            xgb_frac_nres=("pred_frac_nres_full", "mean"),
            xgb_n=("pred_frac_res_full", "count"),
        )
        .reset_index()
    )
    t = tiles.merge(xgb_tiles, on="tile_id", how="left")
    assert len(t) == len(tiles)

    models = {
        "ghsl_built_s": ("ghsl_frac_res", "ghsl_frac_nres"),
        "ae_ridge": ("ridge_frac_res", "ridge_frac_nres"),
        "ae_pls": ("pls_frac_res", "pls_frac_nres"),
        "ae_xgb": ("xgb_frac_res", "xgb_frac_nres"),
    }
    rows = []
    for name, (pr, pn) in models.items():
        rows.append(
            {
                "model": name,
                "scale": "100m",
                **mae_pair(
                    t["soft_gt_frac_res"].to_numpy(float),
                    t["soft_gt_frac_nres"].to_numpy(float),
                    t[pr].to_numpy(float),
                    t[pn].to_numpy(float),
                ),
            }
        )
    metro = []
    for region, sub in t.groupby("region", sort=False):
        row = {"region": region, "n": len(sub)}
        for name, (pr, pn) in models.items():
            row[name] = mae_pair(
                sub["soft_gt_frac_res"].to_numpy(float),
                sub["soft_gt_frac_nres"].to_numpy(float),
                sub[pr].to_numpy(float),
                sub[pn].to_numpy(float),
            )
        metro.append(row)
    metro = _decorate_metro(metro)

    t.to_parquet(run_dir / "artifacts" / "tiles_section3_with_xgb.parquet", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "artifacts" / "mae_tiles_100m.csv", index=False)
    atomic_json(run_dir / "artifacts" / "mae_tiles_metro.json", metro)
    log(run_dir, "100 m soft MAE (same 25,220 tiles as Ridge Section 3):")
    for r in rows:
        log(
            run_dir,
            f"  {r['model']:14s} mean={r['mean']:.4f} res={r['res']:.4f} nres={r['nres']:.4f} n={r['n']}",
        )
    return {"n": len(t), "soft_pooled": {r["model"]: r for r in rows}, "metro_soft": metro}


def make_figures(run_dir: Path, building: dict, tiles: dict) -> None:
    order = ["ghsl_built_s", "ae_ridge", "ae_pls", "ae_xgb"]
    labels = {
        "ghsl_built_s": "GHS-BUILT-S",
        "ae_ridge": "AE Ridge",
        "ae_pls": "AE PLS",
        "ae_xgb": "AE XGB",
    }
    colors = {
        "ghsl_built_s": "#2E7D32",
        "ae_ridge": "#1F6FA8",
        "ae_pls": "#C2691A",
        "ae_xgb": "#8E44AD",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, block, title in [
        (axes[0], building["soft_pooled"], f"Buildings (n={building['n']:,})"),
        (axes[1], tiles["soft_pooled"], f"100 m tiles (n={tiles['n']:,})"),
    ]:
        vals = [block[m]["mean"] for m in order]
        ax.barh([labels[m] for m in order], vals, color=[colors[m] for m in order])
        for y, v in enumerate(vals):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("mae_mean_res_nres (soft GT)")
        ax.set_title(title)
        ax.set_xlim(0, max(vals) * 1.25)
    fig.suptitle("Week 7 · Section 3 MAE on Ridge eval sets · XGB vs baselines")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "section3_mae_headline.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xgb-oof", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=ROOT / "outputs" / "week7_section3_xgb_mae")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / stamp
    for sub in ("artifacts", "checkpoints", "figures", "logs", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=4,
    )
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"argv={sys.argv}\n")

    if not COHORT_BUILD.exists() or not COHORT_TILE.exists():
        raise FileNotFoundError("Missing ridge_section3_* cohort files")

    xgb_oof = resolve_xgb_oof(args.xgb_oof)
    log(run_dir, f"XGB OOF={xgb_oof}")
    log(run_dir, f"Building cohort={COHORT_BUILD} Tile cohort={COHORT_TILE}")
    log(run_dir, f"Frozen Ridge table={W6_BUILD}")
    log(run_dir, f"Frozen 100m tiles={W6_100M / 'artifacts' / 'tiles_100m.parquet'}")

    write_progress(run_dir, stage="agg_xgb_buildings", completed=0)
    xgb_b = aggregate_xgb_to_buildings(xgb_oof, run_dir)
    write_progress(run_dir, stage="score_buildings", completed=1)
    building = score_buildings(run_dir, xgb_b)
    write_progress(run_dir, stage="score_tiles", completed=2)
    tiles = score_tiles(run_dir, xgb_oof)
    write_progress(run_dir, stage="figures", completed=3)
    make_figures(run_dir, building, tiles)

    def _wins(metro: list[dict], key: str = "mean") -> int:
        """Count metros where XGB mean beats Ridge mean."""
        return sum(1 for r in metro if r["ae_xgb"][key] + 1e-12 < r["ae_ridge"][key])

    board_metrics = {
        "building": {
            "n_shared": building["n"],
            "soft_pooled": {
                "built_s": building["soft_pooled"]["ghsl_built_s"],
                "ridge": building["soft_pooled"]["ae_ridge"],
                "pls": building["soft_pooled"]["ae_pls"],
                "xgb": building["soft_pooled"]["ae_xgb"],
            },
            "metro_soft": building["metro_soft"],
            "xgb_beats_ridge": _wins(building["metro_soft"]),
            "soft_wins_built_s": sum(
                1
                for r in building["metro_soft"]
                if r["bs"]["mean"]
                <= min(r["rg"]["mean"], r["pl"]["mean"], r["xg"]["mean"]) + 1e-12
            ),
        },
        "tile_100m": {
            "n_tiles": tiles["n"],
            "soft_pooled": {
                "built_s": tiles["soft_pooled"]["ghsl_built_s"],
                "ridge": tiles["soft_pooled"]["ae_ridge"],
                "pls": tiles["soft_pooled"]["ae_pls"],
                "xgb": tiles["soft_pooled"]["ae_xgb"],
            },
            "metro_soft": tiles["metro_soft"],
            "xgb_beats_ridge": _wins(tiles["metro_soft"]),
            "soft_wins_built_s": sum(
                1
                for r in tiles["metro_soft"]
                if r["bs"]["mean"]
                <= min(r["rg"]["mean"], r["pl"]["mean"], r["xg"]["mean"]) + 1e-12
            ),
        },
        "protocol": "section3_ridge_eval_sets_v1_plus_xgb",
        "xgb_oof": str(xgb_oof),
    }
    atomic_json(run_dir / "artifacts" / "section4_xgb_metrics.json", board_metrics)

    summary = {
        "status": "completed",
        "protocol": "section3_ridge_eval_sets_v1",
        "note": "Exact building IDs and tile IDs from Week 6 Section 3 Ridge board",
        "xgb_oof": str(xgb_oof),
        "building": building,
        "tiles_100m": tiles,
        "runtime_seconds": round(time.time() - t0, 1),
        "artifacts": {
            "mae_buildings": "artifacts/mae_buildings.csv",
            "mae_tiles": "artifacts/mae_tiles_100m.csv",
            "figure": "figures/section3_mae_headline.png",
            "section4_metrics": "artifacts/section4_xgb_metrics.json",
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(run_dir, status="completed", stage="done", completed=4, total=4, percent_complete=100.0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    latest = args.out_root / "latest_run"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    log(run_dir, f"DONE · {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
