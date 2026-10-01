#!/usr/bin/env python3
"""Run GHSL GHS-BUILT-C FUN vs model validation (RES / NRES only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.bootstrap import cluster_bootstrap, summarize_bootstrap
from validation.building_eval import evaluate_buildings
from validation.ghsl import ensure_tiles, open_ghsl, tile_ids_for_bounds
from validation.io import audit_gt, load_config, load_gt_buildings, load_predictions
from validation.plots import plot_comparison_bars, plot_coverage, plot_per_aoi_macro_f1
from validation.pixel_eval import evaluate_pixels
from validation.report import _flatten_functional, disagreement_buildings, write_comparison_table, write_summary_json
from validation.spatial import aoi_bounds_wgs84, held_out_claim_ok, to_ghsl_crs, validate_spatial_independence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "validation" / "config.yaml"))
    ap.add_argument("--skip-pixel", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    import os

    print(f"CPU cores={os.cpu_count()}  config.workers={cfg.get('workers')}", flush=True)
    print("Loading GT + predictions …", flush=True)
    gt = load_gt_buildings(cfg["paths"]["gt_buildings"], cfg)
    preds = load_predictions(cfg["paths"]["model_predictions"])
    audit = audit_gt(gt, cfg)
    print(json.dumps(audit, indent=2), flush=True)
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2))

    # Spatial independence (optional training geometries)
    train_path = cfg["paths"].get("training_buildings")
    train_gdf = None
    if train_path and Path(train_path).exists():
        train_gdf = load_gt_buildings(train_path, cfg)
    independence = validate_spatial_independence(
        gt, train_gdf, buffer_km=float(cfg["spatial_independence"]["buffer_km"])
    )
    independence.to_csv(out_dir / "spatial_independence.csv", index=False)
    held_ok, held_msg = held_out_claim_ok(
        independence, bool(cfg["spatial_independence"]["refuse_held_out_label_on_leakage"])
    )
    print(f"Held-out claim: {held_ok} · {held_msg}", flush=True)
    if not held_ok and cfg["spatial_independence"]["refuse_held_out_label_on_leakage"]:
        # Still run metrics, but mark summary as not held-out
        pass

    # Ensure GHSL tiles covering all AOIs
    print("Ensuring GHSL tiles …", flush=True)
    gt_m = to_ghsl_crs(gt)
    needed = set()
    for aoi, sub in gt_m.groupby("aoi_id"):
        minx, miny, maxx, maxy = sub.total_bounds
        needed.update(tile_ids_for_bounds(minx, miny, maxx, maxy, cfg))
    ensure_tiles(sorted(needed), cfg)
    ghsl_ds = open_ghsl(cfg)
    print(f"GHSL tiles open: {len(ghsl_ds.tiles)} CRS={ghsl_ds.crs}", flush=True)

    comparison_rows: list[dict] = []
    summary_payload: dict = {
        "held_out_ok": held_ok,
        "held_out_message": held_msg,
        "dataset_audit": audit,
        "aoi_bounds_wgs84": aoi_bounds_wgs84(gt),
    }

    # Building-level
    if cfg["building_validation"]["enabled"]:
        print("Building-level evaluation …", flush=True)
        per_bldg, bldg_summary = evaluate_buildings(gt, preds, ghsl_ds, cfg)
        per_bldg.to_parquet(out_dir / "per_building_predictions.parquet", index=False)
        comparison_rows.extend(_flatten_functional("building", bldg_summary))
        summary_payload["building"] = bldg_summary
        per_aoi_b = pd.DataFrame(bldg_summary["per_aoi"])
        per_aoi_b["level"] = "building"

        disagreement_buildings(per_bldg, gt, out_dir / "disagreement_buildings.gpkg")

        if not args.skip_bootstrap and len(per_bldg):
            print("Bootstrap (building, functional) …", flush=True)
            func = per_bldg[per_bldg["ghsl_covered"]].copy()
            boot_m = cluster_bootstrap(func, "gt_label", "model_prediction", "aoi_id", cfg)
            boot_g = cluster_bootstrap(func, "gt_label", "ghsl_prediction", "aoi_id", cfg)
            boot_m["system"] = "model"
            boot_g["system"] = "ghsl"
            boot = pd.concat([boot_m, boot_g], ignore_index=True)
            boot.to_parquet(out_dir / "bootstrap_results.parquet", index=False)
            summary_payload["bootstrap_building"] = {
                "model": summarize_bootstrap(boot_m, cfg["bootstrap"]["confidence_level"]).to_dict(orient="records"),
                "ghsl": summarize_bootstrap(boot_g, cfg["bootstrap"]["confidence_level"]).to_dict(orient="records"),
            }

        plot_coverage(bldg_summary["end_to_end"], out_dir, "building")
        plot_per_aoi_macro_f1(per_aoi_b, out_dir, "building")
    else:
        per_aoi_b = pd.DataFrame()

    # Pixel-level
    per_aoi_p = pd.DataFrame()
    if cfg["pixel_validation"]["enabled"] and not args.skip_pixel:
        print("Pixel-level evaluation …", flush=True)
        per_pix, pix_summary = evaluate_pixels(gt, preds, ghsl_ds, cfg)
        if len(per_pix):
            per_pix.to_parquet(out_dir / "per_pixel_predictions.parquet", index=False)
            comparison_rows.extend(_flatten_functional("pixel", pix_summary))
            summary_payload["pixel"] = pix_summary
            per_aoi_p = pd.DataFrame(pix_summary["per_aoi"])
            per_aoi_p["level"] = "pixel"
            plot_coverage(pix_summary["end_to_end"], out_dir, "pixel")
            plot_per_aoi_macro_f1(per_aoi_p, out_dir, "pixel")
        else:
            summary_payload["pixel"] = pix_summary

    per_aoi = pd.concat([per_aoi_b, per_aoi_p], ignore_index=True) if len(per_aoi_b) or len(per_aoi_p) else pd.DataFrame()
    if len(per_aoi):
        per_aoi.to_csv(out_dir / "per_aoi_metrics.csv", index=False)

    csv_path, md_path = write_comparison_table(comparison_rows, out_dir)
    write_summary_json(summary_payload, out_dir)

    # Bars from functional metrics
    bar_df = pd.DataFrame([r for r in comparison_rows if r["benchmark"] == "functional"])
    if len(bar_df):
        plot_comparison_bars(bar_df.rename(columns={})[["level", "system", "metric", "value"]], out_dir)

    ghsl_ds.close()
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(md_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
