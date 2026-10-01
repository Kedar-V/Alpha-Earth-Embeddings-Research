#!/usr/bin/env python3
"""Prepare validation GT + model prediction parquets from the Week-2 CNN LORO run.

Uses:
  datasets/building_bbox_grid/buildings.parquet  (polygons + y_res)
  outputs/week2_cnn_bbox/cnn_oof_predictions.parquet  (LORO OOF)

RES = single_family | multi_family (y_res == 1). MIXED/UNKNOWN are not created.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--buildings",
        default=str(ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"),
    )
    ap.add_argument(
        "--predictions",
        default=str(ROOT / "outputs" / "week2_cnn_bbox" / "cnn_oof_predictions.parquet"),
    )
    ap.add_argument("--gt-out", default=str(ROOT / "data" / "validation" / "gt_buildings.parquet"))
    ap.add_argument("--pred-out", default=str(ROOT / "data" / "predictions" / "model_predictions.parquet"))
    args = ap.parse_args()

    b = pd.read_parquet(args.buildings)
    p = pd.read_parquet(args.predictions)
    b["id"] = b["id"].astype(str)
    p["id"] = p["id"].astype(str)

    need = {"id", "region", "wkt", "y_res"}
    missing = need - set(b.columns)
    if missing:
        raise SystemExit(f"buildings missing columns: {missing}")

    gt = pd.DataFrame(
        {
            "building_id": b["id"],
            "aoi_id": b["region"].astype(str),
            "geometry": b["wkt"].astype(str),
            "label": b["y_res"].map({1: "RES", 0: "NRES"}).astype(str),
            "confidence": 1.0,
            "source": "overture_class_y_res",
        }
    )
    # Keep only RES/NRES (already binary); drop any unexpected
    gt = gt[gt["label"].isin(["RES", "NRES"])].copy()

    pred = p.merge(b[["id"]], on="id", how="inner")
    # Approximate p_res from binary decision + pred_prob when true residential softmax unavailable.
    # pred_prob is max multiclass softmax; use as confidence for the predicted binary class.
    conf = pred["pred_prob"].astype(float) if "pred_prob" in pred.columns else 0.5
    pred_res = pred["pred_res"].astype(int)
    p_res = conf.where(pred_res == 1, 1.0 - conf)
    out_pred = pd.DataFrame(
        {
            "building_id": pred["id"].astype(str),
            "prediction": pred_res.map({1: "RES", 0: "NRES"}).astype(str),
            "p_res": p_res.astype(float),
            "p_nres": (1.0 - p_res).astype(float),
        }
    )

    # Inner-join universe
    common = set(gt["building_id"]) & set(out_pred["building_id"])
    gt = gt[gt["building_id"].isin(common)].reset_index(drop=True)
    out_pred = out_pred[out_pred["building_id"].isin(common)].reset_index(drop=True)

    Path(args.gt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(args.gt_out, index=False)
    out_pred.to_parquet(args.pred_out, index=False)

    print(f"Wrote {args.gt_out}  n={len(gt)}  RES={(gt.label=='RES').sum()}  NRES={(gt.label=='NRES').sum()}")
    print(f"Wrote {args.pred_out}  n={len(out_pred)}")
    print("AOIs:", sorted(gt["aoi_id"].unique()))


if __name__ == "__main__":
    main()
