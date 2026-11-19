#!/usr/bin/env python3
"""Which part of the week-9 Sentinel-2 / GHSL-heuristic features carries the gain?

Same protocol as scripts/run_week9_ae_s2_nres_loro.py (docs/eval_protocol_apples_to_apples.md):
  TRAIN  week-9 training table (86,158 buildings), leave-one-region-out
  EVAL   locked Section 3/4 building IDs (71,497)
  METRIC 0.5 * (MAE_res + MAE_nres) on soft fractions
  HEAD   StandardScaler + MultiOutput XGBRegressor, identical hyper-parameters

Also reports single-feature separation (AUC, residential vs non-residential) on the locked buildings.
Run:  .venv/bin/python presentation/analysis_s2_ablation.py [--run DIR]
Writes presentation/figures/s2_ablation.csv and s2_feature_auc.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/week9_ae_s2_nres_loro/20261113_055624/artifacts"
OUT = Path(__file__).resolve().parent / "figures"
SEED = 42

AE = [f"A{i:02d}" for i in range(64)]
FOOTPRINT = ["fp_area_m2", "fp_equiv_diam_m", "fp_compactness"]
IMAGERY = [
    "s2_blue",
    "s2_green",
    "s2_red",
    "s2_nir",
    "s2_brightness",
    "s2_blue_dom",
    "s2_ndvi",
    "s2_contrast_small",
    "s2_contrast_large",
    "s2_contrast_ratio",
]
SCORE = ["s2_nres_score"]
ALL_S2 = IMAGERY + FOOTPRINT + SCORE + ["s2_n_pix", "s2_ok"]  # the 16 week-9 features

SETS = {
    "AE only (64)": AE,
    "Footprint shape only (3)": FOOTPRINT,
    "S2 imagery only (10)": IMAGERY,
    "Fuzzy NRES score only (1)": SCORE,
    "AE + footprint shape": AE + FOOTPRINT,
    "AE + S2 imagery": AE + IMAGERY,
    "AE + all 16 S2 features (week 9)": AE + ALL_S2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="Path to week9 artifacts/ dir (or the run dir containing artifacts/).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="Directory for s2_ablation.csv / s2_feature_auc.csv",
    )
    args = ap.parse_args()
    run = args.run
    if (run / "artifacts" / "train_buildings_ae_s2.parquet").exists():
        run = run / "artifacts"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(run / "train_buildings_ae_s2.parquet")
    locked = pd.read_parquet(
        run / "buildings_section3_with_ae_s2.parquet",
        columns=["id", "y_res", "soft_gt_frac_res", "soft_gt_frac_nres", "ghsl_frac_res", "ghsl_frac_nres"],
    )
    assert len(locked) == 71497
    regions = sorted(train.region.unique())

    def run_fold(cols):
        oof = np.full((len(train), 2), np.nan, np.float32)
        for hold in regions:
            tr = (train.region != hold).to_numpy()
            Xtr = np.nan_to_num(train.loc[tr, cols].to_numpy(np.float32), nan=0.0)
            Xte = np.nan_to_num(train.loc[~tr, cols].to_numpy(np.float32), nan=0.0)
            Ytr = train.loc[tr, ["soft_gt_frac_res", "soft_gt_frac_nres"]].to_numpy(np.float32)
            sc = StandardScaler().fit(Xtr)
            model = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    n_jobs=max(1, (os.cpu_count() or 8) // 2),
                    random_state=SEED,
                    tree_method="hist",
                )
            )
            model.fit(sc.transform(Xtr), Ytr)
            oof[~tr] = np.clip(model.predict(sc.transform(Xte)), 0, 1)
        pred = pd.DataFrame({"id": train.id, "p_res": oof[:, 0], "p_nres": oof[:, 1]})
        m = locked.merge(pred, on="id", how="left")
        cov = m.p_res.notna().mean()
        m = m.dropna(subset=["p_res"])
        mr = mean_absolute_error(m.soft_gt_frac_res, m.p_res)
        mn = mean_absolute_error(m.soft_gt_frac_nres, m.p_nres)
        return 0.5 * (mr + mn), mr, mn, cov, len(m)

    rows = []
    for name, cols in SETS.items():
        mean, mr, mn, cov, n = run_fold(cols)
        rows.append(
            {
                "features": name,
                "n_features": len(cols),
                "mae_mean": round(mean, 4),
                "mae_res": round(mr, 4),
                "mae_nres": round(mn, 4),
                "coverage": round(cov, 4),
                "n_scored": n,
            }
        )
        print(rows[-1], flush=True)
    g = locked
    ghsl = 0.5 * (
        mean_absolute_error(g.soft_gt_frac_res, g.ghsl_frac_res)
        + mean_absolute_error(g.soft_gt_frac_nres, g.ghsl_frac_nres)
    )
    rows.append(
        {
            "features": "GHS-BUILT-S (baseline)",
            "n_features": 0,
            "mae_mean": round(ghsl, 4),
            "mae_res": None,
            "mae_nres": None,
            "coverage": 1.0,
            "n_scored": len(g),
        }
    )
    pd.DataFrame(rows).to_csv(args.out_dir / "s2_ablation.csv", index=False)

    # single-feature separation on locked buildings (higher AUC = separates res / non-res better)
    feat = pd.read_parquet(
        run / "buildings_section3_with_ae_s2.parquet",
        columns=["y_res", "s2_ok"] + IMAGERY + FOOTPRINT + SCORE,
    )
    feat = feat[feat.s2_ok > 0.5]
    auc = []
    for c in IMAGERY + FOOTPRINT + SCORE:
        d = feat[[c, "y_res"]].dropna()
        a = roc_auc_score(1 - d.y_res, d[c])  # positive class = non-residential
        auc.append(
            {
                "feature": c,
                "auc_nres_high": round(a, 4),
                "separation": round(max(a, 1 - a), 4),
                "higher_means": "non-residential" if a >= 0.5 else "residential",
                "median_res": float(d.loc[d.y_res == 1, c].median()),
                "median_nres": float(d.loc[d.y_res == 0, c].median()),
            }
        )
    pd.DataFrame(auc).sort_values("separation", ascending=False).to_csv(
        args.out_dir / "s2_feature_auc.csv", index=False
    )
    print(pd.DataFrame(auc).sort_values("separation", ascending=False).to_string(index=False))
    print(f"wrote {args.out_dir / 's2_ablation.csv'}", flush=True)
    print(f"run={run}", flush=True)


if __name__ == "__main__":
    main()
