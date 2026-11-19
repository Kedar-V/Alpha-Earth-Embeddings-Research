#!/usr/bin/env python3
"""Week 16: does the balancing rule change AE vs GHSL? Same model, same metric, three building sets.

  A  current      frozen Section-3 test ids: 1:1 by building count per box, THEN soft/AE/GHSL/PLS filters
  C  order fixed  soft/AE/GHSL filters FIRST, then 1:1 by building count per box (seed 42)
  B  area 1:1     soft/AE/GHSL filters FIRST, then 1:1 by footprint area per box (seed 42)
  Bw area 1:1 + w same buildings as B; training rows weighted so each class has equal total weight

Per building:
  soft GT  = mean week-4 cell coverage at inside-polygon AE cells (week-6 stage_ae, unchanged)
  GHSL     = mean GHS-BUILT-S soft fractions under the footprint, own Mollweide grid (week-6 unbalanced run)
  model    = Ridge on the mean 64-d AlphaEarth embedding of inside-polygon cells, leave-one-box-out,
             trained and scored inside each arm (no reweighting)
Metric: mae_mean_res_nres = 0.5*(MAE_res + MAE_nres), also per class, class-balanced and area-weighted.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("w6", ROOT / "scripts/eval_week6_overture_polygon_mae.py")
w6 = importlib.util.module_from_spec(spec); sys.modules["w6"] = w6; spec.loader.exec_module(w6)

UNBAL = ROOT / "outputs/week6_overture_polygon_mae/20261110_034500/artifacts/per_building.parquet"
TEST_IDS = ROOT / "outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet"
SEED = 42
A = [f"A{i:02d}" for i in range(64)]

run = ROOT / "outputs/week16_balance_area_loro" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
for d in ["checkpoints", "artifacts", "logs"]:
    (run / d).mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/week16_balance_area_loro/latest").unlink(missing_ok=True)
(ROOT / "outputs/week16_balance_area_loro/latest").symlink_to(run.name)
T0 = time.time()


def log(m):
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    with open(run / "logs/run.log", "a") as f:
        f.write(line + "\n")


def progress(stage, done, total):
    (run / "progress.json").write_text(json.dumps(dict(stage=stage, completed=done, total=total, elapsed_s=round(time.time() - T0, 1))))


# 1 all labelled buildings + area
progress("load", 0, 5)
b = pd.read_parquet(w6.BUILDINGS, columns=["id", "region", "crs", "y_res", "area_m2_approx"])
b["id"] = b["id"].astype(str)
log(f"labelled buildings {len(b):,}")

# 2 soft GT for all of them (week-6 code path)
progress("soft_gt", 1, 5)
soft = w6.stage_ae(run, b, workers=8)[["id", "soft_gt_frac_res", "soft_gt_frac_nres", "soft_gt_covered", "soft_gt_n_cells"]]
log(f"soft GT covered {int(soft.soft_gt_covered.sum()):,}")

# 3 GHSL from the unbalanced week-6 run (identical values to the balanced run on shared ids)
u = pd.read_parquet(UNBAL, columns=["id", "ghsl_frac_res", "ghsl_frac_nres", "ghsl_covered"])
u["id"] = u["id"].astype(str)

# 4 mean AE embedding over inside-polygon cells
progress("embeddings", 2, 5)
cells = pd.read_parquet(w6.CELLS, columns=["id", "inside_polygon"] + A, filters=[("inside_polygon", "==", True)])
cells["id"] = cells["id"].astype(str)
emb = cells.groupby("id", sort=False)[A].mean().astype("float32")
emb["n_in"] = cells.groupby("id", sort=False).size()
del cells
log(f"buildings with inside-polygon embeddings {len(emb):,}")

df = b.merge(soft, on="id", how="left").merge(u, on="id", how="left").merge(emb, left_on="id", right_index=True, how="left")
df["has_emb"] = df["n_in"].fillna(0) > 0
df["ok"] = df.soft_gt_covered.fillna(False).astype(bool) & df.ghsl_covered.fillna(False).astype(bool) & df.has_emb
log(f"pass soft ∩ AE ∩ GHSL: {int(df.ok.sum()):,} (res {int((df.ok & (df.y_res == 1)).sum()):,}, non-res {int((df.ok & (df.y_res == 0)).sum()):,})")


def bal_count(d, rng):
    out = []
    for _, s in d.groupby("region", sort=True):
        r, n = s[s.y_res == 1], s[s.y_res == 0]; k = min(len(r), len(n))
        out += [r.loc[rng.choice(r.index, k, replace=False)], n.loc[rng.choice(n.index, k, replace=False)]]
    return pd.concat(out)


def bal_area(d, rng):
    """Per box: keep all of the class with less footprint area; cut the other at random until its area matches."""
    out = []
    for _, s in d.groupby("region", sort=True):
        r, n = s[s.y_res == 1], s[s.y_res == 0]
        big, small = (r, n) if r.area_m2_approx.sum() > n.area_m2_approx.sum() else (n, r)
        target = small.area_m2_approx.sum()
        perm = big.loc[rng.permutation(big.index)]
        cum = perm.area_m2_approx.cumsum().to_numpy()
        k = int(np.searchsorted(cum, target)) + 1          # first prefix that reaches the target
        out += [small, perm.iloc[:min(k, len(perm))]]
    return pd.concat(out)


ok = df[df.ok]
test_ids = set(pd.read_parquet(TEST_IDS, columns=["id"]).id.astype(str))
arms = {
    "A": df[df.id.isin(test_ids) & df.ok],
    "C": bal_count(ok, np.random.default_rng(SEED)),
    "B": bal_area(ok, np.random.default_rng(SEED)),
}
arms["Bw"] = arms["B"]
WEIGHTED = {"Bw"}
log("arm sizes " + json.dumps({k: len(v) for k, v in arms.items()}) + f" (A frozen ids found with full coverage: {len(arms['A']):,} of {len(test_ids):,})")


def loro(d, weighted=False):
    X = d[A].to_numpy("float32"); pr = np.zeros(len(d)); pn = np.zeros(len(d))
    yr = d.y_res.to_numpy()
    for reg in sorted(d.region.unique()):
        te = (d.region == reg).to_numpy()
        for y, out in [("soft_gt_frac_res", pr), ("soft_gt_frac_nres", pn)]:
            sw = None
            if weighted:   # equal total weight per class in the training fold
                c = yr[~te]; sw = np.where(c == 1, 0.5 / (c == 1).sum(), 0.5 / (c == 0).sum()) * len(c)
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[~te], d[y].to_numpy()[~te], ridge__sample_weight=sw)
            out[te] = m.predict(X[te])
    return np.clip(pr, 0, 1), np.clip(pn, 0, 1)


def mae(d, pr, pn, w=None):
    e = 0.5 * (np.abs(pr - d.soft_gt_frac_res.to_numpy()) + np.abs(pn - d.soft_gt_frac_nres.to_numpy()))
    return float(np.average(e, weights=w))


results, preds = {}, []
progress("loro", 3, 5)
for k, d in arms.items():
    d = d.reset_index(drop=True)
    pr, pn = loro(d, weighted=k in WEIGHTED)
    gr, gn = d.ghsl_frac_res.to_numpy(), d.ghsl_frac_nres.to_numpy()
    res = (d.y_res == 1).to_numpy()
    row = dict(n=len(d), res=int(res.sum()), nres=int((~res).sum()),
               area_res=float(d.area_m2_approx[res].sum()), area_nres=float(d.area_m2_approx[~res].sum()))
    for name, (p1, p2) in {"ae": (pr, pn), "ghsl": (gr, gn)}.items():
        row[name] = dict(pooled=mae(d, p1, p2), res=mae(d[res], p1[res], p2[res]), nres=mae(d[~res], p1[~res], p2[~res]),
                         area_w=mae(d, p1, p2, d.area_m2_approx.to_numpy()))
        row[name]["class_bal"] = 0.5 * (row[name]["res"] + row[name]["nres"])
        row[name]["by_region"] = {r: mae(d[d.region == r], p1[(d.region == r).to_numpy()], p2[(d.region == r).to_numpy()]) for r in sorted(d.region.unique())}
    results[k] = row
    log(f"arm {k}: n={len(d):,} res={row['res']:,} AE pooled {row['ae']['pooled']:.4f} GHSL {row['ghsl']['pooled']:.4f} | "
        f"class-bal {row['ae']['class_bal']:.4f} vs {row['ghsl']['class_bal']:.4f} | area-w {row['ae']['area_w']:.4f} vs {row['ghsl']['area_w']:.4f}")
    preds.append(pd.DataFrame(dict(arm=k, id=d.id, region=d.region, y_res=d.y_res, area=d.area_m2_approx,
                                   soft_res=d.soft_gt_frac_res, soft_nres=d.soft_gt_frac_nres, ae_res=pr, ae_nres=pn, ghsl_res=gr, ghsl_nres=gn)))

progress("write", 4, 5)
pd.concat(preds).to_parquet(run / "artifacts/arm_predictions.parquet", index=False)
filt = dict(labelled=dict(res=int((df.y_res == 1).sum()), nres=int((df.y_res == 0).sum())),
            passed=dict(res=int((df.ok & (df.y_res == 1)).sum()), nres=int((df.ok & (df.y_res == 0)).sum()),
                        area_res=float(df.area_m2_approx[df.ok & (df.y_res == 1)].sum()), area_nres=float(df.area_m2_approx[df.ok & (df.y_res == 0)].sum())))
summary = dict(arms=results, filter=filt, seed=SEED, model="Ridge(alpha=1) on standardized mean AE embedding, LORO by box",
               metric="0.5*(|res err| + |nres err|) vs soft GT, mean over buildings", run_dir=str(run.relative_to(ROOT)))
(run / "summary.json").write_text(json.dumps(summary, indent=1))
progress("done", 5, 5)
log(f"done → {run / 'summary.json'}")
