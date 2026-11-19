#!/usr/bin/env python3
"""Data for the canvas "balance" step: how the current (week 6+) building sets are formed.

Stages, all read from stored artifacts (nothing re-sampled here):
  1 labelled   datasets/building_bbox_grid/buildings.parquet (Track-A footprints, y_res)
  2 pool       week-6 balanced_buildings / per_building_with_pls (1:1 per metro by building count, seed 42)
  3 train      week-9 train_buildings_ae_s2 (pool rows with soft GT; used by weeks 8-15, no reweighting)
  4 test       week-7 frozen Section-3 building ids (pool rows covered by soft GT, AE, GHSL and PLS)
For each building in the canvas window, records the last stage it reached and why it stopped.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
H = Path(__file__).resolve().parent
W6 = ROOT / "outputs/week6_overture_polygon_mae/20261110_041557/artifacts"

lab = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet", columns=["id", "region", "y_res", "area_m2_approx"])
lab["id"] = lab["id"].astype(str)
pool = pd.read_parquet(W6 / "per_building_with_pls.parquet",
                       columns=["id", "region", "y_res", "soft_gt_covered", "ae_covered", "ghsl_covered", "pls_covered"])
pool["id"] = pool["id"].astype(str)
train_ids = set(pd.read_parquet(ROOT / "outputs/week9_ae_s2_nres_loro/latest/artifacts/train_buildings_ae_s2.parquet", columns=["id"]).id.astype(str))
test_ids = set(pd.read_parquet(ROOT / "outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet", columns=["id"]).id.astype(str))
area = dict(zip(lab.id, lab.area_m2_approx))

pool["train"] = pool.id.isin(train_ids)
pool["test"] = pool.id.isin(test_ids)
assert pool.train.sum() == len(train_ids) and pool.test.sum() == len(test_ids)


def why(r):
    """First coverage the building is missing, in the order the filters matter."""
    for k, name in [("soft_gt_covered", "no soft truth"), ("ae_covered", "no AlphaEarth value"),
                    ("ghsl_covered", "no GHSL value"), ("pls_covered", "no PLS value")]:
        if not bool(r[k]):
            return name
    return "other"


pool["why"] = [None if t else why(r) for t, (_, r) in zip(pool.test, pool.iterrows())]


def split(df, area_col=True):
    r, n = df[df.y_res == 1], df[df.y_res == 0]
    a = lambda d: float(sum(area.get(i, 0) or 0 for i in d.id))
    out = dict(res=int(len(r)), nres=int(len(n)))
    if area_col:
        out.update(a_res=round(a(r)), a_nres=round(a(n)))
    return out


regions = []
for reg in sorted(lab.region.unique()):
    L, P = lab[lab.region == reg], pool[pool.region == reg]
    regions.append(dict(name=reg, labelled=split(L), pool=split(P), train=split(P[P.train]), test=split(P[P.test]),
                        drop_why={k: dict(res=int(((P.why == k) & (P.y_res == 1)).sum()), nres=int(((P.why == k) & (P.y_res == 0)).sum()))
                                  for k in P.why.dropna().unique()}))
tot = dict(labelled=split(lab), pool=split(pool), train=split(pool[pool.train]), test=split(pool[pool.test]),
           drop_why={k: dict(res=int(((pool.why == k) & (pool.y_res == 1)).sum()), nres=int(((pool.why == k) & (pool.y_res == 0)).sum()))
                     for k in pool.why.dropna().unique()})

# per-building stage for the canvas window
win = json.loads((H / "data/gt_canvas_data.json").read_text())["buildings"]
lab_ids = set(lab.id)
P = pool.set_index("id")
stage = {}
for b in win:
    i = str(b["id"])
    if b["status"] == "excluded" or i not in lab_ids:
        continue
    if i not in P.index:
        stage[i] = dict(s="cut")
    else:
        r = P.loc[i]
        stage[i] = dict(s="test" if r.test else "train" if r.train else "pool", why=r.why)
nyc = pool[pool.region == "NYC_NE"]
print({k: sum(1 for v in stage.values() if v["s"] == k) for k in ["cut", "pool", "train", "test"]})
print(tot)

# week-16 rerun: filters first, then area 1:1 (arm B), same model for every arm
W16 = ROOT / "outputs/week16_balance_area_loro/latest"
w16 = json.loads((W16 / "summary.json").read_text())
arm = pd.read_parquet(W16 / "artifacts/arm_predictions.parquet", columns=["arm", "id"])
inB, inC = set(arm[arm.arm == "B"].id.astype(str)), set(arm[arm.arm == "C"].id.astype(str))
ck = pd.read_parquet(W16 / "checkpoints/ae_by_building.parquet", columns=["id", "soft_gt_covered", "soft_gt_n_cells"])
nsoft = dict(zip(ck.id.astype(str), ck.soft_gt_n_cells))
soft_ok = set(ck[ck.soft_gt_covered].id.astype(str))
for i, v in stage.items():
    v["b"] = 1 if i in inB else 0 if i in soft_ok else -1          # -1 = fails the filter (no cell centre inside)
    v["c"] = 1 if i in inC else 0
    v["n"] = int(nsoft.get(i, 0))                              # stored cell centres inside (whole footprint, not clipped)
# per-box view of the fix: after the filter, and after area 1:1 (arm B)
okids = set(ck[ck.soft_gt_covered].id.astype(str))
u16 = pd.read_parquet(ROOT / "outputs/week6_overture_polygon_mae/20261110_034500/artifacts/per_building.parquet", columns=["id", "ghsl_covered"])
okids &= set(u16[u16.ghsl_covered].id.astype(str))      # same filter as the week-16 run (soft ∩ AE embedding ∩ GHSL; embedding ⇔ soft here)
fx = []
for reg in sorted(lab.region.unique()):
    L = lab[lab.region == reg]
    Pp, Bb = L[L.id.isin(okids)], L[L.id.isin(inB)]
    d = dict(name=reg, labelled=split(L), passed=split(Pp), area1=split(Bb))
    d["cut"] = "residential" if d["passed"]["a_res"] > d["passed"]["a_nres"] else "non_residential"
    fx.append(d)
assert sum(r["passed"]["res"] + r["passed"]["nres"] for r in fx) == w16["filter"]["passed"]["res"] + w16["filter"]["passed"]["nres"], "filter mismatch"
print([(r["name"], r["cut"], r["passed"]["res"], r["area1"]["res"], r["passed"]["nres"], r["area1"]["nres"]) for r in fx])
print("window arm B", {k: sum(1 for v in stage.values() if v["b"] == k) for k in (1, 0, -1)})

out = dict(regions=regions, total=tot, window=stage, seed=42, fix_regions=fx, w16=dict(arms=w16["arms"], filter=w16["filter"], model=w16["model"], run=w16["run_dir"]),
           src=dict(labelled="datasets/building_bbox_grid/buildings.parquet", pool=str((W6 / "per_building_with_pls.parquet").relative_to(ROOT)),
                    train="outputs/week9_ae_s2_nres_loro/latest/artifacts/train_buildings_ae_s2.parquet",
                    test="outputs/week7_section3_xgb_mae/ridge_section3_building_ids.parquet",
                    code="scripts/eval_week6_overture_polygon_mae.py: balance_buildings_per_region"))
(H / "data/balance.json").write_text(json.dumps(out, separators=(",", ":")))
print("KB", round((H / "data/balance.json").stat().st_size / 1e3))
