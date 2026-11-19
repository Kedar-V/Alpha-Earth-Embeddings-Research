#!/usr/bin/env python3
"""Data for the GT flow canvas (fixed flow, week 17). Every number comes from a stored artifact.

  window geometry + imagery  data/gt_canvas_data.json (build_gt_canvas_data.py: Overture 2026-10-21.0 snapshot,
                             Esri World Imagery, Sentinel-2 2018; geometry checked against stored GT to 5e-5)
  flow numbers               outputs/week17_fixed_flow_tiles/latest (scripts/run_week17_fixed_flow_tiles.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
R = (ROOT / "outputs/week17_fixed_flow_tiles/latest").resolve()
assert (R / "DONE").exists(), f"week-17 run not finished: {R}"
S = json.loads((R / "summary.json").read_text())
old = json.loads((H / "data/gt_canvas_data.json").read_text())
M = old["meta"]
W = M["window"]

to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
utm_merc = Transformer.from_crs(M["utm"], "EPSG:3857", always_xy=True)
moll_merc = Transformer.from_crs("ESRI:54009", "EPSG:3857", always_xy=True)
mx0, my0 = to_merc.transform(W[0], W[3])


def ring(xs, ys, tr):
    X, Y = tr.transform(xs, ys)
    return [[round(float(x - mx0), 2), round(float(my0 - y), 2)] for x, y in zip(X, Y)]


def square(cx, cy, tr, h=5.0):
    return ring([cx - h, cx + h, cx + h, cx - h], [cy + h, cy + h, cy - h, cy - h], tr)


# buildings in the window: geometry from the snapshot, flow status from week 17
bw = pd.read_parquet(R / "artifacts/buildings.parquet").set_index("id")
blds = []
for b in old["buildings"]:
    o = dict(id=b["id"], rings=b["rings"], cls=b["cls"], area=b["area"])
    if b["status"] == "excluded":
        o["s"] = "untagged" if b["reason"] == "no use tag" else "unmapped"
    else:
        r = bw.loc[b["id"]]
        o.update(s=b["status"], ok=bool(r.ok), kept=bool(r.kept), area_true=round(float(r.area), 1))
    blds.append(o)

def two_class(pr, pn):
    """res vs non-res only: larger of the two shares; no building signal at all = wrong."""
    pr, pn = np.nan_to_num(pr), np.nan_to_num(pn)
    return np.where(pr + pn == 0, "none", np.where(pr >= pn, "res", "nres"))


def score2(path):
    d = pd.read_parquet(path, columns=["region", "tc", "pr", "pn", "bal"]); d = d[d.bal]
    ok = two_class(d.pr.to_numpy(), d.pn.to_numpy()) == d.tc.to_numpy()
    out = dict(n=int(len(d)), right=int(ok.sum()))
    for c in ["res", "nres"]:
        m = (d.tc == c).to_numpy(); out[c] = dict(n=int(m.sum()), right=int(ok[m].sum()))
    return out


def tiles(df, tr, xcol, ycol):
    out = []
    for r in df.itertuples():
        out.append(dict(ring=square(getattr(r, xcol), getattr(r, ycol), tr), tr=round(float(r.tr), 4), tn=round(float(r.tn), 4), tc=r.tc,
                        pr=None if pd.isna(r.pr) else round(float(r.pr), 4), pn=None if pd.isna(r.pn) else round(float(r.pn), 4),
                        pc=r.pc, ok=bool(r.ok3), ok2=bool(two_class(np.array([r.pr]), np.array([r.pn]))[0] == r.tc), bal=bool(r.bal)))
    return out

wa = pd.read_parquet(R / "artifacts/window_ae_tiles.parquet")
wg = pd.read_parquet(R / "artifacts/window_ghsl_tiles.parquet")
ae_t = tiles(wa, utm_merc, "utm_x", "utm_y")
gh_t = tiles(wg, moll_merc, "mx", "my")

# sanity: tiles whose centre falls inside a kept building in the window match the in-scope sets
out = dict(meta=dict(window=W, width_m=M["width_m"], height_m=M["height_m"], release=M["release"], utm=M["utm"], run=str(R.relative_to(ROOT)),
                     seed=S["config"]["seed"], share_max=S["config"]["share_max"], cell_m=S["config"]["cell_m"]),
           img=old["img"], buildings=blds, ae=ae_t, ghsl=gh_t,
           flow=dict(buildings=S["buildings"], ae=S["ae"], ghsl=S["ghsl"],
                     two=dict(ae=score2(R / "artifacts/ae_tiles.parquet"), ghsl=score2(R / "artifacts/ghsl_tiles.parquet"))))
print("res-vs-nres:", json.dumps(out["flow"]["two"]))
(H / "data/flow_data.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print("window buildings", pd.Series([b["s"] for b in blds]).value_counts().to_dict(),
      "kept", sum(1 for b in blds if b.get("kept")), "| AE tiles", len(ae_t), "GHSL tiles", len(gh_t))
print("MB", round((H / "data/flow_data.json").stat().st_size / 1e6, 2))
