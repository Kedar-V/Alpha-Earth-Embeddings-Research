#!/usr/bin/env python3
"""Data for the interactive "GT build" canvas: one NYC window, every step from real sources.

Steps and sources (nothing simulated):
  1 Overture snapshot   all buildings, release 2026-10-21.0, read from the public S3 bucket (cached to
                        presentation/canvas/data/overture_nyc_2026-10-21.0.parquet; --refresh re-queries)
  2 Labels              scripts/build_building_type_dataset.py CLASS_TO_LABEL; residential = single/multi family
  3 Tiles               AlphaEarth 10 m grid (UTM 18N, edges on 10 m multiples, as in building_cells.parquet)
  4 Coverage fractions  week-4 rule: area(union(RES) ∩ cell)/cell.area, same for NRES (build_week4_pixel_gt.py);
                        recomputed here and checked against datasets/week4_pixel_gt/pixels.parquet
  5 Sampling            pure / mixed from the fractions; kept-for-training from datasets/week4_pixel_gt_balanced
  6 AlphaEarth          64-D embeddings from datasets/building_bbox_grid/building_cells.parquet; held-out
                        model call from outputs/week4_pixel_balanced_train/oof_predictions_full_gt.parquet
  7 GHSL                GHS-BUILT-S + NRES E2018 R2023A 10 m tiles in Mollweide (data/ghsl/*), decoded with
                        scripts/eval_week6_overture_polygon_mae.raw_to_soft; per-cell = value at AE cell centre
                        (week-4 method); per-building = mean under the footprint (week-6 _soft_under_geom)
  Imagery               Esri World Imagery tiles (z19) and the week-9 Sentinel-2 summer-2018 mosaic
Run:  .venv/bin/python presentation/canvas/build_gt_canvas_data.py [--refresh]
"""
import argparse
import base64
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from shapely import wkt as swkt
from shapely.geometry import box
from shapely.ops import transform as stransform, unary_union

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
RELEASE = "2026-10-21.0"
SNAP = DATA / f"overture_nyc_{RELEASE}.parquet"
W = (-73.9980, 40.7349, -73.9944, 40.7376)          # lon/lat window (Greenwich Village / Chelsea)
UTM, MERC, MOLL = "EPSG:32618", "EPSG:3857", "ESRI:54009"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


btd = load("btd", ROOT / "scripts/build_building_type_dataset.py")
w6 = load("w6", ROOT / "scripts/eval_week6_overture_polygon_mae.py")
CLASS_TO_LABEL = btd.CLASS_TO_LABEL
RES_LABELS = {"single_family", "multi_family"}

ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true"); args = ap.parse_args()

# ---------------------------------------------------------------- 1 Overture snapshot
if args.refresh or not SNAP.exists():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    m = 0.0006
    q = f"""SELECT id, subtype, class, height, num_floors, names.primary AS name, sources[1].dataset AS src,
                   ST_AsText(geometry) AS wkt
            FROM read_parquet('s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*',
                              hive_partitioning=1)
            WHERE bbox.xmin < {W[2]+m} AND bbox.xmax > {W[0]-m} AND bbox.ymin < {W[3]+m} AND bbox.ymax > {W[1]-m}"""
    con.execute(q).df().to_parquet(SNAP)
snap = pd.read_parquet(SNAP)

to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
to_merc = Transformer.from_crs("EPSG:4326", MERC, always_xy=True)
utm_to_merc = Transformer.from_crs(UTM, MERC, always_xy=True)
moll_to_merc = Transformer.from_crs(MOLL, MERC, always_xy=True)
utm_to_moll = Transformer.from_crs(UTM, MOLL, always_xy=True)
moll_to_utm = Transformer.from_crs(MOLL, UTM, always_xy=True)

# display frame: Web Mercator metres, origin at the window's NW corner, y pointing down
mx0, my0 = to_merc.transform(W[0], W[3]); mx1, my1 = to_merc.transform(W[2], W[1])
MW, MH = mx1 - mx0, my0 - my1


def disp(x, y):
    return [round(float(x - mx0), 2), round(float(my0 - y), 2)]


def ring_disp(coords, tr):
    xs, ys = tr.transform([c[0] for c in coords], [c[1] for c in coords])
    return [disp(a, b) for a, b in zip(xs, ys)]


# ---------------------------------------------------------------- 2 labels
geoms_ll = [swkt.loads(w) for w in snap.wkt]
geoms_utm = [stransform(lambda x, y, z=None: to_utm.transform(x, y), g) for g in geoms_ll]
win_ll = box(*W)
buildings = []
for (_, r), g, gu in zip(snap.iterrows(), geoms_ll, geoms_utm):
    if not g.intersects(win_ll.buffer(0.0003)):
        continue
    cls = r["class"]
    lab = CLASS_TO_LABEL.get(cls) if isinstance(cls, str) else None
    if lab is None:
        status, reason = "excluded", ("no use tag" if not isinstance(cls, str) else f"class '{cls}' not in the mapping")
    else:
        status, reason = ("res" if lab in RES_LABELS else "nres"), None
    rings = []
    for gg in getattr(g, "geoms", [g]):
        rings.append(ring_disp(list(gg.exterior.coords), to_merc))
    buildings.append(dict(id=r["id"], cls=cls if isinstance(cls, str) else None, label=lab, status=status, reason=reason,
                          src=r["src"], name=r["name"] if isinstance(r["name"], str) else None,
                          floors=None if pd.isna(r["num_floors"]) else int(r["num_floors"]),
                          area=round(gu.area, 1), rings=rings, _utm=gu))
res_u = unary_union([b["_utm"] for b in buildings if b["status"] == "res"])
nres_u = unary_union([b["_utm"] for b in buildings if b["status"] == "nres"])

# ---------------------------------------------------------------- 3-4 AE grid + coverage fractions
ux, uy = to_utm.transform([W[0], W[2], W[0], W[2]], [W[1], W[1], W[3], W[3]])
gx0, gx1 = math.floor(min(ux) / 10) * 10, math.ceil(max(ux) / 10) * 10
gy0, gy1 = math.floor(min(uy) / 10) * 10, math.ceil(max(uy) / 10) * 10
cells = {}
for x in range(gx0, gx1, 10):
    for y in range(gy0, gy1, 10):
        cb = box(x, y, x + 10, y + 10)
        fr = res_u.intersection(cb).area / 100.0 if not res_u.is_empty else 0.0
        fn = nres_u.intersection(cb).area / 100.0 if not nres_u.is_empty else 0.0
        cells[(x + 5, y + 5)] = dict(cx=x + 5, cy=y + 5, ring=ring_disp([(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)], utm_to_merc),
                                     fr=round(fr, 4), fn=round(fn, 4))

# stored week-4 GT for the same cells (match on cell centre)
px = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet",
                     columns=["pixel_id", "region", "pixel_lon", "pixel_lat", "frac_res_full", "frac_nres_full"])
px = px[(px.region == "NYC_NE") & px.pixel_lon.between(W[0] - 0.0005, W[2] + 0.0005) & px.pixel_lat.between(W[1] - 0.0005, W[3] + 0.0005)]
pxx, pxy = to_utm.transform(px.pixel_lon.values, px.pixel_lat.values)
px = px.assign(cx=np.floor(pxx / 10) * 10 + 5, cy=np.floor(pxy / 10) * 10 + 5)
bal = pd.read_parquet(ROOT / "datasets/week4_pixel_gt_balanced/pixels.parquet", columns=["pixel_id", "region", "building_ids"],
                      filters=[("region", "==", "NYC_NE")])
bal_ids = set(bal.pixel_id)
bal_bids = dict(zip(bal.pixel_id, bal.building_ids.astype(str)))   # which footprints each training cell came from
oof = pd.read_parquet(ROOT / "outputs/week4_pixel_balanced_train/oof_predictions_full_gt.parquet",
                      columns=["pixel_id", "region", "p_res", "y_pred"], filters=[("region", "==", "NYC_NE")]).set_index("pixel_id")
diffs = []
for r in px.itertuples(index=False):
    c = cells.get((r.cx, r.cy))
    if c is None:
        continue
    c["pid"] = str(r.pixel_id)
    c["stored_fr"], c["stored_fn"] = round(r.frac_res_full, 4), round(r.frac_nres_full, 4)
    diffs.append(max(abs(c["fr"] - r.frac_res_full), abs(c["fn"] - r.frac_nres_full)))
    c["kept"] = r.pixel_id in bal_ids
    if c["kept"]:
        c["bids"] = [t for t in __import__("re").split(r"[,;| ]+", bal_bids[r.pixel_id]) if t]
    if r.pixel_id in oof.index:
        c["p_res"] = round(float(oof.loc[r.pixel_id, "p_res"]), 3)
for c in cells.values():
    fr, fn = c["fr"], c["fn"]
    c["stratum"] = "empty" if fr == 0 and fn == 0 else ("mixed" if fr > 0 and fn > 0 else ("res" if fr > 0 else "nres"))

# AlphaEarth-only held-out predictions for every stored cell, mixed included (week 7 XGBoost, NYC held out)
w7 = pd.read_parquet(ROOT / "outputs/week7_xgb_score_full_gt_all_cells/20261113_020006/artifacts/oof_predictions_full_gt_all_cells.parquet",
                     columns=["pixel_id", "region", "pred_frac_res_full", "pred_frac_nres_full", "pred_frac_empty_full"],
                     filters=[("region", "==", "NYC_NE")]).set_index("pixel_id")
for c in cells.values():
    if "pid" in c and c["pid"] in w7.index:
        q = w7.loc[c["pid"]]
        c["ae_pr"], c["ae_pn"], c["ae_pe"] = (round(float(q.pred_frac_res_full), 3), round(float(q.pred_frac_nres_full), 3),
                                              round(float(q.pred_frac_empty_full), 3))

# ---------------------------------------------------------------- 6 AlphaEarth embeddings
A = [f"A{i:02d}" for i in range(64)]
ids = [b["id"] for b in buildings if b["status"] != "excluded"]
bc = pd.read_parquet(ROOT / "datasets/building_bbox_grid/building_cells.parquet", columns=["id", "utm_x", "utm_y"] + A,
                     filters=[("id", "in", ids)]).drop_duplicates(["utm_x", "utm_y"])
nyc_sample = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet", columns=["region"] + A,
                             filters=[("region", "==", "NYC_NE")]).sample(40000, random_state=0)[A].to_numpy(float)
mu = nyc_sample.mean(0); _, _, Vt = np.linalg.svd(nyc_sample - mu, full_matrices=False); pcs = Vt[:3]
z_ref = (nyc_sample - mu) @ pcs.T; lo, hi = np.percentile(z_ref, [2, 98], axis=0)
n_ae = 0
for r in bc.itertuples(index=False):
    c = cells.get((r.utm_x, r.utm_y))
    if c is None:
        continue
    v = np.array([getattr(r, a) for a in A], float)
    rgb = np.clip(((v - mu) @ pcs.T - lo) / (hi - lo), 0, 1)
    c["ae"] = [round(float(t), 3) for t in v]
    c["rgb"] = [int(255 * t) for t in rgb]
    n_ae += 1

# ---------------------------------------------------------------- 7 GHSL (native Mollweide grid)
tile = None
cxm, cym = utm_to_moll.transform((gx0 + gx1) / 2, (gy0 + gy1) / 2)
for p in sorted((ROOT / "data/ghsl/built_s_tiles").glob("GHS_BUILT_S_E2018_*.tif")):
    with rasterio.open(p) as d:
        if d.bounds.left <= cxm <= d.bounds.right and d.bounds.bottom <= cym <= d.bounds.top:
            tile = p
nres_path = ROOT / "data/ghsl/built_s_nres_tiles" / tile.name.replace("GHS_BUILT_S_E2018", "GHS_BUILT_S_NRES_E2018")
ds_s, ds_n = rasterio.open(tile), rasterio.open(nres_path)
mxs, mys = utm_to_moll.transform([gx0, gx1, gx0, gx1], [gy0, gy0, gy1, gy1])
win = from_bounds(min(mxs) - 20, min(mys) - 20, max(mxs) + 20, max(mys) + 20, transform=ds_s.transform)
win = Window(int(win.col_off), int(win.row_off), int(math.ceil(win.width)) + 1, int(math.ceil(win.height)) + 1)
bu = ds_s.read(1, window=win); nr = ds_n.read(1, window=win); T = ds_s.window_transform(win)
fr_g, fn_g, valid_g = w6.raw_to_soft(bu.astype(float), nr.astype(float))
ghsl = []
for i in range(bu.shape[0]):
    for j in range(bu.shape[1]):
        x0, y0 = T * (j, i); x1, y1 = T * (j + 1, i + 1)
        ghsl.append(dict(ring=ring_disp([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], moll_to_merc),
                         bu=int(bu[i, j]), nres=int(nr[i, j]), fr=round(float(fr_g[i, j]), 3), fn=round(float(fn_g[i, j]), 3),
                         ok=bool(valid_g[i, j])))
# ground truth on GHSL's own grid: same footprints, coverage per GHSL cell in equal-area Mollweide
res_m = stransform(lambda x, y, z=None: utm_to_moll.transform(x, y), res_u) if not res_u.is_empty else res_u
nres_m = stransform(lambda x, y, z=None: utm_to_moll.transform(x, y), nres_u) if not nres_u.is_empty else nres_u
k = 0
for i in range(bu.shape[0]):
    for j in range(bu.shape[1]):
        x0, y0 = T * (j, i); x1, y1 = T * (j + 1, i + 1)
        cb = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        g = ghsl[k]; k += 1
        g["gt_r"] = round(res_m.intersection(cb).area / cb.area, 4) if not res_m.is_empty else 0.0
        g["gt_n"] = round(nres_m.intersection(cb).area / cb.area, 4) if not nres_m.is_empty else 0.0
        ux_c, uy_c = moll_to_utm.transform((x0 + x1) / 2, (y0 + y1) / 2)
        g["in"] = bool(gx0 <= ux_c <= gx1 and gy0 <= uy_c <= gy1)     # same ground area as the AE grid

# week-4 method: GHSL value at each AE cell centre; offset to the nearest GHSL cell centre
offs = []
for c in cells.values():
    xm, ym = utm_to_moll.transform(c["cx"], c["cy"])
    row, col = ~T * (xm, ym); row, col = int(col), int(row)
    if 0 <= row < bu.shape[0] and 0 <= col < bu.shape[1]:
        c["g_fr"], c["g_fn"] = round(float(fr_g[row, col]), 3), round(float(fn_g[row, col]), 3)
        gcx, gcy = T * (col + 0.5, row + 0.5)
        ux2, uy2 = moll_to_utm.transform(gcx, gcy)
        c["g_ctr"] = disp(*utm_to_merc.transform(ux2, uy2)); offs.append(math.hypot(ux2 - c["cx"], uy2 - c["cy"]))
# week-6 method: mean GHSL soft fractions under each footprint (native Mollweide)
pb = pd.read_parquet(ROOT / "outputs/week6_overture_polygon_mae/20261110_041557/artifacts/per_building_with_pls.parquet",
                     columns=["id", "ghsl_frac_res", "ghsl_frac_nres", "ae_frac_res", "ae_frac_nres", "soft_gt_frac_res", "soft_gt_frac_nres"])
pb = pb.set_index("id")
gd = []
for b in buildings:
    if b["status"] == "excluded":
        continue
    gm = stransform(lambda x, y, z=None: utm_to_moll.transform(x, y), b["_utm"])
    z = w6._soft_under_geom(gm, ds_s, ds_n)
    b["g_fr"], b["g_fn"] = (round(z["ghsl_frac_res"], 3), round(z["ghsl_frac_nres"], 3)) if z["ghsl_covered"] else (None, None)
    b["g_npix"] = int(z["ghsl_n_pixels"])
    if b["id"] in pb.index:
        s = pb.loc[b["id"]]
        nn = lambda v: None if pd.isna(v) else round(float(v), 3)   # missing stored values -> null (valid JSON)
        b["w6"] = dict(g_fr=nn(s.ghsl_frac_res), g_fn=nn(s.ghsl_frac_nres), ae_fr=nn(s.ae_frac_res), ae_fn=nn(s.ae_frac_nres))
        if b["g_fr"] is not None:
            gd.append(max(abs(b["g_fr"] - s.ghsl_frac_res), abs(b["g_fn"] - s.ghsl_frac_nres)))

# ---------------------------------------------------------------- own-grid scoring (each product vs truth on its own tiles)
from sklearn.metrics import f1_score


def own_grid_metrics(gr, gn, pr, pn, pe):
    gr, gn, pr, pn, pe = (np.asarray(v, float) for v in (gr, gn, pr, pn, pe))
    ge = np.clip(1 - gr - gn, 0, 1)
    dec = lambda r, n: np.where(r + n < SHARE_MAX_, "empty", np.where(r >= n, "res", "nres"))
    yt, yp = dec(gr, gn), dec(np.clip(pr, 0, 1), np.clip(pn, 0, 1))
    mr, mn, me = (float(np.mean(np.abs(np.clip(a, 0, 1) - b))) for a, b in ((pr, gr), (pn, gn), (pe, ge)))
    return dict(n=int(len(gr)), mae=round(0.5 * (mr + mn), 4), mae_res=round(mr, 4), mae_nres=round(mn, 4), mae_empty=round(me, 4),
                acc3=round(float(np.mean(yt == yp)), 4), f1_3=round(float(f1_score(yt, yp, average="macro")), 4),
                n_class={k: int(np.sum(yt == k)) for k in ("res", "nres", "empty")})


import re as _re0
SHARE_MAX_ = float(_re0.search(r"BUILDING_SHARE_MAX\s*=\s*([0-9.]+)", (ROOT / "scripts/week3_pixel_3class_trainlib.py").read_text()).group(1))
ae_dom = [c for c in cells.values() if (c["fr"] + c["fn"]) > 0 and "ae_pr" in c]
gh_dom = [g for g in ghsl if g["in"] and (g["gt_r"] + g["gt_n"]) > 0 and g["ok"]]
OWN = dict(
    ae=own_grid_metrics([c["fr"] for c in ae_dom], [c["fn"] for c in ae_dom], [c["ae_pr"] for c in ae_dom],
                        [c["ae_pn"] for c in ae_dom], [c["ae_pe"] for c in ae_dom]),
    ghsl=own_grid_metrics([g["gt_r"] for g in gh_dom], [g["gt_n"] for g in gh_dom], [g["fr"] for g in gh_dom],
                          [g["fn"] for g in gh_dom], [1 - g["fr"] - g["fn"] for g in gh_dom]),
    ae_cells_touching=sum(1 for c in cells.values() if (c["fr"] + c["fn"]) > 0),
    ghsl_cells_touching=sum(1 for g in ghsl if g["in"] and (g["gt_r"] + g["gt_n"]) > 0))

# ---------------------------------------------------------------- 8 per-building scoring (week-6 fair test, latest model)
# Frozen Section-3 buildings that fall in the window; predictions from the week-14 run (ViT + S2 & shape),
# GHSL = mean under the footprint on its own grid (week-6), soft GT = mean AE-cell coverage inside the
# footprint, hard GT = 1/0 from the building's own tag (no grid).
w14 = pd.read_parquet(ROOT / "outputs/week14_ae_grid_vit_s2_loro/20261113_075223/artifacts/buildings_section3_with_ae_grid_vit_s2.parquet",
                      columns=["id", "y_res", "soft_gt_frac_res", "soft_gt_frac_nres", "vit_frac_res", "vit_frac_nres",
                               "ghsl_frac_res", "ghsl_frac_nres"]).set_index("id")
r3 = lambda v: None if pd.isna(v) else round(float(v), 3)
hr_all = (w14.y_res == 1).astype(float)
def _mae(pr, pn, gr, gn):
    return round(float(0.5 * ((pr.clip(0, 1) - gr).abs().mean() + (pn.clip(0, 1) - gn).abs().mean())), 4)
HEAD = dict(n=len(w14),
            soft=dict(vit=_mae(w14.vit_frac_res, w14.vit_frac_nres, w14.soft_gt_frac_res, w14.soft_gt_frac_nres),
                      ghsl=_mae(w14.ghsl_frac_res, w14.ghsl_frac_nres, w14.soft_gt_frac_res, w14.soft_gt_frac_nres)),
            hard=dict(vit=_mae(w14.vit_frac_res, w14.vit_frac_nres, hr_all, 1 - hr_all),
                      ghsl=_mae(w14.ghsl_frac_res, w14.ghsl_frac_nres, hr_all, 1 - hr_all)))
n_locked = 0
for b in buildings:
    if b["id"] in w14.index:
        s = w14.loc[b["id"]]; n_locked += 1
        b["test"] = dict(y=int(s.y_res), sr=r3(s.soft_gt_frac_res), sn=r3(s.soft_gt_frac_nres),
                         vr=r3(s.vit_frac_res), vn=r3(s.vit_frac_nres), gr=r3(s.ghsl_frac_res), gn=r3(s.ghsl_frac_nres))

# empty as a third class (later weeks): week-11 three-head model + building empty truth; week-3 decode threshold
import re as _re
SHARE_MAX = float(_re.search(r"BUILDING_SHARE_MAX\s*=\s*([0-9.]+)", (ROOT / "scripts/week3_pixel_3class_trainlib.py").read_text()).group(1))
w11 = pd.read_parquet(ROOT / "outputs/week11_ae_s2_empty_head_loro/20261113_063942/artifacts/buildings_section3_with_ae_s2_empty.parquet",
                      columns=["id", "soft_gt_frac_empty", "s2e_frac_res", "s2e_frac_nres", "s2e_frac_empty", "ghsl_frac_res", "ghsl_frac_nres"]).set_index("id")
for b in buildings:
    if "test" in b and b["id"] in w11.index:
        s11 = w11.loc[b["id"]]
        b["test"].update(se=r3(s11.soft_gt_frac_empty), er=r3(s11.s2e_frac_res), en=r3(s11.s2e_frac_nres), ee=r3(s11.s2e_frac_empty))
EMPTY_HEAD = dict(n=len(w11),
                  model=round(float((w11.s2e_frac_empty.clip(0, 1) - w11.soft_gt_frac_empty).abs().mean()), 4),
                  ghsl=round(float(((1 - w11.ghsl_frac_res - w11.ghsl_frac_nres).clip(0, 1) - w11.soft_gt_frac_empty).abs().mean()), 4))
w3 = json.loads((ROOT / "datasets/week3_pixel_gt_v2/manifest.json").read_text())["label_counts_by_region"]
W3_NYC = {k: int(v.get("NYC_NE", 0)) for k, v in w3.items()}

# cells whose centre lies inside each labelled footprint (how many training cells a building can give)
from shapely.geometry import Point
for b in buildings:
    if b["status"] == "excluded":
        continue
    b["n_inside"] = sum(1 for c in cells.values() if b["_utm"].contains(Point(c["cx"], c["cy"])))

# metro-wide effect of the "fully covered only" rule (stored week-4 GT, NYC)
nyc = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet", columns=["region", "frac_res_full", "frac_nres_full"],
                      filters=[("region", "==", "NYC_NE")])
FULL = 0.999
nyc_rule = dict(pure_res=int(((nyc.frac_res_full > 0) & (nyc.frac_nres_full == 0)).sum()),
                pure_nres=int(((nyc.frac_nres_full > 0) & (nyc.frac_res_full == 0)).sum()),
                full_res=int(((nyc.frac_res_full >= FULL) & (nyc.frac_nres_full == 0)).sum()),
                full_nres=int(((nyc.frac_nres_full >= FULL) & (nyc.frac_res_full == 0)).sum()))

# ---------------------------------------------------------------- imagery
PX = 1400                                                 # output width in pixels
H_PX = int(round(PX * MH / MW))


def esri_mosaic():
    z = 19
    def tile_xy(lon, lat):
        n = 2 ** z; x = (lon + 180) / 360 * n
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
        return x, y
    tx0, ty0 = tile_xy(W[0], W[3]); tx1, ty1 = tile_xy(W[2], W[1])
    canvas = Image.new("RGB", ((int(tx1) - int(tx0) + 1) * 256, (int(ty1) - int(ty0) + 1) * 256))
    for tx in range(int(tx0), int(tx1) + 1):
        for ty in range(int(ty0), int(ty1) + 1):
            u = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}"
            im = Image.open(io.BytesIO(requests.get(u, timeout=30, headers={"User-Agent": "research-figure"}).content)).convert("RGB")
            canvas.paste(im, ((tx - int(tx0)) * 256, (ty - int(ty0)) * 256))
    crop = canvas.crop((round((tx0 - int(tx0)) * 256), round((ty0 - int(ty0)) * 256),
                        round((tx1 - int(tx0)) * 256), round((ty1 - int(ty0)) * 256)))
    return crop.resize((PX, H_PX), Image.LANCZOS)


def s2_mosaic():
    src = rasterio.open(ROOT / "outputs/week9_ae_s2_nres_loro/20261113_055624/artifacts/s2_mosaics/NYC_NE_bgrn.tif")
    dst = np.zeros((3, H_PX, PX), np.float32)
    dst_T = rasterio.transform.from_bounds(mx0, my1, mx1, my0, PX, H_PX)
    for k, band in enumerate([3, 2, 1]):                    # red, green, blue
        reproject(rasterio.band(src, band), dst[k], dst_transform=dst_T, dst_crs=MERC, resampling=Resampling.nearest)
    v = dst[dst > 0]; lo_, hi_ = np.percentile(v, [2, 98])
    rgb = (np.clip((dst - lo_) / (hi_ - lo_), 0, 1) ** 0.8 * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(rgb)


def b64(im, fmt, q=82):
    buf = io.BytesIO(); im.save(buf, fmt, quality=q) if fmt == "JPEG" else im.save(buf, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


for b in buildings:
    b.pop("_utm")
counts = pd.Series([b["status"] for b in buildings]).value_counts().to_dict()
strata = pd.Series([c["stratum"] for c in cells.values()]).value_counts().to_dict()
out = dict(
    meta=dict(release=RELEASE, window=W, width_m=round(MW, 1), height_m=round(MH, 1), utm=UTM, ghsl_tile=tile.name,
              n_buildings=len(buildings), counts=counts, strata=strata, n_cells=len(cells), n_ae=n_ae,
              n_stored=len(diffs), max_frac_diff=round(max(diffs), 6) if diffs else None,
              ghsl_offset_median=round(float(np.median(offs)), 2), n_ghsl=len(ghsl),
              ghsl_building_check=dict(n=len(gd), max_diff=round(max(gd), 6) if gd else None),
              n_test_buildings=n_locked, nyc_rule=nyc_rule, full_threshold=FULL,
              share_max=SHARE_MAX, empty_head=EMPTY_HEAD, week3_nyc=W3_NYC, own_grid=OWN,
              headline=HEAD,
              kept=dict(res=sum(1 for c in cells.values() if c.get("kept") and c["stratum"] == "res"),
                        nres=sum(1 for c in cells.values() if c.get("kept") and c["stratum"] == "nres"),
                        res_all=sum(1 for c in cells.values() if "pid" in c and c["stratum"] == "res"),
                        nres_all=sum(1 for c in cells.values() if "pid" in c and c["stratum"] == "nres"))),
    buildings=buildings, cells=list(cells.values()), ghsl=ghsl,
    img=dict(esri=b64(esri_mosaic(), "JPEG"), s2=b64(s2_mosaic(), "PNG"), px=PX, py=H_PX),
)
(DATA / "gt_canvas_data.json").write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print(json.dumps(out["meta"], indent=1))
print("json MB", round((DATA / "gt_canvas_data.json").stat().st_size / 1e6, 2))
