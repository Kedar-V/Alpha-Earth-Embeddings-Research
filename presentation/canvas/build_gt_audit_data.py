#!/usr/bin/env python3
"""Metro-wide GT audit numbers for the canvas (answers the GT review questions with real data).

Every number comes from the project's stored tables; nothing is simulated.
  regions       scripts/build_building_type_dataset.py US_REGIONS (lon/lat boxes) + buildings.parquet
  composition   datasets/week4_pixel_gt/pixels.parquet (every AE cell touching a labelled footprint)
  clustering    datasets/week4_pixel_gt_balanced/pixels.parquet (the training cells) via building_ids
  balancing     buildings.parquet (footprint areas) + the two pixel tables above
  test set      outputs/week14_ae_grid_vit_s2_loro/.../buildings_section3_with_ae_grid_vit_s2.parquet
Writes presentation/canvas/data/gt_audit.json
Run:  .venv/bin/python presentation/canvas/build_gt_audit_data.py
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data" / "gt_audit.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


btd = load("btd", ROOT / "scripts/build_building_type_dataset.py")
REGIONS = btd.US_REGIONS
ORDER = list(REGIONS)
r4 = lambda v: round(float(v), 4)

# ------------------------------------------------------------------ regions: boxes, buildings, cells
B = pd.read_parquet(ROOT / "datasets/building_bbox_grid/buildings.parquet",
                    columns=["id", "region", "y_res", "area_m2_approx", "class", "source_primary"])
P = pd.read_parquet(ROOT / "datasets/week4_pixel_gt/pixels.parquet",
                    columns=["region", "frac_res_full", "frac_nres_full", "n_buildings"])
K = pd.read_parquet(ROOT / "datasets/week4_pixel_gt_balanced/pixels.parquet",
                    columns=["region", "stratum", "frac_res_full", "frac_nres_full", "building_ids"])
T = pd.read_parquet(ROOT / "outputs/week14_ae_grid_vit_s2_loro/20261113_075223/artifacts/buildings_section3_with_ae_grid_vit_s2.parquet")
man = json.loads((ROOT / "datasets/week4_pixel_gt_balanced/manifest.json").read_text())

P["pure_res"] = (P.frac_res_full > 0) & (P.frac_nres_full == 0)
P["pure_nres"] = (P.frac_nres_full > 0) & (P.frac_res_full == 0)
P["mixed"] = (P.frac_res_full > 0) & (P.frac_nres_full > 0)

regions = []
for name in ORDER:
    x0, y0, x1, y1 = REGIONS[name]
    lat = (y0 + y1) / 2
    w_km = (x1 - x0) * 111.32 * math.cos(math.radians(lat)); h_km = (y1 - y0) * 110.57
    b = B[B.region == name]; p = P[P.region == name]; t = T[T.region == name]
    regions.append(dict(
        name=name, box=[x0, y0, x1, y1], w_km=round(w_km, 1), h_km=round(h_km, 1), area_km2=round(w_km * h_km),
        b_res=int((b.y_res == 1).sum()), b_nres=int((b.y_res == 0).sum()),
        a_res_km2=round(b[b.y_res == 1].area_m2_approx.sum() / 1e6, 2), a_nres_km2=round(b[b.y_res == 0].area_m2_approx.sum() / 1e6, 2),
        med_res=round(float(b[b.y_res == 1].area_m2_approx.median())), med_nres=round(float(b[b.y_res == 0].area_m2_approx.median())),
        cells=int(len(p)), c_res=int(p.pure_res.sum()), c_nres=int(p.pure_nres.sum()), c_mixed=int(p.mixed.sum()),
        kept=int(man["counts_by_region"][name]["residential"]),
        cut="residential" if p.pure_res.sum() > p.pure_nres.sum() else "non-residential",
        t_res=int((t.y_res == 1).sum()), t_nres=int((t.y_res == 0).sum())))

# ------------------------------------------------------------------ composition (res / non-res / empty)
def simplex_hist(fr, fn, n=10):
    """counts on a triangular lattice: i = res bin, j = non-res bin (i + j <= n)."""
    i = np.clip(np.floor(fr * n + 1e-9), 0, n).astype(int); j = np.clip(np.floor(fn * n + 1e-9), 0, n).astype(int)
    over = i + j > n; j[over] = n - i[over]
    H = np.zeros((n + 1, n + 1), int); np.add.at(H, (i, j), 1)
    return [[int(i_), int(j_), int(H[i_, j_])] for i_ in range(n + 1) for j_ in range(n + 1 - i_) if H[i_, j_]]


EDGES = [0, 0.01, 0.25, 0.5, 0.75, 1.0001]
def empty_bins(s):
    e = (1 - s.frac_res_full - s.frac_nres_full).clip(0, 1)
    c = pd.cut(e, EDGES, right=False).value_counts(sort=False)
    return [r4(v / max(len(s), 1)) for v in c.values]


def comp(p):
    e = (1 - p.frac_res_full - p.frac_nres_full).clip(0, 1)
    m = p[p.mixed]
    rs = (m.frac_res_full / (m.frac_res_full + m.frac_nres_full))
    return dict(n=int(len(p)), pure_res=int(p.pure_res.sum()), pure_nres=int(p.pure_nres.sum()), mixed=int(p.mixed.sum()),
                empty_bins=dict(res=empty_bins(p[p.pure_res]), nres=empty_bins(p[p.pure_nres]), mixed=empty_bins(m)),
                mean_empty=dict(res=r4(e[p.pure_res].mean()), nres=r4(e[p.pure_nres].mean()), mixed=r4(e[p.mixed].mean()) if len(m) else None),
                mixed_res_share=[r4(v) for v in rs.quantile([.1, .25, .5, .75, .9]).values] if len(m) else None,
                multi_building=r4((p.n_buildings > 1).mean()),
                tern=tern_bins(p))


def tern_bins(p, n=20):
    """[mean res share, mean non-res share, count, kind] per bin: pure cells along their edge, mixed on a 2-D lattice."""
    out = []
    for kind, mask, key in (("res", p.pure_res, "frac_res_full"), ("nres", p.pure_nres, "frac_nres_full")):
        s = p[mask]; b = np.clip(np.floor(s[key] * n), 0, n - 1)
        for _, g in s.groupby(b):
            out.append([r4(g.frac_res_full.mean()), r4(g.frac_nres_full.mean()), int(len(g)), kind])
    m = p[p.mixed]; m10 = 10
    bi = np.clip(np.floor(m.frac_res_full * m10), 0, m10 - 1); bj = np.clip(np.floor(m.frac_nres_full * m10), 0, m10 - 1)
    for _, g in m.groupby([bi, bj]):
        out.append([r4(g.frac_res_full.mean()), r4(g.frac_nres_full.mean()), int(len(g)), "mixed"])
    return out


composition = dict(all=comp(P), nyc=comp(P[P.region == "NYC_NE"]), edges=[0, 1, 25, 50, 75, 100])
# the frozen buildings' soft truth also has an empty share (week 11)
w11 = pd.read_parquet(ROOT / "outputs/week11_ae_s2_empty_head_loro/20261113_063942/artifacts/buildings_section3_with_ae_s2_empty.parquet",
                      columns=["y_res_x", "soft_gt_frac_empty"]).rename(columns={"y_res_x": "y_res"})
composition["building_empty"] = dict(res=r4(w11[w11.y_res == 1].soft_gt_frac_empty.mean()), nres=r4(w11[w11.y_res == 0].soft_gt_frac_empty.mean()))

# ------------------------------------------------------------------ clustering on the actual training cells
K["bids"] = K.building_ids.astype(str).str.split(r"[,;| ]+")
K["w"] = 1.0 / K.bids.str.len()
E = K[["region", "stratum", "bids", "w"]].explode("bids")
per_b = E.groupby(["stratum", "region", "bids"]).w.sum().rename("k").reset_index()


def lorenz(k):
    v = np.sort(k)[::-1]; cs = np.cumsum(v) / v.sum(); n = len(v)
    xs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return [[x, r4(cs[max(0, int(math.ceil(x * n)) - 1)])] for x in xs]


def cluster(df):
    k = df.k.to_numpy(); tot = k.sum()
    v = np.sort(k)[::-1]
    caps = {}
    for cap in (1, 5, 10, 25):
        kc = np.minimum(k, cap); caps[str(cap)] = dict(cells=int(round(kc.sum())), kish=int(round(kc.sum() ** 2 / (kc ** 2).sum())))
    return dict(buildings=int(len(k)), cells=int(round(tot)), kish=int(round(tot ** 2 / (k ** 2).sum())),
                top1=r4(v[:max(1, len(v) // 100)].sum() / tot), top10=r4(v[:max(1, len(v) // 10)].sum() / tot),
                median=r4(np.median(k)), max=int(round(v[0])), lorenz=lorenz(k), caps=caps)


clustering = {s: cluster(per_b[per_b.stratum == s]) for s in ("residential", "non_residential")}
clustering["by_region"] = {}
for name in ORDER:
    clustering["by_region"][name] = {s: {kk: vv for kk, vv in cluster(per_b[(per_b.stratum == s) & (per_b.region == name)]).items()
                                         if kk in ("buildings", "cells", "kish", "top10", "max")} for s in ("residential", "non_residential")}
# largest single contributor in NYC
nyc_nres = per_b[(per_b.region == "NYC_NE") & (per_b.stratum == "non_residential")].sort_values("k", ascending=False).head(1)
clustering["nyc_top"] = dict(id=str(nyc_nres.bids.iloc[0]), cells=int(round(nyc_nres.k.iloc[0])))

# ------------------------------------------------------------------ balancing: what each unit implies
K["built"] = K.frac_res_full + K.frac_nres_full
bal = []
for name in ORDER:
    b = B[B.region == name]; k = K[K.region == name]
    rr, nn = b[b.y_res == 1], b[b.y_res == 0]
    kr, kn = k[k.stratum == "residential"], k[k.stratum == "non_residential"]
    bal.append(dict(
        name=name,
        # current: equal cells -> built area and buildings behind them
        cell_area_ratio=r4(kn.frac_nres_full.sum() / kr.frac_res_full.sum()),
        cell_bld_res=int(per_b[(per_b.region == name) & (per_b.stratum == "residential")].shape[0]),
        cell_bld_nres=int(per_b[(per_b.region == name) & (per_b.stratum == "non_residential")].shape[0]),
        # equal building counts -> non-res area (and cells) per res
        count_area_ratio=r4(nn.area_m2_approx.mean() / rr.area_m2_approx.mean()),
        # equal area -> residential buildings per non-res building
        area_bld_ratio=r4(nn.area_m2_approx.mean() / rr.area_m2_approx.mean()),
        natural_area_ratio=r4(rr.area_m2_approx.sum() / nn.area_m2_approx.sum()),
        mean_cover_res=r4(kr.frac_res_full.mean()), mean_cover_nres=r4(kn.frac_nres_full.mean())))
kr_all, kn_all = K[K.stratum == "residential"], K[K.stratum == "non_residential"]
balancing = dict(by_region=bal, cell_area_ratio=r4(kn_all.frac_nres_full.sum() / kr_all.frac_res_full.sum()),
                 mean_cover_res=r4(kr_all.frac_res_full.mean()), mean_cover_nres=r4(kn_all.frac_nres_full.mean()),
                 n_per_class=int(man["n_residential"]))

# ------------------------------------------------------------------ frozen test set: mix and class-weighted error
MODELS = [("vit", "ViT + S2 & shape (week 14)"), ("ghsl", "GHSL"), ("ae", "AlphaEarth Ridge"), ("pls", "PLS"),
          ("xgb", "XGBoost"), ("cnn", "CNN"), ("ring", "Rings"), ("s2", "S2 & shape")]
hr = (T.y_res == 1).astype(float)


def err(m, truth):
    pr, pn = T[f"{m}_frac_res"].clip(0, 1), T[f"{m}_frac_nres"].clip(0, 1)
    gr, gn = (T.soft_gt_frac_res, T.soft_gt_frac_nres) if truth == "soft" else (hr, 1 - hr)
    return 0.5 * ((pr - gr).abs() + (pn - gn).abs())


scores = []
for m, label in MODELS:
    if f"{m}_frac_res" not in T:
        continue
    row = dict(key=m, label=label)
    for truth in ("soft", "hard"):
        e = err(m, truth); ok = e.notna()
        er, en = e[ok & (T.y_res == 1)].mean(), e[ok & (T.y_res == 0)].mean()
        row[truth] = dict(pooled=r4(e[ok].mean()), res=r4(er), nres=r4(en), balanced=r4(0.5 * (er + en)), n=int(ok.sum()))
    scores.append(row)
test = dict(n=int(len(T)), res=int((T.y_res == 1).sum()), nres=int((T.y_res == 0).sum()), scores=scores)

# ------------------------------------------------------------------ aerial thumbnail of each study box (Esri World Imagery)
import base64, io
import requests
from PIL import Image, ImageDraw


def thumb(box_, z=12, width=320):
    x0, y0, x1, y1 = box_
    def txy(lon, lat):
        n = 2 ** z
        return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    pad = 0.15 * max(x1 - x0, y1 - y0)
    ax, ay = txy(x0 - pad, y1 + pad); bx, by = txy(x1 + pad, y0 - pad)
    cv = Image.new("RGB", ((int(bx) - int(ax) + 1) * 256, (int(by) - int(ay) + 1) * 256))
    for tx in range(int(ax), int(bx) + 1):
        for ty in range(int(ay), int(by) + 1):
            u = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}"
            im = Image.open(io.BytesIO(requests.get(u, timeout=30, headers={"User-Agent": "research-figure"}).content)).convert("RGB")
            cv.paste(im, ((tx - int(ax)) * 256, (ty - int(ay)) * 256))
    ox, oy = int(ax) * 256, int(ay) * 256
    crop = cv.crop((round(ax * 256) - ox, round(ay * 256) - oy, round(bx * 256) - ox, round(by * 256) - oy))
    sc = width / crop.width; crop = crop.resize((width, round(crop.height * sc)), Image.LANCZOS)
    to = lambda lon, lat: ((txy(lon, lat)[0] - ax) * 256 * sc, (txy(lon, lat)[1] - ay) * 256 * sc)
    (p0, p1) = to(x0, y1), to(x1, y0)
    buf = io.BytesIO(); crop.save(buf, "JPEG", quality=72)
    return dict(img="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w=crop.width, h=crop.height,
                box=[round(p0[0], 1), round(p0[1], 1), round(p1[0], 1), round(p1[1], 1)], to=to)


CANVAS_W = (-73.9980, 40.7349, -73.9944, 40.7376)
for r in regions:
    t = thumb(r["box"]); to = t.pop("to")
    if r["name"] == "NYC_NE":
        a, b = to(CANVAS_W[0], CANVAS_W[3]), to(CANVAS_W[2], CANVAS_W[1]); t["window"] = [round((a[0] + b[0]) / 2, 1), round((a[1] + b[1]) / 2, 1)]
    r["thumb"] = t

out = dict(regions=regions, composition=composition, clustering=clustering, balancing=balancing, test=test,
           release=btd.RELEASE, n_labels=len(btd.CLASS_TO_LABEL),
           res_classes=sorted(k for k, v in btd.CLASS_TO_LABEL.items() if v in ("single_family", "multi_family")),
           nres_classes=sorted(k for k, v in btd.CLASS_TO_LABEL.items() if v not in ("single_family", "multi_family")),
           src_share=B.source_primary.value_counts(normalize=True).round(4).to_dict())
OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
print(json.dumps({k: out[k] for k in ("test", "balancing")}, indent=1)[:4000])
print(json.dumps({s: {k: v for k, v in clustering[s].items() if k != "lorenz"} for s in ("residential", "non_residential")}, indent=1))
print(len(composition["all"]["tern"]), len(composition["nyc"]["tern"]))
print("KB", round(OUT.stat().st_size / 1e3, 1))
