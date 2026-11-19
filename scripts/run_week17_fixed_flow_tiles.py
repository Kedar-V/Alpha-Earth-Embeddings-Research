#!/usr/bin/env python3
"""Week 17: the fixed flow, scored as tiles right on each product's own grid.

  1 labelled   Overture Track-A footprints (datasets/building_bbox_grid/buildings.parquet)
  2 area       true footprint area in the box's UTM zone (replaces area_m2_approx, which is Web-Mercator area)
  3 filter     keep buildings with soft truth (>=1 AlphaEarth cell centre inside), an AlphaEarth embedding and a GHSL value
  4 balance    per box: keep all of the class with less footprint area; shuffle the other (seed 42) and keep
               buildings in that order until their area first reaches the smaller class's area
  5 tiles      each grid on its own: AlphaEarth 10 m UTM cells / GHSL 10 m Mollweide cells whose centre lies
               inside a kept building. Truth = area(union(class footprints) ∩ tile) / tile area over all labelled
               footprints (same rule as datasets/week4_pixel_gt). Class: empty if res+nres < 0.15, else larger share.
  6 balance    per box and grid: 1:1 residential / non-residential tiles by truth class (seed 42);
               truth-empty tiles (centre inside a footprint but < 15% covered) are counted and left out
  7 AlphaEarth XGBoost (week-7 settings) on the 64-d embedding -> (res, nres) shares, leave-one-box-out,
               trained on the balanced AlphaEarth tiles of the other nine boxes
  8 score      prediction class by the same rule; count tiles right per grid, box and class

Run layout (resume by passing --run-dir of an unfinished run; finished stages/boxes/folds are reused):
  job.log · job.pid · launch.txt · environment.txt · progress.json · summary.json · DONE | FAILED
  checkpoints/{buildings,ae_tiles}.parquet · checkpoints/ghsl/<box>.parquet
  folds/<nn>_<box>/{model.json,preds.parquet,metrics.json,DONE} · artifacts/ · failures/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
W16_SOFT = ROOT / "outputs/week16_balance_area_loro/latest/checkpoints/ae_by_building.parquet"   # week-6 stage_ae on all 492,044
UNBAL = ROOT / "outputs/week6_overture_polygon_mae/20261110_034500/artifacts/per_building.parquet"
W4 = ROOT / "datasets/week4_pixel_gt/pixels.parquet"
SEED, SHARE_MAX, CELL = 42, 0.15, 10.0
MOLL = "ESRI:54009"
A = [f"A{i:02d}" for i in range(64)]
WINDOW = (-73.9980, 40.7349, -73.9944, 40.7376)          # canvas window (lon/lat)
STAGES = ["area", "filter", "balance_buildings", "ae_tiles", "ghsl_tiles", "balance_tiles", "ae_model", "score", "export"]

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", type=Path, default=None, help="resume an unfinished run")
args = ap.parse_args()
run = args.run_dir or ROOT / "outputs/week17_fixed_flow_tiles" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
for d in ["checkpoints/ghsl", "artifacts", "folds", "failures"]:
    (run / d).mkdir(parents=True, exist_ok=True)
for m in ["DONE", "FAILED"]:
    (run / m).unlink(missing_ok=True)
latest = ROOT / "outputs/week17_fixed_flow_tiles/latest"
latest.unlink(missing_ok=True); latest.symlink_to(run.name)
(run / "job.pid").write_text(str(os.getpid()))
(run / "launch.txt").write_text(f"started_at={datetime.now(timezone.utc).isoformat()}\nargv={sys.argv}\ncwd={os.getcwd()}\n")
T0 = time.time(); STARTED = datetime.now(timezone.utc).isoformat()
STATE = dict(stage="init", stage_index=0, completed=0, total=None, failed=0, unit="", last_progress_at=STARTED, stage_t0=T0)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def gpu_stats():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return [dict(zip(["gpu", "util_pct", "mem_used_mb", "mem_total_mb"], map(int, l.split(", ")))) for l in out]
    except Exception:
        return None


def progress(status="running"):
    p = psutil.Process()
    el = time.time() - STATE["stage_t0"]
    done, tot = STATE["completed"], STATE["total"]
    rate = done / el if el > 0 and done else None
    obj = dict(status=status, stage=STATE["stage"], stage_index=STATE["stage_index"], stages=STAGES, started_at=STARTED, updated_at=now(),
               last_progress_at=STATE["last_progress_at"], completed=done, total=tot, unit=STATE["unit"], failed=STATE["failed"],
               percent_complete=round(100 * done / tot, 2) if tot else None, rate_per_second=round(rate, 3) if rate else None,
               eta_seconds=round((tot - done) / rate) if rate and tot else None, elapsed_seconds=round(time.time() - T0, 1),
               resources=dict(rss_gb=round(p.memory_info().rss / 1e9, 2), cpu_pct=psutil.cpu_percent(None), load_avg=os.getloadavg(),
                              ram_used_pct=psutil.virtual_memory().percent, gpus=gpu_stats()),
               output=str(run))
    atomic_json(run / "progress.json", obj)


def stage(name, total=None, unit=""):
    STATE.update(stage=name, stage_index=STAGES.index(name) + 1, completed=0, total=total, unit=unit, stage_t0=time.time(), last_progress_at=now())
    log(f"== stage {STATE['stage_index']}/{len(STAGES)}: {name}" + (f" ({total} {unit})" if total else ""))
    progress()


def tick(n=1):
    STATE["completed"] += n; STATE["last_progress_at"] = now(); progress()


def log(m):
    line = f"[{now()}] [{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    with open(run / "job.log", "a") as f:
        f.write(line + "\n")


def cls3(r, n):
    r, n = np.nan_to_num(r), np.nan_to_num(n)
    return np.where(r + n < SHARE_MAX, "empty", np.where(r >= n, "res", "nres"))


def main():
    import pyproj, rasterio, rasterio.features, rasterio.merge, shapely, xgboost
    from shapely import STRtree
    spec = importlib.util.spec_from_file_location("w6", ROOT / "scripts/eval_week6_overture_polygon_mae.py")
    w6 = importlib.util.module_from_spec(spec); sys.modules["w6"] = w6; spec.loader.exec_module(w6)
    (run / "environment.txt").write_text("\n".join(f"{k}={v}" for k, v in dict(
        python=sys.executable, host=platform.node(), cpus=os.cpu_count(), ram_gb=round(psutil.virtual_memory().total / 1e9, 1),
        gpus=json.dumps(gpu_stats()), xgboost=xgboost.__version__, shapely=shapely.__version__, rasterio=rasterio.__version__,
        pyproj=pyproj.__version__, pandas=pd.__version__, numpy=np.__version__, seed=SEED, share_max=SHARE_MAX).items()) + "\n")

    # ------------------------------------------------ 1-4 buildings (checkpointed)
    ck_b = run / "checkpoints/buildings.parquet"
    b = pd.read_parquet(w6.BUILDINGS, columns=["id", "region", "crs", "y_res", "wkt", "area_m2_approx"])
    b["id"] = b["id"].astype(str)
    geom = shapely.from_wkt(b.wkt.to_numpy())
    if ck_b.exists():
        stage("area"); log(f"reuse {ck_b}")
        b = b.merge(pd.read_parquet(ck_b).drop(columns=["region", "y_res", "area_m2_approx"]), on="id", how="left")
    else:
        stage("area", total=b.crs.nunique(), unit="UTM zones")
        b["area"] = np.nan
        for crs, idx in b.groupby("crs").groups.items():
            tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            pos = b.index.get_indexer(idx)
            b.loc[idx, "area"] = shapely.area(shapely.transform(geom[pos], lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))))
            tick()
        assert b.area.notna().all()
        stage("filter")
        soft = pd.read_parquet(W16_SOFT, columns=["id", "soft_gt_covered"]); soft["id"] = soft.id.astype(str)
        u = pd.read_parquet(UNBAL, columns=["id", "ghsl_covered"]); u["id"] = u.id.astype(str)
        emb_ids = set(pd.read_parquet(w6.CELLS, columns=["id"], filters=[("inside_polygon", "==", True)]).id.astype(str))
        b = b.merge(soft, on="id", how="left").merge(u, on="id", how="left")
        b["has_emb"] = b.id.isin(emb_ids)
        b["ok"] = b.soft_gt_covered.fillna(False).astype(bool) & b.ghsl_covered.fillna(False).astype(bool) & b.has_emb
        stage("balance_buildings", total=b.region.nunique(), unit="boxes")
        rng = np.random.default_rng(SEED); keep = []
        for reg, s in b[b.ok].groupby("region", sort=True):
            r, n = s[s.y_res == 1], s[s.y_res == 0]
            big, small = (r, n) if r.area.sum() > n.area.sum() else (n, r)
            perm = big.loc[rng.permutation(big.index)]
            k = int(np.searchsorted(perm.area.cumsum().to_numpy(), small.area.sum())) + 1
            keep += list(small.id) + list(perm.id.iloc[:min(k, len(perm))]); tick()
        b["kept"] = b.id.isin(set(keep))
        b[["id", "region", "y_res", "area", "area_m2_approx", "soft_gt_covered", "ghsl_covered", "has_emb", "ok", "kept"]].to_parquet(ck_b, index=False)
    ratio = (b.area_m2_approx / b.area).groupby(b.region).median()
    K = b[b.kept]
    log("area_m2_approx / true area by box: " + json.dumps(ratio.round(3).to_dict()))
    log(f"filter: {int(b.ok.sum()):,} pass (res {int((b.ok & (b.y_res == 1)).sum()):,} · non-res {int((b.ok & (b.y_res == 0)).sum()):,})")
    log(f"area 1:1: {len(K):,} kept (res {int((K.y_res == 1).sum()):,} · non-res {int((K.y_res == 0).sum()):,}) · res area share {K.area[K.y_res == 1].sum() / K.area.sum():.4f}")

    # ------------------------------------------------ 5a AlphaEarth tiles (checkpointed)
    ck_ae = run / "checkpoints/ae_tiles.parquet"
    stage("ae_tiles")
    if ck_ae.exists():
        ae = pd.read_parquet(ck_ae); log(f"reuse {ck_ae}")
    else:
        cells = pd.read_parquet(w6.CELLS, columns=["id", "pixel_lon", "pixel_lat", "utm_x", "utm_y"] + A, filters=[("inside_polygon", "==", True)])
        cells["id"] = cells.id.astype(str)
        cells = cells[cells.id.isin(set(K.id))].copy()
        cells["k"] = w6.lonlat_key(cells.pixel_lon.to_numpy(), cells.pixel_lat.to_numpy())
        n_dup = int(cells.k.duplicated().sum()); cells = cells.drop_duplicates("k")
        cells["region"] = cells.id.map(K.set_index("id").region)
        pix = pd.read_parquet(W4, columns=["pixel_id", "pixel_lon", "pixel_lat", "frac_res_full", "frac_nres_full"])
        pix["k"] = w6.lonlat_key(pix.pixel_lon.to_numpy(), pix.pixel_lat.to_numpy()); pix = pix.drop_duplicates("k")
        ae = cells.merge(pix[["k", "pixel_id", "frac_res_full", "frac_nres_full"]], on="k", how="left")
        miss = int(ae.frac_res_full.isna().sum())
        log(f"AlphaEarth tiles in scope {len(ae):,} (cells shared by two kept buildings counted once: {n_dup:,}) · missing stored truth {miss:,}")
        if miss:
            ae[ae.frac_res_full.isna()][["k", "id", "region"]].to_json(run / "failures/ae_missing_truth.jsonl", orient="records", lines=True)
        ae = ae[ae.frac_res_full.notna()].rename(columns={"frac_res_full": "tr", "frac_nres_full": "tn"}).copy()
        ae["tc"] = cls3(ae.tr.to_numpy(), ae.tn.to_numpy())
        ae.to_parquet(ck_ae, index=False)
    tick()

    # ------------------------------------------------ 5b GHSL tiles, exact truth on the Mollweide grid (checkpoint per box)
    regions = sorted(b.region.unique())
    stage("ghsl_tiles", total=len(regions), unit="boxes")
    mos = w6.DualMosaic.open()
    to_moll = pyproj.Transformer.from_crs("EPSG:4326", MOLL, always_xy=True)
    gm = None
    parts = []
    for reg in regions:
        ck = run / "checkpoints/ghsl" / f"{reg}.parquet"
        if ck.exists():
            parts.append(pd.read_parquet(ck)); log(f"GHSL {reg}: reuse checkpoint"); tick(); continue
        if gm is None:
            gm = shapely.transform(geom, lambda xy: np.column_stack(to_moll.transform(xy[:, 0], xy[:, 1])))
        t1 = time.time()
        s = b[b.region == reg]; kg = gm[s.index[s.kept].to_numpy()]
        minx, miny, maxx, maxy = shapely.total_bounds(kg); pad = 20.0
        bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        idx = mos.datasets_for_bounds(*bounds)
        bu, T = rasterio.merge.merge([mos.s_ds[i] for i in idx], bounds=bounds, nodata=255)
        nr, T2 = rasterio.merge.merge([mos.nres_ds[i] for i in idx], bounds=bounds, nodata=255)
        assert T == T2 and abs(T.a - CELL) < 1e-6 and abs(T.e + CELL) < 1e-6, (T, T2)
        bu, nr = bu[0], nr[0]
        burn = rasterio.features.rasterize(((g, 1) for g in kg), out_shape=bu.shape, transform=T, fill=0, all_touched=False, dtype="uint8")
        rows, cols = np.nonzero(burn)
        x0 = T.c + cols * CELL; y1 = T.f - rows * CELL
        boxes = shapely.box(x0, y1 - CELL, x0 + CELL, y1)
        tr_t = np.zeros(len(boxes)); tn_t = np.zeros(len(boxes))
        for yv, out in [(1, tr_t), (0, tn_t)]:
            fp = gm[s.index[s.y_res == yv].to_numpy()]
            ci, fi = STRtree(fp).query(boxes, predicate="intersects")
            if not len(ci):
                continue
            o = np.argsort(ci, kind="stable"); ci, fi = ci[o], fi[o]
            st = np.r_[0, np.nonzero(np.diff(ci))[0] + 1]; en = np.r_[st[1:], len(ci)]
            one = (en - st) == 1
            out[ci[st[one]]] = shapely.area(shapely.intersection(boxes[ci[st[one]]], fp[fi[st[one]]]))
            for a0, a1 in zip(st[~one], en[~one]):
                out[ci[a0]] = shapely.area(shapely.intersection(boxes[ci[a0]], shapely.union_all(fp[fi[a0:a1]])))
        tr_t /= CELL * CELL; tn_t /= CELL * CELL
        g_fr, g_fn, valid = w6.raw_to_soft(bu[rows, cols], nr[rows, cols])
        d = pd.DataFrame(dict(region=reg, mx=x0 + CELL / 2, my=y1 - CELL / 2, tr=tr_t, tn=tn_t, pr=g_fr, pn=g_fn, valid=valid))
        d.to_parquet(ck, index=False); parts.append(d)
        log(f"GHSL {reg}: {len(d):,} tiles in scope · {int((~valid).sum())} no-data · {time.time() - t1:.1f}s"); tick()
    gh = pd.concat(parts, ignore_index=True)
    gh["tc"] = cls3(gh.tr.to_numpy(), gh.tn.to_numpy())

    # ------------------------------------------------ 6 tile balance per box and grid
    stage("balance_tiles", total=2, unit="grids")

    def bal_tiles(df, valid_col=None):
        df = df.copy(); df["bal"] = False
        for reg in sorted(df.region.unique()):
            rng = np.random.default_rng(SEED)
            s = df[(df.region == reg) & (df.tc != "empty") & (df[valid_col] if valid_col else True)]
            r, n = s.index[s.tc == "res"], s.index[s.tc == "nres"]; m = min(len(r), len(n))
            df.loc[rng.choice(r, m, replace=False), "bal"] = True
            df.loc[rng.choice(n, m, replace=False), "bal"] = True
        return df

    ae = bal_tiles(ae); tick()
    gh = bal_tiles(gh, "valid"); tick()
    log(f"tiles 1:1 · AlphaEarth {int(ae.bal.sum()):,} of {len(ae):,} · GHSL {int(gh.bal.sum()):,} of {len(gh):,}")

    # ------------------------------------------------ 7 AlphaEarth model, leave-one-box-out (one folder per fold)
    from xgboost import XGBRegressor
    stage("ae_model", total=len(regions), unit="folds")
    ae["pr"] = np.nan; ae["pn"] = np.nan
    X = ae[A].to_numpy("float32"); Y = ae[["tr", "tn"]].to_numpy("float32")
    for i, reg in enumerate(regions):
        fd = run / "folds" / f"{i:02d}_{reg}"; fd.mkdir(exist_ok=True)
        tem = (ae.region == reg).to_numpy()
        if (fd / "DONE").exists():
            p = pd.read_parquet(fd / "preds.parquet")
            ae.loc[tem, "pr"], ae.loc[tem, "pn"] = p.pr.to_numpy(), p.pn.to_numpy()
            log(f"fold {reg}: reuse"); tick(); continue
        t1 = time.time()
        trm = (ae.region != reg).to_numpy() & ae.bal.to_numpy()
        m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=1.0, objective="reg:squarederror", multi_strategy="one_output_per_tree", tree_method="hist",
                         device=f"cuda:{i % 2}", random_state=SEED, n_jobs=8, verbosity=0)
        m.fit(X[trm], Y[trm])
        p = np.clip(m.predict(X[tem]), 0, 1)
        if not np.isfinite(p).all():
            raise RuntimeError(f"non-finite predictions in fold {reg}")
        ae.loc[tem, "pr"], ae.loc[tem, "pn"] = p[:, 0], p[:, 1]
        m.save_model(fd / "model.json")
        pd.DataFrame(dict(k=ae.k[tem].to_numpy(), pr=p[:, 0], pn=p[:, 1])).to_parquet(fd / "preds.parquet", index=False)
        hb = ae[tem & ae.bal.to_numpy()]; pc = cls3(hb.pr.to_numpy(), hb.pn.to_numpy())
        met = dict(box=reg, device=f"cuda:{i % 2}", n_train=int(trm.sum()), n_pred=int(tem.sum()), n_scored_balanced=int(len(hb)),
                   mae_res=float(np.abs(hb.pr - hb.tr).mean()), mae_nres=float(np.abs(hb.pn - hb.tn).mean()),
                   tiles_right=int((pc == hb.tc.to_numpy()).sum()), seconds=round(time.time() - t1, 1))
        atomic_json(fd / "metrics.json", met); (fd / "DONE").write_text(now())
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(met) + "\n")
        log(f"fold {reg}: train {met['n_train']:,} → predict {met['n_pred']:,} · balanced right {met['tiles_right']:,}/{met['n_scored_balanced']:,} · {met['seconds']}s"); tick()

    # ------------------------------------------------ 8 score
    stage("score")
    ae["pc"] = cls3(ae.pr.to_numpy(), ae.pn.to_numpy()); ae["ok3"] = ae.pc == ae.tc
    gh["pc"] = np.where(gh.valid, cls3(gh.pr.to_numpy(), gh.pn.to_numpy()), "nodata"); gh["ok3"] = gh.pc == gh.tc

    def score(df):
        s = df[df.bal]
        out = dict(n=int(len(s)), right=int(s.ok3.sum()))
        for c in ["res", "nres"]:
            q = s[s.tc == c]; out[c] = dict(n=int(len(q)), right=int(q.ok3.sum()), pred={k: int(v) for k, v in q.pc.value_counts().items()})
        out["by_region"] = {r: dict(n=int(len(q)), right=int(q.ok3.sum()), res_right=int(q[q.tc == "res"].ok3.sum()), nres_right=int(q[q.tc == "nres"].ok3.sum()))
                            for r, q in s.groupby("region")}
        return out

    def scope(df):
        return dict(n=int(len(df)), res=int((df.tc == "res").sum()), nres=int((df.tc == "nres").sum()), empty=int((df.tc == "empty").sum()),
                    by_region={r: dict(n=int(len(q)), res=int((q.tc == "res").sum()), nres=int((q.tc == "nres").sum()),
                                       empty=int((q.tc == "empty").sum()), bal=int(q.bal.sum())) for r, q in df.groupby("region")})

    res = dict(ae=dict(scope=scope(ae), score=score(ae)), ghsl=dict(scope=scope(gh), score=score(gh), nodata=int((~gh.valid).sum())))
    for k in ["ae", "ghsl"]:
        s = res[k]["score"]
        log(f"{k}: {s['right']:,} / {s['n']:,} right · res {s['res']['right']:,}/{s['res']['n']:,} · non-res {s['nres']['right']:,}/{s['nres']['n']:,}")
    f = lambda d: dict(res=int((d.y_res == 1).sum()), nres=int((d.y_res == 0).sum()), a_res=float(d.area[d.y_res == 1].sum()), a_nres=float(d.area[d.y_res == 0].sum()))
    res["buildings"] = {reg: dict(labelled=f(s), passed=f(s[s.ok]), kept=f(s[s.kept]), approx_ratio=float(ratio[reg]),
                                  cut="residential" if s[s.ok & (s.y_res == 1)].area.sum() > s[s.ok & (s.y_res == 0)].area.sum() else "non_residential")
                        for reg, s in b.groupby("region", sort=True)}
    res["config"] = dict(seed=SEED, share_max=SHARE_MAX, cell_m=CELL, run=str(run.relative_to(ROOT)))

    # ------------------------------------------------ 9 export
    stage("export")
    lo0, la0, lo1, la1 = WINDOW; pd_ = 0.0003
    wa = ae[ae.pixel_lon.between(lo0 - pd_, lo1 + pd_) & ae.pixel_lat.between(la0 - pd_, la1 + pd_)]
    xa, ya = to_moll.transform([lo0 - pd_, lo1 + pd_], [la0 - pd_, la1 + pd_])
    wg = gh[(gh.region == "NYC_NE") & gh.mx.between(min(xa), max(xa)) & gh.my.between(min(ya), max(ya))]
    wa[["pixel_id", "id", "pixel_lon", "pixel_lat", "utm_x", "utm_y", "tr", "tn", "tc", "pr", "pn", "pc", "ok3", "bal"]].to_parquet(run / "artifacts/window_ae_tiles.parquet", index=False)
    wg[["mx", "my", "tr", "tn", "tc", "pr", "pn", "pc", "ok3", "bal", "valid"]].to_parquet(run / "artifacts/window_ghsl_tiles.parquet", index=False)
    ae.drop(columns=A).to_parquet(run / "artifacts/ae_tiles.parquet", index=False)
    gh.to_parquet(run / "artifacts/ghsl_tiles.parquet", index=False)
    b[["id", "region", "y_res", "area", "area_m2_approx", "ok", "kept"]].to_parquet(run / "artifacts/buildings.parquet", index=False)
    res.update(status="completed", runtime_seconds=round(time.time() - T0, 1), finished_at=now(),
               window=dict(ae_tiles=int(len(wa)), ghsl_tiles=int(len(wg))),
               artifacts=sorted(str(p.relative_to(run)) for p in (run / "artifacts").iterdir()))
    atomic_json(run / "summary.json", res)
    (run / "DONE").write_text(now())
    STATE["completed"] = 1; progress("completed")
    log(f"done → {run}/summary.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        (run / "failures/traceback.txt").write_text(tb)
        (run / "FAILED").write_text(f"{now()}\n{e!r}\n")
        log("FAILED: " + repr(e)); progress("failed")
        raise
