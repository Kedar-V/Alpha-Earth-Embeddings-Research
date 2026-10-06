#!/usr/bin/env python3
"""Build per-city PLS boundary walks for the embedding globe viewer.

For each city:
  - centroid of high-conf correct res  (P(res) > 0.9)
  - centroid of high-conf correct nres (P(res) < 0.1)
  - 12-step linear path in PLS-32 (NN retrieval) and 64-D (LORO LR P(res))
  - nearest real patch per step from that city's viz sample

USA uses week4 fold joblibs; IND/JPN use intl extract LORO folds.

Observability (submit-job):
  viz/usa_embedding_globe/runs/boundary_<stamp>/
    job.log, job.pid, progress.json, summary.json, DONE, launch.txt
    artifacts/, checkpoints/, failures/

Durable outputs:
  viz/usa_embedding_globe/data/boundary_walks.json
  viz/usa_embedding_globe/data/boundary_walks_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz" / "usa_embedding_globe"
DATA = OUT / "data"
POINTS = DATA / "points.json"
INTL_FOLD_DIR = DATA / "intl_folds"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from export_usa_embedding_globe_sample import (  # noqa: E402
    A_COLS,
    CITY_TO_WEEK4_REGION,
    WEEK4_CKPT_DIR,
    atomic_json,
    connect,
    update_progress,
)

N_STEPS = 12  # endpoints + 10 interior
MIN_ENDPOINT = 3
PLS_DIMS = 32
PLS3_DIMS = 3
HI_RES = 0.9
HI_NRES = 0.1  # P(res) < this
MID_RES = 0.7
MID_NRES = 0.3
WALK_PROBA_MODEL = "pls3_lr"  # path + endpoint P(res); NN still PLS-32


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _proba(p: dict) -> float | None:
    if p.get("proba_res") is None:
        return None
    try:
        return float(p["proba_res"])
    except (TypeError, ValueError):
        return None


def pick_endpoint_indices(meta: list[dict]) -> tuple[list[int], list[int], dict]:
    """Choose res / nres centroid membership with falling-back selectivity.

    Prefer high-conf correct cells; if a city lacks those (common under weak
    IND LORO), widen until both sides have enough cells. Last resort uses GT
    extremes by model P(res) so every city with both classes can walk.
    """
    n = len(meta)
    correct_res = [
        i for i, p in enumerate(meta)
        if p.get("correct") is True and p.get("gt") == 1 and _proba(p) is not None
    ]
    correct_nres = [
        i for i, p in enumerate(meta)
        if p.get("correct") is True and p.get("gt") == 0 and _proba(p) is not None
    ]
    gt_res = [i for i, p in enumerate(meta) if p.get("gt") == 1 and _proba(p) is not None]
    gt_nres = [i for i, p in enumerate(meta) if p.get("gt") == 0 and _proba(p) is not None]

    def hi(idxs, side: str, thr: float) -> list[int]:
        if side == "res":
            return [i for i in idxs if _proba(meta[i]) > thr]
        return [i for i in idxs if _proba(meta[i]) < thr]

    def topk(idxs, side: str, k: int) -> list[int]:
        if not idxs:
            return []
        key = (lambda i: _proba(meta[i])) if side == "res" else (lambda i: -_proba(meta[i]))
        # res: highest P(res); nres: lowest P(res) == highest -P
        ordered = sorted(idxs, key=key, reverse=True)
        return ordered[: max(1, min(k, len(ordered)))]

    ladder = [
        ("correct_hi_0.9_0.1", hi(correct_res, "res", HI_RES), hi(correct_nres, "nres", HI_NRES), MIN_ENDPOINT),
        ("correct_mid_0.7_0.3", hi(correct_res, "res", MID_RES), hi(correct_nres, "nres", MID_NRES), MIN_ENDPOINT),
        ("correct_any", correct_res, correct_nres, MIN_ENDPOINT),
        ("correct_any_min1", correct_res, correct_nres, 1),
        ("gt_extreme_top3", topk(gt_res, "res", MIN_ENDPOINT), topk(gt_nres, "nres", MIN_ENDPOINT), 1),
        ("gt_any", gt_res, gt_nres, 1),
    ]
    for mode, res_idx, nres_idx, need in ladder:
        if len(res_idx) >= need and len(nres_idx) >= need:
            return res_idx, nres_idx, {
                "endpoint_mode": mode,
                "n_res_pool": len(res_idx),
                "n_nres_pool": len(nres_idx),
                "min_required": need,
                "n_meta": n,
            }
    return [], [], {
        "endpoint_mode": "failed",
        "n_res_pool": 0,
        "n_nres_pool": 0,
        "n_correct_res": len(correct_res),
        "n_correct_nres": len(correct_nres),
        "n_gt_res": len(gt_res),
        "n_gt_nres": len(gt_nres),
        "n_meta": n,
    }


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float(np.linalg.norm(a - b))
    return float(1.0 - np.dot(a, b) / (na * nb))


def load_embeddings(cell_ids: list[int]) -> dict[int, np.ndarray]:
    emb_sql = ", ".join(f"c.{a}" for a in A_COLS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT c.cell_id, {emb_sql}
                FROM alphaearth_cells c
                WHERE c.cell_id = ANY(%s)
                """,
                (cell_ids,),
            )
            rows = list(cur.fetchall())
    out: dict[int, np.ndarray] = {}
    for r in rows:
        cid = int(r["cell_id"])
        out[cid] = np.asarray([float(r[a]) for a in A_COLS], dtype=np.float64)
    return out


def resolve_fold(city_id: str, intl_fold_dir: Path | None = None) -> tuple[Path | None, str | None, str]:
    """Return (checkpoint_path, fold_id, method) for a city.

    Prefer extract LORO folds with PLS-3 LR (USA/IND/JPN). Fall back to week4
    64-D LR only if the intl fold is missing.
    """
    intl_fold_dir = Path(intl_fold_dir or INTL_FOLD_DIR)
    intl_path = intl_fold_dir / f"fold_{city_id}.joblib"
    if intl_path.exists():
        return intl_path, city_id, "intl_loro_pls3"
    if city_id in CITY_TO_WEEK4_REGION:
        region = CITY_TO_WEEK4_REGION[city_id]
        path = WEEK4_CKPT_DIR / f"fold_{region}.joblib"
        return path, region, "week4_baseline_lr_loro"
    return None, None, "unknown"


def build_city_walk(
    city_id: str,
    city_name: str,
    pts: list[dict],
    emb: dict[int, np.ndarray],
    intl_fold_dir: Path | None = None,
) -> dict | None:
    ckpt, region, method = resolve_fold(city_id, intl_fold_dir=intl_fold_dir)
    if ckpt is None:
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "skipped",
            "reason": "no_fold_map",
        }
    if not ckpt.exists():
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "skipped",
            "reason": f"missing {ckpt.name}",
        }

    fold = joblib.load(ckpt)
    mu = np.asarray(fold["mu"], dtype=np.float64).reshape(-1)
    sd = np.asarray(fold["sd"], dtype=np.float64).reshape(-1)
    sd = np.clip(sd, 1e-6, None)
    R_full = np.asarray(fold["pls_da"]["x_rotations"], dtype=np.float64)
    R = R_full[:, :PLS_DIMS]
    use_pls3 = (
        WALK_PROBA_MODEL == "pls3_lr"
        and fold.get("pls3_lr_w") is not None
        and R_full.shape[1] >= PLS3_DIMS
    )
    if use_pls3:
        w = np.asarray(fold["pls3_lr_w"], dtype=np.float64).reshape(-1)[:PLS3_DIMS]
        b = float(fold["pls3_lr_b"])
        proba_space = "pls3"
        method = f"{method}_pls3"
    else:
        w = np.asarray(fold["baseline_lr_w"], dtype=np.float64).reshape(-1)
        b = float(fold["baseline_lr_b"])
        proba_space = "lr64"
        method = f"{method}_lr64"

    ids = []
    X = []
    meta = []
    for p in pts:
        cid = int(p["id"])
        if cid not in emb:
            continue
        ids.append(cid)
        X.append(emb[cid])
        meta.append(dict(p))
    if len(ids) < MIN_ENDPOINT * 2:
        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "skipped",
            "reason": "too_few_cells",
            "n_cells": len(ids),
        }

    X = np.stack(X, axis=0)
    Xs = (X - mu) / sd
    T = Xs @ R  # (n, PLS_DIMS)
    T3 = T[:, :PLS3_DIMS]

    # Rescore sample with the walk probability model (PLS-3 LR preferred).
    if use_pls3:
        logits = T3 @ w + b
    else:
        logits = Xs @ w + b
    cell_proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    cell_pred = (cell_proba >= 0.5).astype(np.int64)
    for i, p in enumerate(meta):
        gt = p.get("gt")
        p["proba_res"] = float(cell_proba[i])
        p["pred"] = int(cell_pred[i])
        p["correct"] = bool(int(cell_pred[i]) == int(gt)) if gt is not None else None

    res_idx, nres_idx, ep_info = pick_endpoint_indices(meta)
    ep_info["proba_model"] = proba_space
    if not res_idx or not nres_idx:
        return {
            "city_id": city_id,
            "city_name": city_name,
            "week4_region": CITY_TO_WEEK4_REGION.get(city_id),
            "fold_id": region,
            "status": "skipped",
            "reason": "insufficient_endpoint_cells",
            "endpoint_selection": ep_info,
            "n_hi_res": 0,
            "n_hi_nres": 0,
        }

    if use_pls3:
        c_res = T3[res_idx].mean(axis=0)
        c_nres = T3[nres_idx].mean(axis=0)
    else:
        c_res = Xs[res_idx].mean(axis=0)
        c_nres = Xs[nres_idx].mean(axis=0)
    c_res_pls = T[res_idx].mean(axis=0)
    c_nres_pls = T[nres_idx].mean(axis=0)

    ts = np.linspace(0.0, 1.0, N_STEPS)
    frames = []
    prev_nn = None
    for step, t in enumerate(ts):
        # Path score in PLS-3 (or 64-D fallback); NN retrieval in PLS-32.
        x_t = (1.0 - t) * c_res + t * c_nres
        z_t = (1.0 - t) * c_res_pls + t * c_nres_pls
        proba = float(sigmoid(np.dot(x_t, w) + b))

        dists = np.asarray([cosine_dist(z_t, T[i]) for i in range(len(ids))], dtype=np.float64)
        order = np.argsort(dists)
        nn_i = int(order[0])
        if prev_nn is not None and ids[nn_i] == prev_nn and len(order) > 1:
            nn_i = int(order[1])
        p = meta[nn_i]
        cid = ids[nn_i]
        prev_nn = cid
        frames.append(
            {
                "step": step,
                "t": float(t),
                "proba_res": proba,
                "pred_frac_res": proba,
                "pred_frac_nres": float(1.0 - proba),
                "nn_cell_id": cid,
                "nn_dist": float(dists[nn_i]),
                "nn_proba_res": float(p["proba_res"]),
                "gt": p.get("gt"),
                "pred": p.get("pred"),
                "correct": p.get("correct"),
                "purity": p.get("purity"),
                "patch": p.get("patch"),
                "context": p.get("context"),
                "context500": p.get("context500"),
                "lon": p.get("lon"),
                "lat": p.get("lat"),
                "pls_xyz": [float(z_t[0]), float(z_t[1]), float(z_t[2])],
            }
        )

    return {
        "city_id": city_id,
        "city_name": city_name,
        "country": city_id.split("_", 1)[0],
        "week4_region": CITY_TO_WEEK4_REGION.get(city_id),
        "fold_id": region,
        "pred_model": method,
        "proba_space": proba_space,
        "status": "ok",
        "n_cells": len(ids),
        "n_hi_res": len(res_idx),
        "n_hi_nres": len(nres_idx),
        "n_steps": N_STEPS,
        "thresholds": {
            "hi_res": HI_RES,
            "hi_nres": HI_NRES,
            "mid_res": MID_RES,
            "mid_nres": MID_NRES,
        },
        "endpoint_selection": ep_info,
        "endpoint_proba": {
            "start_model": frames[0]["proba_res"],
            "end_model": frames[-1]["proba_res"],
        },
        "frames": frames,
        "checkpoint": str(ckpt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=os.environ.get("RUN_DIR", ""))
    ap.add_argument("--points", type=str, default=str(POINTS))
    ap.add_argument("--intl-fold-dir", type=str, default=str(INTL_FOLD_DIR))
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Write walks JSON here (default: viz/.../data/boundary_walks.json)",
    )
    ap.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city_id filter (default: all cities in points)",
    )
    ap.add_argument("--workers", type=int, default=0, help="Parallel city walks")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = OUT / "runs" / f"boundary_{stamp}"
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    n_cpus = os.cpu_count() or 8
    workers = int(args.workers or min(32, n_cpus))
    intl_fold_dir = Path(args.intl_fold_dir)
    update_progress(
        run_dir,
        status="running",
        stage="load_points",
        started_at=started,
        completed=0,
        failed=0,
        total=20,
        percent_complete=0.0,
        resource_allocation={
            "cpus": n_cpus,
            "gpus": 0,
            "gpu_used": False,
            "workers": workers,
        },
        output=str(DATA),
        job_run_dir=str(run_dir),
    )
    print(f"run_dir={run_dir} workers={workers}", flush=True)

    points_path = Path(args.points)
    if not points_path.exists():
        update_progress(run_dir, status="failed", stage="load_points")
        raise SystemExit(f"missing {points_path}")

    payload = json.loads(points_path.read_text())
    pts = payload["points"]
    cell_ids = [int(p["id"]) for p in pts]
    print(f"loaded {len(pts)} points from {points_path}", flush=True)

    update_progress(run_dir, stage="load_embeddings", total=len(cell_ids))
    print("fetching 64-D embeddings from Postgres …", flush=True)
    emb = load_embeddings(cell_ids)
    miss = [i for i in cell_ids if i not in emb]
    print(f"embeddings ok={len(emb)} missing={len(miss)}", flush=True)
    if miss:
        (run_dir / "failures" / "missing_embeddings.txt").write_text(
            "\n".join(map(str, miss)) + "\n"
        )

    by_city: dict[str, list[dict]] = {}
    names: dict[str, str] = {}
    for p in pts:
        by_city.setdefault(p["city_id"], []).append(p)
        names[p["city_id"]] = p["city_name"]

    cities = sorted(by_city.keys())
    if args.cities:
        want = set(args.cities)
        missing = sorted(want - set(cities))
        if missing:
            raise SystemExit(f"--cities not in points: {missing}")
        cities = sorted(want)
    update_progress(run_dir, stage="walks", total=len(cities), completed=0, failed=0)
    walks_by_id: dict[str, dict] = {}
    ok = skip = fail = 0
    done = 0

    def _one(city_id: str) -> dict:
        return build_city_walk(
            city_id,
            names[city_id],
            by_city[city_id],
            emb,
            intl_fold_dir=intl_fold_dir,
        )

    print(f"building walks for {len(cities)} cities workers={workers} …", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, cid): cid for cid in cities}
        for fut in as_completed(futs):
            city_id = futs[fut]
            done += 1
            try:
                walk = fut.result()
            except Exception as e:
                fail += 1
                (run_dir / "failures" / f"{city_id}.txt").write_text(str(e) + "\n")
                walk = {
                    "city_id": city_id,
                    "city_name": names[city_id],
                    "status": "failed",
                    "reason": str(e),
                }
                print(f"  [{done}/{len(cities)}] FAILED {city_id}: {e}", flush=True)
            else:
                if walk and walk.get("status") == "ok":
                    ok += 1
                    ep = walk["endpoint_proba"]
                    print(
                        f"  [{done}/{len(cities)}] ok {city_id} "
                        f"n_hi_res={walk['n_hi_res']} n_hi_nres={walk['n_hi_nres']} "
                        f"P(res) {ep['start_model']:.3f} → {ep['end_model']:.3f}",
                        flush=True,
                    )
                else:
                    skip += 1
                    print(
                        f"  [{done}/{len(cities)}] skip {city_id}: {walk.get('reason')}",
                        flush=True,
                    )
            walks_by_id[city_id] = walk
            atomic_json(run_dir / "checkpoints" / f"{city_id}.json", walk)
            (run_dir / "checkpoints" / f"{city_id}.DONE").write_text("ok\n")
            update_progress(
                run_dir,
                stage="walks",
                completed=done,
                success=ok,
                skipped=skip,
                failed=fail,
                total=len(cities),
                percent_complete=round(100.0 * done / len(cities), 2),
                last_city=city_id,
            )

    walks = [walks_by_id[c] for c in cities]
    update_progress(run_dir, stage="write_outputs")
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": {
            "endpoints": "centroid high-conf correct res/nres (scored with walk proba model)",
            "pls_nn": "LORO PLS-DA dims 1-32 cosine NN",
            "proba": "LORO logistic on PLS dims 1-3 (centroid path); fallback lr64 if pls3 missing",
            "n_steps": N_STEPS,
            "hi_res": HI_RES,
            "hi_nres": HI_NRES,
            "min_endpoint_cells": MIN_ENDPOINT,
            "corpus": "viz sample per city",
            "intl_fold_dir": str(intl_fold_dir),
            "walk_proba_model": WALK_PROBA_MODEL,
        },
        "n_cities": len(walks),
        "n_ok": ok,
        "n_skipped": skip,
        "n_failed": fail,
        "cities": walks,
    }
    out_path = Path(args.out) if args.out else (DATA / "boundary_walks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, out)
    summary = {
        "status": "completed" if fail == 0 else "completed_with_failures",
        "n_cities": len(walks),
        "n_ok": ok,
        "n_skipped": skip,
        "n_failed": fail,
        "runtime_seconds": round(time.time() - t0, 1),
        "boundary_walks": str(out_path),
        "per_city": [
            {
                "city_id": w.get("city_id"),
                "city_name": w.get("city_name"),
                "status": w.get("status"),
                "n_hi_res": w.get("n_hi_res"),
                "n_hi_nres": w.get("n_hi_nres"),
                "endpoint_mode": (w.get("endpoint_selection") or {}).get("endpoint_mode"),
                "proba_space": w.get("proba_space"),
                "p_start": (w.get("endpoint_proba") or {}).get("start_model"),
                "p_end": (w.get("endpoint_proba") or {}).get("end_model"),
                "reason": w.get("reason"),
            }
            for w in walks
        ],
    }
    if not args.out:
        atomic_json(DATA / "boundary_walks_summary.json", summary)
    else:
        atomic_json(out_path.with_name(out_path.stem + "_summary.json"), summary)
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "artifacts" / "boundary_walks.json", out)
    update_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        completed=len(cities),
        failed=fail,
        percent_complete=100.0,
        runtime_seconds=summary["runtime_seconds"],
    )
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
