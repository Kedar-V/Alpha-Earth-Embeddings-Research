#!/usr/bin/env python3
"""Three-country falsification: S_c ≈ S_G ⊕ S_L,c

Stages:
  01_spectrum_nulls
  02_sg_sl_prediction
  03_local_swap
  04_rank_overlap
  05_figures_report
  06_notebook
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_ext import (  # noqa: E402
    average_projection_basis,
    average_projection_spectrum,
    balance_cap_by_city,
    binary_metrics,
    concat_orthonormal,
    fit_lr,
    global_overlap,
    loco_bases,
    orthonormalize,
    pca_basis,
    pls_rotations,
    project_eval,
    random_orthogonal,
    residual_basis,
    stratified_split_idx,
)

COUNTRIES = ("USA", "JPN", "IND")
PAIRS = [
    ("USA", "JPN"),
    ("USA", "IND"),
    ("JPN", "USA"),
    ("JPN", "IND"),
    ("IND", "USA"),
    ("IND", "JPN"),
]
KS = list(range(1, 21))
R_GRID_FIXED = (1, 2, 3, 5)

_CACHE: dict = {}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(run_dir: Path, msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kw) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    prev["updated_at"] = now()
    prev["last_progress_at"] = prev["updated_at"]
    atomic_json(path, prev)


def stage_done(run_dir: Path, name: str) -> bool:
    return (run_dir / "checkpoints" / f"{name}.DONE").exists()


def mark_done(run_dir: Path, name: str, payload: dict | None = None) -> None:
    p = run_dir / "checkpoints" / f"{name}.DONE"
    p.write_text(json.dumps(payload or {"status": "ok", "completed_at": now()}, indent=2) + "\n")


def load_extract(extract_dir: Path):
    X = np.load(extract_dir / "X.npy", mmap_mode="r")
    y = np.load(extract_dir / "y.npy").astype(np.int64)
    regions = np.load(extract_dir / "regions.npy", allow_pickle=True).astype(str)
    countries = np.load(extract_dir / "countries.npy", allow_pickle=True).astype(str)
    return X, y, regions, countries


def init_cache(extract_dir: Path):
    global _CACHE
    X, y, regions, countries = load_extract(extract_dir)
    _CACHE = {"X": X, "y": y, "regions": regions, "countries": countries}


def country_slice(country: str, max_per_class: int, seed: int):
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    mask = countries == country
    Xm = np.asarray(X[mask], dtype=np.float32)
    ym = y[mask]
    rm = regions[mask]
    rng = np.random.default_rng(seed)
    return balance_cap_by_city(Xm, ym, rm, max_per_class, rng)


def load_prior_bases(prior_run: Path, seed: int) -> dict[str, np.ndarray]:
    sub = prior_run / "artifacts" / "subspaces"
    return {c: orthonormalize(np.load(sub / f"U_pls_{c}_seed{seed}.npy")) for c in COUNTRIES}


def fit_country_pls(country: str, max_per: int, seed: int, k_max: int = 20, shuffle: bool = False) -> np.ndarray:
    Xm, ym, _ = country_slice(country, max_per, seed)
    if shuffle:
        ym = np.random.default_rng(seed + 991).permutation(ym)
    rng = np.random.default_rng(seed)
    tr, _ = stratified_split_idx(ym, 0.2, rng)
    sc = StandardScaler().fit(Xm[tr])
    R = pls_rotations(sc.transform(Xm[tr]), ym[tr], n_components=k_max)
    return orthonormalize(R)


def fit_country_pca(country: str, max_per: int, seed: int, k_max: int = 20) -> np.ndarray:
    Xm, ym, _ = country_slice(country, max_per, seed)
    rng = np.random.default_rng(seed)
    tr, _ = stratified_split_idx(ym, 0.2, rng)
    sc = StandardScaler().fit(Xm[tr])
    return orthonormalize(pca_basis(sc.transform(Xm[tr]), n_components=k_max))


# ---------------------------------------------------------------------------
# Stage 01 — spectrum + nulls
# ---------------------------------------------------------------------------
def _spectrum_real_job(payload: dict) -> list[dict]:
    prior = Path(payload["prior_run"])
    seed = payload["seed"]
    bases = load_prior_bases(prior, seed)
    rows = []
    for k in KS:
        evals, _ = average_projection_spectrum([bases[c] for c in COUNTRIES], k)
        gaps = [float(evals[i] - evals[i + 1]) for i in range(min(5, len(evals) - 1))]
        rows.append(
            {
                "control": "real_pls",
                "seed": seed,
                "k": k,
                "global_overlap": global_overlap(evals, k),
                "lambda_1": float(evals[0]),
                "lambda_2": float(evals[1]) if len(evals) > 1 else float("nan"),
                "lambda_3": float(evals[2]) if len(evals) > 2 else float("nan"),
                "gap_1_2": gaps[0] if gaps else float("nan"),
                "gap_2_3": gaps[1] if len(gaps) > 1 else float("nan"),
                "argmax_gap_le5": int(np.argmax(gaps) + 1) if gaps else 1,
                "evals_json": json.dumps([float(x) for x in evals[:20]]),
            }
        )
    return rows


def _spectrum_null_job(payload: dict) -> list[dict]:
    """Fit null bases once (k_max=20), emit spectrum rows for requested ks."""
    control = payload["control"]
    seed = payload["seed"]
    max_per = payload["max_per_class"]
    ks = payload["ks"]
    draw = payload.get("draw", 0)
    rng = np.random.default_rng(seed * 10007 + draw * 17 + 3)

    if control == "random":
        # independent random bases per k (true null for each Grassmann)
        rows = []
        for k in ks:
            bases_k = [random_orthogonal(64, k, np.random.default_rng(seed * 10007 + draw * 17 + k + i)) for i, _ in enumerate(COUNTRIES)]
            evals, _ = average_projection_spectrum(bases_k, k)
            rows.append(
                {
                    "control": control,
                    "seed": seed,
                    "k": k,
                    "draw": draw,
                    "global_overlap": global_overlap(evals, k),
                    "lambda_1": float(evals[0]),
                    "lambda_2": float(evals[1]) if len(evals) > 1 else float("nan"),
                    "lambda_3": float(evals[2]) if len(evals) > 2 else float("nan"),
                    "gap_1_2": float(evals[0] - evals[1]) if len(evals) > 1 else float("nan"),
                    "evals_json": json.dumps([float(x) for x in evals[:20]]),
                }
            )
        return rows
    elif control == "label_shuffle":
        bases = [fit_country_pls(c, max_per, seed + 500 + draw, k_max=20, shuffle=True) for c in COUNTRIES]
    elif control == "pca":
        bases = [fit_country_pca(c, max_per, seed, k_max=20) for c in COUNTRIES]
    elif control == "bootstrap":
        bases = []
        for ci, c in enumerate(COUNTRIES):
            Xm, ym, rm = country_slice(c, max_per, seed)
            cities = np.unique(rm)
            boot_cities = rng.choice(cities, size=len(cities), replace=True)
            parts_x, parts_y = [], []
            for bc in boot_cities:
                m = rm == bc
                parts_x.append(Xm[m])
                parts_y.append(ym[m])
            Xb = np.concatenate(parts_x)
            yb = np.concatenate(parts_y)
            tr, _ = stratified_split_idx(yb, 0.2, np.random.default_rng(seed + draw + ci))
            sc = StandardScaler().fit(Xb[tr])
            R = pls_rotations(sc.transform(Xb[tr]), yb[tr], n_components=20)
            bases.append(orthonormalize(R))
    else:
        raise ValueError(control)

    rows = []
    for k in ks:
        evals, _ = average_projection_spectrum([b[:, :k] for b in bases], k)
        rows.append(
            {
                "control": control,
                "seed": seed,
                "k": k,
                "draw": draw,
                "global_overlap": global_overlap(evals, k),
                "lambda_1": float(evals[0]),
                "lambda_2": float(evals[1]) if len(evals) > 1 else float("nan"),
                "lambda_3": float(evals[2]) if len(evals) > 2 else float("nan"),
                "gap_1_2": float(evals[0] - evals[1]) if len(evals) > 1 else float("nan"),
                "evals_json": json.dumps([float(x) for x in evals[:20]]),
            }
        )
    return rows


def choose_r_star(real_df: pd.DataFrame, ctrl_df: pd.DataFrame) -> dict:
    """Pick r* from mean spectrum at k=20 (or max k): argmax gap among first 5, vs nulls."""
    # use k=max available rows averaged over seeds — spectrum of P_bar for k=20
    k_ref = int(real_df.k.max())
    sub = real_df[real_df.k == k_ref]
    # average evals across seeds
    evals_list = [json.loads(s) for s in sub.evals_json]
    mean_evals = np.mean(np.array(evals_list), axis=0)
    gaps = [float(mean_evals[i] - mean_evals[i + 1]) for i in range(min(5, len(mean_evals) - 1))]
    r_cand = int(np.argmax(gaps) + 1)

    # null 95th percentile for lambda_i under shuffle+random at k=k_ref
    null = ctrl_df[(ctrl_df.k == k_ref) & (ctrl_df.control.isin(["label_shuffle", "random"]))]
    passes = []
    for i in range(r_cand):
        col = f"lambda_{i+1}" if i < 3 else None
        if col and col in null.columns and null[col].notna().any():
            thr = float(np.nanpercentile(null[col], 95))
            passes.append(bool(mean_evals[i] > thr))
        else:
            # use evals_json for nulls
            null_vals = []
            for s in null.evals_json:
                e = json.loads(s)
                if len(e) > i:
                    null_vals.append(e[i])
            thr = float(np.percentile(null_vals, 95)) if null_vals else 0.0
            passes.append(bool(mean_evals[i] > thr))

    # if leading components don't beat null, fall back to smallest r that does, else 2
    if all(passes):
        r_star = r_cand
    else:
        r_star = 1
        for i, ok in enumerate(passes, 1):
            if ok:
                r_star = i
            else:
                break
        if r_star < 1:
            r_star = 2

    return {
        "r_star": int(r_star),
        "r_cand_gap": int(r_cand),
        "k_ref": k_ref,
        "mean_evals_top10": [float(x) for x in mean_evals[:10]],
        "gaps_top5": gaps,
        "lambda_beats_null_95": passes,
        "rule": "argmax gap among first 5 eigens of mean P_bar(k=20); require λ_i > 95th pct shuffle+random",
    }


def stage_spectrum(run_dir: Path, args) -> Path:
    name = "01_spectrum_nulls"
    out = run_dir / "artifacts" / "avgproj_spectrum.csv"
    out_c = run_dir / "artifacts" / "avgproj_spectrum_controls.csv"
    out_r = run_dir / "artifacts" / "r_star.json"
    if stage_done(run_dir, name) and out.exists() and out_r.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running", completed=0)
    t0 = time.time()

    # Real spectra
    real_jobs = [{"prior_run": str(args.prior_run), "seed": s} for s in args.seeds]
    real_rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(real_jobs))) as pool:
        for part in pool.map(_spectrum_real_job, real_jobs):
            real_rows.extend(part)
    real_df = pd.DataFrame(real_rows)
    real_df.to_csv(out, index=False)
    log(run_dir, f"{name} real spectrum rows={len(real_df)}")

    # Nulls: one job fits bases once, emits all requested k spectra
    null_jobs = []
    k_null = [1, 2, 3, 5, 9, 15, 20]
    k_boot = [2, 5, 9, 20]
    for seed in args.seeds:
        for control in ("random", "pca", "label_shuffle"):
            null_jobs.append(
                {
                    "control": control,
                    "seed": seed,
                    "ks": k_null,
                    "max_per_class": args.max_per_class,
                    "draw": 0,
                }
            )
        for b in range(args.n_bootstrap):
            null_jobs.append(
                {
                    "control": "bootstrap",
                    "seed": seed,
                    "ks": k_boot,
                    "max_per_class": args.max_per_class,
                    "draw": b,
                }
            )

    ctrl_rows = []
    n_workers = min(args.workers, len(null_jobs))
    log(run_dir, f"{name} null jobs={len(null_jobs)} workers={n_workers}")
    write_progress(run_dir, stage=name, total=len(null_jobs), completed=0)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_spectrum_null_job, j): j for j in null_jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                ctrl_rows.extend(fut.result())
            except Exception as e:
                (run_dir / "failures" / f"spectrum_{done}.txt").write_text(traceback.format_exc())
                log(run_dir, f"null job failed: {e}")
            done += 1
            if done % 20 == 0 or done == len(null_jobs):
                write_progress(
                    run_dir,
                    stage=name,
                    completed=done,
                    total=len(null_jobs),
                    percent_complete=round(100 * done / len(null_jobs), 2),
                    rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
                )
                log(run_dir, f"{name} nulls {done}/{len(null_jobs)}")
    ctrl_df = pd.DataFrame(ctrl_rows)
    ctrl_df.to_csv(out_c, index=False)

    r_star = choose_r_star(real_df, ctrl_df)
    atomic_json(out_r, r_star)
    log(run_dir, f"{name} r_star={r_star['r_star']}")
    mark_done(run_dir, name, {"r_star": r_star, "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Stage 02 — SG / SL prediction table
# ---------------------------------------------------------------------------
def _sg_sl_job(payload: dict) -> list[dict]:
    prior = Path(payload["prior_run"])
    seed = payload["seed"]
    r = payload["r"]
    max_per = payload["max_per_class"]
    bases = load_prior_bases(prior, seed)
    k_task = max(r, 9)
    rows = []

    # Pre-slice all countries once
    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        data[c] = (Xm, ym, tr, te)

    # Descriptive S_G
    U_g_desc = average_projection_basis([bases[c] for c in COUNTRIES], r)

    # Within-country (descriptive S_G)
    for c in COUNTRIES:
        Xm, ym, tr, te = data[c]
        Xtr, ytr = Xm[tr], ym[tr]
        Xte, yte = Xm[te], ym[te]
        U_c = bases[c][:, :k_task]
        U_loc = residual_basis(U_c, U_g_desc)
        reps = {
            "global": U_g_desc[:, :r],
            "local": U_loc[:, : min(r, U_loc.shape[1])] if U_loc.shape[1] else U_g_desc[:, :0],
            "global_plus_local": concat_orthonormal(U_g_desc[:, :r], U_loc[:, : min(r, max(U_loc.shape[1], 0))]),
            "full_task": bases[c][:, :r],
            "lr64": None,
        }
        for name, U in reps.items():
            if name == "lr64":
                sc = StandardScaler().fit(Xtr)
                pred, proba, _ = fit_lr(sc.transform(Xtr), ytr, sc.transform(Xte), seed=seed)
                m = binary_metrics(yte, pred, proba)
            else:
                m = project_eval(U, Xtr, ytr, Xte, yte, seed)
            rows.append(
                {
                    "mode": "descriptive",
                    "protocol": "within_country",
                    "seed": seed,
                    "r": r,
                    "train_country": c,
                    "eval_country": c,
                    "representation": name,
                    **{f"m_{k}": v for k, v in m.items()},
                }
            )

    # Cross-country LOCO
    for src, tgt in PAIRS:
        Xm_s, ym_s, tr_s, _ = data[src]
        Xm_t, ym_t, _, _ = data[tgt]
        # eval on full capped target (same as prior transfer protocol)
        Xtr, ytr = Xm_s[tr_s], ym_s[tr_s]
        Xte, yte = Xm_t, ym_t
        U_g_loco = average_projection_basis(loco_bases(bases, tgt), r)
        U_src = bases[src][:, :k_task]
        U_loc = residual_basis(U_src, U_g_loco)
        reps = {
            "global": U_g_loco[:, :r],
            "local": U_loc[:, : min(r, U_loc.shape[1])] if U_loc.shape[1] else U_g_loco[:, :0],
            "global_plus_local": concat_orthonormal(U_g_loco[:, :r], U_loc[:, : min(r, max(U_loc.shape[1], 0))]),
            "full_task": bases[src][:, :r],
            "lr64": None,
        }
        for name, U in reps.items():
            if name == "lr64":
                sc = StandardScaler().fit(Xtr)
                pred, proba, _ = fit_lr(sc.transform(Xtr), ytr, sc.transform(Xte), seed=seed)
                m = binary_metrics(yte, pred, proba)
            else:
                m = project_eval(U, Xtr, ytr, Xte, yte, seed)
            rows.append(
                {
                    "mode": "loco",
                    "protocol": "cross_country",
                    "seed": seed,
                    "r": r,
                    "train_country": src,
                    "eval_country": tgt,
                    "representation": name,
                    **{f"m_{k}": v for k, v in m.items()},
                }
            )
    return rows


def stage_sg_sl(run_dir: Path, args) -> Path:
    name = "02_sg_sl_prediction"
    out = run_dir / "artifacts" / "sg_sl_prediction_table.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    r_star = json.loads((run_dir / "artifacts" / "r_star.json").read_text())["r_star"]
    r_values = sorted(set(R_GRID_FIXED) | {r_star})
    jobs = []
    for seed in args.seeds:
        for r in r_values:
            jobs.append(
                {
                    "prior_run": str(args.prior_run),
                    "seed": seed,
                    "r": r,
                    "max_per_class": args.max_per_class,
                }
            )
    rows = []
    t0 = time.time()
    n_workers = min(args.workers, len(jobs))
    log(run_dir, f"{name}: {len(jobs)} jobs on {n_workers} workers; r={r_values}")
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_sg_sl_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            write_progress(
                run_dir,
                stage=name,
                completed=done,
                total=len(jobs),
                percent_complete=round(100 * done / len(jobs), 2),
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
                current_item=f"seed={futs[fut]['seed']} r={futs[fut]['r']}",
            )
            if done % 5 == 0 or done == len(jobs):
                pd.DataFrame(rows).to_csv(out.with_suffix(".partial.csv"), index=False)
                log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Stage 03 — local swap
# ---------------------------------------------------------------------------
def _swap_job(payload: dict) -> list[dict]:
    prior = Path(payload["prior_run"])
    seed = payload["seed"]
    r = payload["r"]
    target = payload["target"]
    donor = payload["donor"]
    max_per = payload["max_per_class"]
    bases = load_prior_bases(prior, seed)
    k_task = max(r, 9)

    Xm, ym, _ = country_slice(target, max_per, seed)
    rng = np.random.default_rng(seed)
    tr, te = stratified_split_idx(ym, 0.2, rng)
    Xtr, ytr = Xm[tr], ym[tr]
    Xte, yte = Xm[te], ym[te]

    rows = []
    for mode, U_g in (
        ("descriptive", average_projection_basis([bases[c] for c in COUNTRIES], r)),
        ("loco", average_projection_basis(loco_bases(bases, target), r)),
    ):
        U_loc = residual_basis(bases[donor][:, :k_task], U_g)
        U = concat_orthonormal(U_g[:, :r], U_loc[:, : min(r, max(U_loc.shape[1], 0))])
        m = project_eval(U, Xtr, ytr, Xte, yte, seed)
        rows.append(
            {
                "mode": mode,
                "seed": seed,
                "r": r,
                "target": target,
                "donor": donor,
                "matched": int(target == donor),
                **{f"m_{k}": v for k, v in m.items()},
            }
        )
    return rows


def stage_swap(run_dir: Path, args) -> Path:
    name = "03_local_swap"
    out = run_dir / "artifacts" / "local_swap_matrix.csv"
    out_s = run_dir / "artifacts" / "local_swap_summary.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    r_star = json.loads((run_dir / "artifacts" / "r_star.json").read_text())["r_star"]
    r_values = sorted(set([r_star, 2, 3]))
    jobs = []
    for seed in args.seeds:
        for r in r_values:
            for tgt in COUNTRIES:
                for donor in COUNTRIES:
                    jobs.append(
                        {
                            "prior_run": str(args.prior_run),
                            "seed": seed,
                            "r": r,
                            "target": tgt,
                            "donor": donor,
                            "max_per_class": args.max_per_class,
                        }
                    )
    rows = []
    t0 = time.time()
    n_workers = min(args.workers, len(jobs))
    log(run_dir, f"{name}: {len(jobs)} jobs on {n_workers} workers")
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_swap_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            write_progress(
                run_dir,
                stage=name,
                completed=done,
                total=len(jobs),
                percent_complete=round(100 * done / len(jobs), 2),
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
            )
            if done % 10 == 0 or done == len(jobs):
                log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # Summary: diagonal advantage
    # NOTE: use bracket access for column "mode" (DataFrame.mode is a method)
    summ = []
    for mode in ("descriptive", "loco"):
        for r in r_values:
            sub = df[(df["mode"] == mode) & (df["r"] == r)]
            for tgt in COUNTRIES:
                matched = sub[(sub["target"] == tgt) & (sub["donor"] == tgt)]["m_macro_f1"]
                mismatched = sub[(sub["target"] == tgt) & (sub["donor"] != tgt)]["m_macro_f1"]
                adv = []
                for seed in args.seeds:
                    m = float(
                        sub[(sub["target"] == tgt) & (sub["donor"] == tgt) & (sub["seed"] == seed)][
                            "m_macro_f1"
                        ].mean()
                    )
                    mm = float(
                        sub[(sub["target"] == tgt) & (sub["donor"] != tgt) & (sub["seed"] == seed)][
                            "m_macro_f1"
                        ].mean()
                    )
                    adv.append(m - mm)
                adv = np.asarray(adv, dtype=float)
                rng = np.random.default_rng(0)
                boots = [
                    float(np.mean(rng.choice(adv, size=len(adv), replace=True)))
                    for _ in range(1000)
                    if np.isfinite(adv).all()
                ]
                summ.append(
                    {
                        "mode": mode,
                        "r": r,
                        "target": tgt,
                        "matched_f1_mean": float(matched.mean()) if len(matched) else float("nan"),
                        "mismatched_f1_mean": float(mismatched.mean()) if len(mismatched) else float("nan"),
                        "diagonal_advantage_mean": float(np.nanmean(adv)),
                        "diagonal_advantage_std": float(np.nanstd(adv)),
                        "diagonal_advantage_ci95_lo": float(np.percentile(boots, 2.5)) if boots else float("nan"),
                        "diagonal_advantage_ci95_hi": float(np.percentile(boots, 97.5)) if boots else float("nan"),
                        "n_seeds": len(args.seeds),
                    }
                )
    pd.DataFrame(summ).to_csv(out_s, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Stage 04 — rank vs GlobalOverlap
# ---------------------------------------------------------------------------
def _rank_job(payload: dict) -> list[dict]:
    prior = Path(payload["prior_run"])
    seed = payload["seed"]
    source = payload["source"]
    max_per = payload["max_per_class"]
    bases = load_prior_bases(prior, seed)

    # spectrum overlaps for all k
    overlaps = {}
    for k in KS:
        evals, _ = average_projection_spectrum([bases[c] for c in COUNTRIES], k)
        overlaps[k] = global_overlap(evals, k)

    Xm, ym, _ = country_slice(source, max_per, seed)
    rng = np.random.default_rng(seed)
    tr, te = stratified_split_idx(ym, 0.2, rng)
    Xtr, ytr = Xm[tr], ym[tr]
    Xte_id, yte_id = Xm[te], ym[te]

    # OOD targets
    targets = [t for s, t in PAIRS if s == source]
    ood_data = {}
    for tgt in targets:
        Xt, yt, _ = country_slice(tgt, max_per, seed + 17)
        ood_data[tgt] = (Xt, yt)

    rows = []
    prev_id = prev_ood = prev_ov = None
    for k in KS:
        U = bases[source][:, :k]
        m_id = project_eval(U, Xtr, ytr, Xte_id, yte_id, seed)
        ood_f1s = []
        for tgt, (Xt, yt) in ood_data.items():
            mo = project_eval(U, Xtr, ytr, Xt, yt, seed)
            ood_f1s.append(mo["macro_f1"])
        f1_ood = float(np.mean(ood_f1s))
        ov = overlaps[k]
        rows.append(
            {
                "seed": seed,
                "source": source,
                "k": k,
                "f1_within": m_id["macro_f1"],
                "f1_ood": f1_ood,
                "global_overlap": ov,
                "delta_f1_within": (m_id["macro_f1"] - prev_id) if prev_id is not None else float("nan"),
                "delta_f1_ood": (f1_ood - prev_ood) if prev_ood is not None else float("nan"),
                "delta_global_overlap": (ov - prev_ov) if prev_ov is not None else float("nan"),
            }
        )
        prev_id, prev_ood, prev_ov = m_id["macro_f1"], f1_ood, ov
    return rows


def stage_rank(run_dir: Path, args) -> Path:
    name = "04_rank_overlap"
    out = run_dir / "artifacts" / "rank_overlap.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    jobs = []
    for seed in args.seeds:
        for source in COUNTRIES:
            jobs.append(
                {
                    "prior_run": str(args.prior_run),
                    "seed": seed,
                    "source": source,
                    "max_per_class": args.max_per_class,
                }
            )
    rows = []
    t0 = time.time()
    n_workers = min(args.workers, len(jobs))
    log(run_dir, f"{name}: {len(jobs)} jobs on {n_workers} workers")
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_rank_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            write_progress(
                run_dir,
                stage=name,
                completed=done,
                total=len(jobs),
                percent_complete=round(100 * done / len(jobs), 2),
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
            )
            log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Stages 05–06
# ---------------------------------------------------------------------------
def stage_figures_report(run_dir: Path, args) -> Path:
    name = "05_figures_report"
    if stage_done(run_dir, name) and (run_dir / "REPORT.md").exists():
        log(run_dir, f"skip {name}")
        return run_dir / "REPORT.md"
    write_progress(run_dir, stage=name, status="running")
    # Import by file path — avoid colliding with global_local_subspace/make_figures_and_report.py
    import importlib.util

    mod_path = Path(__file__).resolve().parent / "make_figures_and_report.py"
    spec = importlib.util.spec_from_file_location("tcf_make_figures_and_report", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    report = mod.build_figures_and_report(run_dir)
    mark_done(run_dir, name, {"report": str(report)})
    return report


def stage_notebook(run_dir: Path, args) -> Path:
    name = "06_notebook"
    nb_path = ROOT / "week5" / "notebooks" / "04_three_country_falsification.ipynb"
    if stage_done(run_dir, name) and nb_path.exists():
        log(run_dir, f"skip {name}")
        return nb_path
    write_progress(run_dir, stage=name, status="running")
    import importlib.util

    mod_path = Path(__file__).resolve().parent / "make_figures_and_report.py"
    spec = importlib.util.spec_from_file_location("tcf_make_figures_and_report", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    path = mod.build_notebook(run_dir)
    mark_done(run_dir, name, {"notebook": str(path)})
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--prior-run",
        type=Path,
        default=ROOT / "week5/runs/global_local_subspace_20261005_173516",
    )
    ap.add_argument(
        "--extract-dir",
        type=Path,
        default=ROOT
        / "week4/international_geo_generalization/runs/intl_transfer_20261005_142709/extract",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--only-stage", type=str, default="")
    args = ap.parse_args()

    run_dir = args.run_dir
    for d in ("checkpoints", "artifacts", "figures", "failures", "artifacts/subspaces"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    cfg = {
        "prior_run": str(args.prior_run),
        "extract_dir": str(args.extract_dir),
        "workers": args.workers,
        "max_per_class": args.max_per_class,
        "seeds": args.seeds,
        "n_bootstrap": args.n_bootstrap,
        "hypothesis": "S_c ≈ S_G ⊕ S_L,c",
        "created_at": now(),
    }
    atomic_json(run_dir / "config.json", cfg)
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=now(),
        output_location=str(run_dir),
        resource_allocation={"workers": args.workers, "cpus": os.cpu_count(), "gpus": 0},
    )

    try:
        log(run_dir, f"loading extract {args.extract_dir}")
        init_cache(args.extract_dir)
        stages = [
            ("01_spectrum_nulls", stage_spectrum),
            ("02_sg_sl_prediction", stage_sg_sl),
            ("03_local_swap", stage_swap),
            ("04_rank_overlap", stage_rank),
            ("05_figures_report", stage_figures_report),
            ("06_notebook", stage_notebook),
        ]
        if args.only_stage:
            stages = [s for s in stages if s[0] == args.only_stage]
            if not stages:
                raise SystemExit(f"unknown stage {args.only_stage}")

        t0 = time.time()
        for i, (sname, fn) in enumerate(stages, 1):
            log(run_dir, f"=== stage {i}/{len(stages)}: {sname} ===")
            write_progress(run_dir, stage=sname, status="running", stage_index=i, n_stages=len(stages))
            try:
                fn(run_dir, args)
            except Exception:
                tb = traceback.format_exc()
                (run_dir / "failures" / f"{sname}.txt").write_text(tb)
                log(run_dir, f"FAILED {sname}\n{tb}")
                write_progress(run_dir, status="failed", stage=sname)
                (run_dir / "FAILED").write_text(tb)
                return 1

        summary = {
            "status": "completed",
            "runtime_seconds": round(time.time() - t0, 1),
            "run_dir": str(run_dir),
            "config": cfg,
            "r_star": json.loads((run_dir / "artifacts" / "r_star.json").read_text()),
            "report": str(run_dir / "REPORT.md"),
            "notebook": str(ROOT / "week5/notebooks/04_three_country_falsification.ipynb"),
            "figures": sorted(p.name for p in (run_dir / "figures").glob("*.png")),
        }
        atomic_json(run_dir / "summary.json", summary)
        (run_dir / "DONE").write_text("ok\n")
        write_progress(run_dir, status="completed", stage="done", percent_complete=100.0)
        log(run_dir, "DONE")
        return 0
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "FAILED").write_text(tb)
        write_progress(run_dir, status="failed")
        log(run_dir, "FAILED\n" + tb)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
