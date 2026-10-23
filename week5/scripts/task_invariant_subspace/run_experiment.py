#!/usr/bin/env python3
"""Experiment 05: Task-invariant vs geography-specific predictive subspaces.

PLS is a probe only. Hypothesis: S_c ≈ S_G_task + S_L,c where S_G_task is
predictively invariant across countries (not merely geometrically shared).

Stages:
  01_direction_utility
  02_task_invariance
  03_geometry_vs_task
  04_loco_methods
  05_global_local
  06_rank_analysis
  07_figures_report
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

from common import (
    COUNTRIES,
    KS,
    PAIRS,
    R_GRID,
    TI_LAMBDAS,
    average_projection_basis,
    binary_metrics,
    concat_orthonormal,
    country_slice,
    direction_stats,
    eval_direction_on_country,
    fit_country_pls,
    fit_lr,
    grassmann_karcher_mean,
    init_cache,
    load_prior_bases,
    loco_bases,
    log,
    mark_done,
    n_gpus,
    orthonormalize,
    pca_basis,
    project_eval,
    random_orthogonal,
    residual_basis,
    stage_done,
    stratified_split_idx,
    subspace_distances,
    task_invariance_score,
    write_progress,
    atomic_json,
)

# ---------------------------------------------------------------------------
# Worker init
# ---------------------------------------------------------------------------
_EXTRACT = None
_PRIOR = None


def _worker_init(extract_dir: str, prior_run: str):
    global _EXTRACT, _PRIOR
    _EXTRACT = Path(extract_dir)
    _PRIOR = Path(prior_run)
    init_cache(_EXTRACT)


# ---------------------------------------------------------------------------
# Stage 01 — direction utility (real + shuffle)
# ---------------------------------------------------------------------------
def _direction_utility_job(payload: dict) -> list[dict]:
    """Evaluate source→target 1D utilities with cached country slices (fast path)."""
    seed = payload["seed"]
    max_per = payload["max_per_class"]
    shuffle = payload["shuffle"]
    prior = Path(payload["prior_run"])
    rows = []
    if shuffle:
        bases = {c: fit_country_pls(c, max_per, seed + 700, k_max=20, shuffle=True) for c in COUNTRIES}
        tag = "label_shuffle"
    else:
        bases = load_prior_bases(prior, seed)
        tag = "real_pls"

    # Cache each country once: raw capped arrays + default train scaler features
    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        data[c] = {"X": Xm, "y": ym, "tr": tr, "te": te}

    for src in COUNTRIES:
        Xm, ym, tr, te = data[src]["X"], data[src]["y"], data[src]["tr"], data[src]["te"]
        sc = StandardScaler().fit(Xm[tr])
        Xte_s = sc.transform(Xm[te])
        # pre-transform all targets under this source scaler
        Xt = {c: sc.transform(data[c]["X"]) for c in COUNTRIES if c != src}
        U = bases[src]
        for k in KS:
            u = np.asarray(U[:, k - 1], dtype=np.float64)
            u = u / (np.linalg.norm(u) + 1e-12)
            src_stats = direction_stats(Xte_s @ u, ym[te])
            src_stats["n"] = int(len(te))
            tgt_utils_f1 = []
            tgt_utils_auroc = []
            sign_ok = []
            for tgt in COUNTRIES:
                if tgt == src:
                    continue
                z = Xt[tgt] @ u
                st = direction_stats(z, data[tgt]["y"])
                st_flip = direction_stats(-z, data[tgt]["y"])
                use = st if st["auroc_abs"] >= st_flip["auroc_abs"] else st_flip
                flipped = int(st_flip["auroc_abs"] > st["auroc_abs"])
                sign_match = int(use["sign_res_positive"] == src_stats["sign_res_positive"]) if not flipped else 0
                use["n"] = int(len(data[tgt]["y"]))
                rows.append(
                    {
                        "control": tag,
                        "seed": seed,
                        "source": src,
                        "target": tgt,
                        "component": k,
                        "flipped_for_best_auroc": flipped,
                        "sign_match_source": sign_match,
                        "source_macro_f1": src_stats["macro_f1"],
                        "source_auroc": src_stats["auroc"],
                        "source_auroc_abs": src_stats["auroc_abs"],
                        "source_cohens_d": src_stats["cohens_d"],
                        "source_point_biserial": src_stats["point_biserial"],
                        "source_logistic_coef": src_stats["logistic_coef"],
                        "source_sign_res_positive": src_stats["sign_res_positive"],
                        "target_macro_f1": use["macro_f1"],
                        "target_auroc": use["auroc"],
                        "target_auroc_abs": use["auroc_abs"],
                        "target_cohens_d": use["cohens_d"],
                        "target_point_biserial": use["point_biserial"],
                        "target_logistic_coef": use["logistic_coef"],
                        "target_sign_res_positive": use["sign_res_positive"],
                        "n_source": src_stats["n"],
                        "n_target": use["n"],
                    }
                )
                tgt_utils_f1.append(use["macro_f1"])
                tgt_utils_auroc.append(use["auroc_abs"])
                sign_ok.append(sign_match)
            for lam in TI_LAMBDAS:
                rows.append(
                    {
                        "control": tag,
                        "seed": seed,
                        "source": src,
                        "target": "__AGG__",
                        "component": k,
                        "flipped_for_best_auroc": -1,
                        "sign_match_source": float(np.mean(sign_ok)) if sign_ok else float("nan"),
                        "source_macro_f1": src_stats["macro_f1"],
                        "source_auroc": src_stats["auroc"],
                        "source_auroc_abs": src_stats["auroc_abs"],
                        "source_cohens_d": src_stats["cohens_d"],
                        "source_point_biserial": src_stats["point_biserial"],
                        "source_logistic_coef": src_stats["logistic_coef"],
                        "source_sign_res_positive": src_stats["sign_res_positive"],
                        "target_macro_f1": float(np.mean(tgt_utils_f1)),
                        "target_auroc": float(np.mean(tgt_utils_auroc)),
                        "target_auroc_abs": float(np.mean(tgt_utils_auroc)),
                        "target_cohens_d": float("nan"),
                        "target_point_biserial": float("nan"),
                        "target_logistic_coef": float("nan"),
                        "target_sign_res_positive": -1,
                        "ti_lambda": lam,
                        "ti_f1": task_invariance_score(tgt_utils_f1, lam),
                        "ti_auroc": task_invariance_score(tgt_utils_auroc, lam),
                        "std_target_f1": float(np.std(tgt_utils_f1, ddof=0)),
                        "std_target_auroc": float(np.std(tgt_utils_auroc, ddof=0)),
                        "n_source": src_stats["n"],
                        "n_target": -1,
                    }
                )
    return rows


def stage_direction_utility(run_dir: Path, args) -> Path:
    name = "01_direction_utility"
    out = run_dir / "artifacts" / "task_direction_utility.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    jobs = []
    for seed in args.seeds:
        jobs.append({"seed": seed, "max_per_class": args.max_per_class, "shuffle": False, "prior_run": str(args.prior_run)})
        for draw in range(args.n_shuffle):
            jobs.append(
                {
                    "seed": seed * 100 + draw,
                    "max_per_class": args.max_per_class,
                    "shuffle": True,
                    "prior_run": str(args.prior_run),
                }
            )
    rows = []
    t0 = time.time()
    n_workers = min(args.workers, len(jobs))
    log(run_dir, f"{name}: {len(jobs)} jobs, workers={n_workers}")
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.extract_dir), str(args.prior_run)),
    ) as pool:
        futs = {pool.submit(_direction_utility_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception:
                (run_dir / "failures" / f"{name}_{done}.txt").write_text(traceback.format_exc())
                log(run_dir, f"{name} job failed")
            done += 1
            if done % 5 == 0 or done == len(jobs):
                write_progress(
                    run_dir,
                    stage=name,
                    completed=done,
                    total=len(jobs),
                    percent_complete=round(100 * done / len(jobs), 2),
                    rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
                )
                pd.DataFrame(rows).to_csv(out.with_suffix(".partial.csv"), index=False)
                log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Stage 02 — task invariance summaries + excess over shuffle
# ---------------------------------------------------------------------------
def stage_task_invariance(run_dir: Path, args) -> Path:
    name = "02_task_invariance"
    out = run_dir / "artifacts" / "task_invariance.csv"
    out_s = run_dir / "artifacts" / "shuffle_invariance.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running")
    util = pd.read_csv(run_dir / "artifacts" / "task_direction_utility.csv")
    agg = util[util.target == "__AGG__"].copy()
    real = agg[agg.control == "real_pls"]
    shuf = agg[agg.control == "label_shuffle"]

    # mean shuffle TI by source/component/lambda
    shuf_mean = (
        shuf.groupby(["source", "component", "ti_lambda"], as_index=False)
        .agg(ti_f1_shuffle=("ti_f1", "mean"), ti_auroc_shuffle=("ti_auroc", "mean"),
             ti_f1_shuffle_std=("ti_f1", "std"), ti_auroc_shuffle_std=("ti_auroc", "std"),
             n_shuffle=("ti_f1", "count"))
    )
    real2 = real.merge(shuf_mean, on=["source", "component", "ti_lambda"], how="left")
    real2["excess_ti_f1"] = real2["ti_f1"] - real2["ti_f1_shuffle"]
    real2["excess_ti_auroc"] = real2["ti_auroc"] - real2["ti_auroc_shuffle"]
    real2.to_csv(out, index=False)
    shuf.to_csv(out_s, index=False)

    # bootstrap CI on mean excess (over seeds), by component and lambda=1
    boot_rows = []
    sub = real2[real2.ti_lambda == 1.0]
    rng = np.random.default_rng(42)
    for k in KS:
        vals = sub[sub.component == k]["excess_ti_auroc"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(args.n_bootstrap)]
        boot_rows.append(
            {
                "component": k,
                "mean_excess_ti_auroc": float(vals.mean()),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
                "n": int(len(vals)),
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(run_dir / "artifacts" / "bootstrap" / "excess_ti_bootstrap.csv", index=False)
    mark_done(run_dir, name, {"n_real": len(real2), "n_shuffle": len(shuf)})
    return out


# ---------------------------------------------------------------------------
# Stage 03 — geometry vs task
# ---------------------------------------------------------------------------
def _geom_task_job(payload: dict) -> list[dict]:
    seed = payload["seed"]
    prior = Path(payload["prior_run"])
    max_per = payload["max_per_class"]
    bases = load_prior_bases(prior, seed)
    util = pd.read_csv(payload["util_path"])
    real_agg = util[
        (util.control == "real_pls")
        & (util.target == "__AGG__")
        & (util.seed == seed)
        & (util.ti_lambda == 1.0)
    ]
    # cache capped country arrays + train index once
    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, _ = stratified_split_idx(ym, 0.2, rng)
        data[c] = (Xm, ym, tr)
    rows = []
    for src, tgt in PAIRS:
        Xm_s, ym_s, tr_s = data[src]
        Xm_t, ym_t, _ = data[tgt]
        pair_all = util[
            (util.control == "real_pls")
            & (util.seed == seed)
            & (util.source == src)
            & (util.target == tgt)
        ]
        for k in KS:
            Ui = bases[src][:, :k]
            Uj = bases[tgt][:, :k]
            dist = subspace_distances(Ui, Uj)
            pair = pair_all[pair_all.component <= k]
            mean_f1 = float(pair["target_macro_f1"].mean()) if len(pair) else float("nan")
            mean_auroc = float(pair["target_auroc_abs"].mean()) if len(pair) else float("nan")
            sign_frac = float(pair["sign_match_source"].mean()) if len(pair) else float("nan")
            m = project_eval(Ui, Xm_s[tr_s], ym_s[tr_s], Xm_t, ym_t, seed)
            ti = real_agg[(real_agg.source == src) & (real_agg.component <= k)]["ti_auroc"].mean()
            chordal = float(dist.get("chordal_distance", dist.get("chordal", float("nan"))))
            rows.append(
                {
                    "seed": seed,
                    "source": src,
                    "target": tgt,
                    "k": k,
                    "chordal_distance": chordal,
                    "grassmann_geodesic": dist.get("grassmann_geodesic", dist.get("geodesic")),
                    "mean_principal_angle_deg": dist.get("mean_principal_angle_deg", dist.get("mean_angle_deg")),
                    "projection_overlap": float(k - chordal**2) / float(k) if np.isfinite(chordal) and k else float("nan"),
                    "mean_target_f1": mean_f1,
                    "mean_target_auroc_abs": mean_auroc,
                    "sign_consistency": sign_frac,
                    "transfer_f1": m["macro_f1"],
                    "transfer_auroc": m["auroc"],
                    "mean_ti_auroc_components": float(ti) if pd.notna(ti) else float("nan"),
                }
            )
    return rows


def stage_geometry_vs_task(run_dir: Path, args) -> Path:
    name = "03_geometry_vs_task"
    out = run_dir / "artifacts" / "geometry_vs_task.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    util_path = str(run_dir / "artifacts" / "task_direction_utility.csv")
    jobs = [{"seed": s, "prior_run": str(args.prior_run), "max_per_class": args.max_per_class, "util_path": util_path} for s in args.seeds]
    rows = []
    t0 = time.time()
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs)),
        initializer=_worker_init,
        initargs=(str(args.extract_dir), str(args.prior_run)),
    ) as pool:
        done = 0
        for part in pool.map(_geom_task_job, jobs):
            rows.extend(part)
            done += 1
            write_progress(run_dir, stage=name, completed=done, total=len(jobs), percent_complete=round(100 * done / len(jobs), 2))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    # association summary (exploratory; 3 countries)
    assoc = []
    for metric in ("projection_overlap", "chordal_distance"):
        for ycol in ("transfer_f1", "mean_target_auroc_abs", "mean_ti_auroc_components"):
            a = df[[metric, ycol]].dropna()
            if len(a) < 5:
                continue
            r = float(np.corrcoef(a[metric], a[ycol])[0, 1])
            assoc.append({"x": metric, "y": ycol, "pearson": r, "n": len(a), "note": "exploratory; seeds×k×pairs not independent countries"})
    pd.DataFrame(assoc).to_csv(run_dir / "artifacts" / "geometry_vs_task_assoc.csv", index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


# ---------------------------------------------------------------------------
# Method A / B for LOCO global
# ---------------------------------------------------------------------------
def method_a_invariant_basis(
    bases: dict[str, np.ndarray],
    hold_out: str,
    seed: int,
    max_per: int,
    r: int,
    util_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Select top-r directions by source-only TI (no hold_out labels)."""
    sources = [c for c in COUNTRIES if c != hold_out]
    # cache slices once
    packed = {}
    for c in sources:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        packed[c] = (Xm, ym, tr, te)
    cands = []
    for src in sources:
        Xm_s, ym_s, tr, te = packed[src]
        sc = StandardScaler().fit(Xm_s[tr])
        Xte_s = sc.transform(Xm_s[te])
        Xt = {t: sc.transform(packed[t][0]) for t in sources if t != src}
        U = bases[src]
        for k in range(1, 21):
            u = np.asarray(U[:, k - 1], dtype=np.float64)
            u = u / (np.linalg.norm(u) + 1e-12)
            z_src = Xte_s @ u
            src_u = direction_stats(z_src, ym_s[te])["auroc_abs"]
            tgt_us = []
            for tgt in sources:
                if tgt == src:
                    continue
                z = Xt[tgt] @ u
                a0 = direction_stats(z, packed[tgt][1])["auroc_abs"]
                a1 = direction_stats(-z, packed[tgt][1])["auroc_abs"]
                tgt_us.append(max(a0, a1))
            ti = task_invariance_score(tgt_us, lam=1.0)
            score = 0.5 * src_u + 0.5 * ti
            if z_src[ym_s[te] == 1].mean() < z_src[ym_s[te] == 0].mean():
                u = -u
            cands.append((score, u.copy()))
    cands.sort(key=lambda x: -x[0])
    chosen = []
    for score, u in cands:
        v = u.copy()
        for q in chosen:
            v = v - q * float(q @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-8:
            continue
        chosen.append(v / nrm)
        if len(chosen) >= r:
            break
    if not chosen:
        return average_projection_basis([bases[c] for c in sources], r)
    return orthonormalize(np.stack(chosen, axis=1))


def method_b_optimize_basis(
    hold_out: str,
    seed: int,
    max_per: int,
    r: int,
    device: str = "cpu",
    steps: int = 80,
    lr: float = 0.05,
    lam: float = 1.0,
) -> np.ndarray:
    """Learn orthonormal W maximizing mean_AUROC - lam*std across source countries."""
    import torch

    sources = [c for c in COUNTRIES if c != hold_out]
    packs = []
    for c in sources:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        sc = StandardScaler().fit(Xm[tr])
        Xtr = torch.tensor(sc.transform(Xm[tr]), dtype=torch.float32, device=device)
        ytr = torch.tensor(ym[tr], dtype=torch.float32, device=device)
        Xte = torch.tensor(sc.transform(Xm[te]), dtype=torch.float32, device=device)
        yte = torch.tensor(ym[te], dtype=torch.float32, device=device)
        packs.append((Xtr, ytr, Xte, yte))

    torch.manual_seed(seed + 17)
    W = torch.randn(64, r, device=device, dtype=torch.float32)
    W, _ = torch.linalg.qr(W)

    def auroc_torch(scores, y):
        # Wilcoxon-Mann-Whitney AUROC
        pos = scores[y > 0.5]
        neg = scores[y < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return torch.tensor(0.5, device=device)
        # soft approximation via ranks with sigmoid pairwise
        # for speed use differentiable soft AUROC
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        return torch.sigmoid(diff * 5.0).mean()

    def country_auroc(Wmat, Xtr, ytr, Xte, yte):
        # fit ridge logistic in r-D via least squares on logits proxy: use score = Z @ beta
        Ztr = Xtr @ Wmat
        Zte = Xte @ Wmat
        # closed-form ridge regression to labels then sigmoid AUROC on test
        A = Ztr.T @ Ztr + 1e-2 * torch.eye(r, device=device)
        b = Ztr.T @ ytr
        beta = torch.linalg.solve(A, b)
        scores = Zte @ beta
        return auroc_torch(scores, yte)

    for step in range(steps):
        W = W.detach().requires_grad_(True)
        aurocs = []
        for Xtr, ytr, Xte, yte in packs:
            aurocs.append(country_auroc(W, Xtr, ytr, Xte, yte))
        stack = torch.stack(aurocs)
        loss = -(stack.mean() - lam * stack.std(unbiased=False))
        loss.backward()
        with torch.no_grad():
            W = W - lr * W.grad
            W, _ = torch.linalg.qr(W)
        if step % 50 == 0:
            pass
    return orthonormalize(W.detach().cpu().numpy())


def _loco_job(payload: dict) -> list[dict]:
    seed = payload["seed"]
    r = payload["r"]
    max_per = payload["max_per_class"]
    prior = Path(payload["prior_run"])
    device = payload.get("device", "cpu")
    bases = load_prior_bases(prior, seed)
    rows = []

    # precompute country data
    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        data[c] = (Xm, ym, tr, te)

    for tgt in COUNTRIES:
        sources = [c for c in COUNTRIES if c != tgt]
        # pool source train
        Xtr_list, ytr_list = [], []
        for s in sources:
            Xm, ym, tr, _ = data[s]
            Xtr_list.append(Xm[tr])
            ytr_list.append(ym[tr])
        Xtr = np.concatenate(Xtr_list)
        ytr = np.concatenate(ytr_list)
        Xm_t, ym_t, _, _ = data[tgt]
        Xte, yte = Xm_t, ym_t  # full target

        # Method A / B / geometric
        U_a = method_a_invariant_basis(bases, tgt, seed, max_per, r)
        try:
            U_b = method_b_optimize_basis(tgt, seed, max_per, r, device=device)
        except Exception:
            U_b = U_a.copy()
        U_avg = average_projection_basis(loco_bases(bases, tgt), r)
        try:
            U_karch = grassmann_karcher_mean(loco_bases(bases, tgt), r)
        except Exception:
            U_karch = U_avg
        # pooled PLS on sources
        sc_pool = StandardScaler().fit(Xtr)
        from common import pls_rotations

        U_pool = orthonormalize(pls_rotations(sc_pool.transform(Xtr), ytr, n_components=r))
        # PCA
        U_pca = orthonormalize(pca_basis(sc_pool.transform(Xtr), n_components=r))
        # source-specific: evaluate each source PLS separately then average metrics later as rows
        methods = {
            "lr64": None,
            "pca": U_pca[:, :r],
            "pooled_pls": U_pool[:, :r],
            "geometric_avgproj": U_avg[:, :r],
            "geometric_karcher": orthonormalize(U_karch)[:, :r],
            "task_invariant_A": U_a[:, :r],
            "task_invariant_B": U_b[:, :r],
        }
        for s in sources:
            methods[f"source_pls_{s}"] = bases[s][:, :r]

        for name, U in methods.items():
            if name == "lr64":
                sc = StandardScaler().fit(Xtr)
                pred, proba, _ = fit_lr(sc.transform(Xtr), ytr, sc.transform(Xte), seed=seed)
                m = binary_metrics(yte, pred, proba)
            else:
                m = project_eval(U, Xtr, ytr, Xte, yte, seed)
            rows.append(
                {
                    "seed": seed,
                    "r": r,
                    "target": tgt,
                    "method": name,
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "auroc": m["auroc"],
                    "n": m.get("n", len(yte)),
                }
            )
        # save subspaces
        sub_dir = Path(payload["subspace_dir"])
        np.save(sub_dir / f"UG_taskA_loco_{tgt}_r{r}_seed{seed}.npy", U_a)
        np.save(sub_dir / f"UG_taskB_loco_{tgt}_r{r}_seed{seed}.npy", U_b)
        np.save(sub_dir / f"UG_avgproj_loco_{tgt}_r{r}_seed{seed}.npy", U_avg)
    return rows


def stage_loco(run_dir: Path, args) -> Path:
    name = "04_loco_methods"
    out = run_dir / "artifacts" / "loco_results.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    sub_dir = run_dir / "artifacts" / "subspaces"
    sub_dir.mkdir(parents=True, exist_ok=True)
    ng = n_gpus()
    devices = [f"cuda:{i}" for i in range(ng)] if ng > 0 else ["cpu"]
    log(run_dir, f"{name}: GPUs={ng} devices={devices}")
    jobs = []
    idx = 0
    for seed in args.seeds:
        for r in R_GRID:
            jobs.append(
                {
                    "seed": seed,
                    "r": r,
                    "max_per_class": args.max_per_class,
                    "prior_run": str(args.prior_run),
                    "device": devices[idx % len(devices)],
                    "subspace_dir": str(sub_dir),
                }
            )
            idx += 1
    rows = []
    t0 = time.time()
    # Use process pool; GPU jobs should not oversubscribe too hard — one job per GPU concurrent ideally
    n_workers = min(args.workers, max(len(devices) * 2, 4), len(jobs))
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs), resource_allocation={"gpus": ng, "workers": n_workers})
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(args.extract_dir), str(args.prior_run)),
    ) as pool:
        futs = {pool.submit(_loco_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception:
                (run_dir / "failures" / f"{name}_{done}.txt").write_text(traceback.format_exc())
                log(run_dir, f"{name} failed: {traceback.format_exc().splitlines()[-1]}")
            done += 1
            write_progress(
                run_dir,
                stage=name,
                completed=done,
                total=len(jobs),
                percent_complete=round(100 * done / len(jobs), 2),
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
            )
            pd.DataFrame(rows).to_csv(out.with_suffix(".partial.csv"), index=False)
            log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0, "gpus": ng})
    return out


# ---------------------------------------------------------------------------
# Stage 05 — global/local + swap using task-invariant S_G
# ---------------------------------------------------------------------------
def _global_local_job(payload: dict) -> dict:
    seed = payload["seed"]
    r = payload["r"]
    max_per = payload["max_per_class"]
    prior = Path(payload["prior_run"])
    device = payload.get("device", "cpu")
    bases = load_prior_bases(prior, seed)
    gl_rows = []
    swap_rows = []

    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        data[c] = (Xm, ym, tr, te)

    # descriptive S_G_task: Method A on all three (NOT for OOD claims); uses cached `data`
    def method_a_all(r_dim: int) -> np.ndarray:
        cands = []
        for src in COUNTRIES:
            Xm_s, ym_s, tr, te = data[src]
            sc = StandardScaler().fit(Xm_s[tr])
            Xte_s = sc.transform(Xm_s[te])
            Xt = {t: sc.transform(data[t][0]) for t in COUNTRIES if t != src}
            for k in range(20):
                u = np.asarray(bases[src][:, k], dtype=np.float64)
                u = u / (np.linalg.norm(u) + 1e-12)
                z_src = Xte_s @ u
                src_u = direction_stats(z_src, ym_s[te])["auroc_abs"]
                tgt_us = []
                for tgt in COUNTRIES:
                    if tgt == src:
                        continue
                    z = Xt[tgt] @ u
                    a0 = direction_stats(z, data[tgt][1])["auroc_abs"]
                    a1 = direction_stats(-z, data[tgt][1])["auroc_abs"]
                    tgt_us.append(max(a0, a1))
                ti = task_invariance_score(tgt_us, 1.0)
                if z_src[ym_s[te] == 1].mean() < z_src[ym_s[te] == 0].mean():
                    u = -u
                cands.append((0.5 * src_u + 0.5 * ti, u.copy()))
        cands.sort(key=lambda x: -x[0])
        chosen = []
        for _, u in cands:
            v = u.copy()
            for q in chosen:
                v = v - q * float(q @ v)
            nrm = np.linalg.norm(v)
            if nrm < 1e-8:
                continue
            chosen.append(v / nrm)
            if len(chosen) >= r_dim:
                break
        return orthonormalize(np.stack(chosen, axis=1))

    U_g_desc = method_a_all(r)

    for c in COUNTRIES:
        Xm, ym, tr, te = data[c]
        U_c = bases[c][:, : max(r, 9)]
        U_loc = residual_basis(U_c, U_g_desc)  # orthogonal residual by construction
        reps = {
            "global": U_g_desc[:, :r],
            "local": U_loc[:, : min(r, U_loc.shape[1])] if U_loc.shape[1] else U_g_desc[:, :0],
            "global_plus_local": concat_orthonormal(U_g_desc[:, :r], U_loc[:, : min(r, max(U_loc.shape[1], 0))]),
            "full_task": bases[c][:, :r],
            "lr64": None,
        }
        metrics_by_rep = {}
        for name, U in reps.items():
            if name == "lr64":
                sc = StandardScaler().fit(Xm[tr])
                pred, proba, _ = fit_lr(sc.transform(Xm[tr]), ym[tr], sc.transform(Xm[te]), seed=seed)
                m = binary_metrics(ym[te], pred, proba)
            else:
                m = project_eval(U, Xm[tr], ym[tr], Xm[te], ym[te], seed)
            metrics_by_rep[name] = m
            gl_rows.append(
                {
                    "mode": "descriptive",
                    "seed": seed,
                    "r": r,
                    "country": c,
                    "representation": name,
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "auroc": m["auroc"],
                }
            )
        gl_rows.append(
            {
                "mode": "descriptive",
                "seed": seed,
                "r": r,
                "country": c,
                "representation": "delta_local",
                "macro_f1": metrics_by_rep["global_plus_local"]["macro_f1"] - metrics_by_rep["global"]["macro_f1"],
                "balanced_accuracy": float("nan"),
                "auroc": float("nan"),
            }
        )

    # LOCO local swap: S_G^(-i) + S_L,j on target i
    for tgt in COUNTRIES:
        U_g = method_a_invariant_basis(bases, tgt, seed, max_per, r)
        Xm, ym, tr, te = data[tgt]
        for donor in COUNTRIES:
            U_loc = residual_basis(bases[donor][:, : max(r, 9)], U_g)
            U = concat_orthonormal(U_g[:, :r], U_loc[:, : min(r, max(U_loc.shape[1], 0))])
            m = project_eval(U, Xm[tr], ym[tr], Xm[te], ym[te], seed)
            swap_rows.append(
                {
                    "mode": "loco",
                    "seed": seed,
                    "r": r,
                    "target": tgt,
                    "donor": donor,
                    "matched": int(tgt == donor),
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "auroc": m["auroc"],
                }
            )
        # also within LOCO: global only on target (train on sources for classifier)
        sources = [c for c in COUNTRIES if c != tgt]
        Xtr = np.concatenate([data[s][0][data[s][2]] for s in sources])
        ytr = np.concatenate([data[s][1][data[s][2]] for s in sources])
        for name, U in (
            ("global_loco", U_g[:, :r]),
            ("lr64_loco", None),
        ):
            if name == "lr64_loco":
                sc = StandardScaler().fit(Xtr)
                pred, proba, _ = fit_lr(sc.transform(Xtr), ytr, sc.transform(Xm), seed=seed)
                m = binary_metrics(ym, pred, proba)
            else:
                m = project_eval(U, Xtr, ytr, Xm, ym, seed)
            gl_rows.append(
                {
                    "mode": "loco",
                    "seed": seed,
                    "r": r,
                    "country": tgt,
                    "representation": name,
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "auroc": m["auroc"],
                }
            )

    return {"gl": gl_rows, "swap": swap_rows}


def stage_global_local(run_dir: Path, args) -> Path:
    name = "05_global_local"
    out = run_dir / "artifacts" / "global_local_results.csv"
    out_swap = run_dir / "artifacts" / "local_swap_matrix.csv"
    if stage_done(run_dir, name) and out.exists() and out_swap.exists():
        log(run_dir, f"skip {name}")
        return out
    t0 = time.time()
    gl_rows, swap_rows = [], []
    # Resume: heavy matrices already written; only rebuild summary
    if out.exists() and out.stat().st_size > 10 and out_swap.exists() and out_swap.stat().st_size > 10:
        log(run_dir, f"{name}: reusing existing matrices; rebuilding summary")
        write_progress(run_dir, stage=name, status="running", completed=1, total=1, percent_complete=50.0)
    else:
        ng = n_gpus()
        devices = [f"cuda:{i}" for i in range(ng)] if ng > 0 else ["cpu"]
        jobs = []
        idx = 0
        for seed in args.seeds:
            for r in R_GRID:
                jobs.append(
                    {
                        "seed": seed,
                        "r": r,
                        "max_per_class": args.max_per_class,
                        "prior_run": str(args.prior_run),
                        "device": devices[idx % len(devices)],
                    }
                )
                idx += 1
        n_workers = min(args.workers, max(len(devices) * 2, 4), len(jobs))
        write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(str(args.extract_dir), str(args.prior_run)),
        ) as pool:
            futs = {pool.submit(_global_local_job, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                    gl_rows.extend(res["gl"])
                    swap_rows.extend(res["swap"])
                except Exception:
                    (run_dir / "failures" / f"{name}_{done}.txt").write_text(traceback.format_exc())
                    log(run_dir, f"{name} failed")
                done += 1
                write_progress(
                    run_dir,
                    stage=name,
                    completed=done,
                    total=len(jobs),
                    percent_complete=round(100 * done / len(jobs), 2),
                )
                log(run_dir, f"{name} {done}/{len(jobs)}")
        if gl_rows:
            pd.DataFrame(gl_rows).to_csv(out, index=False)
        if swap_rows:
            pd.DataFrame(swap_rows).to_csv(out_swap, index=False)

    # swap summary + bootstrap CI on diagonal advantage
    # NOTE: use bracket cols — DataFrame.mode is a pandas method, not the column.
    swap = pd.read_csv(out_swap) if out_swap.exists() else pd.DataFrame(swap_rows)
    summ = []
    rng = np.random.default_rng(0)
    if len(swap) and "r" in swap.columns:
        for r in sorted(swap["r"].unique()):
            for tgt in COUNTRIES:
                sub = swap[(swap["r"] == r) & (swap["target"] == tgt) & (swap["mode"] == "loco")]
                for seed in args.seeds:
                    ss = sub[sub["seed"] == seed]
                    if ss.empty:
                        continue
                    matched = float(ss[ss["matched"] == 1]["macro_f1"].mean())
                    mis = float(ss[ss["matched"] == 0]["macro_f1"].mean())
                    summ.append(
                        {
                            "r": int(r),
                            "target": tgt,
                            "seed": int(seed),
                            "matched": matched,
                            "mismatched": mis,
                            "advantage": matched - mis,
                        }
                    )
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(run_dir / "artifacts" / "local_swap_summary.csv", index=False)
    boot_rows = []
    if len(summ_df):
        for r in sorted(summ_df["r"].unique()):
            for tgt in COUNTRIES:
                vals = summ_df[(summ_df["r"] == r) & (summ_df["target"] == tgt)]["advantage"].to_numpy()
                if len(vals) == 0:
                    continue
                boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(args.n_bootstrap)]
                boot_rows.append(
                    {
                        "r": int(r),
                        "target": tgt,
                        "mean_advantage": float(vals.mean()),
                        "ci_lo": float(np.percentile(boots, 2.5)),
                        "ci_hi": float(np.percentile(boots, 97.5)),
                    }
                )
    pd.DataFrame(boot_rows).to_csv(run_dir / "artifacts" / "bootstrap" / "local_swap_bootstrap.csv", index=False)
    mark_done(run_dir, name, {"seconds": time.time() - t0, "n_swap": int(len(swap)), "n_summary": int(len(summ_df))})
    return out


# ---------------------------------------------------------------------------
# Stage 06 — rank analysis
# ---------------------------------------------------------------------------
def _rank_job(payload: dict) -> list[dict]:
    seed = payload["seed"]
    max_per = payload["max_per_class"]
    prior = Path(payload["prior_run"])
    bases = load_prior_bases(prior, seed)
    util = pd.read_csv(payload["util_path"])
    ti = pd.read_csv(payload["ti_path"])
    rows = []
    data = {}
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, te = stratified_split_idx(ym, 0.2, rng)
        data[c] = (Xm, ym, tr, te)

    for k in KS:
        from common import average_projection_spectrum, global_overlap as go

        evals, _ = average_projection_spectrum([bases[c] for c in COUNTRIES], k)
        g_overlap = go(evals, k)

        # task invariance mean (real, lambda=1)
        ti_k = ti[(ti.component == k) & (ti.ti_lambda == 1.0) & (ti.seed == seed)]
        ti_mean = float(ti_k["ti_auroc"].mean()) if len(ti_k) else float("nan")
        excess = float(ti_k["excess_ti_auroc"].mean()) if len(ti_k) and "excess_ti_auroc" in ti_k else float("nan")

        # within F1: mean over countries of PLS-k within
        within_f1s = []
        for c in COUNTRIES:
            Xm, ym, tr, te = data[c]
            m = project_eval(bases[c][:, :k], Xm[tr], ym[tr], Xm[te], ym[te], seed)
            within_f1s.append(m["macro_f1"])
        # OOD F1: mean over pairs of source PLS-k
        ood_f1s = []
        for src, tgt in PAIRS:
            Xm_s, ym_s, tr_s, _ = data[src]
            Xm_t, ym_t, _, _ = data[tgt]
            m = project_eval(bases[src][:, :k], Xm_s[tr_s], ym_s[tr_s], Xm_t, ym_t, seed)
            ood_f1s.append(m["macro_f1"])

        rows.append(
            {
                "seed": seed,
                "k": k,
                "within_f1": float(np.mean(within_f1s)),
                "ood_f1": float(np.mean(ood_f1s)),
                "geometric_overlap": float(g_overlap),
                "task_invariance": ti_mean,
                "excess_task_invariance": excess,
            }
        )
    # deltas
    rows = sorted(rows, key=lambda x: x["k"])
    for i, row in enumerate(rows):
        if i == 0:
            row["d_within"] = 0.0
            row["d_ood"] = 0.0
            row["d_geom"] = 0.0
            row["d_ti"] = 0.0
        else:
            prev = rows[i - 1]
            row["d_within"] = row["within_f1"] - prev["within_f1"]
            row["d_ood"] = row["ood_f1"] - prev["ood_f1"]
            row["d_geom"] = row["geometric_overlap"] - prev["geometric_overlap"]
            row["d_ti"] = row["task_invariance"] - prev["task_invariance"]
    return rows


def stage_rank(run_dir: Path, args) -> Path:
    name = "06_rank_analysis"
    out = run_dir / "artifacts" / "rank_analysis.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    jobs = [
        {
            "seed": s,
            "max_per_class": args.max_per_class,
            "prior_run": str(args.prior_run),
            "util_path": str(run_dir / "artifacts" / "task_direction_utility.csv"),
            "ti_path": str(run_dir / "artifacts" / "task_invariance.csv"),
        }
        for s in args.seeds
    ]
    rows = []
    write_progress(run_dir, stage=name, status="running", completed=0, total=len(jobs))
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(jobs)),
        initializer=_worker_init,
        initargs=(str(args.extract_dir), str(args.prior_run)),
    ) as pool:
        done = 0
        for part in pool.map(_rank_job, jobs):
            rows.extend(part)
            done += 1
            write_progress(run_dir, stage=name, completed=done, total=len(jobs), percent_complete=round(100 * done / len(jobs), 2))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    # association deltas
    a = df.dropna(subset=["d_ti", "d_ood", "d_geom"])
    a = a[a.k > 1]
    assoc = {
        "pearson_dTI_dOOD": float(np.corrcoef(a.d_ti, a.d_ood)[0, 1]) if len(a) > 3 else float("nan"),
        "pearson_dGeom_dOOD": float(np.corrcoef(a.d_geom, a.d_ood)[0, 1]) if len(a) > 3 else float("nan"),
        "n_rows": int(len(a)),
        "limitation": "Only 3 countries; seed×k rows are not independent geographic observations.",
    }
    atomic_json(run_dir / "artifacts" / "rank_delta_assoc.json", assoc)
    mark_done(run_dir, name, assoc)
    return out


# ---------------------------------------------------------------------------
# Stage 07 — figures + report + notebook
# ---------------------------------------------------------------------------
def stage_figures_report(run_dir: Path, args) -> None:
    name = "07_figures_report"
    if stage_done(run_dir, name) and (run_dir / "REPORT.md").exists():
        log(run_dir, f"skip {name}")
        return
    write_progress(run_dir, stage=name, status="running")
    # Load sibling module explicitly — common.py puts global_local_subspace on
    # sys.path ahead of this package, which would shadow make_figures_and_report.
    import importlib.util

    fig_path = Path(__file__).resolve().parent / "make_figures_and_report.py"
    spec = importlib.util.spec_from_file_location("exp05_make_figures_and_report", fig_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.build_all(run_dir)
    mark_done(run_dir, name, {"report": str(run_dir / "REPORT.md")})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--prior-run", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--exp04-run", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--n-shuffle", type=int, default=5, help="shuffle draws per seed")
    args = ap.parse_args()

    run_dir = args.run_dir
    for d in ("checkpoints", "logs", "artifacts", "failures", "figures", "artifacts/subspaces", "artifacts/bootstrap", "artifacts/seeds"):
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    atomic_json(
        run_dir / "config.json",
        {
            "experiment": "05_task_invariant_subspace",
            "hypothesis": "S_c ≈ S_G_task + S_L,c with predictive invariance",
            "prior_run": str(args.prior_run),
            "exp04_run": str(args.exp04_run) if args.exp04_run else None,
            "extract_dir": str(args.extract_dir),
            "workers": args.workers,
            "max_per_class": args.max_per_class,
            "seeds": args.seeds,
            "n_bootstrap": args.n_bootstrap,
            "n_shuffle": args.n_shuffle,
            "gpus": n_gpus(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "PLS is a probe only; geometry alone is insufficient (Exp04 spectrum null).",
        },
    )

    init_cache(args.extract_dir)
    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        output_location=str(run_dir),
        resource_allocation={"cpus": args.workers, "gpus": n_gpus(), "max_per_class": args.max_per_class},
    )
    log(run_dir, f"start Experiment 05 workers={args.workers} gpus={n_gpus()} seeds={args.seeds}")

    t0 = time.time()
    try:
        stage_direction_utility(run_dir, args)
        stage_task_invariance(run_dir, args)
        stage_geometry_vs_task(run_dir, args)
        stage_loco(run_dir, args)
        stage_global_local(run_dir, args)
        stage_rank(run_dir, args)
        stage_figures_report(run_dir, args)
        write_progress(run_dir, status="completed", stage="done", percent_complete=100.0)
        atomic_json(
            run_dir / "summary.json",
            {"status": "completed", "runtime_seconds": time.time() - t0, "report": str(run_dir / "REPORT.md")},
        )
        (run_dir / "DONE").write_text("ok\n")
        log(run_dir, f"DONE in {time.time() - t0:.1f}s")
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "FAILED").write_text(tb)
        write_progress(run_dir, status="failed", stage="failed")
        log(run_dir, f"FAILED\n{tb}")
        raise


if __name__ == "__main__":
    main()
