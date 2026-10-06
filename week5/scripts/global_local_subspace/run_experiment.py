#!/usr/bin/env python3
"""Global ⊕ local AlphaEarth subspace experiment (cross-country).

Tests whether task-relevant structure decomposes as:
  S_country ≈ S_global ⊕ S_local,country

Stages (checkpointed DONE markers under run_dir/checkpoints/):
  01_transfer_k_sweep
  02_within_vs_cross
  03_country_subspaces_angles
  04_geometry_vs_transfer
  05_component_stability
  06_global_consensus
  07_shared_vs_local
  08_controls
  09_figures_report
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

from geometry import (  # noqa: E402
    SEED_DEFAULT,
    average_projection_basis,
    balance_cap_by_city,
    binary_metrics,
    fit_lr,
    grassmann_karcher_mean,
    orthonormalize,
    pca_basis,
    pls_rotations,
    principal_angles_rad,
    random_orthogonal,
    residual_basis,
    stratified_split_idx,
    subspace_distances,
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


def country_slice(X, y, regions, countries, country: str, max_per_class: int | None, seed: int):
    mask = countries == country
    Xm = np.asarray(X[mask], dtype=np.float32)
    ym = y[mask]
    rm = regions[mask]
    rng = np.random.default_rng(seed)
    return balance_cap_by_city(Xm, ym, rm, max_per_class, rng)


def fit_supervised_basis(X_tr: np.ndarray, y_tr: np.ndarray, method: str, k_max: int, seed: int):
    scaler = StandardScaler().fit(X_tr)
    Xs = scaler.transform(X_tr)
    if method == "pls":
        R = pls_rotations(Xs, y_tr, n_components=k_max)
        U = orthonormalize(R)
    elif method == "pca":
        U = orthonormalize(pca_basis(Xs, n_components=k_max))
    elif method == "random":
        U = random_orthogonal(Xs.shape[1], k_max, np.random.default_rng(seed))
    else:
        raise ValueError(method)
    return scaler, U


def eval_projection(scaler, U, X_tr, y_tr, X_te, y_te, k: int, seed: int) -> dict:
    Uk = U[:, : min(k, U.shape[1])]
    Ztr = scaler.transform(X_tr) @ Uk
    Zte = scaler.transform(X_te) @ Uk
    pred, proba, _ = fit_lr(Ztr, y_tr, Zte, seed=seed)
    return binary_metrics(y_te, pred, proba)


def eval_lr64(X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    scaler = StandardScaler().fit(X_tr)
    pred, proba, _ = fit_lr(scaler.transform(X_tr), y_tr, scaler.transform(X_te), seed=seed)
    return binary_metrics(y_te, pred, proba)


# ---------------------------------------------------------------------------
# Stage workers (fork-friendly via globals set before pool)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _transfer_job(payload: dict) -> list[dict]:
    """One (source, target, seed) transfer k-sweep."""
    src, tgt, seed = payload["source"], payload["target"], payload["seed"]
    max_per = payload["max_per_class"]
    shuffle_labels = payload.get("shuffle_labels", False)
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    Xs, ys, rs = country_slice(X, y, regions, countries, src, max_per, seed)
    Xt, yt, rt = country_slice(X, y, regions, countries, tgt, max_per, seed + 17)
    if shuffle_labels:
        rng = np.random.default_rng(seed + 99)
        ys = rng.permutation(ys)
    # within-source split for in-domain curve
    rng = np.random.default_rng(seed)
    tr_idx, te_src_idx = stratified_split_idx(ys, 0.2, rng)
    X_tr, y_tr = Xs[tr_idx], ys[tr_idx]
    X_te_src, y_te_src = Xs[te_src_idx], ys[te_src_idx]

    rows = []
    # lr64
    m_ood = eval_lr64(X_tr, y_tr, Xt, yt, seed)
    m_id = eval_lr64(X_tr, y_tr, X_te_src, y_te_src, seed)
    rows.append(
        {
            "source_country": src,
            "target_country": tgt,
            "seed": seed,
            "method": "lr64",
            "k": 64,
            "protocol": "cross_country",
            **{f"ood_{k}": v for k, v in m_ood.items()},
            **{f"id_{k}": v for k, v in m_id.items()},
            "transfer_drop_f1": m_id["macro_f1"] - m_ood["macro_f1"],
        }
    )
    for method in ("pls", "pca", "random"):
        scaler, U = fit_supervised_basis(X_tr, y_tr, method, k_max=20, seed=seed)
        for k in KS:
            if k > U.shape[1]:
                continue
            mo = eval_projection(scaler, U, X_tr, y_tr, Xt, yt, k, seed)
            mi = eval_projection(scaler, U, X_tr, y_tr, X_te_src, y_te_src, k, seed)
            rows.append(
                {
                    "source_country": src,
                    "target_country": tgt,
                    "seed": seed,
                    "method": method,
                    "k": k,
                    "protocol": "cross_country",
                    **{f"ood_{kk}": vv for kk, vv in mo.items()},
                    **{f"id_{kk}": vv for kk, vv in mi.items()},
                    "transfer_drop_f1": mi["macro_f1"] - mo["macro_f1"],
                }
            )
    return rows


def _loro_city_job(payload: dict) -> list[dict]:
    """Within-country LORO city holdout for one (country, held_out, seed)."""
    country, held, seed = payload["country"], payload["held_out"], payload["seed"]
    max_per = payload["max_per_class"]
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    Xm, ym, rm = country_slice(X, y, regions, countries, country, max_per, seed)
    te = rm == held
    tr = ~te
    if te.sum() == 0 or tr.sum() == 0:
        return []
    X_tr, y_tr = Xm[tr], ym[tr]
    X_te, y_te = Xm[te], ym[te]
    rows = []
    m64 = eval_lr64(X_tr, y_tr, X_te, y_te, seed)
    rows.append(
        {
            "country": country,
            "held_out": held,
            "seed": seed,
            "method": "lr64",
            "k": 64,
            "protocol": "loro_city",
            **m64,
        }
    )
    for method in ("pls", "pca"):
        scaler, U = fit_supervised_basis(X_tr, y_tr, method, k_max=20, seed=seed)
        for k in KS:
            if k > U.shape[1]:
                continue
            m = eval_projection(scaler, U, X_tr, y_tr, X_te, y_te, k, seed)
            rows.append(
                {
                    "country": country,
                    "held_out": held,
                    "seed": seed,
                    "method": method,
                    "k": k,
                    "protocol": "loro_city",
                    **m,
                }
            )
    return rows


def stage_transfer_k_sweep(run_dir: Path, args) -> Path:
    name = "01_transfer_k_sweep"
    out = run_dir / "artifacts" / "transfer_k_sweep.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running", completed=0)
    jobs = []
    for src, tgt in PAIRS:
        for seed in args.seeds:
            jobs.append({"source": src, "target": tgt, "seed": seed, "max_per_class": args.max_per_class})
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_transfer_job, j): j for j in jobs}
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
                current_item=f"{futs[fut]['source']}->{futs[fut]['target']} seed={futs[fut]['seed']}",
            )
            if done % 3 == 0 or done == len(jobs):
                pd.DataFrame(rows).to_csv(out.with_suffix(".partial.csv"), index=False)
                log(run_dir, f"{name} {done}/{len(jobs)}")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(df), "seconds": time.time() - t0})
    return out


def stage_within_vs_cross(run_dir: Path, args) -> Path:
    name = "02_within_vs_cross"
    out = run_dir / "artifacts" / "within_vs_cross.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running")
    # LORO city jobs
    jobs = []
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    for country in COUNTRIES:
        cities = sorted(np.unique(regions[countries == country]).tolist())
        for held in cities:
            for seed in args.seeds:
                jobs.append(
                    {
                        "country": country,
                        "held_out": held,
                        "seed": seed,
                        "max_per_class": args.max_per_class,
                    }
                )
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_loro_city_job, j): j for j in jobs}
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
                current_item=f"{futs[fut]['country']}/{futs[fut]['held_out']}",
            )
    # Also summarize cross from stage1
    transfer = pd.read_csv(run_dir / "artifacts" / "transfer_k_sweep.csv")
    # reshape for plotting convenience: mean over seeds
    loro = pd.DataFrame(rows)
    loro.to_csv(out, index=False)
    # Combined summary table
    summary_rows = []
    for country in COUNTRIES:
        g = loro[(loro.country == country) & (loro.method == "pls")]
        for k, sub in g.groupby("k"):
            summary_rows.append(
                {
                    "country": country,
                    "setting": "loro_city",
                    "method": "pls",
                    "k": int(k),
                    "macro_f1_mean": float(sub.macro_f1.mean()),
                    "macro_f1_std": float(sub.macro_f1.std(ddof=0)),
                }
            )
        # in-domain 80/20 from transfer file where source==target... we stored id_* for each cross pair
        # use mean id metrics when source==country averaged over targets
        t = transfer[(transfer.source_country == country) & (transfer.method == "pls")]
        for k, sub in t.groupby("k"):
            summary_rows.append(
                {
                    "country": country,
                    "setting": "within_country_20pct",
                    "method": "pls",
                    "k": int(k),
                    "macro_f1_mean": float(sub["id_macro_f1"].mean()),
                    "macro_f1_std": float(sub["id_macro_f1"].std(ddof=0)),
                }
            )
            summary_rows.append(
                {
                    "country": country,
                    "setting": "cross_country_ood",
                    "method": "pls",
                    "k": int(k),
                    "macro_f1_mean": float(sub["ood_macro_f1"].mean()),
                    "macro_f1_std": float(sub["ood_macro_f1"].std(ddof=0)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(run_dir / "artifacts" / "within_vs_cross_summary.csv", index=False)
    mark_done(run_dir, name, {"n_loro_rows": len(loro), "seconds": time.time() - t0})
    return out


def stage_country_subspaces(run_dir: Path, args) -> Path:
    name = "03_country_subspaces_angles"
    out_dir = run_dir / "artifacts" / "subspaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage_done(run_dir, name) and (run_dir / "artifacts" / "principal_angles.csv").exists():
        log(run_dir, f"skip {name}")
        return out_dir
    write_progress(run_dir, stage=name, status="running")
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    angle_rows = []
    basis_meta = {}
    # Fit country PLS on full (capped) country data for each seed, save U
    for seed in args.seeds:
        bases = {}
        for country in COUNTRIES:
            Xm, ym, rm = country_slice(X, y, regions, countries, country, args.max_per_class, seed)
            # use 80% train for subspace fit (match transfer protocol)
            rng = np.random.default_rng(seed)
            tr, _ = stratified_split_idx(ym, 0.2, rng)
            scaler, U = fit_supervised_basis(Xm[tr], ym[tr], "pls", k_max=20, seed=seed)
            path = out_dir / f"U_pls_{country}_seed{seed}.npy"
            np.save(path, U)
            np.save(out_dir / f"scaler_mean_{country}_seed{seed}.npy", scaler.mean_)
            np.save(out_dir / f"scaler_scale_{country}_seed{seed}.npy", scaler.scale_)
            bases[country] = U
            basis_meta[f"{country}_seed{seed}"] = {"path": str(path), "shape": list(U.shape)}
        for a, b in [("USA", "JPN"), ("USA", "IND"), ("JPN", "IND")]:
            for k in KS:
                d = subspace_distances(bases[a][:, :k], bases[b][:, :k])
                ang = d.pop("principal_angles_deg")
                angle_rows.append(
                    {
                        "country_a": a,
                        "country_b": b,
                        "seed": seed,
                        "k": k,
                        **d,
                        "principal_angles_deg_json": json.dumps(ang),
                    }
                )
        write_progress(run_dir, stage=name, current_item=f"seed={seed}", completed=seed)
    pd.DataFrame(angle_rows).to_csv(run_dir / "artifacts" / "principal_angles.csv", index=False)
    atomic_json(out_dir / "manifest.json", basis_meta)
    mark_done(run_dir, name, {"n_angle_rows": len(angle_rows)})
    return out_dir


# Canonical unordered pair order used when writing principal_angles.csv
ANGLE_PAIR_ORDER = {"USA": 0, "JPN": 1, "IND": 2}


def _angle_pair_key(c1: str, c2: str) -> tuple[str, str]:
    """Match stage-03 storage order (USA–JPN, USA–IND, JPN–IND), not lexical sort."""
    a, b = sorted([c1, c2], key=lambda c: ANGLE_PAIR_ORDER[c])
    return a, b


def stage_geometry_vs_transfer(run_dir: Path, args) -> Path:
    name = "04_geometry_vs_transfer"
    out = run_dir / "artifacts" / "geometry_vs_transfer.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running")
    transfer = pd.read_csv(run_dir / "artifacts" / "transfer_k_sweep.csv")
    angles = pd.read_csv(run_dir / "artifacts" / "principal_angles.csv")
    # map unordered country pairs to ordered transfer
    rows = []
    for (src, tgt) in PAIRS:
        a, b = _angle_pair_key(src, tgt)
        for seed in args.seeds:
            for k in KS:
                g = angles[
                    (angles.country_a == a)
                    & (angles.country_b == b)
                    & (angles.seed == seed)
                    & (angles.k == k)
                ]
                t = transfer[
                    (transfer.source_country == src)
                    & (transfer.target_country == tgt)
                    & (transfer.seed == seed)
                    & (transfer.method == "pls")
                    & (transfer.k == k)
                ]
                if g.empty or t.empty:
                    continue
                gr = g.iloc[0]
                tr = t.iloc[0]
                rows.append(
                    {
                        "source_country": src,
                        "target_country": tgt,
                        "seed": seed,
                        "k": k,
                        "ood_macro_f1": tr["ood_macro_f1"],
                        "id_macro_f1": tr["id_macro_f1"],
                        "transfer_drop_f1": tr["transfer_drop_f1"],
                        "chordal_distance": gr["chordal_distance"],
                        "grassmann_geodesic": gr["grassmann_geodesic"],
                        "mean_principal_angle_deg": gr["mean_principal_angle_deg"],
                        "max_principal_angle_deg": gr["max_principal_angle_deg"],
                        "projection_frobenius": gr["projection_frobenius"],
                    }
                )
    if not rows:
        raise RuntimeError("geometry_vs_transfer produced 0 rows — angle/transfer key mismatch")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # correlations
    corr_rows = []
    for metric in ("chordal_distance", "grassmann_geodesic", "mean_principal_angle_deg", "max_principal_angle_deg"):
        for target in ("ood_macro_f1", "transfer_drop_f1"):
            x = df[metric].to_numpy(dtype=float)
            yv = df[target].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(yv)
            if mask.sum() < 5:
                continue
            pear = float(np.corrcoef(x[mask], yv[mask])[0, 1])
            # spearman
            rx = pd.Series(x[mask]).rank().to_numpy()
            ry = pd.Series(yv[mask]).rank().to_numpy()
            spear = float(np.corrcoef(rx, ry)[0, 1])
            # bootstrap CI for pearson
            rng = np.random.default_rng(0)
            boots = []
            idx = np.where(mask)[0]
            for _ in range(1000):
                b = rng.choice(idx, size=len(idx), replace=True)
                if np.std(x[b]) < 1e-12 or np.std(yv[b]) < 1e-12:
                    continue
                boots.append(float(np.corrcoef(x[b], yv[b])[0, 1]))
            lo, hi = (np.percentile(boots, [2.5, 97.5]).tolist() if boots else [float("nan"), float("nan")])
            corr_rows.append(
                {
                    "distance_metric": metric,
                    "performance_metric": target,
                    "pearson": pear,
                    "spearman": spear,
                    "pearson_ci95_lo": lo,
                    "pearson_ci95_hi": hi,
                    "n": int(mask.sum()),
                }
            )
    pd.DataFrame(corr_rows).to_csv(run_dir / "artifacts" / "geometry_transfer_correlations.csv", index=False)
    mark_done(run_dir, name, {"n_rows": len(df)})
    return out


def stage_component_stability(run_dir: Path, args) -> Path:
    name = "05_component_stability"
    out = run_dir / "artifacts" / "component_stability.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running")
    transfer = pd.read_csv(run_dir / "artifacts" / "transfer_k_sweep.csv")
    angles = pd.read_csv(run_dir / "artifacts" / "principal_angles.csv")
    rows = []
    # marginal Δ performance when k increases
    for (src, tgt) in PAIRS:
        for seed in args.seeds:
            t = transfer[
                (transfer.source_country == src)
                & (transfer.target_country == tgt)
                & (transfer.seed == seed)
                & (transfer.method == "pls")
            ].sort_values("k")
            a, b = _angle_pair_key(src, tgt)
            g = angles[
                (angles.country_a == a) & (angles.country_b == b) & (angles.seed == seed)
            ].sort_values("k")
            if t.empty:
                continue
            prev_ood = prev_id = prev_chord = None
            for _, r in t.iterrows():
                k = int(r.k)
                gr = g[g.k == k]
                chord = float(gr.chordal_distance.iloc[0]) if len(gr) else float("nan")
                rows.append(
                    {
                        "source_country": src,
                        "target_country": tgt,
                        "seed": seed,
                        "k": k,
                        "ood_macro_f1": r.ood_macro_f1,
                        "id_macro_f1": r.id_macro_f1,
                        "chordal_distance": chord,
                        "delta_ood_f1": (r.ood_macro_f1 - prev_ood) if prev_ood is not None else float("nan"),
                        "delta_id_f1": (r.id_macro_f1 - prev_id) if prev_id is not None else float("nan"),
                        "delta_chordal": (chord - prev_chord) if prev_chord is not None and np.isfinite(chord) else float("nan"),
                    }
                )
                prev_ood, prev_id, prev_chord = r.ood_macro_f1, r.id_macro_f1, chord
    pd.DataFrame(rows).to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(rows)})
    return out


def _consensus_precompute_job(payload: dict) -> dict:
    """Per-seed: pooled PLS + consensus bases for all k (writes npy under sub_dir)."""
    seed = payload["seed"]
    max_per = payload["max_per_class"]
    sub_dir = Path(payload["sub_dir"])
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    bases = {c: np.load(sub_dir / f"U_pls_{c}_seed{seed}.npy") for c in COUNTRIES}
    Xs_list, ys_list = [], []
    for c in COUNTRIES:
        Xm, ym, _ = country_slice(X, y, regions, countries, c, max_per, seed)
        rng = np.random.default_rng(seed)
        tr, _ = stratified_split_idx(ym, 0.2, rng)
        Xs_list.append(Xm[tr])
        ys_list.append(ym[tr])
    Xpool = np.concatenate(Xs_list)
    ypool = np.concatenate(ys_list)
    scaler_pool, U_pool = fit_supervised_basis(Xpool, ypool, "pls", k_max=20, seed=seed)
    np.save(sub_dir / f"U_pls_pooled_seed{seed}.npy", U_pool)
    np.save(sub_dir / f"scaler_mean_pooled_seed{seed}.npy", scaler_pool.mean_)
    np.save(sub_dir / f"scaler_scale_pooled_seed{seed}.npy", scaler_pool.scale_)
    for k in KS:
        U_avg = average_projection_basis([bases[c] for c in COUNTRIES], k)
        U_karch = grassmann_karcher_mean([bases[c][:, :k] for c in COUNTRIES], k)
        np.save(sub_dir / f"U_consensus_avgproj_k{k}_seed{seed}.npy", U_avg)
        np.save(sub_dir / f"U_consensus_karcher_k{k}_seed{seed}.npy", U_karch)
    return {"seed": seed, "status": "ok"}


def _consensus_eval_job(payload: dict) -> list[dict]:
    """Per (seed, source, target): evaluate pooled/consensus/oracle across all k."""
    seed, src, tgt = payload["seed"], payload["source"], payload["target"]
    max_per = payload["max_per_class"]
    sub_dir = Path(payload["sub_dir"])
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    U_pool = np.load(sub_dir / f"U_pls_pooled_seed{seed}.npy")
    # rebuild pooled scaler from saved mean/scale
    sc_pool = StandardScaler()
    sc_pool.mean_ = np.load(sub_dir / f"scaler_mean_pooled_seed{seed}.npy")
    sc_pool.scale_ = np.load(sub_dir / f"scaler_scale_pooled_seed{seed}.npy")
    sc_pool.n_features_in_ = sc_pool.mean_.shape[0]
    sc_pool.var_ = sc_pool.scale_ ** 2

    Xs, ys, _ = country_slice(X, y, regions, countries, src, max_per, seed)
    Xt, yt, _ = country_slice(X, y, regions, countries, tgt, max_per, seed + 17)
    rng = np.random.default_rng(seed)
    tr, _ = stratified_split_idx(ys, 0.2, rng)
    X_tr, y_tr = Xs[tr], ys[tr]
    sc_src = StandardScaler().fit(X_tr)

    # oracle basis once (target labels for basis only)
    rng_t = np.random.default_rng(seed + 3)
    tr_t, _ = stratified_split_idx(yt, 0.2, rng_t)
    _, U_o = fit_supervised_basis(Xt[tr_t], yt[tr_t], "pls", k_max=20, seed=seed)

    rows = []
    for k in KS:
        U_avg = np.load(sub_dir / f"U_consensus_avgproj_k{k}_seed{seed}.npy")
        U_karch = np.load(sub_dir / f"U_consensus_karcher_k{k}_seed{seed}.npy")
        for method, U, use_pool_scaler in (
            ("pooled_pls", U_pool, True),
            ("consensus_avgproj", U_avg, False),
            ("consensus_karcher", U_karch, False),
            ("oracle_target_pls", U_o, False),
        ):
            sc = sc_pool if use_pool_scaler else sc_src
            Uk = U[:, : min(k, U.shape[1])]
            Ztr = sc.transform(X_tr) @ Uk
            Zte = sc.transform(Xt) @ Uk
            pred, proba, _ = fit_lr(Ztr, y_tr, Zte, seed=seed)
            m = binary_metrics(yt, pred, proba)
            rows.append(
                {
                    "source_country": src,
                    "target_country": tgt,
                    "seed": seed,
                    "method": method,
                    "k": k,
                    **{f"ood_{kk}": vv for kk, vv in m.items()},
                }
            )
    return rows


def stage_global_consensus(run_dir: Path, args) -> Path:
    name = "06_global_consensus"
    out = run_dir / "artifacts" / "global_consensus_transfer.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running", completed=0)
    sub_dir = run_dir / "artifacts" / "subspaces"
    t0 = time.time()

    # Phase A: precompute bases (parallel by seed)
    pre_jobs = [
        {"seed": seed, "max_per_class": args.max_per_class, "sub_dir": str(sub_dir)} for seed in args.seeds
    ]
    log(run_dir, f"{name} phase A: precompute bases for {len(pre_jobs)} seeds")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pre_jobs))) as pool:
        futs = list(pool.map(_consensus_precompute_job, pre_jobs))
    log(run_dir, f"{name} phase A done: {len(futs)} seeds")

    # Phase B: evaluate transfers (parallel by seed × pair)
    eval_jobs = []
    for seed in args.seeds:
        for src, tgt in PAIRS:
            eval_jobs.append(
                {
                    "seed": seed,
                    "source": src,
                    "target": tgt,
                    "max_per_class": args.max_per_class,
                    "sub_dir": str(sub_dir),
                }
            )
    rows: list[dict] = []
    n_workers = min(args.workers, len(eval_jobs))
    log(run_dir, f"{name} phase B: {len(eval_jobs)} eval jobs on {n_workers} workers")
    write_progress(
        run_dir,
        stage=name,
        status="running",
        completed=0,
        total=len(eval_jobs),
        resource_allocation={"workers": n_workers, "cpus": os.cpu_count(), "gpus": 0},
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_consensus_eval_job, j): j for j in eval_jobs}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            write_progress(
                run_dir,
                stage=name,
                completed=done,
                total=len(eval_jobs),
                percent_complete=round(100 * done / len(eval_jobs), 2),
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
                current_item=f"{futs[fut]['source']}->{futs[fut]['target']} seed={futs[fut]['seed']}",
                eta_seconds=round((len(eval_jobs) - done) / max(done / max(time.time() - t0, 1e-6), 1e-9)),
            )
            if done % 5 == 0 or done == len(eval_jobs):
                pd.DataFrame(rows).to_csv(out.with_suffix(".partial.csv"), index=False)
                log(run_dir, f"{name} eval {done}/{len(eval_jobs)}")
    pd.DataFrame(rows).to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(rows), "seconds": time.time() - t0, "workers": n_workers})
    return out


def _shared_local_job(payload: dict) -> list[dict]:
    """Per (seed, source, target): global vs local residual variants."""
    seed, src, tgt = payload["seed"], payload["source"], payload["target"]
    max_per = payload["max_per_class"]
    sub_dir = Path(payload["sub_dir"])
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    bases = {c: np.load(sub_dir / f"U_pls_{c}_seed{seed}.npy") for c in COUNTRIES}
    Xs, ys, _ = country_slice(X, y, regions, countries, src, max_per, seed)
    Xt, yt, _ = country_slice(X, y, regions, countries, tgt, max_per, seed + 17)
    rng = np.random.default_rng(seed)
    tr, _ = stratified_split_idx(ys, 0.2, rng)
    X_tr, y_tr = Xs[tr], ys[tr]
    sc = StandardScaler().fit(X_tr)
    rows = []
    for k_g in (2, 3, 5, 9):
        U_g = average_projection_basis([bases[c] for c in COUNTRIES], k_g)
        U_c = bases[src][:, : max(k_g, 9)]
        U_loc = residual_basis(U_c, U_g)
        variants = {
            "global_only": U_g[:, :k_g],
            "local_residual_only": U_loc[:, : min(k_g, U_loc.shape[1])] if U_loc.shape[1] else U_g[:, :0],
            "global_plus_local": (
                orthonormalize(
                    np.concatenate(
                        [U_g[:, :k_g], U_loc[:, : min(k_g, max(U_loc.shape[1], 0))]],
                        axis=1,
                    )
                )
                if U_loc.shape[1]
                else U_g[:, :k_g]
            ),
            "source_pls": bases[src][:, :k_g],
        }
        for vname, U in variants.items():
            if U.shape[1] == 0:
                continue
            Ztr = sc.transform(X_tr) @ U
            Zte = sc.transform(Xt) @ U
            pred, proba, _ = fit_lr(Ztr, y_tr, Zte, seed=seed)
            m = binary_metrics(yt, pred, proba)
            rows.append(
                {
                    "source_country": src,
                    "target_country": tgt,
                    "seed": seed,
                    "k_global": k_g,
                    "variant": vname,
                    "k_used": int(U.shape[1]),
                    **m,
                }
            )
        m64 = eval_lr64(X_tr, y_tr, Xt, yt, seed)
        rows.append(
            {
                "source_country": src,
                "target_country": tgt,
                "seed": seed,
                "k_global": k_g,
                "variant": "lr64",
                "k_used": 64,
                **m64,
            }
        )
    return rows


def stage_shared_vs_local(run_dir: Path, args) -> Path:
    name = "07_shared_vs_local"
    out = run_dir / "artifacts" / "shared_vs_local.csv"
    if stage_done(run_dir, name) and out.exists() and out.stat().st_size > 10:
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running", completed=0)
    sub_dir = run_dir / "artifacts" / "subspaces"
    jobs = []
    for seed in args.seeds:
        for src, tgt in PAIRS:
            jobs.append(
                {
                    "seed": seed,
                    "source": src,
                    "target": tgt,
                    "max_per_class": args.max_per_class,
                    "sub_dir": str(sub_dir),
                }
            )
    rows: list[dict] = []
    t0 = time.time()
    n_workers = min(args.workers, len(jobs))
    log(run_dir, f"{name}: {len(jobs)} jobs on {n_workers} workers")
    write_progress(
        run_dir,
        stage=name,
        total=len(jobs),
        resource_allocation={"workers": n_workers, "cpus": os.cpu_count(), "gpus": 0},
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_shared_local_job, j): j for j in jobs}
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
                current_item=f"{futs[fut]['source']}->{futs[fut]['target']} seed={futs[fut]['seed']}",
                eta_seconds=round((len(jobs) - done) / max(done / max(time.time() - t0, 1e-6), 1e-9)),
            )
            if done % 5 == 0 or done == len(jobs):
                log(run_dir, f"{name} {done}/{len(jobs)}")
    pd.DataFrame(rows).to_csv(out, index=False)
    mark_done(run_dir, name, {"n_rows": len(rows), "seconds": time.time() - t0, "workers": n_workers})
    return out


def stage_controls(run_dir: Path, args) -> Path:
    name = "08_controls"
    out = run_dir / "artifacts" / "label_shuffle_control.csv"
    if stage_done(run_dir, name) and out.exists():
        log(run_dir, f"skip {name}")
        return out
    write_progress(run_dir, stage=name, status="running")
    # label-shuffled transfer for PLS/PCA on a subset of pairs/seeds
    jobs = []
    for src, tgt in PAIRS:
        for seed in args.seeds[: min(3, len(args.seeds))]:
            jobs.append(
                {
                    "source": src,
                    "target": tgt,
                    "seed": seed,
                    "max_per_class": args.max_per_class,
                    "shuffle_labels": True,
                }
            )
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_transfer_job, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            part = fut.result()
            for r in part:
                r["control"] = "label_shuffle"
            rows.extend(part)
            done += 1
            write_progress(run_dir, stage=name, completed=done, total=len(jobs))
    pd.DataFrame(rows).to_csv(out, index=False)
    # Also compute angles under label shuffle for one seed
    X, y, regions, countries = _CACHE["X"], _CACHE["y"], _CACHE["regions"], _CACHE["countries"]
    angle_rows = []
    seed = args.seeds[0]
    bases = {}
    for country in COUNTRIES:
        Xm, ym, rm = country_slice(X, y, regions, countries, country, args.max_per_class, seed)
        rng = np.random.default_rng(seed + 123)
        ym_s = rng.permutation(ym)
        tr, _ = stratified_split_idx(ym_s, 0.2, np.random.default_rng(seed))
        _, U = fit_supervised_basis(Xm[tr], ym_s[tr], "pls", k_max=20, seed=seed)
        bases[country] = U
        np.save(run_dir / "artifacts" / "subspaces" / f"U_pls_shuffled_{country}_seed{seed}.npy", U)
    for a, b in [("USA", "JPN"), ("USA", "IND"), ("JPN", "IND")]:
        for k in KS:
            d = subspace_distances(bases[a][:, :k], bases[b][:, :k])
            d.pop("principal_angles_deg")
            angle_rows.append({"country_a": a, "country_b": b, "seed": seed, "k": k, "control": "label_shuffle", **d})
    pd.DataFrame(angle_rows).to_csv(run_dir / "artifacts" / "principal_angles_label_shuffle.csv", index=False)
    mark_done(run_dir, name, {"n_rows": len(rows), "seconds": time.time() - t0})
    return out


def stage_figures_report(run_dir: Path, args) -> Path:
    name = "09_figures_report"
    if stage_done(run_dir, name) and (run_dir / "REPORT.md").exists():
        log(run_dir, f"skip {name}")
        return run_dir / "REPORT.md"
    write_progress(run_dir, stage=name, status="running")
    # import locally to keep worker imports light
    from make_figures_and_report import build_all

    report = build_all(run_dir)
    mark_done(run_dir, name, {"report": str(report)})
    return report


def init_cache(extract_dir: Path):
    global _CACHE
    X, y, regions, countries = load_extract(extract_dir)
    _CACHE = {"X": X, "y": y, "regions": regions, "countries": countries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument(
        "--extract-dir",
        type=Path,
        default=ROOT
        / "week4/international_geo_generalization/runs/intl_transfer_20261005_142709/extract",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--max-per-class", type=int, default=25000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--only-stage", type=str, default="")
    args = ap.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "failures").mkdir(exist_ok=True)

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    cfg = {
        "extract_dir": str(args.extract_dir),
        "workers": args.workers,
        "max_per_class": args.max_per_class,
        "seeds": args.seeds,
        "ks": KS,
        "pairs": PAIRS,
        "countries": list(COUNTRIES),
        "hypothesis": "S_country ≈ S_global ⊕ S_local,country",
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
            ("01_transfer_k_sweep", stage_transfer_k_sweep),
            ("02_within_vs_cross", stage_within_vs_cross),
            ("03_country_subspaces_angles", stage_country_subspaces),
            ("04_geometry_vs_transfer", stage_geometry_vs_transfer),
            ("05_component_stability", stage_component_stability),
            ("06_global_consensus", stage_global_consensus),
            ("07_shared_vs_local", stage_shared_vs_local),
            ("08_controls", stage_controls),
            ("09_figures_report", stage_figures_report),
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
            "artifacts": sorted(p.name for p in (run_dir / "artifacts").glob("*")),
            "figures": sorted(p.name for p in (run_dir / "figures").glob("*.png")),
            "report": str(run_dir / "REPORT.md"),
        }
        # key numbers
        try:
            transfer = pd.read_csv(run_dir / "artifacts" / "transfer_k_sweep.csv")
            key = []
            for src, tgt in PAIRS:
                for method, k in (("lr64", 64), ("pls", 2), ("pls", 9)):
                    sub = transfer[
                        (transfer.source_country == src)
                        & (transfer.target_country == tgt)
                        & (transfer.method == method)
                        & (transfer.k == k)
                    ]
                    if sub.empty:
                        continue
                    key.append(
                        {
                            "pair": f"{src}->{tgt}",
                            "method": method,
                            "k": k,
                            "ood_macro_f1_mean": float(sub.ood_macro_f1.mean()),
                            "ood_macro_f1_std": float(sub.ood_macro_f1.std(ddof=0)),
                        }
                    )
            summary["key_transfer"] = key
        except Exception as e:
            summary["key_transfer_error"] = str(e)

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
