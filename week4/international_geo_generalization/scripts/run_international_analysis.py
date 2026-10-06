#!/usr/bin/env python3
"""Week4-style international transfer analysis (USA / India / Japan).

Protocols (threshold 0.5; StandardScaler fit on train only; seed 42):
  1) Within-country LORO (leave-one-city-out) for USA, IND, JPN
  2) Country→country transfer (train all cities in source country; test target)
  3) City→city transfer matrix (train all of A, test all of B; diagonal = 80/20)

Models: lr64, pca-k={2,9}, pls-k={2,9} — matches week4 stage-3 transfer suite.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
MODELS = (("lr64", "lr64", 64), ("pca", "pca", 2), ("pca", "pca", 9), ("pls", "pls", 2), ("pls", "pls", 9))

_STATE: dict = {}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(path: Path, msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prev["last_progress_at"] = prev["updated_at"]
    atomic_json(path, prev)


def binary_metrics(y, pred, proba) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    proba = np.asarray(proba, dtype=np.float64)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_residential": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_nonresidential": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
        "n_residential": int((y == 1).sum()),
        "n_nonresidential": int((y == 0).sum()),
    }
    if len(np.unique(y)) > 1 and np.isfinite(proba).all():
        out["auroc"] = float(roc_auc_score(y, proba))
    else:
        out["auroc"] = float("nan")
    return out


def fit_lr(X_tr, y_tr, X_te):
    clf = LogisticRegression(solver="newton-cholesky", C=1.0, max_iter=200, random_state=SEED)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return pred, proba, clf


def pca_basis(X_tr: np.ndarray) -> np.ndarray:
    cov = (X_tr.T @ X_tr) / max(X_tr.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, np.argsort(evals)[::-1]].astype(np.float64)


def pls_rotations(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    yv = yv - yv.mean()
    n_comp = int(min(n_components, X.shape[1], max(X.shape[0] - 1, 1)))
    XtX = X.T @ X
    XtY = X.T @ yv
    p = X.shape[1]
    W = np.zeros((p, n_comp))
    P = np.zeros((p, n_comp))
    for k in range(n_comp):
        w = XtY.copy()
        nrm = np.linalg.norm(w)
        if nrm < 1e-12:
            W, P = W[:, :k], P[:, :k]
            break
        w = w / nrm
        for _ in range(80):
            t_var = float(w @ (XtX @ w))
            if t_var < 1e-18:
                break
            c_scalar = float(XtY @ w) / t_var
            if abs(c_scalar) < 1e-18:
                break
            c_unit = 1.0 if c_scalar >= 0 else -1.0
            w_new = XtY * c_unit
            nrm = np.linalg.norm(w_new)
            if nrm < 1e-12:
                break
            w_new = w_new / nrm
            if np.linalg.norm(w_new - w) < 1e-8:
                w = w_new
                break
            w = w_new
        t_var = max(float(w @ (XtX @ w)), 1e-18)
        p_k = (XtX @ w) / t_var
        c_scalar = float(XtY @ w) / t_var
        Xtw = XtX @ w
        XtX = XtX - np.outer(p_k, Xtw) - np.outer(Xtw, p_k) + t_var * np.outer(p_k, p_k)
        XtY = XtY - c_scalar * Xtw - p_k * float(w @ XtY) + t_var * c_scalar * p_k
        W[:, k] = w
        P[:, k] = p_k
    if W.shape[1] == 0:
        raise RuntimeError("PLS produced no components")
    return W @ np.linalg.pinv(P.T @ W)


def scale_train(X_tr, X_te):
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), scaler


def _split80(y_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y_local == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(0.2 * len(idx))))
        te.append(idx[:n_te])
        tr.append(idx[n_te:])
    return np.concatenate(tr), np.concatenate(te)


def _fit_bundle(X_tr: np.ndarray, y_tr: np.ndarray):
    X_tr_s, _, scaler = scale_train(X_tr, X_tr[:1])
    pca = pca_basis(X_tr_s)
    pls = pls_rotations(X_tr_s, y_tr, n_components=9)
    Zp = X_tr_s @ pca
    Zl = X_tr_s @ pls
    models = {}
    _, _, clf = fit_lr(X_tr_s, y_tr, X_tr_s[:2])
    models[("lr64", 64)] = clf
    for k in (2, 9):
        _, _, clf_p = fit_lr(Zp[:, :k], y_tr, Zp[:2, :k])
        _, _, clf_l = fit_lr(Zl[:, :k], y_tr, Zl[:2, :k])
        models[("pca", k)] = clf_p
        models[("pls", k)] = clf_l
    return scaler, pca, pls, models


def _score(bundle, X_te: np.ndarray, y_te: np.ndarray) -> list[dict]:
    scaler, pca, pls, models = bundle
    Xs = scaler.transform(X_te)
    rows = []
    for kind, k in (("lr64", 64), ("pca", 2), ("pca", 9), ("pls", 2), ("pls", 9)):
        if kind == "lr64":
            Z = Xs
        elif kind == "pca":
            Z = Xs @ pca[:, :k]
        else:
            Z = Xs @ pls[:, :k]
        clf = models[(kind, k)]
        proba = clf.predict_proba(Z)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        m = binary_metrics(y_te, pred, proba)
        m.update({"model": kind, "k": k})
        rows.append(m)
    return rows


def _loro_fold(held_out: str) -> dict:
    t0 = time.time()
    X = _STATE["X"]
    y = _STATE["y"]
    regions = _STATE["regions"]
    mask_country = _STATE["country_mask"]
    test_idx = np.where(mask_country & (regions == held_out))[0]
    train_idx = np.where(mask_country & (regions != held_out))[0]
    if held_out in set(regions[train_idx].tolist()):
        raise AssertionError("LORO leakage")
    X_tr = np.asarray(X[train_idx], dtype=np.float64)
    X_te = np.asarray(X[test_idx], dtype=np.float64)
    y_tr, y_te = y[train_idx], y[test_idx]
    X_tr_s, X_te_s, _ = scale_train(X_tr, X_te)
    rows = []
    pred, proba, _ = fit_lr(X_tr_s, y_tr, X_te_s)
    m = binary_metrics(y_te, pred, proba)
    m.update({"held_out": held_out, "model": "lr64", "k": 64})
    rows.append(m)
    pca = pca_basis(X_tr_s)
    pls = pls_rotations(X_tr_s, y_tr, n_components=9)
    for kind, basis in (("pca", pca), ("pls", pls)):
        Ztr, Zte = X_tr_s @ basis, X_te_s @ basis
        for k in (2, 9):
            pred, proba, _ = fit_lr(Ztr[:, :k], y_tr, Zte[:, :k])
            mk = binary_metrics(y_te, pred, proba)
            mk.update({"held_out": held_out, "model": kind, "k": k})
            rows.append(mk)
    return {"held_out": held_out, "rows": rows, "seconds": time.time() - t0}


def _city_source(source: str) -> dict:
    t0 = time.time()
    X = _STATE["X"]
    y = _STATE["y"]
    regions = _STATE["regions"]
    cities = _STATE["cities"]
    src = np.where(regions == source)[0]
    local_tr, local_te = _split80(y[src])
    full = _fit_bundle(np.asarray(X[src], dtype=np.float64), y[src])
    diag = _fit_bundle(np.asarray(X[src][local_tr], dtype=np.float64), y[src][local_tr])
    rows = []
    for target in cities:
        if target == source:
            te = src[local_te]
            bundle = diag
            protocol = "within_city_20pct"
        else:
            te = np.where(regions == target)[0]
            bundle = full
            protocol = "train_all_source"
        for m in _score(bundle, np.asarray(X[te], dtype=np.float64), y[te]):
            m.update({"source": source, "target": target, "protocol": protocol})
            rows.append(m)
    return {"source": source, "rows": rows, "seconds": time.time() - t0}


def run_loro(run_dir: Path, log_path: Path, country: str, workers: int) -> pd.DataFrame:
    ckpt = run_dir / "checkpoints" / f"loro_{country}.DONE"
    out = run_dir / "artifacts" / f"loro_{country}_metrics.csv"
    if ckpt.exists() and out.exists():
        log(log_path, f"LORO {country} already DONE")
        return pd.read_csv(out)
    regions = _STATE["regions"]
    countries = _STATE["countries"]
    mask = countries == country
    cities = sorted(np.unique(regions[mask]).tolist())
    _STATE["country_mask"] = mask
    log(log_path, f"LORO {country} cities={cities} workers={min(workers, len(cities))}")
    frames = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(cities))) as pool:
        futs = {pool.submit(_loro_fold, c): c for c in cities}
        done = 0
        for fut in as_completed(futs):
            result = fut.result()
            frames.append(pd.DataFrame(result["rows"]))
            done += 1
            write_progress(
                run_dir,
                stage=f"loro_{country}",
                completed=done,
                total=len(cities),
                percent_complete=round(100.0 * done / len(cities), 2),
                current_item=result["held_out"],
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
            )
            log(log_path, f"LORO {country} {done}/{len(cities)} {result['held_out']} {result['seconds']:.1f}s")
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "country", country)
    atomic_df(out, df)
    summary = (
        df[df["model"] == "lr64"]
        .groupby("country")[["macro_f1", "auroc"]]
        .agg(["mean", "std", "min", "max"])
    )
    atomic_json(
        run_dir / "artifacts" / f"loro_{country}_summary.json",
        {
            "country": country,
            "n_cities": len(cities),
            "lr64_macro_f1_mean": float(df.loc[df.model == "lr64", "macro_f1"].mean()),
            "lr64_macro_f1_std": float(df.loc[df.model == "lr64", "macro_f1"].std(ddof=0)),
            "cities": cities,
            "runtime_seconds": round(time.time() - t0, 1),
        },
    )
    ckpt.write_text("ok\n")
    log(log_path, f"LORO {country} mean_lr64_f1={df.loc[df.model=='lr64','macro_f1'].mean():.4f}")
    return df


def run_country_transfer(run_dir: Path, log_path: Path) -> pd.DataFrame:
    ckpt = run_dir / "checkpoints" / "country_transfer.DONE"
    out = run_dir / "artifacts" / "country_transfer_metrics.csv"
    if ckpt.exists() and out.exists():
        log(log_path, "country transfer already DONE")
        return pd.read_csv(out)
    X, y = _STATE["X"], _STATE["y"]
    countries = _STATE["countries"]
    country_list = sorted(np.unique(countries).tolist())
    rows = []
    t0 = time.time()
    total = len(country_list) ** 2
    done = 0
    for src in country_list:
        src_idx = np.where(countries == src)[0]
        local_tr, local_te = _split80(y[src_idx])
        full = _fit_bundle(np.asarray(X[src_idx], dtype=np.float64), y[src_idx])
        diag = _fit_bundle(np.asarray(X[src_idx][local_tr], dtype=np.float64), y[src_idx][local_tr])
        for tgt in country_list:
            if tgt == src:
                te = src_idx[local_te]
                bundle = diag
                protocol = "within_country_20pct"
            else:
                te = np.where(countries == tgt)[0]
                bundle = full
                protocol = "train_all_source_country"
            for m in _score(bundle, np.asarray(X[te], dtype=np.float64), y[te]):
                m.update({"source_country": src, "target_country": tgt, "protocol": protocol})
                rows.append(m)
            done += 1
            write_progress(
                run_dir,
                stage="country_transfer",
                completed=done,
                total=total,
                percent_complete=round(100.0 * done / total, 2),
                current_item=f"{src}->{tgt}",
            )
            log(
                log_path,
                f"country {src}->{tgt} lr64_f1="
                f"{[r['macro_f1'] for r in rows if r['source_country']==src and r['target_country']==tgt and r['model']=='lr64'][-1]:.4f}",
            )
    df = pd.DataFrame(rows)
    atomic_df(out, df)
    for (model, k), g in df.groupby(["model", "k"]):
        mat = g.pivot(index="source_country", columns="target_country", values="macro_f1")
        mat.to_csv(run_dir / "artifacts" / f"country_transfer_f1_{model}_k{k}.csv")
    ckpt.write_text("ok\n")
    log(log_path, f"country transfer done in {time.time()-t0:.1f}s")
    return df


def run_city_transfer(run_dir: Path, log_path: Path, workers: int) -> pd.DataFrame:
    ckpt = run_dir / "checkpoints" / "city_transfer.DONE"
    out = run_dir / "artifacts" / "city_transfer_metrics.csv"
    if ckpt.exists() and out.exists():
        log(log_path, "city transfer already DONE")
        return pd.read_csv(out)
    cities = _STATE["cities"]
    log(log_path, f"city transfer {len(cities)}x{len(cities)} workers={min(workers, len(cities))}")
    frames = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(cities))) as pool:
        futs = {pool.submit(_city_source, c): c for c in cities}
        done = 0
        for fut in as_completed(futs):
            result = fut.result()
            frames.append(pd.DataFrame(result["rows"]))
            done += 1
            atomic_df(out, pd.concat(frames, ignore_index=True))
            write_progress(
                run_dir,
                stage="city_transfer",
                completed=done,
                total=len(cities),
                percent_complete=round(100.0 * done / len(cities), 2),
                current_item=result["source"],
                rate_per_second=round(done / max(time.time() - t0, 1e-6), 3),
            )
            log(log_path, f"city source {done}/{len(cities)} {result['source']} {result['seconds']:.1f}s")
    df = pd.concat(frames, ignore_index=True)
    atomic_df(out, df)
    # annotate countries
    city_to_country = {
        r: c
        for r, c in zip(_STATE["regions"].tolist(), _STATE["countries"].tolist())
    }
    # unique mapping
    city_to_country = {}
    for r, c in zip(_STATE["regions"], _STATE["countries"]):
        city_to_country[str(r)] = str(c)
    df["source_country"] = df["source"].map(city_to_country)
    df["target_country"] = df["target"].map(city_to_country)
    atomic_df(out, df)
    g = df[(df.model == "lr64")]
    mat = g.pivot(index="source", columns="target", values="macro_f1").reindex(index=cities, columns=cities)
    mat.to_csv(run_dir / "artifacts" / "city_transfer_f1_lr64_k64.csv")
    ckpt.write_text("ok\n")
    return df


def write_results_md(run_dir: Path, loro: dict[str, pd.DataFrame], country_df: pd.DataFrame, city_df: pd.DataFrame) -> str:
    lines = []
    present = sorted(loro.keys())
    if present == ["IND"]:
        lines.append("# India AlphaEarth LORO — 10-city panel")
    elif present == ["JPN"]:
        lines.append("# Japan AlphaEarth LORO — 10-city panel")
    else:
        lines.append("# International AlphaEarth transfer — USA / India / Japan")
    lines.append("")
    lines.append("Week4-parity protocol: pure-exclusive clear cells, per-city 1:1 balance (seed 42),")
    lines.append("64-D logistic regression (and PCA/PLS probes), scaler fit on train only, threshold 0.5.")
    lines.append("")
    lines.append("## Within-country LORO (leave-one-city-out, lr64)")
    lines.append("")
    lines.append("| Country | Mean macro-F1 | Std | Min | Max | n cities |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c, df in loro.items():
        s = df.loc[df.model == "lr64", "macro_f1"]
        lines.append(
            f"| {c} | {s.mean():.3f} | {s.std(ddof=0):.3f} | {s.min():.3f} | {s.max():.3f} | {s.shape[0]} |"
        )
    lines.append("")
    lines.append("### Per-city LORO lr64")
    lines.append("")
    lines.append("| Country | Held-out city | macro-F1 | AUROC | n |")
    lines.append("|---|---|---:|---:|---:|")
    for c, df in loro.items():
        for _, r in df[df.model == "lr64"].sort_values("held_out").iterrows():
            lines.append(
                f"| {c} | {r['held_out']} | {r['macro_f1']:.3f} | {r['auroc']:.3f} | {int(r['n'])} |"
            )
    lines.append("")
    # PLS/PCA LORO summary for single-country focus panels
    for focus, title in (("IND", "India"), ("JPN", "Japan")):
        if focus not in loro or present != [focus]:
            continue
        lines.append(f"### {title} LORO probes (mean macro-F1 over holdouts)")
        lines.append("")
        lines.append("| model | k | mean F1 | std |")
        lines.append("|---|---:|---:|---:|")
        g = loro[focus].groupby(["model", "k"])["macro_f1"]
        for (model, k), s in g:
            lines.append(f"| {model} | {k} | {s.mean():.3f} | {s.std(ddof=0):.3f} |")
        lines.append("")

    lines.append("## Country → country transfer (lr64)")
    lines.append("")
    mat = country_df[country_df.model == "lr64"].pivot(
        index="source_country", columns="target_country", values="macro_f1"
    )
    cols = list(mat.columns)
    lines.append("| source \\ target | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---:" for _ in cols]) + "|")
    for src in mat.index:
        cells = " | ".join(f"{mat.loc[src, c]:.3f}" for c in cols)
        lines.append(f"| {src} | {cells} |")
    lines.append("")
    lines.append("Diagonal = within-country 80/20 stratified split. Off-diagonal = train all source-country cities, test all target-country cities.")
    lines.append("")

    # City-transfer averages (within-country when extract is single-country)
    city_lr = city_df[city_df.model == "lr64"].copy()
    if len(present) == 1:
        cname = {"IND": "India", "JPN": "Japan", "USA": "USA"}.get(present[0], present[0])
        lines.append(f"## Within-{cname} city→city transfer (lr64, off-diagonal)")
        lines.append("")
        g = city_lr[city_lr.source != city_lr.target]
        if len(g):
            lines.append(
                f"Mean F1={g.macro_f1.mean():.3f}, median={g.macro_f1.median():.3f}, "
                f"min={g.macro_f1.min():.3f}, max={g.macro_f1.max():.3f}, n_pairs={len(g)}."
            )
            lines.append("")
            lines.append("| source | target | macro-F1 |")
            lines.append("|---|---|---:|")
            for _, r in g.sort_values("macro_f1").head(15).iterrows():
                lines.append(f"| {r.source} | {r.target} | {r.macro_f1:.3f} |")
            lines.append("")
            lines.append("_Lowest 15 pairs shown; full matrix in `artifacts/city_transfer_f1_lr64_k64.csv`._")
            lines.append("")
    else:
        lines.append("## Cross-country city-transfer averages (lr64, off-diagonal city pairs)")
        lines.append("")
        lines.append("| Source country | Target country | Mean F1 | Median F1 | Min | Max | n pairs |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for src in sorted(city_lr.source_country.dropna().unique()):
            for tgt in sorted(city_lr.target_country.dropna().unique()):
                g = city_lr[
                    (city_lr.source_country == src)
                    & (city_lr.target_country == tgt)
                    & (city_lr.source != city_lr.target)
                ]
                if g.empty:
                    continue
                lines.append(
                    f"| {src} | {tgt} | {g.macro_f1.mean():.3f} | {g.macro_f1.median():.3f} | "
                    f"{g.macro_f1.min():.3f} | {g.macro_f1.max():.3f} | {len(g)} |"
                )
        lines.append("")

    # Narrative takeaways (skip cross-country bullets when extract is single-country)
    def f1(src, tgt):
        r = country_df[
            (country_df.model == "lr64")
            & (country_df.source_country == src)
            & (country_df.target_country == tgt)
        ]
        return float(r.macro_f1.iloc[0]) if len(r) else float("nan")

    def loro_mean(c: str) -> float:
        if c not in loro or loro[c].empty:
            return float("nan")
        return float(loro[c].loc[loro[c].model == "lr64", "macro_f1"].mean())

    present = sorted(loro.keys())
    lines.append("## Takeaways")
    lines.append("")
    if len(present) == 1 and present[0] in ("IND", "JPN"):
        cc = present[0]
        cname = "India" if cc == "IND" else "Japan"
        prior = (
            "prior 5-city India LORO (~0.44)"
            if cc == "IND"
            else "prior 5-city Japan LORO"
        )
        loro_c = loro_mean(cc)
        within = f1(cc, cc)
        per = loro[cc].loc[loro[cc].model == "lr64"].sort_values("macro_f1")
        worst = per.iloc[0]
        best = per.iloc[-1]
        lines.append(
            f"1. **{cname} 10-city LORO:** mean lr64 macro-F1 = **{loro_c:.3f}** "
            f"(within-country 80/20 split {within:.3f}). "
            f"Compare to {prior} and USA week4 LORO (~0.78–0.86)."
        )
        lines.append(
            f"2. **Hardest / easiest holdouts:** {worst['held_out']} F1={worst['macro_f1']:.3f}; "
            f"{best['held_out']} F1={best['macro_f1']:.3f}."
        )
        lines.append(
            "3. **Next:** if mean LORO remains ≪ USA, inspect per-city class balance / AOI coverage "
            "and consider city-conditioned heads before cross-country claims."
        )
    else:
        usa_ind, usa_jpn = f1("USA", "IND"), f1("USA", "JPN")
        ind_usa, jpn_usa = f1("IND", "USA"), f1("JPN", "USA")
        ind_jpn, jpn_ind = f1("IND", "JPN"), f1("JPN", "IND")
        usa_usa, ind_ind, jpn_jpn = f1("USA", "USA"), f1("IND", "IND"), f1("JPN", "JPN")
        loro_usa, loro_ind, loro_jpn = loro_mean("USA"), loro_mean("IND"), loro_mean("JPN")
        lines.append(
            f"1. **Within-country geographic sharing:** LORO mean macro-F1 is "
            f"USA {loro_usa:.3f}, India {loro_ind:.3f}, Japan {loro_jpn:.3f}."
        )
        lines.append(
            f"2. **USA → Asia transfer:** USA→India {usa_ind:.3f}, USA→Japan {usa_jpn:.3f} "
            f"(vs USA within-split {usa_usa:.3f})."
        )
        lines.append(
            f"3. **Asia → USA:** India→USA {ind_usa:.3f}, Japan→USA {jpn_usa:.3f}."
        )
        lines.append(
            f"4. **India ↔ Japan:** India→Japan {ind_jpn:.3f}, Japan→India {jpn_ind:.3f} "
            f"(India within {ind_ind:.3f}, Japan within {jpn_jpn:.3f})."
        )
    lines.append("")
    lines.append(f"_Generated {now()}_")
    text = "\n".join(lines) + "\n"
    (run_dir / "RESULTS.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 4))
    ap.add_argument("--skip-city-transfer", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "failures").mkdir(exist_ok=True)
    log_path = run_dir / "job.log"

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        write_progress(run_dir, status="running", stage="load_extract", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        X = np.load(args.extract_dir / "X.npy", mmap_mode="r")
        y = np.load(args.extract_dir / "y.npy")
        regions = np.load(args.extract_dir / "regions.npy", allow_pickle=True).astype(str)
        countries = np.load(args.extract_dir / "countries.npy", allow_pickle=True).astype(str)
        if X.shape[1] != 64 or len(y) != len(X):
            raise RuntimeError(f"bad extract shapes X={X.shape} y={y.shape}")
        cities = sorted(np.unique(regions).tolist())
        _STATE.update(
            {
                "X": X,
                "y": y.astype(np.int64),
                "regions": regions,
                "countries": countries,
                "cities": cities,
            }
        )
        log(log_path, f"loaded n={len(y)} cities={len(cities)} countries={sorted(set(countries))}")
        t0 = time.time()

        loro = {}
        for country in ("USA", "IND", "JPN"):
            if country not in set(countries):
                continue
            write_progress(run_dir, stage=f"loro_{country}", status="running")
            loro[country] = run_loro(run_dir, log_path, country, args.workers)

        write_progress(run_dir, stage="country_transfer", status="running")
        country_df = run_country_transfer(run_dir, log_path)

        if args.skip_city_transfer:
            log(log_path, "skipping city transfer — using country matrix for cross-country averages")
            city_df = country_df.rename(
                columns={"source_country": "source", "target_country": "target"}
            ).copy()
            city_df["source_country"] = city_df["source"]
            city_df["target_country"] = city_df["target"]
        else:
            write_progress(run_dir, stage="city_transfer", status="running")
            city_df = run_city_transfer(run_dir, log_path, args.workers)

        text = write_results_md(run_dir, loro, country_df, city_df)

        summary = {
            "status": "completed",
            "runtime_seconds": round(time.time() - t0, 1),
            "n_samples": int(len(y)),
            "n_cities": len(cities),
            "extract_dir": str(args.extract_dir),
            "artifacts": str(run_dir / "artifacts"),
            "results_md": str(run_dir / "RESULTS.md"),
            "loro_lr64_mean": {
                c: float(df.loc[df.model == "lr64", "macro_f1"].mean()) for c, df in loro.items()
            },
        }
        # country transfer lr64 matrix into summary
        g = country_df[country_df.model == "lr64"]
        summary["country_transfer_lr64"] = {
            f"{r.source_country}->{r.target_country}": float(r.macro_f1) for r in g.itertuples()
        }
        atomic_json(run_dir / "summary.json", summary)
        (run_dir / "DONE").write_text("ok\n")
        write_progress(run_dir, status="completed", stage="done", percent_complete=100.0)
        log(log_path, "DONE")
        print(text)
        return 0
    except Exception:
        tb = traceback.format_exc()
        (run_dir / "FAILED").write_text(tb)
        (run_dir / "failures" / "job.txt").write_text(tb)
        write_progress(run_dir, status="failed", stage="failed")
        log(log_path, "FAILED\n" + tb)
        raise


if __name__ == "__main__":
    # Linux fork inherits read-only mmap + _STATE (same pattern as week4 geo_subspace scripts).
    raise SystemExit(main())
