#!/usr/bin/env python3
"""Fit leave-one-city-out LR + PLS fold joblibs for globe / boundary walks.

Trains on the balanced international extract (X.npy / y.npy / regions.npy),
one fold per held-out city, in parallel. Writes joblibs with:

  mu, sd
  baseline_lr_w / baseline_lr_b          (64-D LR)
  pls3_lr_w / pls3_lr_b                  (LR on PLS dims 1–3)
  pls_da.x_rotations                     (64 × n_pls)

Default extract:
  week4/international_geo_generalization/runs/intl_transfer_20261005_142709/extract
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACT = ROOT / "runs" / "intl_transfer_20261005_142709" / "extract"
DEFAULT_OUT = (
    ROOT / "viz" / "usa_embedding_globe" / "data" / "intl_folds"
)

USA_CITIES = [
    "USA_ATLANTA",
    "USA_BOSTON",
    "USA_CHICAGO",
    "USA_DENVER",
    "USA_HOUSTON",
    "USA_LOS_ANGELES",
    "USA_NEW_YORK",
    "USA_PHOENIX",
    "USA_SAN_FRANCISCO",
    "USA_SEATTLE",
]
IND_CITIES = [
    "IND_AHMEDABAD",
    "IND_BENGALURU",
    "IND_CHENNAI",
    "IND_DELHI",
    "IND_HYDERABAD",
    "IND_JAIPUR",
    "IND_KOLKATA",
    "IND_LUCKNOW",
    "IND_MUMBAI",
    "IND_PUNE",
]
JPN_CITIES = [
    "JPN_FUKUOKA",
    "JPN_HIROSHIMA",
    "JPN_KOBE",
    "JPN_KYOTO",
    "JPN_NAGOYA",
    "JPN_OSAKA",
    "JPN_SAPPORO",
    "JPN_SENDAI",
    "JPN_TOKYO",
    "JPN_YOKOHAMA",
]
ALL_INTL = IND_CITIES + JPN_CITIES
ALL_CITIES = USA_CITIES + ALL_INTL
PLS3_DIMS = 3


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def pls_rotations(X: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    """Same NIPALS-style PLS as DRIVE international_analysis."""
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


def _fold_has_pls3(path: Path) -> bool:
    try:
        fold = joblib.load(path)
        w = fold.get("pls3_lr_w")
        return w is not None and np.asarray(w).reshape(-1).shape[0] >= PLS3_DIMS
    except Exception:
        return False


def _fit_one_fold(args: tuple) -> dict:
    extract_dir, held_out, out_path, n_pls, seed, force = args
    extract_dir = Path(extract_dir)
    out_path = Path(out_path)
    t0 = time.time()
    if (
        (not force)
        and out_path.exists()
        and out_path.stat().st_size > 1000
        and _fold_has_pls3(out_path)
    ):
        return {
            "held_out": held_out,
            "status": "skipped",
            "path": str(out_path),
            "seconds": round(time.time() - t0, 2),
        }

    X = np.load(extract_dir / "X.npy", mmap_mode="r")
    y = np.load(extract_dir / "y.npy")
    regions = np.load(extract_dir / "regions.npy", allow_pickle=True).astype(str)
    countries = np.load(extract_dir / "countries.npy", allow_pickle=True).astype(str)

    country = held_out.split("_", 1)[0]
    mask_c = countries == country
    te = mask_c & (regions == held_out)
    tr = mask_c & (regions != held_out)
    n_tr = int(tr.sum())
    n_te = int(te.sum())
    if n_tr < 100 or n_te < 10:
        raise RuntimeError(f"{held_out}: n_tr={n_tr} n_te={n_te} too small")

    # Materialize train only (test not needed for fold weights).
    X_tr = np.asarray(X[tr], dtype=np.float64)
    y_tr = np.asarray(y[tr], dtype=np.int64)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    clf64 = LogisticRegression(
        solver="newton-cholesky",
        C=1.0,
        max_iter=200,
        random_state=seed,
    )
    clf64.fit(X_tr_s, y_tr)
    R = pls_rotations(X_tr_s, y_tr, n_components=n_pls)
    T3 = X_tr_s @ R[:, :PLS3_DIMS]
    clf3 = LogisticRegression(
        solver="newton-cholesky",
        C=1.0,
        max_iter=200,
        random_state=seed,
    )
    clf3.fit(T3, y_tr)

    mu = scaler.mean_.astype(np.float64).reshape(1, -1)
    sd = scaler.scale_.astype(np.float64).reshape(1, -1)
    w = np.asarray(clf64.coef_, dtype=np.float64).reshape(-1)
    b = float(np.asarray(clf64.intercept_).reshape(-1)[0])
    w3 = np.asarray(clf3.coef_, dtype=np.float64).reshape(-1)
    b3 = float(np.asarray(clf3.intercept_).reshape(-1)[0])

    payload = {
        "method": "intl_loro_lr64_pls3",
        "country": country,
        "region_held_out": held_out,
        "held_out_city": held_out,
        "n_train": n_tr,
        "n_test": n_te,
        "n_comps": int(R.shape[1]),
        "pls3_dims": PLS3_DIMS,
        "seed": seed,
        "mu": mu,
        "sd": sd,
        "baseline_lr_w": w,
        "baseline_lr_b": b,
        "pls3_lr_w": w3,
        "pls3_lr_b": b3,
        "pls_da": {
            "x_rotations": np.asarray(R, dtype=np.float64),
        },
        "extract_dir": str(extract_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.joblib")
    joblib.dump(payload, tmp, compress=3)
    tmp.replace(out_path)
    return {
        "held_out": held_out,
        "status": "ok",
        "path": str(out_path),
        "n_train": n_tr,
        "n_test": n_te,
        "seconds": round(time.time() - t0, 2),
    }


def fold_path(out_dir: Path, city_id: str) -> Path:
    return Path(out_dir) / f"fold_{city_id}.joblib"


def ensure_intl_folds(
    cities: list[str] | None = None,
    extract_dir: Path | None = None,
    out_dir: Path | None = None,
    n_pls: int = 32,
    seed: int = 42,
    workers: int | None = None,
    force: bool = False,
    progress_cb=None,
) -> dict:
    extract_dir = Path(extract_dir or DEFAULT_EXTRACT)
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    cities = list(cities or ALL_CITIES)
    workers = int(workers or min(len(cities), os.cpu_count() or 8))

    for p in ("X.npy", "y.npy", "regions.npy", "countries.npy"):
        if not (extract_dir / p).exists():
            raise FileNotFoundError(f"missing extract file: {extract_dir / p}")

    jobs = [
        (str(extract_dir), city, str(fold_path(out_dir, city)), n_pls, seed, force)
        for city in cities
    ]
    results = []
    t0 = time.time()
    print(
        f"fitting {len(jobs)} intl LORO folds workers={workers} extract={extract_dir}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fit_one_fold, j): j[1] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            city = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"held_out": city, "status": "failed", "error": str(e)}
                print(f"  FAIL {city}: {e}", flush=True)
            else:
                print(
                    f"  [{i}/{len(jobs)}] {res['held_out']} {res['status']} "
                    f"{res.get('seconds', '?')}s n_train={res.get('n_train')}",
                    flush=True,
                )
            results.append(res)
            if progress_cb:
                progress_cb(i, len(jobs), res)

    failed = [r for r in results if r.get("status") == "failed"]
    summary = {
        "status": "completed" if not failed else "completed_with_failures",
        "n_folds": len(results),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "n_failed": len(failed),
        "extract_dir": str(extract_dir),
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - t0, 1),
        "folds": results,
    }
    atomic_json(out_dir / "manifest.json", summary)
    return summary


def score_with_fold_dir(
    X: np.ndarray,
    y: np.ndarray,
    city_ids: np.ndarray,
    fold_dir: Path,
    city_to_fold: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Score rows with per-city fold joblibs. Returns pred, proba, pls_xyz (3-D), summary."""
    from sklearn.metrics import accuracy_score, f1_score

    fold_dir = Path(fold_dir)
    n = len(y)
    pred = np.full(n, -1, dtype=np.int64)
    proba = np.full(n, np.nan, dtype=np.float64)
    xyz = np.full((n, 3), np.nan, dtype=np.float64)
    fold_rows = []
    missing = []

    by_city: dict[str, list[int]] = {}
    for i, cid in enumerate(city_ids):
        by_city.setdefault(str(cid), []).append(i)

    for city, idxs in sorted(by_city.items()):
        fold_key = (city_to_fold or {}).get(city, city)
        path = fold_dir / f"fold_{fold_key}.joblib"
        if not path.exists():
            missing.append(city)
            continue
        fold = joblib.load(path)
        mu = np.asarray(fold["mu"], dtype=np.float64).reshape(1, -1)
        sd = np.asarray(fold["sd"], dtype=np.float64).reshape(1, -1)
        w = np.asarray(fold["baseline_lr_w"], dtype=np.float64).reshape(-1)
        b = float(fold["baseline_lr_b"])
        R = np.asarray(fold["pls_da"]["x_rotations"], dtype=np.float64)[:, :3]
        idx = np.asarray(idxs, dtype=np.int64)
        Xs = (X[idx] - mu) / np.clip(sd, 1e-6, None)
        logits = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        yhat = (p >= 0.5).astype(np.int64)
        pred[idx] = yhat
        proba[idx] = p
        xyz[idx] = Xs @ R
        fold_rows.append(
            {
                "held_out_city": city,
                "n_test": int(len(idx)),
                "accuracy": float(accuracy_score(y[idx], yhat)),
                "macro_f1": float(f1_score(y[idx], yhat, average="macro", zero_division=0)),
                "checkpoint": str(path),
            }
        )

    if missing:
        raise RuntimeError(f"missing intl folds for: {sorted(set(missing))}")

    ok = pred >= 0
    summary = {
        "method": "intl_loro_lr64",
        "label_definition": "resi_def_v1_pixel_frac_v1_balanced_pure",
        "threshold": 0.5,
        "checkpoint_dir": str(fold_dir),
        "n_labeled": int(n),
        "n_scored": int(ok.sum()),
        "accuracy": float(accuracy_score(y[ok], pred[ok])) if ok.any() else None,
        "macro_f1": float(f1_score(y[ok], pred[ok], average="macro", zero_division=0))
        if ok.any()
        else None,
        "folds": fold_rows,
    }
    return pred, proba, xyz, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=str, default=str(DEFAULT_EXTRACT))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--countries", nargs="+", default=["USA", "IND", "JPN"])
    ap.add_argument("--n-pls", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cities = []
    for c in args.countries:
        c = c.upper()
        if c == "USA":
            cities.extend(USA_CITIES)
        elif c == "IND":
            cities.extend(IND_CITIES)
        elif c == "JPN":
            cities.extend(JPN_CITIES)
        else:
            raise SystemExit(f"unsupported country for intl folds: {c}")

    summary = ensure_intl_folds(
        cities=cities,
        extract_dir=Path(args.extract_dir),
        out_dir=Path(args.out_dir),
        n_pls=args.n_pls,
        seed=args.seed,
        workers=args.workers or None,
        force=args.force,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
