#!/usr/bin/env python3
"""Stage 4: does subspace alignment track transfer better than geography or raw AEF shift?

Inputs are saved matrices from stages 2 and 3. Metro-level N is 10; p-values are
descriptive. Pixel-level metro probes are a separate diagnostic and are not the
geographic unit.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import lib

PAIRS_NOTE = (
    "Correlations use the 45 unordered metro pairs. Permutations shuffle metro labels. "
    "Ten metros is the geographic sample size; pair p-values are not independent evidence."
)


def _upper(mat: pd.DataFrame) -> pd.Series:
    names = list(mat.index)
    data = {}
    for a, b in combinations(names, 2):
        data[tuple(sorted((a, b)))] = float(mat.loc[a, b])
    return pd.Series(data)


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "n_pairs": int(len(x)),
    }


def _permute_spearman(x: np.ndarray, y: np.ndarray, n: int = 5000) -> dict:
    rng = np.random.default_rng(lib.SEED)
    rho0 = float(stats.spearmanr(x, y).statistic)
    count = 0
    boots = []
    for i in range(n):
        yp = rng.permutation(y)
        rho = float(stats.spearmanr(x, yp).statistic)
        count += abs(rho) >= abs(rho0)
        if i < 2000:
            # pair bootstrap: resamples pairs, which ignores metro dependence
            take = rng.integers(0, len(x), len(x))
            boots.append(float(stats.spearmanr(x[take], y[take]).statistic))
    boots = np.asarray(boots)
    return {
        "spearman_rho": rho0,
        "permutation_p": float((count + 1) / (n + 1)),
        "pair_bootstrap_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_perm": n,
    }


def _partial(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> float:
    def resid(a):
        A = np.column_stack([np.ones(len(a)), Z])
        coef, _, _, _ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(stats.pearsonr(rx, ry).statistic)


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _centroids(metros: list[str]) -> pd.DataFrame:
    cols = pd.read_parquet(lib.PIXELS, columns=["region", "pixel_lat", "pixel_lon"])
    rows = []
    for region, g in cols.groupby("region"):
        if region not in metros:
            continue
        rows.append(
            {
                "region": region,
                "lat": float(g["pixel_lat"].median()),
                "lon": float(g["pixel_lon"].median()),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows).set_index("region").loc[metros]


def _distribution_distances(X, y, regions, metros: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(lib.SEED)
    samples = []
    for region in metros:
        idx = np.where(regions == region)[0]
        take = rng.choice(idx, size=min(2000, len(idx)), replace=False)
        samples.append(np.asarray(X[take], dtype=np.float64))
    pooled = np.concatenate(samples)
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-6)
    Z = [(s - mu) / sd for s in samples]
    means = np.stack([z.mean(0) for z in Z])
    n = len(metros)
    mean_d = np.zeros((n, n))
    mmd = np.zeros((n, n))
    sw = np.zeros((n, n))
    # RBF bandwidth from a subset of pooled pairwise distances
    sub = pooled[rng.choice(len(pooled), size=min(800, len(pooled)), replace=False)]
    sub = (sub - mu) / sd
    s2 = np.einsum("ij,ij->i", sub, sub)
    d2 = np.maximum(s2[:, None] + s2[None, :] - 2.0 * (sub @ sub.T), 0.0)
    med = np.median(d2[np.triu_indices(len(sub), 1)])
    gamma = 1.0 / max(med, 1e-6)

    def _sq(u, v):
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u v^T, without a (n, n, p) tensor.
        u2 = np.einsum("ij,ij->i", u, u)
        v2 = np.einsum("ij,ij->i", v, v)
        return np.maximum(u2[:, None] + v2[None, :] - 2.0 * (u @ v.T), 0.0)

    def _mmd(a, b):
        def k(u, v):
            return np.exp(-gamma * _sq(u, v))
        m = len(a)
        ka = k(a, a)
        kb = k(b, b)
        kab = k(a, b)
        np.fill_diagonal(ka, 0)
        np.fill_diagonal(kb, 0)
        return float(ka.sum() / (m * (m - 1)) + kb.sum() / (m * (m - 1)) - 2 * kab.mean())

    dirs = rng.normal(size=(32, Z[0].shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def _sw(a, b):
        acc = 0.0
        for d in dirs:
            pa, pb = np.sort(a @ d), np.sort(b @ d)
            acc += np.mean(np.abs(pa - pb))
        return float(acc / len(dirs))

    for i in range(n):
        for j in range(i + 1, n):
            mean_d[i, j] = mean_d[j, i] = float(np.linalg.norm(means[i] - means[j]))
            mmd[i, j] = mmd[j, i] = _mmd(Z[i], Z[j])
            sw[i, j] = sw[j, i] = _sw(Z[i], Z[j])
    def frame(arr):
        return pd.DataFrame(arr, index=metros, columns=metros)
    return frame(mean_d), frame(mmd), frame(sw)


def _metro_probe(X, y, regions, metros) -> pd.DataFrame:
    rng = np.random.default_rng(lib.SEED)
    idx = []
    for region in metros:
        for lab in (0, 1):
            pool = np.where((regions == region) & (y == lab))[0]
            idx.append(rng.choice(pool, size=min(2000, len(pool)), replace=False))
    sel = np.concatenate(idx)
    Xs = np.asarray(X[sel], dtype=np.float64)
    ys = y[sel]
    rs = regions[sel]
    tr, te = train_test_split(np.arange(len(sel)), test_size=0.2, random_state=lib.SEED, stratify=rs)
    rows = []
    Xtr, Xte = Xs[tr], Xs[te]
    scaler_m = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler_m.transform(Xtr), scaler_m.transform(Xte)
    pca = lib.pca_basis(Xtr_s)
    pls = lib.pls_rotations(Xtr_s, ys[tr], n_components=32)
    reps = {"raw64": (Xtr_s, Xte_s)}
    for k in (2, 9, 32):
        reps[f"pca{k}"] = (Xtr_s @ pca[:, :k], Xte_s @ pca[:, :k])
        reps[f"pls{k}"] = (Xtr_s @ pls[:, :k], Xte_s @ pls[:, :k])
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder().fit(rs)
    for name, (A, B) in reps.items():
        # building function
        pred, proba, _ = lib.fit_lr(A, ys[tr], B)
        b = lib.binary_metrics(ys[te], pred, proba)
        # metro identity — multinomial on the same rows
        clf = __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
            solver="lbfgs", max_iter=400, random_state=lib.SEED
        )
        clf.fit(A, enc.transform(rs[tr]))
        metro_acc = float((clf.predict(B) == enc.transform(rs[te])).mean())
        rows.append(
            {
                "representation": name,
                "building_macro_f1": b["macro_f1"],
                "building_auroc": b["auroc"],
                "metro_accuracy": metro_acc,
                "n_test": int(len(te)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--stage3", required=True)
    args = ap.parse_args()
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "statistical_tests",
            "stage2": args.stage2,
            "stage3": args.stage3,
            "command": " ".join(sys.argv),
            "gpus": 0,
            "note": PAIRS_NOTE,
        },
    )
    log_path = run_dir / "train.log"
    try:
        t0 = time.time()
        s2, s3 = Path(args.stage2), Path(args.stage3)
        transfer = pd.read_csv(s3 / "transfer_f1_lr64_k64.csv", index_col=0)
        metros = list(transfer.index)
        diag = np.diag(transfer.values).copy()
        if np.any(diag <= 0):
            raise AssertionError("non-positive diagonal; retention undefined")
        retention = transfer.values / diag[None, :]
        # R[A,B] = T[A,B] / T[B,B]; columns are targets so divide by target diagonal
        R = pd.DataFrame(retention, index=metros, columns=metros)
        R.to_csv(run_dir / "transfer_retention_lr64.csv")
        sym = (R.values + R.values.T) / 2
        np.fill_diagonal(sym, 1.0)
        Rsym = pd.DataFrame(sym, index=metros, columns=metros)
        Rsym.to_csv(run_dir / "transfer_retention_symmetric_lr64.csv")
        lib.log(log_path, "retention matrices written")

        X, y, regions = lib.load_arrays()
        lib.log(log_path, "centroids")
        cent = _centroids(metros)
        cent.to_csv(run_dir / "metro_centroids.csv")
        geo = np.zeros((len(metros), len(metros)))
        for i, a in enumerate(metros):
            for j, b in enumerate(metros):
                if i < j:
                    geo[i, j] = geo[j, i] = _haversine(cent.loc[a, "lat"], cent.loc[a, "lon"], cent.loc[b, "lat"], cent.loc[b, "lon"])
        geo_df = pd.DataFrame(geo, index=metros, columns=metros)
        geo_df.to_csv(run_dir / "geo_distance_km.csv")

        lib.log(log_path, "distribution distances on 2000 pixels per metro")
        mean_d, mmd, sw = _distribution_distances(X, y, regions, metros)
        mean_d.to_csv(run_dir / "aef_mean_distance.csv")
        mmd.to_csv(run_dir / "aef_mmd.csv")
        sw.to_csv(run_dir / "aef_sliced_wasserstein.csv")

        cosine = pd.read_csv(s2 / "class_direction_cosine.csv", index_col=0).loc[metros, metros]
        y_sym = _upper(Rsym)
        predictors = {
            "class_direction_cosine": _upper(cosine),
            "geo_distance_km": _upper(geo_df),
            "aef_mean_distance": _upper(mean_d),
            "aef_mmd": _upper(mmd),
            "aef_sliced_wasserstein": _upper(sw),
        }
        for k in lib.K_SUBSPACE:
            path = s2 / f"projection_similarity_k{k}.csv"
            if path.exists():
                predictors[f"pls_similarity_k{k}"] = _upper(pd.read_csv(path, index_col=0).loc[metros, metros])
        keys = sorted(y_sym.index)
        yv = np.array([y_sym[k] for k in keys])
        rows = []
        pair_frame = {"pair": [f"{a}|{b}" for a, b in keys], "retention_sym": yv}
        for name, series in predictors.items():
            xv = np.array([series[k] for k in keys])
            pair_frame[name] = xv
            rec = {"predictor": name, **_corr(xv, yv), **_permute_spearman(xv, yv)}
            rows.append(rec)
            lib.log(log_path, f"{name} spearman={rec['spearman_rho']:.3f} perm_p={rec['permutation_p']:.4f}")
        # partial: PLS k=2 and k=9 controlling for MMD and geo distance
        for k in (2, 9):
            key = f"pls_similarity_k{k}"
            if key not in predictors:
                continue
            xv = np.array([predictors[key][kk] for kk in keys])
            Z = np.column_stack(
                [
                    np.array([predictors["aef_mmd"][kk] for kk in keys]),
                    np.array([predictors["geo_distance_km"][kk] for kk in keys]),
                ]
            )
            rows.append(
                {
                    "predictor": f"{key}_partial_given_mmd_and_geo",
                    "pearson_r": _partial(yv, xv, Z),
                    "pearson_p": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "n_pairs": len(keys),
                    "permutation_p": float("nan"),
                    "pair_bootstrap_ci95": [float("nan"), float("nan")],
                    "n_perm": 0,
                }
            )
        corr = pd.DataFrame(rows)
        lib.atomic_df(run_dir / "correlation_table.csv", corr)
        lib.atomic_df(run_dir / "pair_table.csv", pd.DataFrame(pair_frame))

        lib.log(log_path, "metro vs building probe on a stratified pixel subsample")
        probe = _metro_probe(X, y, regions, metros)
        lib.atomic_df(run_dir / "metro_vs_building_probe.csv", probe)
        lib.atomic_json(
            run_dir / "summary.json",
            {
                "status": "completed",
                "runtime_seconds": time.time() - t0,
                "n_metros": len(metros),
                "n_pairs": len(keys),
                "note": PAIRS_NOTE,
            },
        )
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
