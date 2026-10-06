#!/usr/bin/env python3
"""Stage 2: within-metro PLS bases, principal angles, class-mean directions.

PLS for metro r uses only rows from r. A single unsupervised global z-score
(all pixels, no labels) puts every basis in one inner product. Class-mean
directions are also computed in the raw 64-D coordinates, as specified.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import lib

_STATE: dict = {}


def _one_metro(region: str) -> dict:
    X = _STATE["X"]
    y = _STATE["y"]
    regions = _STATE["regions"]
    mu = _STATE["mu"]
    sd = _STATE["sd"]
    ks = _STATE["ks"]
    idx = np.where(regions == region)[0]
    if set(regions[idx].tolist()) != {region}:
        raise AssertionError("metro slice leaked")
    Xs = (np.asarray(X[idx], dtype=np.float64) - mu) / sd
    y_m = y[idx]
    raw = np.asarray(X[idx], dtype=np.float64)
    delta = raw[y_m == 1].mean(axis=0) - raw[y_m == 0].mean(axis=0)
    delta = delta / (np.linalg.norm(delta) + 1e-12)
    rot = lib.pls_rotations(Xs, y_m, n_components=max(ks))
    out = {"region": region, "delta": delta, "n": int(len(idx))}
    for k in ks:
        Q = lib.orthonormalize(rot[:, :k])
        out[f"Q_{k}"] = Q
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "subspace_analysis",
            "smoke": args.smoke,
            "workers": args.workers,
            "k": list(lib.K_SUBSPACE),
            "command": " ".join(sys.argv),
            "gpus": 0,
            "scaling": "unsupervised global z-score, then within-metro PLS on those coordinates",
        },
    )
    log_path = run_dir / "train.log"
    try:
        X, y, regions = lib.load_arrays()
        metros = lib.metro_list(regions)
        ks = list(lib.K_SUBSPACE)
        if args.smoke:
            rng = np.random.default_rng(lib.SEED)
            keep = []
            for region in metros[:3]:
                for lab in (0, 1):
                    idx = np.where((regions == region) & (y == lab))[0]
                    keep.append(rng.choice(idx, size=800, replace=False))
            sel = np.concatenate(keep)
            X = np.asarray(X[sel])
            y = y[sel]
            regions = regions[sel]
            metros = metros[:3]
            ks = [1, 2]
        mu = np.asarray(X, dtype=np.float64).mean(axis=0)
        sd = np.asarray(X, dtype=np.float64).std(axis=0)
        sd = np.maximum(sd, 1e-6)
        _STATE.update({"X": X, "y": y, "regions": regions, "mu": mu, "sd": sd, "ks": ks})
        lib.log(log_path, f"metros={metros} k={ks} workers={args.workers}")
        t0 = time.time()
        packs = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(metros))) as pool:
            futs = {pool.submit(_one_metro, region): region for region in metros}
            for fut in as_completed(futs):
                packs.append(fut.result())
                lib.log(log_path, f"fit {futs[fut]} ({len(packs)}/{len(metros)})")
        packs = sorted(packs, key=lambda d: d["region"])
        names = [p["region"] for p in packs]
        np.savez_compressed(
            run_dir / "class_directions.npz",
            regions=np.array(names),
            delta=np.stack([p["delta"] for p in packs]),
            mu=mu,
            sd=sd,
        )
        C = np.stack([p["delta"] for p in packs])
        cosine = np.clip(C @ C.T, -1, 1)
        pd.DataFrame(cosine, index=names, columns=names).to_csv(run_dir / "class_direction_cosine.csv")
        for k in ks:
            Qs = [p[f"Q_{k}"] for p in packs]
            np.savez_compressed(run_dir / f"bases_k{k}.npz", regions=np.array(names), **{names[i]: Qs[i] for i in range(len(names))})
            n = len(names)
            sim = np.eye(n)
            ang_first = np.zeros((n, n))
            ang_mean = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    ang = lib.principal_angles_deg(Qs[i], Qs[j])
                    s = lib.projection_similarity(Qs[i], Qs[j])
                    sim[i, j] = sim[j, i] = s
                    ang_first[i, j] = ang_first[j, i] = ang[0]
                    ang_mean[i, j] = ang_mean[j, i] = float(ang.mean())
            for arr, stem in (
                (sim, "projection_similarity"),
                (ang_first, "principal_angle_first_deg"),
                (ang_mean, "principal_angle_mean_deg"),
            ):
                pd.DataFrame(arr, index=names, columns=names).to_csv(run_dir / f"{stem}_k{k}.csv")
                np.save(run_dir / f"{stem}_k{k}.npy", arr)
            lib.log(log_path, f"k={k} mean off-diag similarity={sim[np.triu_indices(n, 1)].mean():.3f}")
        lib.atomic_json(
            run_dir / "summary.json",
            {"status": "completed", "runtime_seconds": time.time() - t0, "metros": names, "k": ks},
        )
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
