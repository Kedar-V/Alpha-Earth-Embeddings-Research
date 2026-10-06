#!/usr/bin/env python3
"""Stage 5: linear maps of frozen AlphaEarth, with and without direction alignment.

The held-out metro is never used to fit the scaler, the map, the head, the epoch,
or lambda. Lambda is chosen on one source metro (the median-sized source metro),
then three fresh seeds are trained with that frozen lambda.

Deviation from a full inner LORO: one predetermined source metro is the validation
metro, chosen by sample count, not by validation score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import lib

WIDTHS = (2, 4, 8, 16)
LAMBDAS = (0.1, 1.0, 10.0)
METHODS = ("class", "supcon", "xsupcon", "align")
SEEDS = (0, 1, 2)


def _median_metro(counts: dict[str, int], metros: list[str]) -> str:
    ordered = sorted(metros, key=lambda m: (counts[m], m))
    return ordered[len(ordered) // 2]


def _pixel_split(y_local: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for lab in (0, 1):
        idx = np.where(y_local == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(0.2 * len(idx))))
        te.append(idx[:n_te])
        tr.append(idx[n_te:])
    train = np.concatenate(tr)
    test = np.concatenate(te)
    if np.intersect1d(train, test).size:
        raise AssertionError("within-metro pixel split overlaps")
    return train, test


def _supcon(z, y, region, cross: bool, tau: float = 0.1):
    import torch
    import torch.nn.functional as F

    z = F.normalize(z, dim=1)
    logits = z @ z.T / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    same = y[:, None] == y[None, :]
    if cross:
        pos = same & (region[:, None] != region[None, :]) & ~eye
    else:
        pos = same & ~eye
    denom = (torch.exp(logits) * (~eye)).sum(1).clamp_min(1e-8)
    log_prob = logits - torch.log(denom)
    npos = pos.sum(1)
    valid = npos > 0
    if int(valid.sum()) == 0:
        return z.new_zeros(())
    loss = -(log_prob * pos).sum(1) / npos.clamp_min(1)
    return loss[valid].mean()


def _align(h, y, region):
    import torch
    import torch.nn.functional as F

    deltas = []
    for r in torch.unique(region):
        m = region == r
        pos = h[m & (y == 1)]
        neg = h[m & (y == 0)]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            continue
        deltas.append(F.normalize(pos.mean(0) - neg.mean(0), dim=0))
    if len(deltas) < 2:
        return h.new_zeros(())
    D = torch.stack(deltas, 0)
    sim = D @ D.T
    iu = torch.triu_indices(D.shape[0], D.shape[0], offset=1, device=h.device)
    return (1.0 - sim[iu[0], iu[1]]).mean()


def _f1_from_logits(y, logits) -> dict:
    proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    pred = (proba >= 0.5).astype(np.int64)
    return lib.binary_metrics(y, pred, proba)


def _direction(h, y) -> np.ndarray:
    d = h[y == 1].mean(0) - h[y == 0].mean(0)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.zeros_like(d)
    return d / n


def _gap(h, y) -> float:
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        return float("nan")
    c1 = h[y == 1].mean(0)
    c0 = h[y == 0].mean(0)
    sep = float(np.linalg.norm(c1 - c0))
    r1 = float(np.linalg.norm(h[y == 1] - c1, axis=1).mean())
    r0 = float(np.linalg.norm(h[y == 0] - c0, axis=1).mean())
    return sep / max(0.5 * (r1 + r0), 1e-8)


def _run_task(task: dict) -> dict:
    import torch
    import torch.nn as nn

    gpu = int(task["gpu"])
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(int(task["seed"]))
    np.random.seed(int(task["seed"]))

    X, y, regions = lib.load_arrays()
    metros = lib.metro_list(regions)
    held = task["held_out"]
    if held not in metros:
        raise AssertionError(f"unknown metro {held}")
    sources = [m for m in metros if m != held]
    counts = {m: int((regions == m).sum()) for m in sources}
    val_metro = _median_metro(counts, sources)
    if val_metro == held:
        raise AssertionError("validation metro is the held-out metro")
    train_metros = [m for m in sources if m != val_metro]

    train_idx = []
    within_idx = []
    within_y = []
    within_region = []
    for m in train_metros:
        local = np.where(regions == m)[0]
        tr, te = _pixel_split(y[local], lib.SEED)
        train_idx.append(local[tr])
        within_idx.append(local[te])
        within_y.append(y[local[te]])
        within_region.append(np.full(len(te), m))
    train_idx = np.concatenate(train_idx)
    within_idx = np.concatenate(within_idx)
    within_y = np.concatenate(within_y)
    if held in set(regions[train_idx].tolist()) or val_metro in set(regions[train_idx].tolist()):
        raise AssertionError("train rows include held-out or validation metro")

    if task.get("smoke"):
        rng = np.random.default_rng(lib.SEED)
        train_idx = rng.choice(train_idx, size=min(8000, len(train_idx)), replace=False)

    X_tr = np.asarray(X[train_idx], dtype=np.float32)
    y_tr = y[train_idx]
    r_tr = regions[train_idx]
    mu = X_tr.mean(0)
    sd = np.maximum(X_tr.std(0), 1e-6)
    X_tr = (X_tr - mu) / sd

    code = {m: i for i, m in enumerate(train_metros)}
    by = {}
    for m in train_metros:
        for lab in (0, 1):
            by[(m, lab)] = np.where((r_tr == m) & (y_tr == lab))[0]
            if len(by[(m, lab)]) < 8:
                raise AssertionError(f"too few train rows for {m} class {lab}")

    d = int(task["d"])
    method = task["method"]
    lam = float(task["lambda"])
    max_epochs = int(task["max_epochs"])
    proj = nn.Linear(64, d).to(device)
    head = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(list(proj.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    rt = torch.from_numpy(np.array([code[m] for m in r_tr], dtype=np.int64)).to(device)

    val_i = np.where(regions == val_metro)[0]
    te_i = np.where(regions == held)[0]
    if task.get("smoke"):
        rng = np.random.default_rng(lib.SEED + 1)
        val_i = rng.choice(val_i, size=min(2000, len(val_i)), replace=False)
        te_i = rng.choice(te_i, size=min(2000, len(te_i)), replace=False)
    Xv = torch.from_numpy(((np.asarray(X[val_i], dtype=np.float32) - mu) / sd)).to(device)
    yv = y[val_i]
    Xt_te = torch.from_numpy(((np.asarray(X[te_i], dtype=np.float32) - mu) / sd)).to(device)
    y_te = y[te_i]

    def _eval(Xg, y_np):
        proj.eval()
        head.eval()
        with torch.no_grad():
            logits = head(proj(Xg)).squeeze(1).float().cpu().numpy()
        proj.train()
        head.train()
        return _f1_from_logits(y_np, logits), logits

    history = []
    best_bce = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    rng = np.random.default_rng(int(task["seed"]))
    per_class = 48 if task.get("smoke") else 128
    n_reg = min(4, len(train_metros))
    # A linear map does not need a full pass over 1e6 rows every epoch.
    steps = 20 if task.get("smoke") else 80
    import torch.nn.functional as F

    for epoch in range(1, max_epochs + 1):
        proj.train()
        head.train()
        total = 0.0
        for _ in range(steps):
            chosen = rng.choice(train_metros, size=n_reg, replace=False)
            parts = []
            for m in chosen:
                for lab in (0, 1):
                    pool = by[(m, lab)]
                    take = rng.choice(pool, size=per_class, replace=len(pool) < per_class)
                    parts.append(take)
            ix = np.concatenate(parts)
            xb = Xt[torch.from_numpy(ix).to(device)]
            yb = yt[torch.from_numpy(ix).to(device)]
            rb = rt[torch.from_numpy(ix).to(device)]
            opt.zero_grad(set_to_none=True)
            h = proj(xb)
            logit = head(h).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, yb.float())
            if method == "supcon":
                loss = loss + _supcon(h, yb, rb, cross=False)
            elif method == "xsupcon":
                loss = loss + _supcon(h, yb, rb, cross=True)
            elif method == "align":
                loss = loss + lam * _align(h, yb, rb)
            elif method != "class":
                raise AssertionError(method)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        vm, _ = _eval(Xv, yv)
        with torch.no_grad():
            vlogit = head(proj(Xv)).squeeze(1)
            vbce = float(F.binary_cross_entropy_with_logits(vlogit, torch.from_numpy(yv.astype(np.float32)).to(device)))
        history.append({"epoch": epoch, "train_loss": total / steps, "val_bce": vbce, "val_macro_f1": vm["macro_f1"]})
        if vbce < best_bce - 1e-4:
            best_bce = vbce
            best_epoch = epoch
            patience = 0
            best_state = {
                "proj": {k: v.detach().cpu() for k, v in proj.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            }
        else:
            patience += 1
        if epoch >= 4 and patience >= 2 and not task.get("smoke"):
            break

    if best_state is None:
        raise AssertionError("no checkpoint")
    proj.load_state_dict(best_state["proj"])
    head.load_state_dict(best_state["head"])
    val_m, _ = _eval(Xv, yv)
    te_m, _ = _eval(Xt_te, y_te)
    Xw = torch.from_numpy(((np.asarray(X[within_idx], dtype=np.float32) - mu) / sd)).to(device)
    if task.get("smoke"):
        Xw = Xw[:2000]
        within_y = within_y[:2000]
    within_m, _ = _eval(Xw, within_y)

    # Geometry on a fixed subsample. Held-out rows are scored only, never fit.
    rng_g = np.random.default_rng(lib.SEED)
    geos = []
    dirs = {}
    for m in metros:
        pool = np.where(regions == m)[0]
        take = rng_g.choice(pool, size=min(2000, len(pool)), replace=False)
        Hs = torch.from_numpy(((np.asarray(X[take], dtype=np.float32) - mu) / sd)).to(device)
        with torch.no_grad():
            h = proj(Hs).cpu().numpy()
        dirs[m] = _direction(h, y[take])
        geos.append(_gap(h, y[take]))
    src_dirs = [dirs[m] for m in train_metros]
    cos_src = []
    for i in range(len(src_dirs)):
        for j in range(i + 1, len(src_dirs)):
            cos_src.append(float(np.dot(src_dirs[i], src_dirs[j])))
    held_cos = [float(np.dot(dirs[held], dirs[m])) for m in train_metros]

    # Metro identity among training metros only, on the within-metro pixel holdout.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    tr_m = regions[train_idx]
    # subsample for the probe
    probe_tr, probe_te = [], []
    rng_p = np.random.default_rng(lib.SEED)
    for m in train_metros:
        pool = np.where(r_tr == m)[0]
        pick = rng_p.choice(pool, size=min(800, len(pool)), replace=False)
        cut = int(0.8 * len(pick))
        probe_tr.append(pick[:cut])
        probe_te.append(pick[cut:])
    probe_tr = np.concatenate(probe_tr)
    probe_te = np.concatenate(probe_te)
    with torch.no_grad():
        hp_tr = proj(Xt[torch.from_numpy(probe_tr).to(device)]).cpu().numpy()
        hp_te = proj(Xt[torch.from_numpy(probe_te).to(device)]).cpu().numpy()
    enc = LabelEncoder().fit(r_tr[probe_tr])
    clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=lib.SEED)
    clf.fit(hp_tr, enc.transform(r_tr[probe_tr]))
    metro_acc = float((clf.predict(hp_te) == enc.transform(r_tr[probe_te])).mean())

    ckpt = {
        "proj": best_state["proj"],
        "head": best_state["head"],
        "d": d,
        "method": method,
        "seed": int(task["seed"]),
        "lambda": lam,
        "held_out": held,
        "val_metro": val_metro,
        "best_epoch": best_epoch,
        "best_val_bce": best_bce,
        "scaler_mean": mu,
        "scaler_scale": sd,
        "history": history,
        "arch": "Linear(64,d)+Linear(d,1)",
    }
    tag = f"{held}_{method}_d{d}_lam{lam:g}_seed{int(task['seed'])}"
    ckpt_path = Path(task["run_dir"]) / "checkpoints" / f"{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)

    row = {
        "held_out": held,
        "val_metro": val_metro,
        "method": method,
        "d": d,
        "lambda": lam,
        "seed": int(task["seed"]),
        "phase": task["phase"],
        "best_epoch": best_epoch,
        "best_val_bce": best_bce,
        "val_macro_f1": val_m["macro_f1"],
        "val_auroc": val_m["auroc"],
        "within_macro_f1": within_m["macro_f1"],
        "heldout_macro_f1": te_m["macro_f1"],
        "heldout_auroc": te_m["auroc"],
        "heldout_f1_residential": te_m["f1_residential"],
        "heldout_f1_nonresidential": te_m["f1_nonresidential"],
        "n_heldout": te_m["n"],
        "source_direction_cosine_mean": float(np.mean(cos_src)) if cos_src else float("nan"),
        "heldout_direction_cosine_mean": float(np.mean(held_cos)) if held_cos else float("nan"),
        "gap_mean_over_metros": float(np.nanmean(geos)),
        "metro_accuracy_source_pixels": metro_acc,
        "checkpoint": str(ckpt_path),
    }
    out = Path(task["run_dir"]) / "tasks" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    lib.atomic_json(out, row)
    lib.log(Path(task["run_dir"]) / "train.log", f"{task['phase']} {tag} heldout_f1={te_m['macro_f1']:.3f} val_f1={val_m['macro_f1']:.3f} epoch={best_epoch}")
    return row


def _worker(payload: dict) -> list[dict]:
    rows = []
    for task in payload["tasks"]:
        tag = f"{task['held_out']}_{task['method']}_d{task['d']}_lam{task['lambda']:g}_seed{int(task['seed'])}"
        dest = Path(task["run_dir"]) / "tasks" / f"{tag}.json"
        if dest.exists():
            rows.append(json.loads(dest.read_text()))
            continue
        try:
            rows.append(_run_task(task))
        except Exception:
            fail = Path(task["run_dir"]) / "tasks" / f"{tag}.FAILED"
            fail.parent.mkdir(parents=True, exist_ok=True)
            fail.write_text(traceback.format_exc())
            lib.log(Path(task["run_dir"]) / "train.log", f"FAILED {tag}\n" + traceback.format_exc())
            raise
    return rows


def _select_lambda(rows: list[dict]) -> dict[tuple, float]:
    chosen = {}
    df_keys = {(r["held_out"], int(r["d"])) for r in rows}
    for held, d in df_keys:
        sub = [r for r in rows if r["held_out"] == held and int(r["d"]) == d]
        # higher inner-val F1; tie -> smaller lambda
        sub = sorted(sub, key=lambda r: (-r["val_macro_f1"], r["lambda"]))
        chosen[(held, d)] = float(sub[0]["lambda"])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", type=int, default=0, help="0 means every visible GPU")
    args = ap.parse_args()
    import torch
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_gpu = torch.cuda.device_count()
    if n_gpu < 1:
        raise SystemExit("stage 5 needs a GPU")
    use = n_gpu if args.gpus <= 0 else min(args.gpus, n_gpu)
    names = [torch.cuda.get_device_name(i) for i in range(use)]
    run_dir = lib.init_run(
        Path(args.run_dir),
        {
            "experiment": "geo_projection",
            "command": " ".join(sys.argv),
            "gpus": use,
            "gpu_names": names,
            "widths": list(WIDTHS),
            "lambdas": list(LAMBDAS),
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "selection": "median-sized source metro by count; lambda maximizes that metro's macro-F1; epoch stops on that metro's BCE",
            "smoke": args.smoke,
        },
    )
    log_path = run_dir / "train.log"
    try:
        X, y, regions = lib.load_arrays()
        metros = lib.metro_list(regions)
        if args.smoke:
            metros = metros[:1]
        lib.log(log_path, f"GPUs={use} {names} metros={metros}")
        phase1 = []
        widths = (2,) if args.smoke else WIDTHS
        lambdas = (1.0,) if args.smoke else LAMBDAS
        for i, held in enumerate(metros):
            for d in widths:
                for lam in lambdas:
                    phase1.append(
                        {
                            "held_out": held,
                            "method": "align",
                            "d": d,
                            "lambda": lam,
                            "seed": 42,
                            "phase": "select",
                            "max_epochs": 2 if args.smoke else 8,
                            "gpu": i % use,
                            "run_dir": str(run_dir),
                            "smoke": args.smoke,
                        }
                    )
        if args.smoke:
            phase1 = phase1[:1]
            phase1[0]["method"] = "class"
            phase1[0]["lambda"] = 0.0

        def _shards(tasks):
            buckets = {g: [] for g in range(use)}
            for t in tasks:
                buckets[int(t["gpu"]) % use].append(t)
            return [{"tasks": buckets[g]} for g in range(use) if buckets[g]]

        ctx = torch.multiprocessing.get_context("spawn")
        select_rows = []
        with ProcessPoolExecutor(max_workers=use, mp_context=ctx) as pool:
            futs = [pool.submit(_worker, shard) for shard in _shards(phase1)]
            for fut in as_completed(futs):
                select_rows.extend(fut.result())
        lib.atomic_df(run_dir / "lambda_selection.csv", __import__("pandas").DataFrame(select_rows))
        if args.smoke:
            lib.atomic_json(run_dir / "summary.json", {"status": "smoke", "n": len(select_rows)})
            (run_dir / "DONE").write_text("smoke\n")
            lib.log(log_path, "smoke DONE")
            return

        chosen = _select_lambda(select_rows)
        lib.atomic_json(run_dir / "lambda_chosen.json", {f"{a}|{b}": v for (a, b), v in chosen.items()})
        phase2 = []
        for held in metros:
            for d in WIDTHS:
                lam = chosen[(held, d)]
                for method in METHODS:
                    for seed in SEEDS:
                        phase2.append(
                            {
                                "held_out": held,
                                "method": method,
                                "d": d,
                                "lambda": lam if method == "align" else 0.0,
                                "seed": seed,
                                "phase": "eval",
                                "max_epochs": 8,
                                "gpu": (SEEDS.index(seed) + WIDTHS.index(d)) % use,
                                "run_dir": str(run_dir),
                                "smoke": False,
                            }
                        )
        # Spread eval tasks across GPUs in order.
        for i, t in enumerate(phase2):
            t["gpu"] = i % use
        eval_rows = []
        with ProcessPoolExecutor(max_workers=use, mp_context=ctx) as pool:
            futs = [pool.submit(_worker, shard) for shard in _shards(phase2)]
            for fut in as_completed(futs):
                eval_rows.extend(fut.result())
        df = __import__("pandas").DataFrame(eval_rows)
        lib.atomic_df(run_dir / "metrics.csv", df)
        summary = {"status": "completed", "n_eval": int(len(df)), "gpus": use}
        by = df.groupby(["method", "d"])["heldout_macro_f1"].agg(["mean", "std"]).reset_index()
        summary["heldout_f1_by_method"] = by.to_dict(orient="records")
        lib.atomic_json(run_dir / "summary.json", summary)
        (run_dir / "DONE").write_text("ok\n")
        lib.log(log_path, "DONE")
    except Exception:
        (run_dir / "FAILED").write_text(traceback.format_exc())
        lib.log(log_path, "FAILED\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
