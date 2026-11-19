#!/usr/bin/env python3
"""Score GHS-BUILT-S on Africa city-LORO balanced cells; compare to AE Ridge/XGB OOF.

Reuses tile download + sampling from eval_ghsl_built_s_vs_ae_intl.py.
Default inputs: latest africa_city_loro_gpu run artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ghsl_built_s_vs_ae_intl as ghsl  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    prev["last_progress_at"] = utc_now()
    if "_t0" in prev:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        prev["elapsed_seconds"] = round(elapsed, 1)
        if "completed" in prev and "total" in prev and int(prev["total"]) > 0:
            done, total = int(prev["completed"]), int(prev["total"])
            rate = done / elapsed
            prev["rate_per_second"] = round(rate, 4)
            prev["percent_complete"] = round(100.0 * done / total, 2)
            prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--loro-dir",
        type=Path,
        default=ROOT / "outputs" / "africa_city_loro_gpu" / "20261113_125451",
    )
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    loro_dir = args.loro_dir
    bal_path = loro_dir / "artifacts" / "balanced_pure.parquet"
    oof_path = loro_dir / "artifacts" / "oof_predictions.parquet"
    if not bal_path.exists() or not oof_path.exists():
        raise SystemExit(f"missing LORO artifacts under {loro_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "africa_loro_vs_ghsl" / stamp)
    for sub in ("artifacts", "figures", "checkpoints", "failures", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
    t0 = time.time()
    write_progress(
        run_dir,
        status="running",
        stage="load",
        started_at=utc_now(),
        _t0=t0,
        completed=0,
        total=5,
        loro_dir=str(loro_dir),
        output_location=str(run_dir),
        resource_allocation={"cpus": os.cpu_count(), "gpus": 0},
    )
    log(run_dir, f"compare GHSL vs AE city-LORO · loro={loro_dir}")

    bal = pd.read_parquet(bal_path)
    oof = pd.read_parquet(oof_path)
    # oof.region == city_id
    if "region" in oof.columns and "city_id" not in oof.columns:
        oof = oof.rename(columns={"region": "city_id"})
    log(run_dir, f"loaded balanced n={len(bal):,} oof n={len(oof):,}")
    write_progress(run_dir, stage="tiles", completed=1)

    # Download GHSL tiles covering city bboxes
    tile_set: set[tuple[int, int]] = set()
    for city_id, g in bal.groupby("city_id"):
        tiles = ghsl.tiles_for_bbox(
            float(g.pixel_lon.min()),
            float(g.pixel_lat.min()),
            float(g.pixel_lon.max()),
            float(g.pixel_lat.max()),
        )
        tile_set.update(tiles)
        log(run_dir, f"tiles {city_id}: {tiles}")
    ghsl.download_tiles(run_dir, sorted(tile_set), workers=min(8, os.cpu_count() or 4))
    write_progress(run_dir, stage="sample_ghsl", completed=2, n_tiles=len(tile_set))

    # Sample BUILT-S at AE centers
    sample_df = bal[["pixel_id", "city_id", "pixel_lon", "pixel_lat"]].copy()
    workers = max(1, min(os.cpu_count() or 4, sample_df.city_id.nunique()))
    log(run_dir, f"sampling GHSL workers={workers}")
    raw = ghsl.sample_built_s(sample_df, workers=workers)
    fr = ghsl.raw_to_frac(raw)
    fr.to_parquet(run_dir / "artifacts" / "ghsl_samples.parquet", index=False)
    write_progress(run_dir, stage="score", completed=3)

    merged = oof.merge(
        fr[["pixel_id", "bs_frac_res", "bs_frac_nres", "bs_frac_empty", "built_s_valid"]],
        on="pixel_id",
        how="left",
    )
    # ensure GT cols
    for c in ("frac_res_full", "frac_nres_full"):
        if c not in merged.columns:
            raise SystemExit(f"oof missing {c}")
    if "pred_frac_res" not in merged.columns:
        raise SystemExit("oof missing XGB preds")
    if "ridge_frac_res" not in merged.columns:
        raise SystemExit("oof missing ridge preds")

    rows = []
    for level, keys in (("country", ["country"]), ("city", ["country", "city_id"])):
        for key, sub in merged.groupby(keys):
            if not isinstance(key, tuple):
                key = (key,)
            yt_r = sub["frac_res_full"].to_numpy()
            yt_n = sub["frac_nres_full"].to_numpy()
            for model, pr, pn in (
                ("ae_ridge_loro", "ridge_frac_res", "ridge_frac_nres"),
                ("ae_xgb_loro", "pred_frac_res", "pred_frac_nres"),
                ("ghsl_built_s", "bs_frac_res", "bs_frac_nres"),
            ):
                m = ghsl.mae_rn(yt_r, yt_n, sub[pr].to_numpy(), sub[pn].to_numpy())
                rec = {"level": level, "model": model, **m}
                for k, v in zip(keys, key):
                    rec[k] = v
                rows.append(rec)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / "artifacts" / "compare_metrics.csv", index=False)
    merged.to_parquet(run_dir / "artifacts" / "scored.parquet", index=False)

    country = metrics[metrics.level == "country"].copy()
    pivot = country.pivot_table(index="country", columns="model", values="mae_mean_res_nres")
    pivot.to_csv(run_dir / "artifacts" / "country_mae_pivot.csv")
    log(run_dir, "country MAE mean(res,nres):\n" + pivot.to_string())

    # figure
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    countries = sorted(pivot.index)
    x = np.arange(len(countries))
    w = 0.25
    models = [("ae_ridge_loro", "AE Ridge LORO"), ("ae_xgb_loro", "AE XGB LORO"), ("ghsl_built_s", "GHSL BUILT-S")]
    for i, (col, lab) in enumerate(models):
        vals = [float(pivot.loc[c, col]) if col in pivot.columns else np.nan for c in countries]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel("MAE mean(res, nres)")
    ax.set_title("Africa city-LORO · AE vs GHSL (same balanced pure cells)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = run_dir / "figures" / "ae_vs_ghsl_country.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    # RESULTS.md
    lines = [
        "# Africa city-LORO vs GHSL BUILT-S",
        "",
        f"Same balanced pure cells as `{loro_dir.name}` (EGY+NGA, 1:1 per city).",
        "AE = leave-one-city-out OOF; GHSL = BUILT-S sampled at AE 10 m centers (no train).",
        "",
        "## Country MAE mean(res, nres)",
        "",
        "| Country | AE Ridge LORO | AE XGB LORO | GHSL BUILT-S | winner |",
        "|---|---:|---:|---:|---|",
    ]
    for c in countries:
        r = float(pivot.loc[c, "ae_ridge_loro"])
        x = float(pivot.loc[c, "ae_xgb_loro"])
        g = float(pivot.loc[c, "ghsl_built_s"])
        win = min(("AE Ridge", r), ("AE XGB", x), ("GHSL", g), key=lambda t: t[1])[0]
        lines.append(f"| {c} | {r:.4f} | {x:.4f} | {g:.4f} | {win} |")

    city = metrics[metrics.level == "city"].copy()
    lines += ["", "## Per-city", "", "| Country | City | AE Ridge | AE XGB | GHSL |", "|---|---|---:|---:|---:|"]
    for (ctry, cid), g in city.groupby(["country", "city_id"]):
        def _v(model: str) -> float:
            s = g[g.model == model]
            return float(s.mae_mean_res_nres.iloc[0]) if len(s) else float("nan")

        lines.append(
            f"| {ctry} | {cid} | {_v('ae_ridge_loro'):.4f} | {_v('ae_xgb_loro'):.4f} | {_v('ghsl_built_s'):.4f} |"
        )
    lines += ["", f"![country]({fig_path.name})", ""]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")

    summary = {
        "status": "completed",
        "loro_dir": str(loro_dir),
        "n": int(len(merged)),
        "country_mae": {
            c: {
                "ae_ridge_loro": float(pivot.loc[c, "ae_ridge_loro"]),
                "ae_xgb_loro": float(pivot.loc[c, "ae_xgb_loro"]),
                "ghsl_built_s": float(pivot.loc[c, "ghsl_built_s"]),
            }
            for c in countries
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "figure": str(fig_path),
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "summary.json", summary)
    write_progress(run_dir, status="completed", stage="done", completed=5, total=5, percent_complete=100.0)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, f"DONE · {run_dir}")
    print(pivot.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
