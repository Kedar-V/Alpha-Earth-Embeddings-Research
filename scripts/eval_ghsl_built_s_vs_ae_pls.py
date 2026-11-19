#!/usr/bin/env python3
"""GHS-BUILT-S vs AE PLS fraction MAE (10 m + 100 m).

AE head: Week-4 PLS LORO OOF
  outputs/week4_pls_loro/oof_predictions.parquet
  pred_frac_{res,nres,empty}_pls

Also scores Ridge on the same pixel_ids for a fair side-by-side.

Uses cached BUILT-S samples at AE centers from the 10 m BUILT-S run.
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
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PLS = ROOT / "outputs" / "week4_pls_loro" / "oof_predictions.parquet"
RIDGE = ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet"
BUILT_S = ROOT / "outputs" / "ghsl_built_s_vs_ae" / "20261103_010752" / "checkpoints" / "built_s_at_ae_centers.parquet"
FUN = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
SHARED = ROOT / "outputs" / "week4_ghsl_pixel" / "predictions_balanced_fractions.parquet"
CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
MIN_CELLS = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def write_progress(run_dir: Path, **kwargs: Any) -> None:
    path = run_dir / "progress.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(kwargs)
    prev["updated_at"] = utc_now()
    if "_t0" in prev:
        prev["elapsed_seconds"] = round(max(0.0, time.time() - float(prev["_t0"])), 1)
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    with (run_dir / "job.log").open("a") as f:
        f.write(line + "\n")


def ensure_bs_fracs(bs: pd.DataFrame) -> pd.DataFrame:
    if "pred_frac_res" in bs.columns:
        return bs
    bu = bs["built_s_raw"].astype(float)
    nr = bs["built_s_nres_raw"].astype(float)
    valid = (bu != 255) & (nr != 255)
    bu_f = np.clip(bu / 100.0, 0, 1)
    is_nres = nr == 1
    out = bs.copy()
    out["pred_frac_res"] = np.where(valid, np.where(is_nres, 0.0, bu_f), np.nan)
    out["pred_frac_nres"] = np.where(valid, np.where(is_nres, bu_f, 0.0), np.nan)
    out["pred_frac_empty"] = np.where(valid, 1.0 - bu_f, np.nan)
    return out


def mae_pack(yt_r, yt_n, yt_e, yp_r, yp_n, yp_e) -> dict[str, float]:
    m = np.isfinite(yp_r) & np.isfinite(yp_n) & np.isfinite(yp_e) & np.isfinite(yt_r)
    if m.sum() == 0:
        return {"n": 0}
    return {
        "n": int(m.sum()),
        "mae_frac_res": float(mean_absolute_error(yt_r[m], yp_r[m])),
        "mae_frac_nres": float(mean_absolute_error(yt_n[m], yp_n[m])),
        "mae_frac_empty": float(mean_absolute_error(yt_e[m], yp_e[m])),
        "mae_mean_res_nres": float(
            0.5 * (mean_absolute_error(yt_r[m], yp_r[m]) + mean_absolute_error(yt_n[m], yp_n[m]))
        ),
        "r2_built": float(r2_score(yt_r[m] + yt_n[m], yp_r[m] + yp_n[m])),
    }


def score_frame(df: pd.DataFrame, eval_name: str, scale: str) -> list[dict[str, Any]]:
    yt_r, yt_n, yt_e = df["frac_res"].to_numpy(), df["frac_nres"].to_numpy(), df["frac_empty"].to_numpy()
    models = [
        ("ae_pls", "pls_frac_res", "pls_frac_nres", "pls_frac_empty"),
        ("ae_w4_ridge", "ridge_frac_res", "ridge_frac_nres", "ridge_frac_empty"),
        ("ghsl_built_s", "bs_frac_res", "bs_frac_nres", "bs_frac_empty"),
    ]
    if "fun_frac_res" in df.columns:
        models.append(("ghsl_fun_onehot", "fun_frac_res", "fun_frac_nres", "fun_frac_empty"))
    rows = []
    for name, a, b, c in models:
        if a not in df.columns:
            continue
        met = mae_pack(yt_r, yt_n, yt_e, df[a].to_numpy(), df[b].to_numpy(), df[c].to_numpy())
        met.update({"model": name, "eval": eval_name, "scale": scale})
        rows.append(met)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (ROOT / "outputs" / "ghsl_built_s_vs_ae_pls" / ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "job.pid").write_text(str(os.getpid()))
    (run_dir / "launch.txt").write_text(f"cmd={' '.join(sys.argv)}\npid={os.getpid()}\nstarted={utc_now()}\n")
    write_progress(
        run_dir,
        status="running",
        stage="load",
        started_at=utc_now(),
        _t0=time.time(),
        output_location=str(run_dir),
        resources={"cpus": os.cpu_count()},
    )
    log(run_dir, f"Run dir {run_dir}")

    write_progress(run_dir, stage="load", last_progress_at=utc_now())
    pls = pd.read_parquet(
        PLS,
        columns=[
            "pixel_id",
            "region",
            "pixel_lon",
            "pixel_lat",
            "frac_res_full",
            "frac_nres_full",
            "frac_empty_full",
            "pred_frac_res_pls",
            "pred_frac_nres_pls",
            "pred_frac_empty_pls",
        ],
    ).rename(
        columns={
            "frac_res_full": "frac_res",
            "frac_nres_full": "frac_nres",
            "frac_empty_full": "frac_empty",
            "pred_frac_res_pls": "pls_frac_res",
            "pred_frac_nres_pls": "pls_frac_nres",
            "pred_frac_empty_pls": "pls_frac_empty",
        }
    )
    ridge = pd.read_parquet(
        RIDGE,
        columns=["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"],
    ).rename(
        columns={
            "pred_frac_res": "ridge_frac_res",
            "pred_frac_nres": "ridge_frac_nres",
            "pred_frac_empty": "ridge_frac_empty",
        }
    )
    bs = ensure_bs_fracs(pd.read_parquet(BUILT_S)).rename(
        columns={
            "pred_frac_res": "bs_frac_res",
            "pred_frac_nres": "bs_frac_nres",
            "pred_frac_empty": "bs_frac_empty",
        }
    )
    fun = pd.read_parquet(FUN, columns=["pixel_id", "ghsl_label"]) if FUN.exists() else None

    df = pls.merge(ridge, on="pixel_id", how="left")
    df = df.merge(bs[["pixel_id", "bs_frac_res", "bs_frac_nres", "bs_frac_empty"]], on="pixel_id", how="inner")
    if fun is not None:
        df = df.merge(fun.drop_duplicates("pixel_id"), on="pixel_id", how="left")
        lab = df["ghsl_label"].fillna("NODATA")
        df["fun_frac_res"] = (lab == "RES").astype(float)
        df["fun_frac_nres"] = (lab == "NRES").astype(float)
        df["fun_frac_empty"] = (~lab.isin(["RES", "NRES"])).astype(float)
    log(run_dir, f"PLS∩BUILT-S n={len(df):,}")

    # Eval A: all PLS OOF cells
    rows = score_frame(df, "pls_oof_all", "10m")
    # Eval B: shared_59k ∩ PLS
    shared_ids = set(pd.read_parquet(SHARED, columns=["pixel_id"])["pixel_id"])
    df_shared = df[df.pixel_id.isin(shared_ids)].copy()
    rows += score_frame(df_shared, "shared_overlap", "10m")
    log(run_dir, f"shared∩PLS n={len(df_shared):,}")

    # 100 m aggregation on PLS cells
    write_progress(run_dir, stage="agg_100m", last_progress_at=utc_now())
    cells = pd.read_parquet(CELLS, columns=["pixel_id", "region", "tile_id", "tile_i", "tile_j"])
    d100 = cells.merge(df, on=["pixel_id", "region"], how="inner")
    d100 = d100.loc[(d100["frac_res"] + d100["frac_nres"]) > 0].copy()
    agg = {
        "n_cells": ("pixel_id", "size"),
        "frac_res": ("frac_res", "mean"),
        "frac_nres": ("frac_nres", "mean"),
        "frac_empty": ("frac_empty", "mean"),
        "pls_frac_res": ("pls_frac_res", "mean"),
        "pls_frac_nres": ("pls_frac_nres", "mean"),
        "pls_frac_empty": ("pls_frac_empty", "mean"),
        "ridge_frac_res": ("ridge_frac_res", "mean"),
        "ridge_frac_nres": ("ridge_frac_nres", "mean"),
        "ridge_frac_empty": ("ridge_frac_empty", "mean"),
        "bs_frac_res": ("bs_frac_res", "mean"),
        "bs_frac_nres": ("bs_frac_nres", "mean"),
        "bs_frac_empty": ("bs_frac_empty", "mean"),
    }
    if "fun_frac_res" in d100.columns:
        agg.update(
            {
                "fun_frac_res": ("fun_frac_res", "mean"),
                "fun_frac_nres": ("fun_frac_nres", "mean"),
                "fun_frac_empty": ("fun_frac_empty", "mean"),
            }
        )
    tiles = d100.groupby(["tile_id", "region"], sort=False).agg(**agg).reset_index()
    tiles = tiles.loc[tiles["n_cells"] >= MIN_CELLS].copy()
    log(run_dir, f"100 m tiles n={len(tiles):,}")
    rows += score_frame(tiles, "100m_min20", "100m")
    tiles.to_parquet(run_dir / "artifacts" / "tiles_100m_pls.parquet", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "mae_summary.csv", index=False)
    for r in summary.itertuples():
        log(run_dir, f"{r.scale:4s} {r.eval:16s} {r.model:16s} mae_rn={r.mae_mean_res_nres:.4f} n={r.n}")

    # figures
    write_progress(run_dir, stage="figures", last_progress_at=utc_now())
    fig_dir = run_dir / "figures"
    for eval_name, title in [
        ("pls_oof_all", "10 m · all PLS OOF cells"),
        ("shared_overlap", "10 m · shared∩PLS"),
        ("100m_min20", f"100 m · min {MIN_CELLS} cells"),
    ]:
        s = summary.loc[summary["eval"] == eval_name].sort_values("mae_mean_res_nres")
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        colors = []
        for m in s["model"]:
            if m == "ae_pls":
                colors.append("#1f77b4")
            elif m.startswith("ae"):
                colors.append("#2ca02c")
            elif "built_s" in m:
                colors.append("#ff7f0e")
            else:
                colors.append("#7f7f7f")
        ax.barh(s["model"], s["mae_mean_res_nres"], color=colors)
        for y, v in enumerate(s["mae_mean_res_nres"]):
            ax.text(v + 0.003, y, f"{v:.3f}", va="center")
        ax.set_xlabel("MAE mean(res, nres) ↓")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / f"mae_ladder_{eval_name}.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # notebook snippet update
    nb_dir = ROOT / "notebooks" / "ghsl_built_s_vs_ae"
    nb_dir.mkdir(exist_ok=True)
    for p in fig_dir.glob("*.png"):
        (nb_dir / "figures" / p.name).parent.mkdir(exist_ok=True)
        (nb_dir / "figures" / p.name).write_bytes(p.read_bytes())

    s10 = summary.loc[summary["eval"] == "pls_oof_all"].set_index("model")
    s100 = summary.loc[summary["eval"] == "100m_min20"].set_index("model")
    pls10 = float(s10.loc["ae_pls", "mae_mean_res_nres"])
    bs10 = float(s10.loc["ghsl_built_s", "mae_mean_res_nres"])
    ridge10 = float(s10.loc["ae_w4_ridge", "mae_mean_res_nres"]) if "ae_w4_ridge" in s10.index else float("nan")
    pls100 = float(s100.loc["ae_pls", "mae_mean_res_nres"])
    bs100 = float(s100.loc["ghsl_built_s", "mae_mean_res_nres"])
    ridge100 = float(s100.loc["ae_w4_ridge", "mae_mean_res_nres"]) if "ae_w4_ridge" in s100.index else float("nan")
    verdict = (
        f"10 m PLS OOF: BUILT-S MAE={bs10:.3f} vs PLS={pls10:.3f} vs Ridge={ridge10:.3f}. "
        f"100 m: BUILT-S={bs100:.3f} vs PLS={pls100:.3f} vs Ridge={ridge100:.3f}."
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GHS-BUILT-S vs AE PLS fraction MAE\n",
                    "\n",
                    f"Run: `{run_dir}`\n",
                    "\n",
                    f"{verdict}\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from IPython.display import Image, display, Markdown\n",
                    f"RUN = Path({str(run_dir)!r})\n",
                    "s = pd.read_csv(RUN / 'mae_summary.csv')\n",
                    "display(s)\n",
                    "for p in sorted((RUN / 'figures').glob('*.png')):\n",
                    "    display(Markdown(f'### {p.name}'))\n",
                    "    display(Image(filename=str(p)))\n",
                ],
            },
        ],
    }
    nb_path = nb_dir / "03_built_s_vs_ae_pls.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1))

    payload = {
        "status": "completed",
        "finished_at": utc_now(),
        "run_dir": str(run_dir),
        "ae_model": "week4_pls_loro OOF (pred_frac_*_pls)",
        "also_scored": ["ae_w4_ridge on same ids", "ghsl_built_s", "ghsl_fun_onehot"],
        "mae_summary": summary.to_dict(orient="records"),
        "verdict": verdict,
        "notebook": str(nb_path),
        "n_pls_cells": int(len(df)),
        "n_shared_overlap": int(len(df_shared)),
        "n_tiles_100m": int(len(tiles)),
    }
    atomic_json(run_dir / "summary.json", payload)
    write_progress(run_dir, status="completed", stage="done", last_progress_at=utc_now(), verdict=verdict)
    (run_dir / "DONE").write_text(utc_now() + "\n")
    log(run_dir, verdict)
    log(run_dir, f"DONE · notebook={nb_path}")


if __name__ == "__main__":
    main()
