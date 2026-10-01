#!/usr/bin/env python3
"""Balance RES vs NRES at the 100 m tile level (not inherited from cell balance).

Tile label = argmax(frac_res, frac_nres) from full Week-4 GT; ties dropped.
Per metro, undersample majority class to match minority (seed 42).
Primary support: ≥5 AE cells/tile (stable majority); also report ≥20.

Models scored by pooling cell preds inside each tile:
  pred = mean(p_res) ≥ τ  (soft heads) or vote-share ≥ τ (hard 0/1).
W4 balanced-train OOF only covers ~17% of cells in these tiles — reported
with coverage, not as the primary comparable head.

Usage:
  .venv/bin/python -u scripts/build_week4_100m_tiles_balanced.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import (  # noqa: E402
    ACCENT,
    MUTED,
    NRES,
    OK,
    RES,
    label_bars_h,
    set_claim_title,
    style_axes,
)

TILES = ROOT / "datasets" / "week4_100m_tiles" / "tiles.parquet"
CELLS = ROOT / "datasets" / "week4_100m_tiles" / "cells_with_tile.parquet"
GHSL = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
OUT_DATA = ROOT / "datasets" / "week4_100m_tiles_balanced"
OUT_EVAL = ROOT / "outputs" / "week4_100m_tiles_balanced"
FIG = OUT_EVAL / "figures"

SEED = 42
MIN_CELLS_PRIMARY = 5
MIN_CELLS_STRICT = 20
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.01), 2)

MODELS = [
    {
        "key": "balanced_train_full",
        "label": "W4 LR (full GT OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet",
        "kind": "p_res",
        "primary": True,
    },
    {
        "key": "balanced_train",
        "label": "W4 LR (balanced OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "kind": "p_res",
        "primary": False,
    },
    {
        "key": "ghsl",
        "label": "GHSL (pixel)",
        "path": GHSL,
        "kind": "ghsl",
        "primary": True,
    },
]


def _label_tiles(tiles: pd.DataFrame) -> pd.DataFrame:
    t = tiles.copy()
    pr = t["frac_res"].to_numpy(dtype=np.float64)
    pn = t["frac_nres"].to_numpy(dtype=np.float64)
    y = np.full(len(t), -1, dtype=np.int64)
    y[pr > pn] = 1
    y[pn > pr] = 0
    t["y"] = y
    return t.loc[t["y"] >= 0].copy()


def balance_tiles(
    tiles: pd.DataFrame, min_cells: int, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (balanced_tiles, before_counts). Undersample per region."""
    pool = tiles.loc[tiles["n_cells"] >= min_cells].copy()
    before = (
        pool.groupby(["region", "y"]).size().unstack(fill_value=0).rename(columns={0: "nres", 1: "res"})
    )
    before["total"] = before.sum(axis=1)

    rng = np.random.default_rng(seed)
    parts = []
    for region, sub in pool.groupby("region", sort=True):
        res = sub.loc[sub["y"] == 1]
        nres = sub.loc[sub["y"] == 0]
        n = int(min(len(res), len(nres)))
        if n == 0:
            continue
        res_i = res.iloc[rng.choice(len(res), size=n, replace=False)]
        nres_i = nres.iloc[rng.choice(len(nres), size=n, replace=False)]
        parts.append(pd.concat([res_i, nres_i], ignore_index=True))
    bal = pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0].copy()
    bal["min_cells"] = min_cells
    bal["balance_seed"] = seed
    return bal, before.reset_index()


def write_balance_figures(
    before: pd.DataFrame, bal: pd.DataFrame, min_cells: int, fig_dir: Path
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # before
    b = before.set_index("region")[["nres", "res"]].sort_index()
    b.index = [str(i).replace("_", " ") for i in b.index]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(b))
    h = 0.35
    ax.barh(y + h / 2, b["nres"], height=h, color=NRES, label="NRES tiles")
    ax.barh(y - h / 2, b["res"], height=h, color=RES, label="RES tiles")
    ax.set_yticks(y)
    ax.set_yticklabels(b.index)
    ax.set_xlabel("100 m tiles")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    maj = "RES" if int(b["res"].sum()) >= int(b["nres"].sum()) else "NRES"
    set_claim_title(
        fig,
        f"Before balance (min≥{min_cells}): {maj} dominates tile counts",
        "Tile label = argmax(frac_res, frac_nres) on full Week-4 GT (unstratified polygons).",
    )
    p = fig_dir / f"imbalanced_tiles_by_region_min{min_cells}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    after = bal.groupby(["region", "y"]).size().unstack(fill_value=0)
    after = after.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "nres", 1: "res"})
    after.index = [str(i).replace("_", " ") for i in after.index]
    after = after.sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(after))
    ax.barh(y + h / 2, after["nres"], height=h, color=NRES, label="NRES tiles")
    ax.barh(y - h / 2, after["res"], height=h, color=RES, label="RES tiles")
    ax.set_yticks(y)
    ax.set_yticklabels(after.index)
    ax.set_xlabel("100 m tiles")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    n = len(bal)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"After per-metro 1:1 undersample: {n:,} tiles ({n // 2:,} per class), min≥{min_cells}",
        f"Seed {SEED}. Same protocol as cell balance, but the unit is the 100 m tile.",
    )
    p = fig_dir / f"balanced_tiles_by_region_min{min_cells}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def _load_p(meta: dict, pixel_ids: pd.Series) -> pd.DataFrame:
    raw = pd.read_parquet(meta["path"])
    base = pixel_ids.to_frame("pixel_id")
    if meta["kind"] == "ghsl":
        sub = raw[["pixel_id", "ghsl_label"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="inner")
        out = out.loc[out["ghsl_label"].isin(["RES", "NRES"])].copy()
        out["p_res"] = (out["ghsl_label"] == "RES").astype(np.float64)
        return out[["pixel_id", "p_res"]]
    if meta["kind"] == "p_res":
        sub = raw[["pixel_id", "p_res"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="inner")
        out["p_res"] = out["p_res"].astype(float).clip(0, 1)
        return out[["pixel_id", "p_res"]]
    sub = raw[["pixel_id", "y_pred"]].drop_duplicates("pixel_id")
    out = base.merge(sub, on="pixel_id", how="inner")
    out["p_res"] = out["y_pred"].astype(float)
    return out[["pixel_id", "p_res"]]


def score_tiles(
    bal: pd.DataFrame,
    cells: pd.DataFrame,
    meta: dict,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Aggregate cell preds → tile mean(p); sweep τ. Keep tiles with ≥1 pred cell."""
    ids = set(bal["tile_id"])
    c = cells.loc[cells["tile_id"].isin(ids), ["pixel_id", "tile_id"]].copy()
    preds = _load_p(meta, c["pixel_id"])
    c = c.merge(preds, on="pixel_id", how="inner")
    g = (
        c.groupby("tile_id", sort=False)
        .agg(n_pred=("pixel_id", "size"), p_mean=("p_res", "mean"))
        .reset_index()
    )
    t = bal.merge(g, on="tile_id", how="inner")
    cov_cells = len(c)
    cov_tiles = t["tile_id"].nunique()
    rows = []
    yt = t["y"].to_numpy(dtype=np.int64)
    for thr in thresholds:
        yp = (t["p_mean"].to_numpy() >= thr).astype(np.int64)
        rows.append(
            {
                "model": meta["key"],
                "label": meta["label"],
                "primary": meta["primary"],
                "threshold": float(thr),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(yt, yp)),
                "n_tiles": int(len(t)),
                "n_pred_cells": int(cov_cells),
                "coverage_tiles": float(cov_tiles / max(len(bal), 1)),
                "min_cells": int(bal["min_cells"].iloc[0]),
                "kind": meta["kind"],
            }
        )
    return pd.DataFrame(rows)


def write_eval_figures(best: pd.DataFrame, sweep: pd.DataFrame, min_cells: int, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # primary ladder @0.5 vs best
    sub = best.loc[best["primary"]].sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    y = np.arange(len(sub))
    h = 0.35
    ax.barh(y + h / 2, sub["f1_at_0.5"], height=h, color=MUTED, label="@ τ=0.50")
    ax.barh(y - h / 2, sub["macro_f1"], height=h, color=RES, label="best τ*")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}  (τ*={r.threshold:.2f})" for r in sub.itertuples()])
    ax.set_xlabel("Macro-F1 · RES vs NRES · balanced 100 m tiles")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    for yi, v, d in zip(y, sub["macro_f1"], sub["delta_vs_0.5"]):
        ax.text(v + 0.01, yi - h / 2, f"{v:.3f} (Δ{d:+.3f})", va="center", fontsize=8)
    best_f = float(sub["macro_f1"].max())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"On 1:1 balanced 100 m tiles (min≥{min_cells}), best F1 = {best_f:.3f}",
        "Unit of balance = tile, not cell. Soft: τ on mean(p_res). Hard: τ = vote-share.",
    )
    p = fig_dir / f"f1_balanced_tiles_ladder_min{min_cells}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # curves for soft primary models
    soft_keys = [m["key"] for m in MODELS if m["kind"] == "p_res" and m["primary"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for key in soft_keys:
        s = sweep.loc[sweep["model"] == key].sort_values("threshold")
        if s.empty:
            continue
        ax.plot(s["threshold"], s["macro_f1"], lw=2, label=s["label"].iloc[0])
        i = s["macro_f1"].idxmax()
        ax.scatter([s.loc[i, "threshold"]], [s.loc[i, "macro_f1"]], s=40, zorder=3)
    ax.axvline(0.5, color="#999", ls="--", lw=1)
    ax.set_xlabel("threshold on mean(p_res)")
    ax.set_ylabel("Macro-F1")
    ax.set_xlim(0.30, 0.70)
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"τ sweep on balanced 100 m tiles (min≥{min_cells})",
        "Dashed = 0.50. τ* chosen on this eval set (optimistic).",
    )
    p = fig_dir / f"f1_balanced_tiles_threshold_curves_min{min_cells}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def run_build() -> dict:
    t0 = time.time()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "figures").mkdir(parents=True, exist_ok=True)

    tiles = _label_tiles(pd.read_parquet(TILES))
    manifests = {"name": "week4_100m_tiles_balanced_v1", "seed": SEED, "variants": {}}
    for min_c in (MIN_CELLS_PRIMARY, MIN_CELLS_STRICT):
        bal, before = balance_tiles(tiles, min_cells=min_c, seed=SEED)
        bal.to_parquet(OUT_DATA / f"tiles_balanced_min{min_c}.parquet", index=False)
        before.to_csv(OUT_DATA / f"counts_before_min{min_c}.csv", index=False)
        after = (
            bal.groupby(["region", "y"]).size().unstack(fill_value=0).reset_index()
        )
        after.to_csv(OUT_DATA / f"counts_after_min{min_c}.csv", index=False)
        figs = write_balance_figures(before, bal, min_c, OUT_DATA / "figures")
        manifests["variants"][f"min{min_c}"] = {
            "min_cells": min_c,
            "n_tiles": int(len(bal)),
            "n_res": int((bal["y"] == 1).sum()),
            "n_nres": int((bal["y"] == 0).sum()),
            "figures": [str(p.relative_to(ROOT)) for p in figs],
        }
        print(
            f"min≥{min_c}: balanced {len(bal):,} tiles "
            f"({(bal.y==1).sum():,} RES / {(bal.y==0).sum():,} NRES)",
            flush=True,
        )

    manifests["elapsed_s"] = round(time.time() - t0, 1)
    manifests["note"] = (
        "Balance unit = 100 m tile (majority of full-GT frac_res vs frac_nres). "
        "Not the cell-balanced subset."
    )
    (OUT_DATA / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    return manifests


def run_eval() -> dict:
    t0 = time.time()
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(CELLS, columns=["pixel_id", "tile_id"])

    all_sweep = []
    all_best = []
    figs_all = []
    for min_c in (MIN_CELLS_PRIMARY, MIN_CELLS_STRICT):
        bal = pd.read_parquet(OUT_DATA / f"tiles_balanced_min{min_c}.parquet")
        print(f"\nEval on balanced tiles min≥{min_c} (n={len(bal):,}) …", flush=True)
        parts = []
        for meta in MODELS:
            if not Path(meta["path"]).exists():
                print(f"  SKIP {meta['label']}: missing {meta['path']}", flush=True)
                continue
            print(f"  {meta['label']} …", flush=True)
            s = score_tiles(bal, cells, meta, THRESHOLDS)
            parts.append(s)
            i = s["macro_f1"].idxmax()
            best = s.loc[i].to_dict()
            r05 = s.loc[(s["threshold"] - 0.5).abs() < 1e-9].iloc[0]
            best["f1_at_0.5"] = float(r05["macro_f1"])
            best["delta_vs_0.5"] = float(best["macro_f1"] - r05["macro_f1"])
            all_best.append(best)
            print(
                f"    @0.50={r05['macro_f1']:.4f}  best@{best['threshold']:.2f}="
                f"{best['macro_f1']:.4f}  cov_tiles={best['coverage_tiles']:.1%}  "
                f"n={best['n_tiles']}",
                flush=True,
            )
        sweep = pd.concat(parts, ignore_index=True)
        sweep.to_csv(OUT_EVAL / f"f1_threshold_sweep_min{min_c}.csv", index=False)
        all_sweep.append(sweep)
        best_df = pd.DataFrame([b for b in all_best if b["min_cells"] == min_c])
        figs_all += write_eval_figures(best_df, sweep, min_c, FIG)

    best_all = pd.DataFrame(all_best)
    best_all.to_csv(OUT_EVAL / "f1_best_summary.csv", index=False)
    pd.concat(all_sweep, ignore_index=True).to_csv(
        OUT_EVAL / "f1_threshold_sweep_all.csv", index=False
    )

    out = {
        "name": "week4_100m_tiles_balanced_eval_v1",
        "protocol": (
            "1:1 RES/NRES balance at 100 m tile level (per-metro undersample). "
            "Tile GT = argmax(frac_res, frac_nres). Pred = mean(p_res) ≥ τ."
        ),
        "seed": SEED,
        "min_cells": [MIN_CELLS_PRIMARY, MIN_CELLS_STRICT],
        "best": best_all.to_dict(orient="records"),
        "figures": [str(p.relative_to(ROOT)) for p in figs_all],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_EVAL / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {OUT_EVAL} in {out['elapsed_s']}s", flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.eval_only:
        run_eval()
    elif args.build_only:
        run_build()
    else:
        run_build()
        run_eval()


if __name__ == "__main__":
    main()
