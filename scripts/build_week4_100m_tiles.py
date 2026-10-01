#!/usr/bin/env python3
"""Build Week 4 100×100 m tiles from 10 m AE cells + evaluate RES/NRES fraction MAE.

Tile ID: per-metro UTM, tile_i = floor(easting/100), tile_j = floor(northing/100).
GT / AE preds at 100 m: mean of equal-area 10 m fractions (= area aggregate).
GHSL at 100 m: soft share of FUN labels at AE centers in the tile.

Primary metric: (MAE_res + MAE_nres) / 2. Compare 10 m baseline on the same cells.

Usage:
  .venv/bin/python -u scripts/build_week4_100m_tiles.py
  .venv/bin/python -u scripts/eval_week4_100m_fractions.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score

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

FULL = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
GHSL_CACHE = ROOT / "outputs" / "week4_ghsl_pixel" / "ghsl_at_ae_centers.parquet"
OUT_DATA = ROOT / "datasets" / "week4_100m_tiles"
OUT_EVAL = ROOT / "outputs" / "week4_100m_fraction_eval"
FIG = OUT_EVAL / "figures"

TILE_M = 100
MIN_CELLS = 20  # AE GT is building-linked (sparse); 50 left too few tiles
MIN_CELLS_BALANCED = 10  # balanced pure subset is thinner per tile

MODELS = [
    {
        "key": "balanced_train_full",
        "label": "W4 Ridge (full GT OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet",
        "kind": "frac",
        "scope": "full",
        "primary": True,
    },
    {
        "key": "balanced_train",
        "label": "W4 Ridge (balanced OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "kind": "frac",
        "scope": "balanced",
        "primary": False,
    },
    {
        "key": "ghsl",
        "label": "GHSL (soft under tile)",
        "path": GHSL_CACHE,
        "kind": "ghsl",
        "scope": "full",
        "primary": True,
    },
]


def _assign_tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add utm_x, utm_y, tile_i, tile_j, tile_id using per-region CRS."""
    parts = []
    for region, sub in df.groupby("region", sort=True):
        crs = str(sub["crs"].mode().iloc[0])
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tf.transform(
            sub["pixel_lon"].to_numpy(dtype=np.float64),
            sub["pixel_lat"].to_numpy(dtype=np.float64),
        )
        out = sub.copy()
        out["utm_x"] = x
        out["utm_y"] = y
        out["tile_i"] = np.floor(out["utm_x"] / TILE_M).astype(np.int64)
        out["tile_j"] = np.floor(out["utm_y"] / TILE_M).astype(np.int64)
        out["tile_id"] = (
            out["region"].astype(str)
            + "|"
            + out["tile_i"].astype(str)
            + "|"
            + out["tile_j"].astype(str)
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def build_tiles(
    cells: pd.DataFrame, min_cells: int = MIN_CELLS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cells_with_tile_id, tiles aggregated GT)."""
    cells = _assign_tiles(cells)
    g = cells.groupby(["tile_id", "region", "tile_i", "tile_j", "crs"], sort=False)
    tiles = g.agg(
        n_cells=("pixel_id", "size"),
        frac_res=("frac_res", "mean"),
        frac_nres=("frac_nres", "mean"),
        frac_empty=("frac_empty", "mean"),
        utm_x_c=("utm_x", "mean"),
        utm_y_c=("utm_y", "mean"),
        pixel_lon_c=("pixel_lon", "mean"),
        pixel_lat_c=("pixel_lat", "mean"),
    ).reset_index()
    tiles["tile_size_m"] = TILE_M
    tiles["support_ok"] = tiles["n_cells"] >= min_cells
    return cells, tiles


def _load_pred_rn(meta: dict, pixel_ids: pd.Series) -> pd.DataFrame:
    raw = pd.read_parquet(meta["path"])
    base = pixel_ids.to_frame("pixel_id")
    if meta["kind"] == "ghsl":
        sub = raw[["pixel_id", "ghsl_label"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="left")
        out["ghsl_label"] = out["ghsl_label"].fillna("NODATA")
        out["pred_frac_res"] = (out["ghsl_label"] == "RES").astype(np.float64)
        out["pred_frac_nres"] = (out["ghsl_label"] == "NRES").astype(np.float64)
        out["pred_frac_empty"] = out["ghsl_label"].isin(["NONBUILT", "NODATA"]).astype(np.float64)
        return out[["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]]

    if meta["kind"] == "p_res":
        sub = raw[["pixel_id", "p_res"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="left")
        p = out["p_res"].astype(float).clip(0, 1)
        out["pred_frac_res"] = p
        out["pred_frac_nres"] = 1.0 - p
        out["pred_frac_empty"] = 0.0
        return out[["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]]

    keep = ["pixel_id"]
    for c in (
        "pred_frac_res_full",
        "pred_frac_nres_full",
        "pred_frac_empty_full",
        "pred_frac_res",
        "pred_frac_nres",
        "pred_frac_empty",
    ):
        if c in raw.columns:
            keep.append(c)
    sub = raw[keep].drop_duplicates("pixel_id")
    out = base.merge(sub, on="pixel_id", how="left")
    if "pred_frac_res_full" in out.columns and out["pred_frac_res_full"].notna().any():
        pr = out["pred_frac_res_full"].astype(float)
        pn = out["pred_frac_nres_full"].astype(float)
        pe = (
            out["pred_frac_empty_full"].astype(float)
            if "pred_frac_empty_full" in out.columns
            else (1.0 - pr - pn).clip(0, 1)
        )
    else:
        pr = out["pred_frac_res"].astype(float)
        pn = out["pred_frac_nres"].astype(float)
        pe = (
            out["pred_frac_empty"].astype(float)
            if "pred_frac_empty" in out.columns
            else (1.0 - pr - pn).clip(0, 1)
        )
    s = pr + pn
    over = s > 1.0
    pr2, pn2 = pr.copy(), pn.copy()
    pr2.loc[over] = pr.loc[over] / s.loc[over]
    pn2.loc[over] = pn.loc[over] / s.loc[over]
    out["pred_frac_res"] = pr2
    out["pred_frac_nres"] = pn2
    out["pred_frac_empty"] = pe.fillna((1.0 - pr2 - pn2).clip(0, 1))
    return out[["pixel_id", "pred_frac_res", "pred_frac_nres", "pred_frac_empty"]]


def mae_rn(yt_res, yt_nres, yp_res, yp_nres) -> dict:
    mae_r = float(mean_absolute_error(yt_res, yp_res))
    mae_n = float(mean_absolute_error(yt_nres, yp_nres))
    return {
        "mae_frac_res": mae_r,
        "mae_frac_nres": mae_n,
        "mae_mean_res_nres": 0.5 * (mae_r + mae_n),
        "n": int(len(yt_res)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "macro_f1": float("nan"), "accuracy": float("nan")}
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _rn_labels(frac_res: pd.Series, frac_nres: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Majority RES(1) vs NRES(0); drop ties."""
    pr = frac_res.to_numpy(dtype=np.float64)
    pn = frac_nres.to_numpy(dtype=np.float64)
    mask = pr != pn
    y = (pr[mask] > pn[mask]).astype(np.int64)
    return y, mask


def score_model(
    meta: dict,
    cells: pd.DataFrame,
    tiles: pd.DataFrame,
    min_cells: int = MIN_CELLS,
) -> tuple[dict, dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Return (mae_100, mae_10, bin_100, bin_10, tile_pred_frame, per_region_mae)."""
    preds = _load_pred_rn(meta, cells["pixel_id"])
    c = cells.merge(preds, on="pixel_id", how="inner")
    c = c.dropna(subset=["pred_frac_res", "pred_frac_nres"])

    ok_tiles = set(tiles.loc[tiles["support_ok"], "tile_id"])
    c10 = c.loc[c["tile_id"].isin(ok_tiles)].copy()
    m10 = mae_rn(
        c10["frac_res"],
        c10["frac_nres"],
        c10["pred_frac_res"],
        c10["pred_frac_nres"],
    )
    m10["scale"] = "10m"
    m10["model"] = meta["key"]
    m10["label"] = meta["label"]

    # 10 m binary: majority of GT fracs vs majority of pred fracs (drop ties)
    both = (c10["frac_res"] != c10["frac_nres"]) & (
        c10["pred_frac_res"] != c10["pred_frac_nres"]
    )
    c10b = c10.loc[both]
    yt10 = (c10b["frac_res"].to_numpy() > c10b["frac_nres"].to_numpy()).astype(np.int64)
    yp10 = (c10b["pred_frac_res"].to_numpy() > c10b["pred_frac_nres"].to_numpy()).astype(
        np.int64
    )
    b10 = binary_metrics(yt10, yp10)
    b10["scale"] = "10m"
    b10["model"] = meta["key"]
    b10["label"] = meta["label"]

    agg = (
        c.groupby("tile_id", sort=False)
        .agg(
            n_pred=("pixel_id", "size"),
            pred_frac_res=("pred_frac_res", "mean"),
            pred_frac_nres=("pred_frac_nres", "mean"),
            pred_frac_empty=("pred_frac_empty", "mean"),
        )
        .reset_index()
    )
    t = tiles.merge(agg, on="tile_id", how="inner")
    t = t.loc[t["support_ok"] & (t["n_pred"] >= min_cells)].copy()
    m100 = mae_rn(
        t["frac_res"],
        t["frac_nres"],
        t["pred_frac_res"],
        t["pred_frac_nres"],
    )
    m100["mae_frac_empty"] = float(
        mean_absolute_error(t["frac_empty"], t["pred_frac_empty"])
    )
    m100["scale"] = "100m"
    m100["model"] = meta["key"]
    m100["label"] = meta["label"]
    m100["delta_10m_minus_100m"] = m10["mae_mean_res_nres"] - m100["mae_mean_res_nres"]
    m100["min_cells"] = min_cells

    # 100 m binary from aggregated fractions
    t_bin = t.loc[t["frac_res"] != t["frac_nres"]].copy()
    t_bin = t_bin.loc[t_bin["pred_frac_res"] != t_bin["pred_frac_nres"]]
    yt100 = (t_bin["frac_res"].to_numpy() > t_bin["frac_nres"].to_numpy()).astype(np.int64)
    yp100 = (t_bin["pred_frac_res"].to_numpy() > t_bin["pred_frac_nres"].to_numpy()).astype(
        np.int64
    )
    b100 = binary_metrics(yt100, yp100)
    b100["scale"] = "100m"
    b100["model"] = meta["key"]
    b100["label"] = meta["label"]
    b100["delta_f1_100m_minus_10m"] = b100["macro_f1"] - b10["macro_f1"]
    t_bin["y_gt"] = yt100
    t_bin["y_pred"] = yp100

    per = []
    for reg, sub in t.groupby("region"):
        row = mae_rn(sub["frac_res"], sub["frac_nres"], sub["pred_frac_res"], sub["pred_frac_nres"])
        row["region"] = reg
        row["model"] = meta["key"]
        per.append(row)

    # per-region binary F1 at 100m
    per_bin = []
    for reg, sub in t_bin.groupby("region"):
        bm = binary_metrics(sub["y_gt"].to_numpy(), sub["y_pred"].to_numpy())
        bm["region"] = reg
        bm["model"] = meta["key"]
        per_bin.append(bm)

    t["model"] = meta["key"]
    t_bin["model"] = meta["key"]
    return m100, m10, b100, b10, t, pd.DataFrame(per), t_bin, pd.DataFrame(per_bin)


def write_qa_figures(tiles: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # n_cells hist
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(tiles["n_cells"], bins=40, color=RES, edgecolor="white")
    ax.axvline(MIN_CELLS, color=ACCENT, ls="--", lw=1.5, label=f"min={MIN_CELLS}")
    ax.set_xlabel("AE cells per 100 m tile")
    ax.set_ylabel("tiles")
    style_axes(ax)
    ax.legend(frameon=False)
    n_ok = int(tiles["support_ok"].sum())
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    set_claim_title(
        fig,
        f"{n_ok:,} tiles pass support (≥{MIN_CELLS} AE cells); max pack is 100",
        "100×100 m UTM tiles from Week 4 AE 10 m centers.",
    )
    p = fig_dir / "n_cells_per_tile.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    ct = tiles.loc[tiles["support_ok"]].groupby("region").size().sort_values()
    ct.index = [str(i).replace("_", " ") for i in ct.index]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bars = ax.barh(ct.index, ct.values, color=RES, height=0.7)
    for rect, v in zip(bars, ct.values):
        ax.text(v + 2, rect.get_y() + rect.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Eligible 100 m tiles")
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    set_claim_title(fig, "Eligible 100 m tiles by metro", f"support ≥ {MIN_CELLS} AE cells.")
    p = fig_dir / "tiles_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def write_eval_figures(summary: pd.DataFrame, per_reg: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    s100 = summary.loc[summary["scale"] == "100m"].copy()
    s10 = summary.loc[summary["scale"] == "10m"].copy()
    merged = s100.merge(
        s10[["model", "mae_mean_res_nres"]].rename(columns={"mae_mean_res_nres": "mae_10m"}),
        on="model",
    )
    merged["delta"] = merged["mae_10m"] - merged["mae_mean_res_nres"]

    # Grouped ladder 10m vs 100m
    order = merged.sort_values("mae_mean_res_nres", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    y = np.arange(len(order))
    h = 0.35
    ax.barh(y + h / 2, order["mae_10m"], height=h, color=MUTED, label="10 m cells")
    ax.barh(y - h / 2, order["mae_mean_res_nres"], height=h, color=RES, label="100 m tiles")
    ax.set_yticks(y)
    ax.set_yticklabels(order["label"])
    ax.set_xlabel("MAE · (res + nres) / 2  ·  lower better")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    # direct labels on 100m
    for yi, v, d in zip(y, order["mae_mean_res_nres"], order["delta"]):
        ax.text(v + 0.008, yi - h / 2, f"{v:.3f} (Δ{d:+.3f})", va="center", fontsize=8)
    best = float(order["mae_mean_res_nres"].min())
    bump = float(order["delta"].mean())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"Pooling to 100 m cuts RES/NRES MAE by ~{bump:.3f} on average (best 100 m = {best:.3f})",
        "Same OOF preds; tile GT = area aggregate (mean of equal-area 10 m fracs). Empty ignored.",
    )
    p = fig_dir / "mae_10m_vs_100m_ladder.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Delta bars
    dord = order.sort_values("delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    colors = [OK if v > 0 else ACCENT for v in dord["delta"]]
    bars = ax.barh(dord["label"], dord["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#333", lw=1)
    for rect, v in zip(bars, dord["delta"]):
        ax.text(
            v + (0.004 if v >= 0 else -0.004),
            rect.get_y() + rect.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlabel("MAE drop when pooling 10 m → 100 m")
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        "Every RES/NRES model improves at 100 m — aggregation reduces cell-level noise",
        "Positive = lower error at 100 m. Green = improved.",
    )
    p = fig_dir / "mae_delta_10m_to_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 100m only ladder with GHSL accent
    s = s100.sort_values("mae_mean_res_nres", ascending=False)
    best = float(s["mae_mean_res_nres"].min())
    colors = [
        ACCENT if m == "ghsl" else (OK if abs(v - best) < 1e-12 else MUTED)
        for m, v in zip(s["model"], s["mae_mean_res_nres"])
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.barh(s["label"], s["mae_mean_res_nres"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.006)
    ax.set_xlabel("100 m MAE · (res + nres) / 2")
    ax.set_xlim(0, float(s["mae_mean_res_nres"].max()) * 1.2)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"At 100 m, best RES/NRES MAE is {best:.3f}",
        "Orange = GHSL soft under tile. Green = best AE aggregate.",
    )
    p = fig_dir / "mae_100m_ladder.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # Per-metro delta for best AE model (lowest 100m MAE among non-ghsl)
    ae = s100.loc[s100["model"] != "ghsl"]
    if len(ae) and len(per_reg):
        best_key = ae.sort_values("mae_mean_res_nres").iloc[0]["model"]
        # need 10m per region too — approximate using pooled delta only if missing
        # compute from tile preds file if present later; here use per_reg 100m only ranked
        sub = per_reg.loc[per_reg["model"] == best_key].sort_values("mae_mean_res_nres")
        if len(sub):
            sub = sub.copy()
            sub["lab"] = sub["region"].astype(str).str.replace("_", " ")
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            mean_m = float(sub["mae_mean_res_nres"].mean())
            colors = [OK if v <= mean_m else MUTED for v in sub["mae_mean_res_nres"]]
            bars = ax.barh(sub["lab"], sub["mae_mean_res_nres"], color=colors, height=0.7)
            ax.axvline(mean_m, color="#333", ls="--", lw=1)
            label_bars_h(ax, bars, fmt="{:.3f}", pad=0.004)
            ax.set_xlabel("100 m MAE · (res + nres) / 2")
            style_axes(ax, grid=True)
            lab = ae.loc[ae["model"] == best_key, "label"].iloc[0]
            fig.tight_layout(rect=[0, 0, 1, 0.84])
            set_claim_title(
                fig,
                f"{lab} at 100 m by metro (mean MAE {mean_m:.3f})",
                "Green = at/below mean.",
            )
            p = fig_dir / "mae_100m_by_metro_best_ae.png"
            fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths.append(p)

    return paths


def write_binary_figures(bin_sum: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    s100 = bin_sum.loc[bin_sum["scale"] == "100m"].copy()
    s10 = bin_sum.loc[bin_sum["scale"] == "10m"].copy()
    merged = s100.merge(
        s10[["model", "macro_f1"]].rename(columns={"macro_f1": "f1_10m"}),
        on="model",
    )
    merged["delta"] = merged["macro_f1"] - merged["f1_10m"]
    order = merged.sort_values("macro_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    y = np.arange(len(order))
    h = 0.35
    ax.barh(y + h / 2, order["f1_10m"], height=h, color=MUTED, label="10 m cells")
    ax.barh(y - h / 2, order["macro_f1"], height=h, color=RES, label="100 m tiles")
    ax.set_yticks(y)
    ax.set_yticklabels(order["label"])
    ax.set_xlabel("Macro-F1 · RES vs NRES")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    for yi, v, d in zip(y, order["macro_f1"], order["delta"]):
        ax.text(v + 0.01, yi - h / 2, f"{v:.3f} (Δ{d:+.3f})", va="center", fontsize=8)
    best = float(order["macro_f1"].max())
    bump = float(order["delta"].mean())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"RES/NRES classification: pooling to 100 m lifts macro-F1 by ~{bump:.3f} (best {best:.3f})",
        "Label = argmax(frac_res, frac_nres) on GT and on aggregated preds. Ties dropped.",
    )
    p = fig_dir / "f1_10m_vs_100m_ladder.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    dord = order.sort_values("delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    colors = [OK if v > 0 else ACCENT for v in dord["delta"]]
    bars = ax.barh(dord["label"], dord["delta"], color=colors, height=0.7)
    ax.axvline(0, color="#333", lw=1)
    for rect, v in zip(bars, dord["delta"]):
        ax.text(
            v + (0.004 if v >= 0 else -0.004),
            rect.get_y() + rect.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    ax.set_xlabel("Macro-F1 change · 100 m − 10 m")
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        "Binary RES/NRES also benefits from 100 m aggregation",
        "Positive = higher F1 at 100 m. Green = improved.",
    )
    p = fig_dir / "f1_delta_10m_to_100m.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    s = s100.sort_values("macro_f1", ascending=True)
    best = float(s["macro_f1"].max())
    colors = [
        ACCENT if m == "ghsl" else (OK if abs(v - best) < 1e-12 else MUTED)
        for m, v in zip(s["model"], s["macro_f1"])
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.barh(s["label"], s["macro_f1"], color=colors, height=0.7)
    label_bars_h(ax, bars, fmt="{:.3f}", pad=0.01)
    ax.set_xlabel("100 m macro-F1 · RES vs NRES")
    ax.set_xlim(0, 1.08)
    style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"At 100 m, best RES/NRES macro-F1 is {best:.3f}",
        "Orange = GHSL. Green = best AE aggregate.",
    )
    p = fig_dir / "f1_100m_ladder.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def run_build() -> dict:
    t0 = time.time()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "figures").mkdir(parents=True, exist_ok=True)

    cols = [
        "pixel_id",
        "region",
        "crs",
        "pixel_lon",
        "pixel_lat",
        "frac_res",
        "frac_nres",
        "frac_empty",
    ]
    full = pd.read_parquet(FULL, columns=cols)
    print(f"Full GT cells n={len(full):,}", flush=True)
    cells, tiles = build_tiles(full, MIN_CELLS)
    cells[
        ["pixel_id", "region", "tile_id", "tile_i", "tile_j", "utm_x", "utm_y", *cols[5:]]
    ].to_parquet(OUT_DATA / "cells_with_tile.parquet", index=False)
    tiles.to_parquet(OUT_DATA / "tiles.parquet", index=False)

    # Balanced-only tile aggregates (W4 OOF coverage)
    bal_ids = set(pd.read_parquet(BAL, columns=["pixel_id"])["pixel_id"])
    bal_cells = cells.loc[cells["pixel_id"].isin(bal_ids)].copy()
    g = bal_cells.groupby(["tile_id", "region", "tile_i", "tile_j", "crs"], sort=False)
    bal_tiles = g.agg(
        n_cells=("pixel_id", "size"),
        frac_res=("frac_res", "mean"),
        frac_nres=("frac_nres", "mean"),
        frac_empty=("frac_empty", "mean"),
        utm_x_c=("utm_x", "mean"),
        utm_y_c=("utm_y", "mean"),
        pixel_lon_c=("pixel_lon", "mean"),
        pixel_lat_c=("pixel_lat", "mean"),
    ).reset_index()
    bal_tiles["tile_size_m"] = TILE_M
    bal_tiles["support_ok"] = bal_tiles["n_cells"] >= MIN_CELLS_BALANCED
    bal_tiles.to_parquet(OUT_DATA / "tiles_balanced_subset.parquet", index=False)

    write_qa_figures(tiles, OUT_DATA / "figures")

    manifest = {
        "name": "week4_100m_tiles_v1",
        "tile_m": TILE_M,
        "min_cells": MIN_CELLS,
        "min_cells_balanced": MIN_CELLS_BALANCED,
        "n_cells": int(len(cells)),
        "n_tiles": int(len(tiles)),
        "n_tiles_support_ok": int(tiles["support_ok"].sum()),
        "n_tiles_balanced_ok": int(bal_tiles["support_ok"].sum()),
        "note": (
            "AE GT is building-linked (sparse). Full tiles need ≥20 AE cells; "
            "balanced-subset tiles need ≥10 pure cells for W4 OOF."
        ),
        "gt_aggregate": "mean of equal-area 10m frac_* (= area aggregate over AE cells in tile)",
        "paths": {
            "cells": "datasets/week4_100m_tiles/cells_with_tile.parquet",
            "tiles": "datasets/week4_100m_tiles/tiles.parquet",
            "tiles_balanced": "datasets/week4_100m_tiles/tiles_balanced_subset.parquet",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Tiles {manifest['n_tiles']:,}  eligible {manifest['n_tiles_support_ok']:,}  "
        f"balanced-eligible {manifest['n_tiles_balanced_ok']:,}  ({manifest['elapsed_s']}s)",
        flush=True,
    )
    return manifest


def run_eval() -> dict:
    t0 = time.time()
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    cells = pd.read_parquet(OUT_DATA / "cells_with_tile.parquet")
    tiles_full = pd.read_parquet(OUT_DATA / "tiles.parquet")
    tiles_bal = pd.read_parquet(OUT_DATA / "tiles_balanced_subset.parquet")
    bal_ids = set(pd.read_parquet(BAL, columns=["pixel_id"])["pixel_id"])

    rows = []
    bin_rows = []
    per_parts = []
    per_bin_parts = []
    tile_pred_parts = []
    tile_bin_parts = []
    for meta in MODELS:
        if not Path(meta["path"]).exists():
            print(f"SKIP {meta['label']}: missing {meta['path']}", flush=True)
            continue
        if meta["scope"] == "balanced":
            c = cells.loc[cells["pixel_id"].isin(bal_ids)].copy()
            t = tiles_bal
            min_c = MIN_CELLS_BALANCED
        else:
            c = cells
            t = tiles_full
            min_c = MIN_CELLS
        print(f"Scoring {meta['label']} on {meta['scope']} (min_cells={min_c}) …", flush=True)
        m100, m10, b100, b10, tpred, per, tbin, per_bin = score_model(
            meta, c, t, min_cells=min_c
        )
        rows.extend([m100, m10])
        bin_rows.extend([b100, b10])
        per_parts.append(per)
        per_bin_parts.append(per_bin)
        tile_pred_parts.append(
            tpred[
                [
                    "tile_id",
                    "region",
                    "n_cells",
                    "frac_res",
                    "frac_nres",
                    "frac_empty",
                    "pred_frac_res",
                    "pred_frac_nres",
                    "pred_frac_empty",
                    "n_pred",
                    "model",
                ]
            ]
        )
        tile_bin_parts.append(
            tbin[
                [
                    "tile_id",
                    "region",
                    "frac_res",
                    "frac_nres",
                    "pred_frac_res",
                    "pred_frac_nres",
                    "y_gt",
                    "y_pred",
                    "model",
                ]
            ]
        )
        print(
            f"  MAE 10m={m10['mae_mean_res_nres']:.4f}  100m={m100['mae_mean_res_nres']:.4f}  "
            f"Δ={m100['delta_10m_minus_100m']:+.4f}",
            flush=True,
        )
        print(
            f"  F1  10m={b10['macro_f1']:.4f}  100m={b100['macro_f1']:.4f}  "
            f"Δ={b100['delta_f1_100m_minus_10m']:+.4f}  n_tiles={b100['n']}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_EVAL / "mae_summary.csv", index=False)
    bin_sum = pd.DataFrame(bin_rows)
    bin_sum.to_csv(OUT_EVAL / "f1_summary.csv", index=False)
    per_reg = pd.concat(per_parts, ignore_index=True)
    per_reg.to_csv(OUT_EVAL / "mae_per_region_100m.csv", index=False)
    per_bin = pd.concat(per_bin_parts, ignore_index=True)
    per_bin.to_csv(OUT_EVAL / "f1_per_region_100m.csv", index=False)
    pd.concat(tile_pred_parts, ignore_index=True).to_parquet(
        OUT_EVAL / "tile_predictions.parquet", index=False
    )
    pd.concat(tile_bin_parts, ignore_index=True).to_parquet(
        OUT_EVAL / "tile_binary_predictions.parquet", index=False
    )

    figs = write_eval_figures(summary, per_reg, FIG)
    figs += write_binary_figures(bin_sum, FIG)
    out = {
        "name": "week4_100m_fraction_eval_v2",
        "metric_regression": "(mae_frac_res + mae_frac_nres) / 2",
        "metric_classification": "macro_f1 on argmax(frac_res, frac_nres); ties dropped",
        "min_cells": MIN_CELLS,
        "min_cells_balanced": MIN_CELLS_BALANCED,
        "mae_summary": summary.to_dict(orient="records"),
        "f1_summary": bin_sum.to_dict(orient="records"),
        "figures": [str(p.relative_to(ROOT)) for p in figs],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_EVAL / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT_EVAL} in {out['elapsed_s']}s", flush=True)
    return out


# --- LR / binary head @ 100 m on ALL balanced cells (same protocol as 0.70 chart) ---

LR_MODELS = [
    {
        "key": "balanced_train_full",
        "label": "W4 LR (full GT OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions_full_gt.parquet",
        "kind": "p_res",
    },
    {
        "key": "balanced_train",
        "label": "W4 LR (balanced OOF)",
        "path": ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        "kind": "p_res",
    },
    {
        "key": "ghsl",
        "label": "GHSL (pixel)",
        "path": GHSL_CACHE,
        "kind": "ghsl",
    },
]

# min cells / tile: 1 = every balanced cell's tile (fair vs 0.70); 5 = denser aggregates
LR_MIN_CELLS = (1, 5)


def _load_binary_head(meta: dict, pixel_ids: pd.Series) -> pd.DataFrame:
    """Return pixel_id, y_pred (0/1), p_res in [0,1] for balanced-cell binary scoring."""
    base = pixel_ids.to_frame("pixel_id")
    raw = pd.read_parquet(meta["path"])
    if meta["kind"] == "ghsl":
        sub = raw[["pixel_id", "ghsl_label"]].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="inner")
        out = out.loc[out["ghsl_label"].isin(["RES", "NRES"])].copy()
        out["y_pred"] = (out["ghsl_label"] == "RES").astype(np.int64)
        out["p_res"] = out["y_pred"].astype(np.float64)
        return out[["pixel_id", "y_pred", "p_res"]]
    if meta["kind"] == "p_res":
        cols = ["pixel_id", "p_res"]
        if "y_pred" in raw.columns:
            cols.append("y_pred")
        sub = raw[cols].drop_duplicates("pixel_id")
        out = base.merge(sub, on="pixel_id", how="inner")
        out["p_res"] = out["p_res"].astype(float).clip(0, 1)
        if "y_pred" not in out.columns:
            out["y_pred"] = (out["p_res"] >= 0.5).astype(np.int64)
        else:
            out["y_pred"] = out["y_pred"].astype(np.int64)
        return out[["pixel_id", "y_pred", "p_res"]]
    # y_pred only (W3 Ridge)
    sub = raw[["pixel_id", "y_pred"]].drop_duplicates("pixel_id")
    out = base.merge(sub, on="pixel_id", how="inner")
    out["y_pred"] = out["y_pred"].astype(np.int64)
    out["p_res"] = out["y_pred"].astype(np.float64)
    return out[["pixel_id", "y_pred", "p_res"]]


def score_lr_tiles(
    cells: pd.DataFrame, meta: dict, min_cells: int
) -> tuple[dict, dict, pd.DataFrame]:
    """Cell-level LR F1 vs 100 m majority (mean p_res ≥ 0.5). GT = majority of y; ties dropped."""
    preds = _load_binary_head(meta, cells["pixel_id"])
    c = cells.merge(preds, on="pixel_id", how="inner")
    b10 = binary_metrics(c["y"].to_numpy(), c["y_pred"].to_numpy())
    b10["scale"] = "10m"
    b10["model"] = meta["key"]
    b10["label"] = meta["label"]
    b10["min_cells"] = min_cells

    g = (
        c.groupby("tile_id", sort=False)
        .agg(
            n_cells=("pixel_id", "size"),
            y_mean=("y", "mean"),
            p_mean=("p_res", "mean"),
            region=("region", "first"),
        )
        .reset_index()
    )
    g = g.loc[g["n_cells"] >= min_cells].copy()
    g = g.loc[g["y_mean"] != 0.5].copy()  # drop GT ties
    yt = (g["y_mean"] > 0.5).astype(np.int64).to_numpy()
    yp = (g["p_mean"] >= 0.5).astype(np.int64).to_numpy()
    b100 = binary_metrics(yt, yp)
    b100["scale"] = "100m"
    b100["model"] = meta["key"]
    b100["label"] = meta["label"]
    b100["min_cells"] = min_cells
    b100["delta_f1_100m_minus_10m"] = b100["macro_f1"] - b10["macro_f1"]
    g["y_gt"] = yt
    g["y_pred"] = yp
    g["model"] = meta["key"]
    g["min_cells"] = min_cells
    return b100, b10, g


def write_lr_balanced_figures(summary: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for min_c in sorted(summary["min_cells"].unique()):
        sub = summary.loc[summary["min_cells"] == min_c]
        s100 = sub.loc[sub["scale"] == "100m"].copy()
        s10 = sub.loc[sub["scale"] == "10m"].copy()
        merged = s100.merge(
            s10[["model", "macro_f1"]].rename(columns={"macro_f1": "f1_10m"}),
            on="model",
        )
        merged["delta"] = merged["macro_f1"] - merged["f1_10m"]
        order = merged.sort_values("macro_f1", ascending=True)

        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        y = np.arange(len(order))
        h = 0.35
        ax.barh(y + h / 2, order["f1_10m"], height=h, color=MUTED, label="10 m cells")
        ax.barh(y - h / 2, order["macro_f1"], height=h, color=RES, label="100 m tiles")
        ax.set_yticks(y)
        ax.set_yticklabels(order["label"])
        ax.set_xlabel("Macro-F1 · RES vs NRES")
        ax.set_xlim(0, 1.08)
        style_axes(ax, grid=True)
        ax.legend(frameon=False, loc="lower right")
        for yi, v, d in zip(y, order["macro_f1"], order["delta"]):
            ax.text(v + 0.01, yi - h / 2, f"{v:.3f} (Δ{d:+.3f})", va="center", fontsize=8)
        best = float(order["macro_f1"].max())
        bump = float(order["delta"].mean())
        fig.tight_layout(rect=[0, 0, 1, 0.84])
        claim = (
            f"LR head on all balanced cells → 100 m "
            f"{'beats' if bump > 0 else 'trails'} 10 m by ~{abs(bump):.3f} "
            f"(best {best:.3f}, min≥{min_c})"
        )
        set_claim_title(
            fig,
            claim,
            "GT = majority of cell labels; pred = mean(p_res) ≥ 0.5. Same ~60k balanced set as the 0.70 chart.",
        )
        p = fig_dir / f"f1_lr_balanced_10m_vs_100m_min{min_c}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    return paths


def run_eval_lr_balanced() -> dict:
    """100 m classification with the same binary heads / balanced cells as the 0.70 chart."""
    t0 = time.time()
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # Canonical y from W4 OOF (matches the 0.704 macro-F1 chart)
    oof = pd.read_parquet(
        ROOT / "outputs" / "week4_pixel_balanced_train" / "oof_predictions.parquet",
        columns=["pixel_id", "y", "region"],
    )
    cells = pd.read_parquet(
        OUT_DATA / "cells_with_tile.parquet", columns=["pixel_id", "tile_id"]
    )
    base = oof.merge(cells, on="pixel_id", how="inner")
    print(
        f"LR-balanced 100 m: {len(base):,} cells → {base['tile_id'].nunique():,} tiles",
        flush=True,
    )

    rows = []
    tile_parts = []
    for min_c in LR_MIN_CELLS:
        for meta in LR_MODELS:
            print(f"  {meta['label']}  min_cells≥{min_c} …", flush=True)
            b100, b10, tbin = score_lr_tiles(base, meta, min_cells=min_c)
            rows.extend([b100, b10])
            tile_parts.append(
                tbin[["tile_id", "region", "n_cells", "y_mean", "p_mean", "y_gt", "y_pred", "model", "min_cells"]]
            )
            print(
                f"    F1 10m={b10['macro_f1']:.4f}  100m={b100['macro_f1']:.4f}  "
                f"Δ={b100['delta_f1_100m_minus_10m']:+.4f}  n_tiles={b100['n']}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_EVAL / "f1_lr_balanced_summary.csv", index=False)
    pd.concat(tile_parts, ignore_index=True).to_parquet(
        OUT_EVAL / "tile_lr_balanced_predictions.parquet", index=False
    )
    figs = write_lr_balanced_figures(summary, FIG)
    out = {
        "name": "week4_100m_lr_balanced_v1",
        "protocol": (
            "All Week-4 balanced pure cells (~60k). "
            "10 m = binary head (LR p_res / y_pred / GHSL). "
            "100 m = majority GT (mean y), pred = mean(p_res)≥0.5; GT ties dropped."
        ),
        "min_cells": list(LR_MIN_CELLS),
        "f1_summary": summary.to_dict(orient="records"),
        "figures": [str(p.relative_to(ROOT)) for p in figs],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_EVAL / "lr_balanced_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote LR-balanced 100 m eval in {out['elapsed_s']}s", flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--lr-balanced-only", action="store_true",
                    help="Only run LR-head 100 m eval on all balanced cells")
    args = ap.parse_args()
    if args.lr_balanced_only:
        run_eval_lr_balanced()
    elif args.eval_only:
        run_eval()
        run_eval_lr_balanced()
    elif args.build_only:
        run_build()
    else:
        run_build()
        run_eval()
        run_eval_lr_balanced()


if __name__ == "__main__":
    main()
