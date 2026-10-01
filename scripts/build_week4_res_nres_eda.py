#!/usr/bin/env python3
"""Week 4 EDA: RES vs NRES pixels, buildings, and footprint area by metro.

Sources
  - datasets/week4_pixel_gt/pixels.parquet — AE 10 m cells (pure counts + area_*_m2)
  - datasets/building_bbox_grid/buildings.parquet — labeled Track-A buildings

Writes outputs/week4_res_nres_eda/{summary tables, figures, manifest}.

Usage:
  .venv/bin/python -u scripts/build_week4_res_nres_eda.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from week4_viz import NRES, RES, set_claim_title, style_axes  # noqa: E402

PIX = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
BLD = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT = ROOT / "outputs" / "week4_res_nres_eda"
FIG = OUT / "figures"


def _region_lab(s: pd.Series | pd.Index) -> list[str]:
    return [str(x).replace("_", " ") for x in s]


def pixel_table(df: pd.DataFrame) -> pd.DataFrame:
    pure_res = (df["frac_res_full"] > 0) & (df["frac_nres_full"] == 0)
    pure_nres = (df["frac_nres_full"] > 0) & (df["frac_res_full"] == 0)
    mixed = (df["frac_res_full"] > 0) & (df["frac_nres_full"] > 0)
    emptyish = (df["frac_res_full"] == 0) & (df["frac_nres_full"] == 0)

    rows = []
    for region, sub in df.groupby("region", sort=True):
        pr = pure_res.loc[sub.index]
        pn = pure_nres.loc[sub.index]
        mx = mixed.loc[sub.index]
        em = emptyish.loc[sub.index]
        rows.append(
            {
                "region": region,
                "n_pixels": int(len(sub)),
                "n_pure_res": int(pr.sum()),
                "n_pure_nres": int(pn.sum()),
                "n_mixed": int(mx.sum()),
                "n_empty_overlap": int(em.sum()),
                "area_res_m2": float(sub["area_res_m2"].sum()),
                "area_nres_m2": float(sub["area_nres_m2"].sum()),
                "pct_pure_res": float(100.0 * pr.mean()),
                "pct_pure_nres": float(100.0 * pn.mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    tot = {
        "region": "ALL",
        "n_pixels": int(out["n_pixels"].sum()),
        "n_pure_res": int(out["n_pure_res"].sum()),
        "n_pure_nres": int(out["n_pure_nres"].sum()),
        "n_mixed": int(out["n_mixed"].sum()),
        "n_empty_overlap": int(out["n_empty_overlap"].sum()),
        "area_res_m2": float(out["area_res_m2"].sum()),
        "area_nres_m2": float(out["area_nres_m2"].sum()),
    }
    tot["pct_pure_res"] = 100.0 * tot["n_pure_res"] / tot["n_pixels"]
    tot["pct_pure_nres"] = 100.0 * tot["n_pure_nres"] / tot["n_pixels"]
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


def building_table(bld: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, sub in bld.groupby("region", sort=True):
        res = sub.loc[sub["y_res"] == 1]
        nres = sub.loc[sub["y_res"] == 0]
        rows.append(
            {
                "region": region,
                "n_buildings": int(len(sub)),
                "n_res_buildings": int(len(res)),
                "n_nres_buildings": int(len(nres)),
                "area_res_m2_bld": float(res["area_m2_approx"].sum()),
                "area_nres_m2_bld": float(nres["area_m2_approx"].sum()),
                "pct_res_buildings": float(100.0 * len(res) / max(len(sub), 1)),
            }
        )
    out = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    tot = {
        "region": "ALL",
        "n_buildings": int(out["n_buildings"].sum()),
        "n_res_buildings": int(out["n_res_buildings"].sum()),
        "n_nres_buildings": int(out["n_nres_buildings"].sum()),
        "area_res_m2_bld": float(out["area_res_m2_bld"].sum()),
        "area_nres_m2_bld": float(out["area_nres_m2_bld"].sum()),
    }
    tot["pct_res_buildings"] = 100.0 * tot["n_res_buildings"] / tot["n_buildings"]
    return pd.concat([out, pd.DataFrame([tot])], ignore_index=True)


def write_figures(pix: pd.DataFrame, bld: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    pmetro = pix.loc[pix["region"] != "ALL"].copy()
    bmetro = bld.loc[bld["region"] != "ALL"].copy()
    labs = _region_lab(pmetro["region"])

    # 1) pure pixel counts
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y = np.arange(len(pmetro))
    h = 0.35
    ax.barh(y + h / 2, pmetro["n_pure_nres"], height=h, color=NRES, label="pure NRES cells")
    ax.barh(y - h / 2, pmetro["n_pure_res"], height=h, color=RES, label="pure RES cells")
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("AE 10 m cells")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    n_r = int(pmetro["n_pure_res"].sum())
    n_n = int(pmetro["n_pure_nres"].sum())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"Pure AE cells: {n_n:,} NRES vs {n_r:,} RES (NRES is {100*n_n/(n_n+n_r):.0f}% of pure cells)",
        "Pure = only one class overlaps the cell (frac_*_full). Every metro is NRES-heavy.",
    )
    p = fig_dir / "pure_pixels_by_region.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 2) building counts
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    labs_b = _region_lab(bmetro["region"])
    y = np.arange(len(bmetro))
    ax.barh(y + h / 2, bmetro["n_nres_buildings"], height=h, color=NRES, label="NRES buildings")
    ax.barh(y - h / 2, bmetro["n_res_buildings"], height=h, color=RES, label="RES buildings")
    ax.set_yticks(y)
    ax.set_yticklabels(labs_b)
    ax.set_xlabel("labeled Track-A buildings")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    br = int(bmetro["n_res_buildings"].sum())
    bn = int(bmetro["n_nres_buildings"].sum())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"Buildings: {bn:,} NRES vs {br:,} RES ({100*bn/(bn+br):.0f}% NRES)",
        "Source: datasets/building_bbox_grid/buildings.parquet (y_res).",
    )
    p = fig_dir / "buildings_by_region.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 3) AE-grid footprint area
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y = np.arange(len(pmetro))
    ax.barh(y + h / 2, pmetro["area_nres_m2"] / 1e6, height=h, color=NRES, label="NRES area")
    ax.barh(y - h / 2, pmetro["area_res_m2"] / 1e6, height=h, color=RES, label="RES area")
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("footprint area on AE grid (km²)")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    ar = float(pmetro["area_res_m2"].sum())
    an = float(pmetro["area_nres_m2"].sum())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"AE-grid footprint area: NRES {an/1e6:.2f} km² vs RES {ar/1e6:.2f} km²",
        "Sum of area_res_m2 / area_nres_m2 over Week-4 AE cells (polygon ∩ cell).",
    )
    p = fig_dir / "ae_footprint_area_by_region.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 4) building footprint area (approx)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y = np.arange(len(bmetro))
    ax.barh(y + h / 2, bmetro["area_nres_m2_bld"] / 1e6, height=h, color=NRES, label="NRES buildings")
    ax.barh(y - h / 2, bmetro["area_res_m2_bld"] / 1e6, height=h, color=RES, label="RES buildings")
    ax.set_yticks(y)
    ax.set_yticklabels(labs_b)
    ax.set_xlabel("building footprint area (km²)")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    ar = float(bmetro["area_res_m2_bld"].sum())
    an = float(bmetro["area_nres_m2_bld"].sum())
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        f"Building footprints: NRES {an/1e6:.2f} km² vs RES {ar/1e6:.2f} km²",
        "Sum of area_m2_approx on labeled Track-A buildings.",
    )
    p = fig_dir / "building_area_by_region.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 5) stacked share of pure pixels
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    share_n = pmetro["n_pure_nres"] / (pmetro["n_pure_res"] + pmetro["n_pure_nres"])
    share_r = 1.0 - share_n
    y = np.arange(len(pmetro))
    ax.barh(y, share_n, color=NRES, label="NRES", height=0.7)
    ax.barh(y, share_r, left=share_n, color=RES, label="RES", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of pure AE cells")
    style_axes(ax, grid=True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    set_claim_title(
        fig,
        "Every metro’s pure-cell mix is mostly NRES",
        "Shares within pure RES+NRES only (mixed/empty excluded).",
    )
    p = fig_dir / "pure_pixel_share_by_region.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 6) totals strip
    tall = pix.loc[pix["region"] == "ALL"].iloc[0]
    ball = bld.loc[bld["region"] == "ALL"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, title, vals, ylab in [
        (
            axes[0],
            "Pure AE cells",
            [tall["n_pure_res"], tall["n_pure_nres"]],
            "cells",
        ),
        (
            axes[1],
            "Buildings",
            [ball["n_res_buildings"], ball["n_nres_buildings"]],
            "buildings",
        ),
        (
            axes[2],
            "AE-grid area (km²)",
            [tall["area_res_m2"] / 1e6, tall["area_nres_m2"] / 1e6],
            "km²",
        ),
    ]:
        bars = ax.bar(["RES", "NRES"], vals, color=[RES, NRES], width=0.65)
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:,.0f}" if v >= 100 else f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        style_axes(ax, grid=True)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    set_claim_title(
        fig,
        "Totals across 10 metros: NRES dominates cells, buildings, and area",
        "Left/right: Week-4 AE GT. Middle: Track-A labeled buildings.",
    )
    p = fig_dir / "totals_res_vs_nres.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("Loading pixels …", flush=True)
    pix_raw = pd.read_parquet(
        PIX,
        columns=[
            "region",
            "frac_res_full",
            "frac_nres_full",
            "area_res_m2",
            "area_nres_m2",
            "pixel_area_m2",
        ],
    )
    print("Loading buildings …", flush=True)
    bld_raw = pd.read_parquet(
        BLD, columns=["id", "region", "y_res", "area_m2_approx", "label", "class"]
    )

    pix = pixel_table(pix_raw)
    bld = building_table(bld_raw)
    merged = pix.merge(bld, on="region", how="outer")

    pix.to_csv(OUT / "pixels_by_region.csv", index=False)
    bld.to_csv(OUT / "buildings_by_region.csv", index=False)
    merged.to_csv(OUT / "summary_by_region.csv", index=False)

    figs = write_figures(pix, bld, FIG)
    tall = pix.loc[pix["region"] == "ALL"].iloc[0]
    ball = bld.loc[bld["region"] == "ALL"].iloc[0]
    manifest = {
        "name": "week4_res_nres_eda_v1",
        "sources": {
            "pixels": str(PIX.relative_to(ROOT)),
            "buildings": str(BLD.relative_to(ROOT)),
        },
        "notes": {
            "pure_pixels": "frac_res_full>0 & frac_nres_full==0 (or reverse)",
            "pixel_area": "sum of polygon∩cell areas on the AE 10 m grid",
            "buildings": "labeled Track-A set in building_bbox_grid (not raw Overture dump)",
            "building_area": "sum of area_m2_approx",
        },
        "totals": {
            "n_pure_res": int(tall["n_pure_res"]),
            "n_pure_nres": int(tall["n_pure_nres"]),
            "n_res_buildings": int(ball["n_res_buildings"]),
            "n_nres_buildings": int(ball["n_nres_buildings"]),
            "area_res_m2_ae": float(tall["area_res_m2"]),
            "area_nres_m2_ae": float(tall["area_nres_m2"]),
            "area_res_m2_bld": float(ball["area_res_m2_bld"]),
            "area_nres_m2_bld": float(ball["area_nres_m2_bld"]),
        },
        "figures": [str(p.relative_to(ROOT)) for p in figs],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["totals"], indent=2), flush=True)
    print(f"Wrote {OUT} in {manifest['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
