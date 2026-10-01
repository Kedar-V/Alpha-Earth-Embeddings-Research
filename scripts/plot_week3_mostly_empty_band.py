#!/usr/bin/env python3
"""Mostly-empty GT band: analysis + Overture overlay samples.

Band: frac_res < 10% AND frac_nres < 10% AND frac_empty > 90%.
Writes figures + JSON under outputs/week3_pixel_3class/.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely import wkt as shapely_wkt
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
PIX = ROOT / "datasets" / "week3_pixel_gt_v2" / "pixels.parquet"
BLD = ROOT / "datasets" / "building_bbox_grid" / "buildings.parquet"
OUT = ROOT / "outputs" / "week3_pixel_3class"
FIG = OUT / "figures"


def parse_ids(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i) for i in x if str(i) not in ("", "nan", "None")]
    s = str(x).strip()
    if s in ("", "[]", "None", "nan"):
        return []
    if s.startswith("["):
        try:
            return [str(i) for i in ast.literal_eval(s)]
        except Exception:
            return []
    return [s]


def is_res_label(lab: str) -> bool:
    lab = str(lab).lower()
    return "res" in lab and "non" not in lab


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PIX)
    fr, fn, fe = df["frac_res_full"], df["frac_nres_full"], df["frac_empty_full"]
    band = (fr < 0.10) & (fn < 0.10) & (fe > 0.90)
    sub = df.loc[band].copy()
    sub["building_share"] = (fr[band] + fn[band]).to_numpy()
    sub["is_true_empty"] = sub["building_share"] <= 0.0
    sub["is_partial"] = ~sub["is_true_empty"]

    n = len(sub)
    n_true = int(sub["is_true_empty"].sum())
    n_partial = int(sub["is_partial"].sum())
    analysis = {
        "filter": "frac_res_full < 0.10 AND frac_nres_full < 0.10 AND frac_empty_full > 0.90",
        "n_band": n,
        "n_true_empty_zero_overlap": n_true,
        "n_partial_footprint": n_partial,
        "pct_should_be_100pct_empty": round(100.0 * n_true / n, 2),
        "pct_should_NOT_be_100pct_empty": round(100.0 * n_partial / n, 2),
        "label_counts_in_band": {k: int(v) for k, v in sub["label"].value_counts().items()},
        "partial_building_share": {
            "mean": float(sub.loc[sub.is_partial, "building_share"].mean()),
            "median": float(sub.loc[sub.is_partial, "building_share"].median()),
            "p90": float(sub.loc[sub.is_partial, "building_share"].quantile(0.9)),
            "max": float(sub.loc[sub.is_partial, "building_share"].max()),
        },
        "by_region": {
            r: {
                "n": int(len(g)),
                "true_empty": int(g.is_true_empty.sum()),
                "partial": int(g.is_partial.sum()),
                "pct_true_empty": round(100.0 * g.is_true_empty.mean(), 1),
            }
            for r, g in sub.groupby("region")
        },
        "figures": [
            "mostly_empty_band_overture_samples.png",
            "mostly_empty_band_partial_share_hist.png",
            "mostly_empty_band_by_region.png",
        ],
    }
    (OUT / "mostly_empty_band_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    bld = pd.read_parquet(BLD)
    bld["geometry"] = bld["wkt"].map(shapely_wkt.loads)
    bg = gpd.GeoDataFrame(bld, geometry="geometry", crs="EPSG:4326")
    id_to_geom = dict(zip(bg["id"].astype(str), bg.geometry))

    def cell_view(row, pad_m=35.0):
        pix = shapely_wkt.loads(row.pixel_wkt)
        lat = row.pixel_lat
        dlat = pad_m / 111_320.0
        dlon = pad_m / (111_320.0 * max(0.2, np.cos(np.deg2rad(lat))))
        view = box(pix.bounds[0] - dlon, pix.bounds[1] - dlat, pix.bounds[2] + dlon, pix.bounds[3] + dlat)
        near = bg[(bg["region"] == row.region) & bg.intersects(view)]
        return pix, view, near

    true_s = sub.loc[sub.is_true_empty].sample(n=min(6, n_true), random_state=42)
    part_s = (
        sub.loc[sub.is_partial]
        .sort_values("building_share", ascending=False)
        .groupby("region", group_keys=False)
        .head(2)
    )
    if len(part_s) > 6:
        part_s = part_s.sample(n=6, random_state=0)
    samples = pd.concat([true_s, part_s])

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, (_, row) in zip(axes.ravel(), samples.iterrows()):
        pix, view, near = cell_view(row)
        minx, miny, maxx, maxy = view.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        for _, br in near.iterrows():
            g = br.geometry
            poly = g if g.geom_type == "Polygon" else g.geoms[0]
            xs, ys = poly.exterior.xy
            color = "#c45c26" if is_res_label(br.get("label", "")) else "#2b6a8f"
            ax.fill(xs, ys, facecolor=color, edgecolor="#222", alpha=0.45, lw=0.6)
        xs, ys = pix.exterior.xy
        face = "#2a6f4e" if row.is_true_empty else "#b8860b"
        ax.fill(xs, ys, facecolor=face, edgecolor="black", alpha=0.35, lw=1.4)
        ax.plot(xs, ys, color="black", lw=1.2)
        ax.set_title(
            f"{row.region} · {row.label}\n"
            f"r={row.frac_res_full:.3f} n={row.frac_nres_full:.3f} e={row.frac_empty_full:.3f}",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        tag = "TRUE EMPTY\n(no overlap)" if row.is_true_empty else "PARTIAL\n(edge clip)"
        ax.text(
            0.03,
            0.97,
            tag,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#444", alpha=0.9),
        )
    handles = [
        Patch(facecolor="#2a6f4e", alpha=0.5, edgecolor="k", label="Pixel · true empty (0% building)"),
        Patch(facecolor="#b8860b", alpha=0.5, edgecolor="k", label="Pixel · partial (<10% building)"),
        Patch(facecolor="#c45c26", alpha=0.45, edgecolor="#222", label="Overture RES"),
        Patch(facecolor="#2b6a8f", alpha=0.45, edgecolor="#222", label="Overture NRES"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        "GT cells with frac_res<10%, frac_nres<10%, frac_empty>90%\n"
        f"{n_true:,}/{n:,} ({100 * n_true / n:.1f}%) zero polygon overlap → ideally 100% FracEmpty; "
        f"{n_partial:,} ({100 * n_partial / n:.1f}%) are real partial footprints",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(FIG / "mostly_empty_band_overture_samples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub.loc[sub.is_partial, "building_share"], bins=20, color="#b8860b", edgecolor="white")
    ax.axvline(0.05, color="#8b3a3a", ls="--", label="5% decode margin")
    ax.axvline(0.10, color="#222", ls=":", label="10% band edge")
    ax.set_xlabel("GT building share (frac_res + frac_nres)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Partial-overlap cells inside mostly-empty band (n={n_partial:,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mostly_empty_band_partial_share_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    reg = (
        sub.groupby("region")
        .agg(true_empty=("is_true_empty", "sum"), partial=("is_partial", "sum"))
        .sort_values("true_empty", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(reg))
    ax.barh(ypos, reg["true_empty"], color="#2a6f4e", label="True empty (0% overlap)")
    ax.barh(ypos, reg["partial"], left=reg["true_empty"], color="#b8860b", label="Partial footprint")
    ax.set_yticks(ypos)
    ax.set_yticklabels(reg.index)
    ax.set_xlabel("Pixels in band (res<10%, nres<10%, empty>90%)")
    ax.set_title("Mostly-empty band: true empty vs partial by metro")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "mostly_empty_band_by_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: analysis[k] for k in analysis if k != "by_region"}, indent=2))
    print("Wrote", FIG)
    return analysis


if __name__ == "__main__":
    main()
