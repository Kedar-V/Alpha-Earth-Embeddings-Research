#!/usr/bin/env python3
"""Build a RES/NRES count-balanced pixel subset from week4 fraction GT.

Strata (pure coverage only):
  residential     : frac_res_full > 0 and frac_nres_full == 0
  non_residential : frac_nres_full > 0 and frac_res_full == 0

Natural (unstratified) fraction GT is RES-majority. Per region: undersample the
majority class to the minority count (seed=42) — typically undersample RES.
Writes datasets/week4_pixel_gt_balanced/{pixels.parquet, manifest.json, figures/}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "week4_pixel_gt" / "pixels.parquet"
OUT = ROOT / "datasets" / "week4_pixel_gt_balanced"
FIG = OUT / "figures"
SEED = 42
LABEL_DEFINITION = "resi_def_v1_pixel_frac_v1_balanced_pure"


def assign_stratum(df: pd.DataFrame) -> pd.Series:
    res = (df["frac_res_full"].astype(float) > 0) & (df["frac_nres_full"].astype(float) == 0)
    nres = (df["frac_nres_full"].astype(float) > 0) & (df["frac_res_full"].astype(float) == 0)
    out = pd.Series("other", index=df.index, dtype=object)
    out.loc[res] = "residential"
    out.loc[nres] = "non_residential"
    return out


def balance_per_region(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """1:1 per metro by undersampling the majority pure class (usually RES)."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for region, sub in df.groupby("region", sort=True):
        res = sub.loc[sub["stratum"] == "residential"]
        nres = sub.loc[sub["stratum"] == "non_residential"]
        n = min(len(res), len(nres))
        if n == 0:
            print(f"  skip {region}: res={len(res)} nres={len(nres)}", flush=True)
            continue
        if len(res) > n:
            pick = rng.choice(res.index.to_numpy(), size=n, replace=False)
            res = res.loc[pick]
        if len(nres) > n:
            pick = rng.choice(nres.index.to_numpy(), size=n, replace=False)
            nres = nres.loc[pick]
        maj = "RES" if len(sub.loc[sub["stratum"] == "residential"]) >= len(
            sub.loc[sub["stratum"] == "non_residential"]
        ) else "NRES"
        bal = pd.concat([res, nres], ignore_index=True)
        bal["balance_n_per_class"] = n
        parts.append(bal)
        print(
            f"  {region}: RES={n} NRES={n} (undersampled {maj}; "
            f"from pure res={int((sub.stratum=='residential').sum())} "
            f"nres={int((sub.stratum=='non_residential').sum())})",
            flush=True,
        )
    if not parts:
        raise RuntimeError("No balanced regions produced")
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def write_qa(df: pd.DataFrame, raw_counts: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    C_RES, C_NRES = "#0072B2", "#E69F00"

    ct = (
        df.groupby(["region", "stratum"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["residential", "non_residential"], fill_value=0)
    )
    ct.index = [str(i).replace("_", " ") for i in ct.index]
    ct = ct.sort_values("residential", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(ct))
    h = 0.38
    ax.barh(y + h / 2, ct["residential"], height=h, color=C_RES, label="residential")
    ax.barh(y - h / 2, ct["non_residential"], height=h, color=C_NRES, label="non-residential")
    ax.set_yticks(y)
    ax.set_yticklabels(ct.index)
    ax.set_xlabel("Pixels")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    equal = bool((ct["residential"] == ct["non_residential"]).all())
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(
        0.01, 0.98,
        "After balancing, every metro has equal RES and NRES pure cells"
        if equal
        else "Balanced counts by metro",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        "Blue = residential, orange = non-residential (majority undersampled to minority).",
        fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "balanced_counts_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    raw = raw_counts.reindex(columns=["residential", "non_residential"], fill_value=0).copy()
    raw.index = [str(i).replace("_", " ") for i in raw.index]
    raw["ratio_res"] = raw["residential"] / raw["non_residential"].clip(lower=1)
    raw = raw.sort_values("ratio_res", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(raw))
    h = 0.38
    ax.barh(y + h / 2, raw["residential"], height=h, color=C_RES, label="residential")
    ax.barh(y - h / 2, raw["non_residential"], height=h, color=C_NRES, label="non-residential")
    ax.set_yticks(y)
    ax.set_yticklabels(raw.index)
    ax.set_xlabel("Pixels")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    med_ratio = float(raw["ratio_res"].median())
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.text(
        0.01, 0.98,
        f"Before balancing, RES outnumbers NRES by ~{med_ratio:.1f}× (median metro)",
        fontsize=12, fontweight="bold", ha="left", va="top", transform=fig.transFigure,
    )
    fig.text(
        0.01, 0.935,
        "Pure cells only. Natural unstratified GT; sorted by RES:NRES ratio.",
        fontsize=9, color="#555555", ha="left", va="top", transform=fig.transFigure,
    )
    p = fig_dir / "imbalanced_counts_by_region.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(p)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    src = pd.read_parquet(SRC)
    src["stratum"] = assign_stratum(src)
    pure = src.loc[src["stratum"].isin(["residential", "non_residential"])].copy()
    raw_counts = (
        pure.groupby(["region", "stratum"]).size().unstack(fill_value=0)
    )
    n_res0 = int((pure.stratum == "residential").sum())
    n_nres0 = int((pure.stratum == "non_residential").sum())
    print(
        f"Source {SRC.relative_to(ROOT)} n={len(src):,}  "
        f"pure_res={n_res0:,}  pure_nres={n_nres0:,}  "
        f"(majority={'RES' if n_res0 >= n_nres0 else 'NRES'})",
        flush=True,
    )

    bal = balance_per_region(pure, seed=args.seed)
    bal["y"] = (bal["stratum"] == "residential").astype(np.int64)
    bal["label_definition"] = LABEL_DEFINITION

    out_path = OUT / "pixels.parquet"
    bal.to_parquet(out_path, index=False)

    fig_paths = write_qa(bal, raw_counts, FIG)
    counts = bal.groupby(["region", "stratum"]).size().unstack(fill_value=0).to_dict()
    manifest = {
        "name": "week4_pixel_gt_balanced_v1",
        "purpose": (
            "Per-region count-balanced pure RES/NRES pixels from week4 fraction GT. "
            "Undersample majority (typically RES under natural unstratified GT) to minority; seed fixed."
        ),
        "label_definition": LABEL_DEFINITION,
        "stratum_rule": {
            "residential": "frac_res_full > 0 and frac_nres_full == 0",
            "non_residential": "frac_nres_full > 0 and frac_res_full == 0",
            "excluded": "mixed overlap or non-pure",
        },
        "balance_rule": (
            f"per region: n = min(n_res, n_nres); undersample majority; seed={args.seed}"
        ),
        "source": str(SRC.relative_to(ROOT)),
        "seed": args.seed,
        "n_source": int(len(src)),
        "n_pure_before": int(len(pure)),
        "n_pure_res_before": n_res0,
        "n_pure_nres_before": n_nres0,
        "n_pixels": int(len(bal)),
        "n_residential": int((bal["stratum"] == "residential").sum()),
        "n_non_residential": int((bal["stratum"] == "non_residential").sum()),
        "n_regions": int(bal["region"].nunique()),
        "regions": sorted(bal["region"].astype(str).unique().tolist()),
        "counts_by_region": {
            r: {
                "residential": int(counts.get("residential", {}).get(r, 0)),
                "non_residential": int(counts.get("non_residential", {}).get(r, 0)),
            }
            for r in sorted(bal["region"].astype(str).unique())
        },
        "frac_means": {
            "frac_res": float(bal["frac_res"].mean()),
            "frac_nres": float(bal["frac_nres"].mean()),
            "frac_empty": float(bal["frac_empty"].mean()),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "paths": {
            "pixels": str(out_path.relative_to(ROOT)),
            "manifest": str((OUT / "manifest.json").relative_to(ROOT)),
            "figures": [str(p.relative_to(ROOT)) for p in fig_paths],
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {out_path} n={len(bal):,}  "
        f"RES={manifest['n_residential']:,} NRES={manifest['n_non_residential']:,}  "
        f"{manifest['elapsed_s']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
