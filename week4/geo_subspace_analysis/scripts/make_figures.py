#!/usr/bin/env python3
"""Redraw the paper figures from saved tables. No model refits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import lib

BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#222222"


def _names(idx):
    return [lib.NAMES.get(x, x) for x in idx]


def _heat(ax, mat: pd.DataFrame, vmin, vmax, cmap, title: str):
    order = list(mat.index)
    arr = mat.loc[order, order].to_numpy()
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(_names(order), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(_names(order), fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--stage4", required=True)
    ap.add_argument("--stage5", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, s4 = Path(args.stage1), Path(args.stage2), Path(args.stage3), Path(args.stage4)

    ksum = pd.read_csv(s1 / "k_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, color, label in (("pls", BLUE, "PLS"), ("pca", ORANGE, "PCA")):
        sub = ksum[ksum["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["macro_f1_mean"], color=color, lw=1.8, marker="o", ms=4, label=label)
    base = float(pd.read_json(s1 / "k_picks.json", typ="series")["baseline_macro_f1_mean"]) if False else None
    picks = __import__("json").loads((s1 / "k_picks.json").read_text())
    base = picks["baseline_macro_f1_mean"]
    ax.axhline(base, color="#333333", lw=0.8, ls="--")
    ax.set_xlabel("Components")
    ax.set_ylabel("LORO mean macro-F1")
    ax.set_title("PLS keeps the 64-D F1 at a lower width than PCA", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "f1_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, color, label in (("pls", BLUE, "PLS"), ("pca", ORANGE, "PCA")):
        sub = ksum[ksum["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["retention_of_mean"], color=color, lw=1.8, marker="o", ms=4, label=label)
    ax.axhline(0.95, color="#333333", lw=0.8, ls="--")
    ax.set_ylim(0.7, 1.02)
    ax.set_xlabel("Components")
    ax.set_ylabel("F1(k) / F1(64-D)")
    ax.set_title("Share of the 64-D F1 retained at width k", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "retention_vs_k.png", dpi=150)
    plt.close(fig)

    base_df = pd.read_csv(s1 / "baseline_loro.csv").sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(_names(base_df["region"]), base_df["macro_f1"], color=BLUE)
    ax.axvline(base_df["macro_f1"].mean(), color="#333333", lw=0.8, ls="--")
    ax.set_xlabel("Macro-F1")
    ax.set_title("64-D logistic regression, leave-one-metro-out", loc="left", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "baseline_f1_by_metro.png", dpi=150)
    plt.close(fig)

    for k in (2, 9):
        sim = pd.read_csv(s2 / f"projection_similarity_k{k}.csv", index_col=0)
        ang = pd.read_csv(s2 / f"principal_angle_first_deg_k{k}.csv", index_col=0)
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
        im0 = _heat(axes[0], sim, 0, 1, "Blues", f"PLS subspace similarity, k={k}")
        im1 = _heat(axes[1], ang, 0, 90, "Oranges", f"First principal angle (degrees), k={k}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"subspace_k{k}.png", dpi=150)
        plt.close(fig)
        dist = 1 - sim.to_numpy()
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)
        order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
        ordered = sim.index.to_numpy()[order]
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = _heat(ax, sim.loc[ordered, ordered], 0, 1, "Blues", f"Metros clustered by PLS subspace distance, k={k}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"subspace_cluster_k{k}.png", dpi=150)
        plt.close(fig)

    cos = pd.read_csv(s2 / "class_direction_cosine.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = _heat(ax, cos, -1, 1, "coolwarm", "Cosine of residential class-mean directions")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "class_direction_cosine.png", dpi=150)
    plt.close(fig)

    R = pd.read_csv(s4 / "transfer_retention_lr64.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = _heat(ax, R, 0.4, 1.05, "Blues", "Transfer retention  T(A→B) / T(B,B), 64-D")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "transfer_retention_lr64.png", dpi=150)
    plt.close(fig)

    for label, fname, title in (
        ("lr64_k64", "transfer_f1_lr64.png", "64-D logistic regression, train A test B"),
        ("pls_k2", "transfer_f1_pls_k2.png", "PLS k=2 then logistic regression, train A test B"),
        ("pls_k9", "transfer_f1_pls_k9.png", "PLS k=9 then logistic regression, train A test B"),
    ):
        T = pd.read_csv(s3 / f"transfer_f1_{label}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = _heat(ax, T, 0.3, 0.95, "Blues", title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    pairs = pd.read_csv(s4 / "pair_table.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(pairs["pls_similarity_k2"], pairs["retention_sym"], s=28, c=BLUE)
    ax.set_xlabel("PLS subspace similarity, k=2")
    ax.set_ylabel("Symmetric transfer retention")
    ax.set_title("Subspace alignment against 64-D transfer", loc="left", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "similarity_vs_retention_k2.png", dpi=150)
    plt.close(fig)

    if "class_direction_cosine" in pairs.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(pairs["class_direction_cosine"], pairs["retention_sym"], s=28, c=BLUE)
        ax.set_xlabel("Cosine of class-mean directions")
        ax.set_ylabel("Symmetric transfer retention")
        ax.set_title("Class-direction alignment against 64-D transfer", loc="left", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "class_direction_vs_retention.png", dpi=150)
        plt.close(fig)

    if "pls_similarity_k9" in pairs.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(pairs["pls_similarity_k9"], pairs["retention_sym"], s=28, c=ORANGE)
        ax.set_xlabel("PLS subspace similarity, k=9")
        ax.set_ylabel("Symmetric transfer retention")
        ax.set_title("k=9 subspace alignment against 64-D transfer", loc="left", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "similarity_vs_retention_k9.png", dpi=150)
        plt.close(fig)

    corr = pd.read_csv(s4 / "correlation_table.csv")
    show = corr[corr["predictor"].isin(
        [
            "pls_similarity_k2",
            "pls_similarity_k9",
            "class_direction_cosine",
            "aef_mmd",
            "aef_mean_distance",
            "geo_distance_km",
        ]
    )].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.barh(show["predictor"], show["spearman_rho"], color=BLUE)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Spearman rho with symmetric transfer retention")
    ax.set_title("What lines up with cross-metro transfer", loc="left", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "predictor_spearman.png", dpi=150)
    plt.close(fig)

    probe = pd.read_csv(s4 / "metro_vs_building_probe.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(probe))
    ax.bar(x - 0.18, probe["building_macro_f1"], width=0.36, color=BLUE, label="Building-function F1")
    ax.bar(x + 0.18, probe["metro_accuracy"], width=0.36, color=ORANGE, label="Metro accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(probe["representation"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Building-function signal and metro identity in the same rows", loc="left", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "metro_vs_building.png", dpi=150)
    plt.close(fig)

    if args.stage5:
        s5 = Path(args.stage5)
        met = pd.read_csv(s5 / "metrics.csv")
        met = met[met["phase"] == "eval"]
        g = met.groupby(["d", "method"])["heldout_macro_f1"].mean().unstack("method")
        order = [m for m in ("class", "supcon", "xsupcon", "align") if m in g.columns]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        g[order].plot(kind="bar", ax=ax, color=[BLUE, ORANGE, "#009E73", "#CC79A7"][: len(order)], width=0.8)
        ax.set_xlabel("Projection width")
        ax.set_ylabel("Held-out metro macro-F1")
        ax.set_title("Learned projections, mean over metros and seeds", loc="left", fontsize=12)
        ax.legend(frameon=False, title=None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "learned_projection_f1.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(met["source_direction_cosine_mean"], met["heldout_macro_f1"], s=18, c=BLUE, alpha=0.7)
        ax.set_xlabel("Mean cosine of source-metro class directions")
        ax.set_ylabel("Held-out metro macro-F1")
        ax.set_title("Source alignment against a metro the map never saw", loc="left", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "learned_alignment_vs_heldout.png", dpi=150)
        plt.close(fig)

    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
