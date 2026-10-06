#!/usr/bin/env python3
"""Build the three-question paper-plots notebook from saved CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PKG = Path(__file__).resolve().parents[1]
S1 = PKG / "outputs/baseline_loro/20261002T0015"
S3 = PKG / "outputs/transfer_matrix/20261002T0015"
PAPER = PKG / "outputs/paper_curves/20261002T1100"
FIG = PKG / "figures/paper"
NB = PKG / "notebooks/01_three_questions.ipynb"

BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#222222"
MUTED = "#555555"

NAMES = {
    "Atlanta_SE": "Atlanta",
    "Boston_NE": "Boston",
    "Chicago_MW": "Chicago",
    "Denver_W": "Denver",
    "Houston_S": "Houston",
    "LA_W": "Los Angeles",
    "NYC_NE": "New York",
    "Phoenix_SW": "Phoenix",
    "SF_W": "San Francisco",
    "Seattle_NW": "Seattle",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")


def _names(idx):
    return [NAMES.get(x, x) for x in idx]


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


def _retention(mat: pd.DataFrame) -> pd.DataFrame:
    a = mat.to_numpy(dtype=float)
    diag = np.diag(a).copy()
    if np.any(diag <= 0):
        raise AssertionError("non-positive diagonal")
    return pd.DataFrame(a / diag[None, :], index=mat.index, columns=mat.columns)


def make_figures() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Retention vs k
    ksum = pd.read_csv(S1 / "k_summary.csv")
    picks = json.loads((S1 / "k_picks.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for model, color, label in (("pls", BLUE, "PLS → LR"), ("pca", ORANGE, "PCA → LR")):
        sub = ksum[ksum["model"] == model].sort_values("k")
        ax.plot(sub["k"], sub["retention_of_mean"], color=color, lw=2.0, marker="o", ms=4.5, label=label)
    ax.axhline(0.95, color="#333333", lw=0.8, ls="--")
    ax.text(64, 0.952, "95% of 64-D F1", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_ylim(0.72, 1.03)
    ax.set_xlabel("Components k")
    ax.set_ylabel(r"$R(k)=F1_k\,/\,F1_{64}$")
    ax.set_title("Building-function signal sits in a short supervised subspace", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    p = FIG / "01_retention_vs_k.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["retention"] = p
    paths["pls_k95"] = picks["pls"]["smallest_k_retention_ge_0.95"]
    paths["pca_k95"] = picks["pca"]["smallest_k_retention_ge_0.95"]
    paths["pls2_ret"] = float(ksum[(ksum.model == "pls") & (ksum.k == 2)]["retention_of_mean"].iloc[0])

    # 2. Pareto: metro accuracy vs building F1
    probe_path = PAPER / "metro_vs_building_probe.csv"
    if not probe_path.exists():
        probe_path = PKG / "outputs/statistical_tests/20261002T0031/metro_vs_building_probe.csv"
    probe = pd.read_csv(probe_path)
    if "kind" not in probe.columns:
        probe = probe.copy()
        probe["kind"] = probe["representation"].map(lambda s: "raw" if s == "raw64" else s[:3])
        probe["k"] = probe["representation"].map(lambda s: 64 if s == "raw64" else int(s[3:]))
    label_ks = {1, 2, 3, 4, 5, 9, 16, 32, 64}
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for kind, color, marker in (("pca", ORANGE, "o"), ("pls", BLUE, "s"), ("raw", "#333333", "D")):
        sub = probe[probe["kind"] == kind].sort_values("k")
        ax.plot(sub["metro_accuracy"], sub["building_macro_f1"], color=color, lw=1.0, alpha=0.35, zorder=2)
        ax.scatter(sub["metro_accuracy"], sub["building_macro_f1"], s=55, c=color, marker=marker, zorder=3, label={"pca": "PCA", "pls": "PLS", "raw": "AEF-64"}[kind])
        for _, r in sub.iterrows():
            if kind == "raw":
                label = "AEF-64"
            elif int(r["k"]) not in label_ks:
                continue
            else:
                label = f"{kind.upper()}-{int(r['k'])}"
            ax.annotate(label, (r["metro_accuracy"], r["building_macro_f1"]), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=MUTED)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.55, 0.95)
    ax.set_xlabel("Metro-identity accuracy")
    ax.set_ylabel("Building-function macro-F1")
    ax.set_title("Ideal corner: keep the building label, drop metro identity", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    p = FIG / "02_pareto_building_vs_metro.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["pareto"] = p

    # 3. Local vs cross F1 vs k (PLS)
    byk = PAPER / "transfer_by_k.csv"
    if byk.exists():
        tb = pd.read_csv(byk)
        pls = tb[tb.model == "pls"].sort_values("k")
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.plot(pls["k"], pls["local_f1_mean"], color=ORANGE, lw=2.0, marker="o", ms=3.5, label="Local (within-metro 80/20)")
        ax.plot(pls["k"], pls["cross_f1_mean"], color=BLUE, lw=2.0, marker="o", ms=3.5, label="Cross-region (train A → test B)")
        # mark k=2 and k=9
        for k, name in ((2, "PLS-2"), (9, "PLS-9")):
            row = pls[pls.k == k].iloc[0]
            ax.scatter([k], [row.cross_f1_mean], s=70, c=BLUE, zorder=4)
            ax.annotate(f"{name}\ncross={row.cross_f1_mean:.3f}", (k, row.cross_f1_mean), textcoords="offset points", xytext=(8, -18 if k == 2 else 8), fontsize=8, color=INK)
        ax.set_xlabel("PLS components k")
        ax.set_ylabel("Macro-F1")
        ax.set_title("A short PLS is a semantic bottleneck: local F1 rises, transfer falls", loc="left", fontsize=12, color=INK)
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        p = FIG / "03_local_vs_cross_vs_k.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths["local_cross"] = p
        paths["pls2_cross"] = float(pls[pls.k == 2].cross_f1_mean.iloc[0])
        paths["pls9_cross"] = float(pls[pls.k == 9].cross_f1_mean.iloc[0])
    else:
        paths["local_cross"] = None

    # 4. Transfer heatmaps + delta
    def load_T(model_k: str, paper_first: bool = True) -> pd.DataFrame:
        candidates = []
        if paper_first and PAPER.exists():
            candidates.append(PAPER / f"transfer_f1_{model_k}.csv")
        candidates.append(S3 / f"transfer_f1_{model_k}.csv")
        for c in candidates:
            if c.exists():
                return pd.read_csv(c, index_col=0)
        raise FileNotFoundError(model_k)

    T64 = load_T("lr64_k64")
    T2 = load_T("pls_k2")
    T9 = load_T("pls_k9")
    delta = T2 - T64
    if (PAPER / "transfer_delta_pls2_minus_lr64.csv").exists():
        delta = pd.read_csv(PAPER / "transfer_delta_pls2_minus_lr64.csv", index_col=0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0))
    im0 = _heat(axes[0, 0], T64, 0.30, 0.95, "Blues", "AEF-64 · train row → test column")
    im1 = _heat(axes[0, 1], T2, 0.30, 0.95, "Blues", "PLS-2 → LR")
    im2 = _heat(axes[1, 0], T9, 0.30, 0.95, "Blues", "PLS-9 → LR")
    lim = float(np.nanmax(np.abs(delta.to_numpy())))
    lim = max(lim, 0.05)
    im3 = _heat(axes[1, 1], delta, -lim, lim, "RdBu_r", "ΔT = PLS-2 − AEF-64")
    for im, ax in ((im0, axes[0, 0]), (im1, axes[0, 1]), (im2, axes[1, 0])):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.02)
    fig.suptitle("Where a 2-D PLS helps transfer, and where it does not", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = FIG / "04_transfer_matrices.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["transfer"] = p

    # 5. Retention heatmaps
    R64 = _retention(T64)
    R2 = _retention(T2)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    im0 = _heat(axes[0], R64, 0.35, 1.05, "Blues", "Retention R(i→j) · AEF-64")
    im1 = _heat(axes[1], R2, 0.35, 1.05, "Blues", "Retention R(i→j) · PLS-2")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle("Transfer as a fraction of the target's own local F1", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = FIG / "05_transfer_retention.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["retention_heat"] = p

    # off-diag means for captions
    def off_mean(M):
        a = M.to_numpy(dtype=float)
        return float(a[~np.eye(len(a), dtype=bool)].mean())

    paths["t64_cross"] = off_mean(T64)
    paths["t2_cross"] = off_mean(T2)
    paths["t9_cross"] = off_mean(T9)
    paths["r64_cross"] = off_mean(R64)
    paths["r2_cross"] = off_mean(R2)
    return paths


def build_notebook(paths: dict) -> Path:
    NB.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s))

    def img(title, path: Path | None, note: str):
        if path is None or not path.exists():
            md(f"### {title}\n\n*Figure pending — waiting on `paper_curves` run.*\n\n{note}")
            return
        rel = Path("..") / "figures" / "paper" / path.name
        md(f"### {title}\n\n{note}\n\n![{title}]({rel})")

    md(
        """# Three questions about AlphaEarth geography

This notebook is the plot layer over the saved geo-subspace runs. It does not refit models.

1. **What information is in AEF?** — how much of the 64-D residential/non-residential signal sits in a short supervised subspace.
2. **How does task-relevant geometry differ by region?** — which components keep the building label and drop metro identity.
3. **Does that geometry explain geographic transfer?** — local F1 vs cross-region F1, transfer matrices, and transfer retention.

Colors: PLS / primary = `#0072B2`, PCA / comparison = `#D55E00`.
"""
    )

    md(
        f"""## 1. What information is in AEF?

Leave-one-metro-out logistic regression on PCA or PLS scores. Scaler, PCA, and PLS are fit on the training metros only. Threshold stays at 0.5.

Normalized performance:

$$R(k)=\\frac{{F1_k}}{{F1_{{64}}}}$$

PLS reaches 95% of the 64-D F1 at **k = {paths.get('pls_k95')}**. PCA needs **k = {paths.get('pca_k95')}**. PLS-2 already retains **{paths.get('pls2_ret', float('nan')):.3f}** of the full-representation F1.

This plot is about *recoverability* of the building-function label, not yet about geographic generalization.
"""
    )
    img(
        "F1 retention vs dimensionality",
        paths.get("retention"),
        "Building-function information is concentrated in a low-dimensional supervised subspace that is not aligned with AlphaEarth's highest-variance directions.",
    )

    md(
        """## 2. How does the task-relevant geometry differ by region?

Same pixel subsample for every representation: predict building function, and separately predict metro identity. Ideal representations sit toward the top-left — high building F1, low metro accuracy.
"""
    )
    img(
        "Pareto: building F1 vs metro accuracy",
        paths.get("pareto"),
        "PLS-2 keeps most of the building label while metro accuracy collapses. Wider maps put geography back in.",
    )

    md(
        f"""## 3. Does that geometry explain geographic transfer?

For every source metro A and target B, train on A and test on B. The diagonal uses an 80/20 within-metro split so the same pixels are not scored twice.

Cross-region F1 is the mean of the off-diagonal. Local F1 is the mean of the diagonal.

PLS-2 cross-region F1 ≈ **{paths.get('pls2_cross', paths.get('t2_cross', float('nan'))):.3f}**. PLS-9 ≈ **{paths.get('pls9_cross', paths.get('t9_cross', float('nan'))):.3f}**. AEF-64 ≈ **{paths.get('t64_cross', float('nan')):.3f}**.
"""
    )
    img(
        "Local vs cross-region F1 against PLS width",
        paths.get("local_cross"),
        "If local F1 rises with k while cross-region F1 peaks early and falls, the extra components are a geographic overfit — a semantic bottleneck at small k.",
    )

    md(
        f"""### Transfer matrices

Rows = training metro. Columns = target metro. Mean off-diagonal: AEF-64 **{paths.get('t64_cross', float('nan')):.3f}**, PLS-2 **{paths.get('t2_cross', float('nan')):.3f}**, PLS-9 **{paths.get('t9_cross', float('nan')):.3f}**.

$\\Delta T = T_{{\\mathrm{{PLS2}}}} - T_{{\\mathrm{{AEF64}}}}$ shows which source→target pairs gain from the bottleneck.
"""
    )
    img(
        "Transfer matrices and ΔT",
        paths.get("transfer"),
        "Blue in ΔT means PLS-2 beats the full 64-D probe on that pair.",
    )

    md(
        f"""### Transfer retention

$$R_{{i\\rightarrow j}} = \\frac{{T_{{i\\rightarrow j}}}}{{T_{{j\\rightarrow j}}}}$$

Mean off-diagonal retention: AEF-64 **{paths.get('r64_cross', float('nan')):.3f}**, PLS-2 **{paths.get('r2_cross', float('nan')):.3f}**.
"""
    )
    img(
        "Transfer-retention heatmaps",
        paths.get("retention_heat"),
        "Retention is the primary generalization quantity when targets differ in difficulty.",
    )

    md(
        """## Data paths

| Plot | Source |
|---|---|
| Retention vs k | `outputs/baseline_loro/20261002T0015/k_summary.csv` |
| Pareto probe | `outputs/paper_curves/20261002T1100/metro_vs_building_probe.csv` |
| Local / cross vs k | `outputs/paper_curves/20261002T1100/transfer_by_k.csv` |
| Transfer matrices | `outputs/paper_curves/20261002T1100/transfer_f1_*.csv` (fallback: `transfer_matrix/20261002T0015`) |

Figures are written under `figures/paper/` and embedded above. Re-run `scripts/build_geo_paper_plots_notebook.py` after the `paper_curves` job finishes if any panel was pending.
"""
    )

    nb["cells"] = cells
    NB.write_text(nbf.writes(nb))
    return NB


def main() -> None:
    paths = make_figures()
    nb = build_notebook(paths)
    print(f"figures → {FIG}")
    print(f"notebook → {nb}")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
