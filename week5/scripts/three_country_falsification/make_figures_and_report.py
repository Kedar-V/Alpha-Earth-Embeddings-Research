#!/usr/bin/env python3
"""Figures, REPORT.md, and notebook for three-country falsification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#CC79A7"
INK = "#1a1a1a"
MUTED = "#555555"
COUNTRIES = ("USA", "JPN", "IND")


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)


def evaluate_criteria(run_dir: Path) -> dict:
    r_star = json.loads((run_dir / "artifacts" / "r_star.json").read_text())
    real = pd.read_csv(run_dir / "artifacts" / "avgproj_spectrum.csv")
    ctrl = pd.read_csv(run_dir / "artifacts" / "avgproj_spectrum_controls.csv")
    pred = pd.read_csv(run_dir / "artifacts" / "sg_sl_prediction_table.csv")
    swap = pd.read_csv(run_dir / "artifacts" / "local_swap_summary.csv")
    rank = pd.read_csv(run_dir / "artifacts" / "rank_overlap.csv")

    # 1) shared spectrum: λ1,λ2 beat null and gap structure
    k_ref = r_star["k_ref"]
    real_k = real[real.k == k_ref]
    null_k = ctrl[(ctrl.k == k_ref) & (ctrl.control.isin(["label_shuffle", "random"]))]
    l1_real = float(real_k.lambda_1.mean())
    l2_real = float(real_k.lambda_2.mean())
    l1_thr = float(np.nanpercentile(null_k.lambda_1, 95)) if len(null_k) else 0.0
    l2_thr = float(np.nanpercentile(null_k.lambda_2, 95)) if len(null_k) else 0.0
    c1 = bool(l1_real > l1_thr and l2_real > l2_thr and r_star["r_star"] >= 1)

    # 2) SG transferable LOCO: global cross F1 close to / better than competitive vs lr64
    cross = pred[
        (pred["mode"] == "loco") & (pred["protocol"] == "cross_country") & (pred["r"] == r_star["r_star"])
    ]
    g = float(cross[cross["representation"] == "global"]["m_macro_f1"].mean())
    lr = float(cross[cross["representation"] == "lr64"]["m_macro_f1"].mean())
    gpl = float(cross[cross["representation"] == "global_plus_local"]["m_macro_f1"].mean())
    c2 = bool(g >= lr - 0.02)  # retains strong transfer (within 2 pts of lr64 or better)

    # 3) SL adds within: global_plus_local > global within
    within = pred[
        (pred["mode"] == "descriptive")
        & (pred["protocol"] == "within_country")
        & (pred["r"] == r_star["r_star"])
    ]
    w_g = float(within[within["representation"] == "global"]["m_macro_f1"].mean())
    w_gl = float(within[within["representation"] == "global_plus_local"]["m_macro_f1"].mean())
    c3 = bool(w_gl > w_g + 0.005)

    # 4) matched local > mismatched (descriptive at r_star)
    # NOTE: column name "mode" — use bracket access (DataFrame.mode is a method)
    sw = swap[(swap["mode"] == "descriptive") & (swap["r"] == r_star["r_star"])]
    if sw.empty:
        sw = swap[swap["mode"] == "descriptive"]
    adv = float(sw["diagonal_advantage_mean"].mean()) if len(sw) else float("nan")
    # require positive advantage and CI mostly above 0 for majority of targets
    n_pos = int((sw["diagonal_advantage_ci95_lo"] > 0).sum()) if len(sw) else 0
    c4 = bool(np.isfinite(adv) and adv > 0 and n_pos >= 2)

    # rank phenomenon support (exploratory)
    rk = rank.dropna(subset=["delta_f1_within", "delta_f1_ood", "delta_global_overlap"])
    # late k: mean deltas for k>=10
    late = rank[rank.k >= 10]
    early = rank[(rank.k >= 2) & (rank.k <= 5)]

    n_pass = sum([c1, c2, c3, c4])
    if n_pass == 4:
        overall = "all_four"
    elif c1 and c2 and c3 and not c4:
        overall = "partial_no_country_specific"
    elif c1 and c2 and not c3:
        overall = "partial_1_2"
    elif n_pass <= 1:
        overall = "contradicts"
    else:
        overall = "partial_mixed"

    return {
        "r_star": r_star["r_star"],
        "c1_shared_spectrum": c1,
        "c2_sg_transferable_loco": c2,
        "c3_sl_adds_within": c3,
        "c4_matched_local": c4,
        "overall": overall,
        "metrics": {
            "lambda1_real": l1_real,
            "lambda2_real": l2_real,
            "lambda1_null95": l1_thr,
            "lambda2_null95": l2_thr,
            "loco_global_f1": g,
            "loco_lr64_f1": lr,
            "loco_global_plus_local_f1": gpl,
            "within_global_f1": w_g,
            "within_global_plus_local_f1": w_gl,
            "diagonal_advantage_mean": adv,
            "targets_ci_above_0": n_pos,
            "late_delta_within": float(late.delta_f1_within.mean()) if len(late) else float("nan"),
            "late_delta_ood": float(late.delta_f1_ood.mean()) if len(late) else float("nan"),
            "early_delta_overlap": float(early.delta_global_overlap.mean()) if len(early) else float("nan"),
        },
    }


def build_figures_and_report(run_dir: Path) -> Path:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    art = run_dir / "artifacts"
    real = pd.read_csv(art / "avgproj_spectrum.csv")
    ctrl = pd.read_csv(art / "avgproj_spectrum_controls.csv")
    pred = pd.read_csv(art / "sg_sl_prediction_table.csv")
    swap_m = pd.read_csv(art / "local_swap_matrix.csv")
    swap_s = pd.read_csv(art / "local_swap_summary.csv")
    rank = pd.read_csv(art / "rank_overlap.csv")
    r_star = json.loads((art / "r_star.json").read_text())
    criteria = evaluate_criteria(run_dir)
    atomic_criteria = run_dir / "artifacts" / "success_criteria.json"
    atomic_criteria.write_text(json.dumps(criteria, indent=2) + "\n")

    # --- Fig 1: spectrum vs nulls at k=20 ---
    k_ref = r_star["k_ref"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    # mean real evals
    evals = np.mean([json.loads(s) for s in real[real.k == k_ref].evals_json], axis=0)
    idx = np.arange(1, min(15, len(evals)) + 1)
    ax.plot(idx, evals[: len(idx)], color=BLUE, lw=2.2, marker="o", ms=4, label="real PLS")
    for control, color, ls in (
        ("label_shuffle", ORANGE, "--"),
        ("random", "#999999", ":"),
        ("pca", TEAL, "-."),
    ):
        sub = ctrl[(ctrl.k == k_ref) & (ctrl.control == control)]
        if sub.empty:
            continue
        arr = np.array([json.loads(s)[: len(idx)] for s in sub.evals_json])
        mean, lo, hi = arr.mean(0), np.percentile(arr, 5, 0), np.percentile(arr, 95, 0)
        ax.plot(idx, mean, color=color, ls=ls, lw=1.6, label=control)
        ax.fill_between(idx, lo, hi, color=color, alpha=0.12)
    ax.axvline(r_star["r_star"], color=PURPLE, ls="--", lw=1.2, label=f"r*={r_star['r_star']}")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel(r"$\lambda_i(\bar P)$")
    ax.set_title(f"Average-projection spectrum (k={k_ref}) vs nulls", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_avgproj_spectrum_vs_nulls.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: SG/SL heatmap ---
    rs = criteria["r_star"]
    rows_h = []
    for protocol, mode in (("within_country", "descriptive"), ("cross_country", "loco")):
        sub = pred[(pred["protocol"] == protocol) & (pred["mode"] == mode) & (pred["r"] == rs)]
        for rep in ("global", "local", "global_plus_local", "full_task", "lr64"):
            rows_h.append(
                {
                    "protocol": protocol,
                    "representation": rep,
                    "macro_f1": float(sub[sub["representation"] == rep]["m_macro_f1"].mean()),
                }
            )
    hdf = pd.DataFrame(rows_h)
    pivot = hdf.pivot(index="representation", columns="protocol", values="macro_f1")
    pivot = pivot.reindex(["global", "local", "global_plus_local", "full_task", "lr64"])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["within", "LOCO cross"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.to_numpy()[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title(f"SG / SL prediction table (r*={rs})", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_sg_sl_prediction_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: swap matrix ---
    sub = swap_m[(swap_m["mode"] == "descriptive") & (swap_m["r"] == rs)]
    if sub.empty:
        sub = swap_m[swap_m["mode"] == "descriptive"]
    # Prefer r with finite F1 if r* rows are empty/NaN
    if not np.isfinite(sub["m_macro_f1"]).any():
        sub = swap_m[swap_m["mode"] == "descriptive"]
        # pick r with most finite values
        counts = sub.groupby("r")["m_macro_f1"].apply(lambda s: s.notna().sum())
        if len(counts) and counts.max() > 0:
            sub = sub[sub["r"] == int(counts.idxmax())]
    mat = np.zeros((3, 3))
    for i, tgt in enumerate(COUNTRIES):
        for j, donor in enumerate(COUNTRIES):
            vals = sub[(sub["target"] == tgt) & (sub["donor"] == donor)]["m_macro_f1"]
            mat[i, j] = float(vals.mean()) if len(vals) else float("nan")
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(mat, cmap="Blues", vmin=np.nanmin(mat) - 0.05, vmax=np.nanmax(mat) + 0.02)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"SL,{c}" for c in COUNTRIES])
    ax.set_yticklabels([f"target={c}" for c in COUNTRIES])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Local-component swap $M_{{ij}}$ (r*={rs}, descriptive)", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_local_swap_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 4: rank curves ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, country in zip(axes, COUNTRIES):
        g = rank[rank.source == country].groupby("k")
        for col, color, label in (
            ("f1_within", BLUE, "within F1"),
            ("f1_ood", ORANGE, "OOD F1"),
            ("global_overlap", TEAL, "GlobalOverlap"),
        ):
            mean = g[col].mean()
            # normalize overlap to [0,1] scale for overlay
            y = mean.values
            if col == "global_overlap":
                y = y / max(y.max(), 1e-9)
                label = "GlobalOverlap (norm)"
            ax.plot(mean.index, y, color=color, lw=1.8, marker="o", ms=3, label=label)
        ax.set_title(country, loc="left", color=INK)
        ax.set_xlabel("k")
        _style(ax)
        if country == "USA":
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Within / OOD F1 and GlobalOverlap vs PLS rank", fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / "04_rank_f1_globaloverlap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 5: success dashboard ---
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    labels = [
        "1 shared spectrum",
        "2 SG LOCO transfer",
        "3 SL adds within",
        "4 matched local",
    ]
    flags = [
        criteria["c1_shared_spectrum"],
        criteria["c2_sg_transferable_loco"],
        criteria["c3_sl_adds_within"],
        criteria["c4_matched_local"],
    ]
    colors = [TEAL if f else ORANGE for f in flags]
    ax.barh(labels[::-1], [1] * 4, color=colors[::-1], edgecolor="none", height=0.6)
    for i, (lab, f) in enumerate(zip(labels[::-1], flags[::-1])):
        ax.text(0.5, i, "PASS" if f else "FAIL", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f"Success criteria — overall: {criteria['overall']}", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_success_criteria_dashboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    report = write_report(run_dir, criteria, r_star, swap_s, pred, rank)
    return report


def write_report(run_dir: Path, criteria: dict, r_star: dict, swap_s: pd.DataFrame, pred: pd.DataFrame, rank: pd.DataFrame) -> Path:
    cfg = json.loads((run_dir / "config.json").read_text())
    m = criteria["metrics"]
    rs = criteria["r_star"]
    lines = []
    lines.append("# Three-country falsification — Global ⊕ Local task subspace")
    lines.append("")
    lines.append(f"**Run:** `{run_dir.name}`")
    lines.append(f"**Hypothesis:** `{cfg['hypothesis']}`")
    lines.append(f"**Prior bases:** `{cfg['prior_run']}`")
    lines.append(f"**Seeds:** {cfg['seeds']} · **max_per_class:** {cfg['max_per_class']} · **r\\*:** {rs}")
    lines.append("")
    lines.append("## Success criteria")
    lines.append("")
    lines.append(f"**Overall: `{criteria['overall']}`**")
    lines.append("")
    lines.append("| # | Criterion | Result | Evidence |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| 1 | Significant shared task-subspace structure | "
        f"{'PASS' if criteria['c1_shared_spectrum'] else 'FAIL'} | "
        f"λ1={m['lambda1_real']:.3f} (null95={m['lambda1_null95']:.3f}), "
        f"λ2={m['lambda2_real']:.3f} (null95={m['lambda2_null95']:.3f}) |"
    )
    lines.append(
        f"| 2 | $S_G$ retains strong LOCO cross-country signal | "
        f"{'PASS' if criteria['c2_sg_transferable_loco'] else 'FAIL'} | "
        f"LOCO global F1={m['loco_global_f1']:.3f} vs lr64={m['loco_lr64_f1']:.3f} |"
    )
    lines.append(
        f"| 3 | $S_{{L,c}}$ adds within-country information | "
        f"{'PASS' if criteria['c3_sl_adds_within'] else 'FAIL'} | "
        f"within SG⊕SL={m['within_global_plus_local_f1']:.3f} > SG={m['within_global_f1']:.3f} |"
    )
    lines.append(
        f"| 4 | Matched local beats mismatched | "
        f"{'PASS' if criteria['c4_matched_local'] else 'FAIL'} | "
        f"mean diagonal advantage={m['diagonal_advantage_mean']:.3f}; "
        f"targets with CI>0: {m['targets_ci_above_0']}/3 |"
    )
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- **all_four** → enough evidence to justify scaling globally")
    lines.append("- **partial_1_2** → transferable common subspace, weak local structure")
    lines.append("- **partial_no_country_specific** → global+residual helps, but residual is not country-specific")
    lines.append("- **contradicts** → shared supervised geometry not supported")
    lines.append("")
    lines.append("## Representation vs OOD")
    lines.append("")
    lines.append("- Descriptive $S_G$ (all three countries): spectrum, within-country, swap structure")
    lines.append("- LOCO $S_G^{(-c)}$: all cross-country transfer claims")
    lines.append("")
    lines.append("## Diagonal advantage by target")
    lines.append("")
    sw = swap_s[(swap_s["mode"] == "descriptive") & (swap_s["r"] == rs)]
    if sw.empty or not sw["matched_f1_mean"].notna().any():
        sw = swap_s[swap_s["mode"] == "descriptive"]
    if len(sw):
        lines.append("| target | r | matched F1 | mismatched F1 | advantage | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for _, r in sw.iterrows():
            def fmt(x):
                return f"{x:.3f}" if pd.notna(x) else "nan"

            lines.append(
                f"| {r.target} | {int(r.r)} | {fmt(r.matched_f1_mean)} | {fmt(r.mismatched_f1_mean)} | "
                f"{fmt(r.diagonal_advantage_mean)} | "
                f"[{fmt(r.diagonal_advantage_ci95_lo)}, {fmt(r.diagonal_advantage_ci95_hi)}] |"
            )
    lines.append("")
    lines.append("## Rank phenomenon (exploratory)")
    lines.append("")
    lines.append(
        f"- Late-k (k≥10) mean ΔF1 within={m['late_delta_within']:+.4f}, "
        f"ΔF1 OOD={m['late_delta_ood']:+.4f}"
    )
    lines.append(f"- Early-k (2–5) mean ΔGlobalOverlap={m['early_delta_overlap']:+.4f}")
    lines.append("")
    lines.append("## Figures")
    for p in sorted((run_dir / "figures").glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Config: `config.json`")
    lines.append(f"- r*: `artifacts/r_star.json`")
    lines.append(f"- Artifacts: `artifacts/*.csv`")
    lines.append(f"- Notebook: `week5/notebooks/04_three_country_falsification.ipynb`")
    lines.append("")
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def build_notebook(run_dir: Path) -> Path:
    report = (run_dir / "REPORT.md").read_text()
    figures = [
        ("01_avgproj_spectrum_vs_nulls.png", "1. Average-projection spectrum vs nulls"),
        ("02_sg_sl_prediction_heatmap.png", "2. SG / SL prediction table"),
        ("03_local_swap_matrix.png", "3. Local-component swap matrix"),
        ("04_rank_f1_globaloverlap.png", "4. Rank phenomenon vs GlobalOverlap"),
        ("05_success_criteria_dashboard.png", "5. Success-criteria dashboard"),
    ]
    rel = f"../runs/{run_dir.name}"
    rel_mirror = f"../../week5/runs/{run_dir.name}"

    def make_nb(rel_prefix: str) -> nbf.NotebookNode:
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        cells = []
        cells.append(
            nbf.v4.new_markdown_cell(
                "# Three-country falsification: $\\mathcal S_c \\approx \\mathcal S_G \\oplus \\mathcal S_{L,c}$\n\n"
                f"**Run directory:** `{run_dir}`\n\n"
                "Full report + analysis figures (no recompute required).\n\n"
                f"- Report: `{rel_prefix}/REPORT.md`\n"
                f"- Config: `{rel_prefix}/config.json`\n"
                f"- Artifacts: `{rel_prefix}/artifacts/`\n"
            )
        )
        cells.append(nbf.v4.new_markdown_cell(report))
        cells.append(nbf.v4.new_markdown_cell("## Figures\n"))
        for fname, title in figures:
            src = f"{rel_prefix}/figures/{fname}"
            cells.append(nbf.v4.new_markdown_cell(f"### {title}\n\n![{title}]({src})\n\n`{fname}`\n"))
        code = f'''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown

RUN = Path(r"{run_dir}")
display(Markdown("### r_star"))
display(json.loads((RUN / "artifacts" / "r_star.json").read_text()))
display(Markdown("### success criteria"))
display(json.loads((RUN / "artifacts" / "success_criteria.json").read_text()))
pred = pd.read_csv(RUN / "artifacts" / "sg_sl_prediction_table.csv")
rs = json.loads((RUN / "artifacts" / "r_star.json").read_text())["r_star"]
summary = (
    pred[pred.r == rs]
    .groupby(["protocol", "mode", "representation"])["m_macro_f1"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
display(Markdown("### SG/SL F1 summary at r*"))
display(summary)
display(Markdown("### Local swap diagonal advantage"))
display(pd.read_csv(RUN / "artifacts" / "local_swap_summary.csv"))
rank = pd.read_csv(RUN / "artifacts" / "rank_overlap.csv").dropna(subset=["delta_f1_ood", "delta_global_overlap"])
if len(rank):
    corr = rank["delta_f1_ood"].corr(rank["delta_global_overlap"])
    display(Markdown(f"### corr(ΔF1_OOD, ΔGlobalOverlap) = **{{corr:.3f}}**"))
display(rank.head(12))
'''
        cells.append(nbf.v4.new_code_cell(code))
        nb.cells = cells
        return nb

    out1 = ROOT / "week5/notebooks/04_three_country_falsification.ipynb"
    out2 = ROOT / "notebooks/week5/04_three_country_falsification.ipynb"
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    nb1 = make_nb(rel)
    nbf.write(nb1, out1)
    # execute
    try:
        from nbclient import NotebookClient

        client = NotebookClient(nb1, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(out1.parent)}})
        client.execute()
        nbf.write(nb1, out1)
    except Exception as e:
        # still write unexecuted notebook
        (run_dir / "failures" / "notebook_execute.txt").write_text(str(e))

    nb2 = make_nb(rel_mirror)
    # copy outputs from nb1 code cell if present
    if len(nb1.cells) == len(nb2.cells):
        for a, b in zip(nb1.cells, nb2.cells):
            if a.cell_type == "code" and "outputs" in a:
                b["outputs"] = a["outputs"]
                b["execution_count"] = a.get("execution_count")
    nbf.write(nb2, out2)

    # verify figure embeds
    text = out1.read_text()
    for fname, _ in figures:
        if fname not in text:
            raise RuntimeError(f"notebook missing figure {fname}")
    return out1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--notebook-only", action="store_true")
    args = ap.parse_args()
    if args.notebook_only:
        print(build_notebook(args.run_dir))
    else:
        print(build_figures_and_report(args.run_dir))
        print(build_notebook(args.run_dir))
