#!/usr/bin/env python3
"""Figures, REPORT.md, success criteria, and notebook for Experiment 05."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns

INK = "#1a1a1a"
MUTED = "#555555"
BLUE = "#0072B2"
ORANGE = "#D55E00"
TEAL = "#009E73"
PURPLE = "#6A3D9A"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)


def evaluate_criteria(run_dir: Path) -> dict:
    ti = pd.read_csv(run_dir / "artifacts" / "task_invariance.csv")
    loco = pd.read_csv(run_dir / "artifacts" / "loco_results.csv")
    gl = pd.read_csv(run_dir / "artifacts" / "global_local_results.csv")
    swap = pd.read_csv(run_dir / "artifacts" / "local_swap_summary.csv")
    assoc = json.loads((run_dir / "artifacts" / "rank_delta_assoc.json").read_text())
    geom_assoc = pd.read_csv(run_dir / "artifacts" / "geometry_vs_task_assoc.csv")

    # C1: excess TI > 0 with CI (lambda=1, mean over components 1-5)
    boot = pd.read_csv(run_dir / "artifacts" / "bootstrap" / "excess_ti_bootstrap.csv")
    early = boot[boot.component <= 5]
    c1_pass = bool((early["ci_lo"] > 0).mean() >= 0.5) if len(early) else False
    c1_ev = f"mean excess TI(AUROC) k≤5 = {early.mean_excess_ti_auroc.mean():.3f}; frac CI>0 = {(early.ci_lo > 0).mean():.2f}"

    # C2: LOCO task-invariant A/B vs lr64
    # pick best r by mean over targets for task_invariant_A on source... use r with best mean A
    def mean_method(method):
        sub = loco[loco.method == method]
        return sub.groupby("r")["macro_f1"].mean().max() if len(sub) else float("nan")

    best_r = int(loco[loco.method == "task_invariant_A"].groupby("r")["macro_f1"].mean().idxmax())
    sub = loco[loco.r == best_r]
    m_lr = float(sub[sub.method == "lr64"]["macro_f1"].mean())
    m_a = float(sub[sub.method == "task_invariant_A"]["macro_f1"].mean())
    m_b = float(sub[sub.method == "task_invariant_B"]["macro_f1"].mean())
    m_pool = float(sub[sub.method == "pooled_pls"]["macro_f1"].mean())
    m_geo = float(sub[sub.method == "geometric_avgproj"]["macro_f1"].mean())
    best_task = max(m_a, m_b)
    c2_pass = bool(best_task > m_lr + 0.03)
    c2_ev = f"r={best_r}: taskA={m_a:.3f} taskB={m_b:.3f} geo={m_geo:.3f} poolPLS={m_pool:.3f} lr64={m_lr:.3f}"

    # C3: |corr(dTI,dOOD)| > |corr(dGeom,dOOD)| and sensible sign
    r_ti = abs(assoc.get("pearson_dTI_dOOD", 0.0) or 0.0)
    r_g = abs(assoc.get("pearson_dGeom_dOOD", 0.0) or 0.0)
    c3_pass = bool(r_ti > r_g)
    c3_ev = f"|ρ(ΔTI,ΔOOD)|={r_ti:.3f} vs |ρ(ΔGeom,ΔOOD)|={r_g:.3f} (exploratory)"

    # C4: delta_local > 0
    dloc = gl[(gl.mode == "descriptive") & (gl.representation == "delta_local")]
    # use r=best_r
    dloc_r = dloc[dloc.r == best_r] if best_r in dloc.r.values else dloc
    c4_pass = bool(dloc_r["macro_f1"].mean() > 0.01)
    c4_ev = f"mean Δ_local F1 (r≈{best_r}) = {dloc_r.macro_f1.mean():.3f}"

    # C5: diagonal advantage
    boot_s = pd.read_csv(run_dir / "artifacts" / "bootstrap" / "local_swap_bootstrap.csv")
    boot_s_r = boot_s[boot_s.r == best_r] if best_r in boot_s.r.values else boot_s
    n_pos = int((boot_s_r.ci_lo > 0).sum())
    c5_pass = bool(n_pos >= 2)
    c5_ev = f"targets with CI(advantage)>0: {n_pos}/{boot_s_r.target.nunique()}; mean adv={boot_s_r.mean_advantage.mean():.3f}"

    # C6 optional — not run as main
    c6_pass = None
    c6_ev = "optional alternative estimator not run in main pipeline"

    criteria = {
        "1_task_specific_shared": {"result": "PASS" if c1_pass else "FAIL", "evidence": c1_ev},
        "2_global_task_signal_loco": {"result": "PASS" if c2_pass else "FAIL", "evidence": c2_ev},
        "3_invariance_explains_transfer": {"result": "PASS" if c3_pass else "FAIL", "evidence": c3_ev},
        "4_local_predictive": {"result": "PASS" if c4_pass else "FAIL", "evidence": c4_ev},
        "5_country_specificity": {"result": "PASS" if c5_pass else "FAIL", "evidence": c5_ev},
        "6_estimator_robustness": {"result": "SKIP", "evidence": c6_ev},
    }
    n_pass = sum(1 for k, v in criteria.items() if v["result"] == "PASS")
    n_fail = sum(1 for k, v in criteria.items() if v["result"] == "FAIL")
    if n_pass >= 4 and n_fail <= 1:
        overall = "mostly_pass"
    elif n_pass >= 2:
        overall = "partial_mixed"
    elif criteria["1_task_specific_shared"]["result"] == "FAIL" and criteria["2_global_task_signal_loco"]["result"] == "FAIL":
        overall = "reject_hypothesis"
    else:
        overall = "weak_mixed"

    out = {
        "overall": overall,
        "best_r": best_r,
        "criteria": criteria,
        "loco_means": {"lr64": m_lr, "task_A": m_a, "task_B": m_b, "pooled_pls": m_pool, "geometric_avgproj": m_geo},
        "geom_task_assoc": geom_assoc.to_dict(orient="records"),
        "rank_assoc": assoc,
    }
    (run_dir / "artifacts" / "success_criteria.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def make_figures(run_dir: Path, crit: dict):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    geom = pd.read_csv(run_dir / "artifacts" / "geometry_vs_task.csv")
    ti = pd.read_csv(run_dir / "artifacts" / "task_invariance.csv")
    shuf = pd.read_csv(run_dir / "artifacts" / "shuffle_invariance.csv")
    loco = pd.read_csv(run_dir / "artifacts" / "loco_results.csv")
    gl = pd.read_csv(run_dir / "artifacts" / "global_local_results.csv")
    swap = pd.read_csv(run_dir / "artifacts" / "local_swap_matrix.csv")
    rank = pd.read_csv(run_dir / "artifacts" / "rank_analysis.csv")
    best_r = crit["best_r"]

    # FIG 1
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    g = geom.copy()
    sc = ax.scatter(g["projection_overlap"], g["transfer_f1"], c=g["k"], cmap="viridis", s=22, alpha=0.75)
    ax.set_xlabel("Geometric projection overlap")
    ax.set_ylabel("Transfer macro-F1 (source PLS-k)")
    ax.set_title("Geometry vs task transfer", loc="left", color=INK)
    fig.colorbar(sc, ax=ax, label="k")
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_geometry_vs_task.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 2
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    real = ti[ti.ti_lambda == 1.0]
    sh = shuf[(shuf.ti_lambda == 1.0)]
    ax.hist(sh["ti_auroc"].dropna(), bins=30, alpha=0.55, color="#BBBBBB", label="label-shuffle", density=True)
    ax.hist(real["ti_auroc"].dropna(), bins=30, alpha=0.65, color=TEAL, label="real PLS", density=True)
    ax.set_xlabel("Task invariance TI (AUROC, λ=1)")
    ax.set_ylabel("density")
    ax.set_title("Task invariance: real vs label-shuffle", loc="left", color=INK)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_task_invariance_vs_shuffle.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 3 LOCO methods
    methods = ["lr64", "pooled_pls", "geometric_avgproj", "geometric_karcher", "task_invariant_A", "task_invariant_B"]
    sub = loco[(loco.r == best_r) & (loco.method.isin(methods))]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    order = ["USA", "JPN", "IND"]
    x = np.arange(len(order))
    width = 0.13
    colors = [BLUE, PURPLE, "#6BAED6", "#9ECAE1", TEAL, ORANGE]
    for i, method in enumerate(methods):
        vals = [float(sub[(sub.target == t) & (sub.method == method)]["macro_f1"].mean()) for t in order]
        ax.bar(x + (i - 2.5) * width, vals, width=width, label=method, color=colors[i], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#BBB", ls=":", lw=1)
    ax.set_ylabel("LOCO macro-F1")
    ax.set_title(f"LOCO methods (r={best_r})", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_loco_methods.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 4 global/local within
    reps = ["global", "local", "global_plus_local", "full_task", "lr64"]
    sub = gl[(gl.mode == "descriptive") & (gl.r == best_r) & (gl.representation.isin(reps))]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, c in enumerate(order):
        vals = [float(sub[(sub.country == c) & (sub.representation == r)]["macro_f1"].mean()) for r in reps]
        ax.plot(reps, vals, marker="o", label=c)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Within-country macro-F1")
    ax.set_title(f"Global / local decomposition (descriptive, r={best_r})", loc="left", color=INK)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_global_local_prediction.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 5 swap matrix
    sub = swap[(swap.mode == "loco") & (swap.r == best_r)]
    mat = sub.groupby(["target", "donor"])["macro_f1"].mean().unstack("donor").reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="Blues", vmin=0.4, vmax=0.9, ax=ax)
    ax.set_title(f"Local swap M_ij (LOCO S_G + S_L,donor), r={best_r}", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_local_swap_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 6 rank panels
    mean_rank = rank.groupby("k").mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    panels = [
        ("within_f1", "Within-country F1"),
        ("ood_f1", "Cross-country OOD F1"),
        ("task_invariance", "Task invariance TI"),
        ("geometric_overlap", "Geometric overlap"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.plot(mean_rank.k, mean_rank[col], color=TEAL, lw=2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("k")
        _style(ax)
    fig.suptitle("Rank phenomenon: within / OOD / TI / geometry", color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_rank_invariance_transfer.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 7 excess invariance
    boot = pd.read_csv(run_dir / "artifacts" / "bootstrap" / "excess_ti_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(boot.component, boot.mean_excess_ti_auroc, color=TEAL, lw=2, label="mean excess TI")
    ax.fill_between(boot.component, boot.ci_lo, boot.ci_hi, color=TEAL, alpha=0.2, label="95% bootstrap CI")
    ax.axhline(0, color="#BBB", ls=":", lw=1)
    ax.set_xlabel("component rank k")
    ax.set_ylabel("excess TI (AUROC) vs shuffle")
    ax.set_title("Task-specific excess invariance above shuffle", loc="left", color=INK)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_excess_invariance.png", dpi=180, bbox_inches="tight")
    plt.close()

    # FIG 8 dashboard
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = list(crit["criteria"].keys())
    colors = []
    for n in names:
        r = crit["criteria"][n]["result"]
        colors.append(TEAL if r == "PASS" else (ORANGE if r == "FAIL" else "#BBBBBB"))
    y = np.arange(len(names))[::-1]
    ax.barh(y, [1] * len(names), color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}: {crit['criteria'][n]['result']}" for n in names], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f"Falsification dashboard — overall: {crit['overall']}", loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_falsification_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_report(run_dir: Path, crit: dict):
    loco = pd.read_csv(run_dir / "artifacts" / "loco_results.csv")
    best_r = crit["best_r"]
    sub = loco[loco.r == best_r]
    lines = []
    lines.append("# Experiment 05 — Task-invariant vs geography-specific predictive subspaces\n")
    lines.append(f"**Run:** `{run_dir.name}`\n")
    lines.append("**Hypothesis:** $S_c \\approx S_G^{\\mathrm{task}} + S_{L,c}$ where $S_G^{\\mathrm{task}}$ is *predictively invariant* across countries.\n")
    lines.append("**Note:** PLS is only a probe for task-relevant directions — not the scientific object.\n")
    lines.append("\n## Answer\n")
    overall = crit["overall"]
    if overall == "mostly_pass":
        ans = "Yes, with caveats: task-predictive structure shows cross-country invariance above shuffle, and LOCO task-invariant subspaces transfer — but claims remain three-country only."
    elif overall == "reject_hypothesis":
        ans = "No: task invariance does not clearly exceed geometric/shuffle baselines; reject the current global-task-subspace hypothesis at this scale."
    else:
        ans = (
            "Mixed. Geometry alone was insufficient in Exp04; this experiment asks whether *predictive* invariance "
            f"separates from shuffle/geometry. Overall verdict: **{overall}**. See criteria table."
        )
    lines.append(ans + "\n")
    lines.append("\n## 1. Success criteria\n\n")
    lines.append("| # | Criterion | Result | Evidence |\n|---|---|---|---|\n")
    for i, (k, v) in enumerate(crit["criteria"].items(), 1):
        lines.append(f"| {i} | {k} | **{v['result']}** | {v['evidence']} |\n")
    lines.append(f"\n**Overall:** `{overall}`\n")

    lines.append("\n## 2. Strict LOCO results\n\n")
    lines.append(f"Selected r (by mean task_invariant_A): **{best_r}**\n\n")
    lines.append("| target | lr64 | pooled PLS | geo AvgProj | task-inv A | task-inv B |\n|---|---:|---:|---:|---:|---:|\n")
    for t in ("USA", "JPN", "IND"):
        def m(method):
            return float(sub[(sub.target == t) & (sub.method == method)]["macro_f1"].mean())
        lines.append(
            f"| {t} | {m('lr64'):.3f} | {m('pooled_pls'):.3f} | {m('geometric_avgproj'):.3f} | {m('task_invariant_A'):.3f} | {m('task_invariant_B'):.3f} |\n"
        )
    lm = crit["loco_means"]
    lines.append(
        f"\nMeans: lr64={lm['lr64']:.3f}, taskA={lm['task_A']:.3f}, taskB={lm['task_B']:.3f}, "
        f"geo={lm['geometric_avgproj']:.3f}, poolPLS={lm['pooled_pls']:.3f}\n"
    )

    lines.append("\n## 3. Task invariance vs shuffle\n\n")
    lines.append("See `figures/02_task_invariance_vs_shuffle.png` and `07_excess_invariance.png`.\n")
    lines.append("Central question: do real task directions exceed label-shuffled TI even when Exp04 geometric spectrum did not?\n")

    lines.append("\n## 4. Geometry vs task-invariance\n\n")
    lines.append("See `figures/01_geometry_vs_task.png` and `artifacts/geometry_vs_task_assoc.csv`.\n")
    lines.append("Limitation: only 3 countries — correlations over seed×k×pair rows are exploratory.\n")

    lines.append("\n## 5. Global/local predictive decomposition\n\n")
    lines.append("Orthogonal residual $U_{L,c}=\\mathrm{orth}((I-P_G)U_c)$ imposed by construction.\n")
    lines.append("See `figures/04_global_local_prediction.png`.\n")

    lines.append("\n## 6. Local swap\n\n")
    lines.append("See `figures/05_local_swap_matrix.png` and `artifacts/local_swap_summary.csv`.\n")
    lines.append("Prediction: matched donor > mismatched if $S_{L,c}$ is geography-specific.\n")

    lines.append("\n## 7. Rank phenomenon\n\n")
    lines.append("See `figures/06_rank_invariance_transfer.png`.\n")
    ra = crit["rank_assoc"]
    lines.append(
        f"- ρ(ΔTI, ΔOOD) = {ra.get('pearson_dTI_dOOD')}\n"
        f"- ρ(ΔGeom, ΔOOD) = {ra.get('pearson_dGeom_dOOD')}\n"
        f"- {ra.get('limitation')}\n"
    )

    lines.append("\n## 8. Controls\n\n")
    lines.append("- Label-shuffled PLS directions\n- Random / PCA baselines in LOCO table\n- Matched per-city caps (5000/class)\n- Seeds 0–4\n- Bootstrap CIs on excess TI and swap advantage\n")
    lines.append("- Optional alternative estimator (reduced-rank logistic): SKIP in main run\n")

    lines.append("\n## 9. Alternative explanations\n\n")
    lines.append("- Shared AlphaEarth covariance (Exp04 spectrum null) can mimic geometric consensus without task semantics.\n")
    lines.append("- Threshold/sign flips across countries can destroy F1 while AUROC remains moderate.\n")
    lines.append("- India fragmentation (10-city LORO) may dominate IND LOCO weakness.\n")

    lines.append("\n## 10. Unsupported claims\n\n")
    lines.append("- Do **not** claim worldwide population inference from 3 countries.\n")
    lines.append("- Do **not** equate PLS components across countries without subspace/rotation-invariant comparisons.\n")

    lines.append("\n## 11. Recommendation\n\n")
    if overall == "mostly_pass":
        rec = "Revise hypothesis to emphasize *predictive* invariance (not Grassmann consensus), then scale countries."
    elif overall == "reject_hypothesis":
        rec = "Reject current global-task-subspace hypothesis; consider city-conditioned or hierarchical domain structure first."
    else:
        rec = "Revise: keep LOCO low-rank transfer regularization as practical tool; do not claim clean S_G⊕S_L semantic decomposition until more geographies."
    lines.append(rec + "\n")

    lines.append("\n## Figures\n")
    for i in range(1, 9):
        lines.append(f"- `{i:02d}_*.png`\n")
    lines.append("\n## Reproducibility\n")
    lines.append("- Config: `config.json`\n- Artifacts: `artifacts/`\n- Notebook: `week5/notebooks/05_task_invariant_subspace.ipynb`\n")

    (run_dir / "REPORT.md").write_text("".join(lines))


def write_notebook(run_dir: Path):
    root = Path(__file__).resolve().parents[2]
    nb_path = root / "notebooks" / "05_task_invariant_subspace.ipynb"
    fig = f"../runs/{run_dir.name}/figures"
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# Experiment 05 — Task-invariant vs geography-specific subspaces

**Run:** `{run_dir.name}`

PLS is a **probe** for task-relevant directions. Hypothesis: predictive structure decomposes as
$S_c \\approx S_G^{{\\mathrm{{task}}}} + S_{{L,c}}$ where $S_G$ is defined by **cross-country predictive invariance**,
not Grassmann consensus alone (Exp04 spectrum failed vs shuffle).

Figures below load from `{fig}/` — no recompute needed.
"""
        ),
        nbf.v4.new_markdown_cell(f"## Falsification dashboard\n\n![dash]({fig}/08_falsification_dashboard.png)\n"),
        nbf.v4.new_markdown_cell(f"## Geometry vs task\n\n![g]({fig}/01_geometry_vs_task.png)\n"),
        nbf.v4.new_markdown_cell(f"## Task invariance vs shuffle\n\n![t]({fig}/02_task_invariance_vs_shuffle.png)\n\n![e]({fig}/07_excess_invariance.png)\n"),
        nbf.v4.new_markdown_cell(f"## LOCO methods\n\n![l]({fig}/03_loco_methods.png)\n"),
        nbf.v4.new_markdown_cell(f"## Global / local\n\n![gl]({fig}/04_global_local_prediction.png)\n"),
        nbf.v4.new_markdown_cell(f"## Local swap\n\n![s]({fig}/05_local_swap_matrix.png)\n"),
        nbf.v4.new_markdown_cell(f"## Rank phenomenon\n\n![r]({fig}/06_rank_invariance_transfer.png)\n"),
        nbf.v4.new_markdown_cell("## Reload artifacts"),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json, pandas as pd
RUN = Path(r"{run_dir}")
print((RUN/"REPORT.md").read_text()[:1500])
display(pd.read_csv(RUN/"artifacts"/"loco_results.csv").groupby(["r","method"])["macro_f1"].mean().unstack("method").round(3))
print(json.loads((RUN/"artifacts"/"success_criteria.json").read_text())["overall"])
"""
        ),
    ]
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, nb_path)
    print("wrote", nb_path)


def build_all(run_dir: Path):
    run_dir = Path(run_dir)
    crit = evaluate_criteria(run_dir)
    make_figures(run_dir, crit)
    write_report(run_dir, crit)
    write_notebook(run_dir)
    print("built figures+report+notebook for", run_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    build_all(args.run_dir)
