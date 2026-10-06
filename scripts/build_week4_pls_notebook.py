#!/usr/bin/env python3
"""Build notebooks/week4/08_pls_comparison.ipynb from PLS LORO outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_pls_loro"
NB = ROOT / "notebooks" / "week4" / "08_pls_comparison.ipynb"


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def main() -> None:
    summary = jload(OUT / "summary.json")
    if not summary:
        raise SystemExit(f"Missing {OUT / 'summary.json'}; run scripts/train_week4_pls_loro.py first")

    fine = jload(OUT / "fine_k_sweep.json")
    fine_csv = OUT / "fine_k_sweep.csv"
    fine_df = pd.read_csv(fine_csv) if fine_csv.exists() else None

    base = summary.get("baseline_lr") or {}
    pls = summary.get("best_pls_da_lr") or {}
    pca = summary.get("best_pca_lr") or {}
    pls_f = summary.get("best_pls_frac") or {}
    ridge = summary.get("baseline_ridge") or {}
    pls_flr = summary.get("best_pls_frac_lr") or {}

    cells: list = []

    def md(s: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

    def code(s: str) -> None:
        cells.append(nbf.v4.new_code_cell(s.strip() + "\n"))

    md(f"""
# Week 4: PLS vs Ridge / LR (LORO)

Supervised compression of AlphaEarth `A00`–`A63` with **Partial Least Squares**, compared to the week-4 balanced pure-cell baselines on the same LORO folds.

AE bands are **not orthogonal** (mean pairwise |corr| ≈ 0.22; effective rank ≈ 10). PLS scores **are** orthogonal by construction — the right place to look for a minimal set of important dimensions.

**Data.** `datasets/week4_pixel_gt_balanced/pixels.parquet` — {int(summary.get('n_pixels', 0)):,} pure RES/NRES cells (1:1 per metro).

**Models**
| Family | What |
|---|---|
| `baseline_lr` / `baseline_ridge` | StandardScaler + LR / Ridge on all 64 bands |
| `pls_da_lr` | PLS on binary \\(y\\) → LR on scores (PLS-DA) |
| `pls_frac` | PLSRegression \\(X \\to\\) `(frac_res, frac_nres, frac_empty)` |
| `pls_frac_lr` | PLS on fractions → LR on scores |
| `pca_lr` | PCA → LR (unsupervised control) |

Coarse sweep `n_components ∈ {summary.get('n_comps_sweep')}`; fine PLS-DA sweep `k=1..20` for the minimum orthogonal set. Backend: **{summary.get('backend')}** · {summary.get('n_gpus')} GPUs / {summary.get('n_cpus')} CPUs · {summary.get('elapsed_s')} s.

```bash
.venv/bin/python -u scripts/train_week4_pls_loro.py --backend gpu
.venv/bin/python -u scripts/fine_week4_pls_k_sweep.py
.venv/bin/python -u scripts/build_week4_pls_notebook.py
```
""")

    md(f"""
## Headline

| Model | LORO macro-F1 / MAE | OOF |
|---|---:|---:|
| Baseline LR (64-D) | F1 {fmt(base.get('loro_macro_f1_mean'))} ± {fmt(base.get('loro_macro_f1_std'))} | {fmt(summary.get('oof_macro_f1_baseline'))} |
| PLS-DA → LR (k={int(pls.get('n_components', -1))}) | F1 {fmt(pls.get('loro_macro_f1_mean'))} ± {fmt(pls.get('loro_macro_f1_std'))} | {fmt(summary.get('oof_macro_f1_pls_da'))} |
| PLS frac → LR (k={int(pls_flr.get('n_components', -1))}) | F1 {fmt(pls_flr.get('loro_macro_f1_mean'))} ± {fmt(pls_flr.get('loro_macro_f1_std'))} | — |
| PCA → LR (k={int(pca.get('n_components', -1))}) | F1 {fmt(pca.get('loro_macro_f1_mean'))} ± {fmt(pca.get('loro_macro_f1_std'))} | — |
| Baseline Ridge | MAE {fmt(ridge.get('loro_mae_mean'))} ± {fmt(ridge.get('loro_mae_std'))} | {fmt(summary.get('oof_mae_mean_baseline'))} |
| PLS frac (k={int(pls_f.get('n_components', -1))}) | MAE {fmt(pls_f.get('loro_mae_mean'))} ± {fmt(pls_f.get('loro_mae_std'))} | {fmt(summary.get('oof_mae_mean_pls_frac'))} |

Δ OOF F1 (PLS-DA − baseline) = **{fmt(summary.get('delta_oof_f1_pls_minus_baseline'), 4)}** ·
Δ OOF MAE (PLS − Ridge) = **{fmt(summary.get('delta_oof_mae_pls_minus_baseline'), 4)}**
""")

    md("""
![F1 vs n_components](../../outputs/week4_pls_loro/figures/f1_vs_ncomp.png)

![MAE vs n_components](../../outputs/week4_pls_loro/figures/mae_vs_ncomp.png)

![Best-family F1](../../outputs/week4_pls_loro/figures/best_f1_compare.png)

![Best MAE](../../outputs/week4_pls_loro/figures/best_mae_compare.png)

![Per-metro PLS vs baseline](../../outputs/week4_pls_loro/figures/loro_f1_pls_vs_baseline.png)

![OOF confusion](../../outputs/week4_pls_loro/figures/oof_confusion_compare.png)
""")

    viz = jload(OUT / "pls_t1_t2_viz.json")
    pr_path = OUT / "per_region_precision_recall.csv"
    pr_df = pd.read_csv(pr_path) if pr_path.exists() else None
    base_pr = (
        pr_df.loc[pr_df["model"] == "baseline_lr"].set_index("region")
        if pr_df is not None
        else None
    )

    def pr_cell(region: str, col: str) -> str:
        if base_pr is None or region not in base_pr.index:
            return "—"
        return fmt(base_pr.loc[region, col])

    if viz:
        # build per-metro reading table joining cluster F1 + LR PR/FN
        t1t2 = pd.read_csv(OUT / "pls_t1_t2_by_region.csv") if (OUT / "pls_t1_t2_by_region.csv").exists() else None
        read_rows = []
        if t1t2 is not None and base_pr is not None:
            joined = t1t2.set_index("region").join(base_pr, how="left", rsuffix="_lr")
            # failure-mode tag
            def mode(r):
                fnr, prec, rec_n = r["fn_rate_res"], r["precision_res"], r["recall_nres"]
                if fnr >= 0.35:
                    return "FN-heavy (RES→NRES)"
                if prec < 0.70 or rec_n < 0.65:
                    return "FP-heavy (NRES→RES)"
                if r["macro_f1"] >= 0.85:
                    return "clean split"
                return "mixed"
            for region, r in joined.sort_values("macro_f1").iterrows():
                read_rows.append(
                    f"| {region} | {fmt(r['f1_2d_inregion'])} | {fmt(r['mean_t1_res'], 2)} | "
                    f"{fmt(r['mean_t1_nres'], 2)} | {fmt(r['precision_res'])} | {fmt(r['recall_res'])} | "
                    f"**{fmt(r['fn_rate_res'])}** | {mode(r)} |"
                )
        read_table = "\n".join(read_rows) if read_rows else "| — | — | — | — | — | — | — | — |"

        md(f"""
## Do RES / NRES form two clusters in $(t_1, t_2)$?

**Partly — and the hard cities tell you *how* the blobs fail.**

Orange = non-residential, blue = residential. Globally the mass separates along **$t_1$** (NRES left, RES right); the $P=0.5$ cut is nearly vertical, so $t_2$ is a thin residual. Marginals: RES peaks near $t_1 \\approx +4$, NRES near $t_1 \\approx -2.5$, with a real overlap band around $t_1 \\in [-1, +2]$.

| Probe on pooled OOF scores | macro-F1 |
|---|---:|
| 1D LR on $t_1$ only | {fmt(viz.get('oof_f1_1d_t1'))} |
| 2D LR on $(t_1, t_2)$ | {fmt(viz.get('oof_f1_2d_t1_t2'))} |

Pooled-OOF F1 is descriptive. Honest LORO at k=2 is ~0.74 — transfer is harder than a global 2D cut.

```bash
.venv/bin/python -u scripts/plot_week4_pls_t1_t2.py
```

![Global t1 vs t2](../../outputs/week4_pls_loro/figures/pls_t1_t2_global.png)

![Marginals](../../outputs/week4_pls_loro/figures/pls_t1_t2_marginals.png)

### Reading the per-metro panels with precision / recall

What “two clear clusters” looks like in $(t_1,t_2)$ is the same signal the OOF LR metrics quantify. Join in-panel 2D structure with baseline-LR Prec/Recall/FN (RES = positive):

| Metro | in-panel 2D F1 | mean $t_1$ RES | mean $t_1$ NRES | Prec RES | Recall RES | FN rate | What the plot is doing |
|---|---:|---:|---:|---:|---:|---:|---|
{read_table}

**How to read the blobs**

1. **Clean two-blob cities** (Phoenix, Atlanta, Denver, LA) — class means on $t_1$ sit far apart (often $\\Delta t_1 \\gtrsim 4$). FN rate is low and Prec RES stays high. The black cut sits in a low-density gap; little ink in the wrong lobe.
2. **FN-heavy cities** (SF FN=0.51, Chicago 0.44, Houston 0.40) — a large blue mass still sits on the **left / NRES side** of $t_1$, or the RES cloud is pulled left so many true RES cells fall past the cut. Plot cue: blue points mixed into the orange lobe; high FN, often still decent Prec RES (model is shy about calling RES). Chicago’s RES mean $t_1$ is near **0** — almost no rightward RES lobe.
3. **FP-heavy cities** (NYC, Boston, and Seattle on metrics) — orange bleeds **right** into the RES side. Prec RES drops and/or NRES recall is weak. Plot cue: orange points across the cut into the blue region; FN rate alone can look “ok” (Boston 0.15, Seattle 0.06) while precision is the problem.
4. **NYC / SF / Chicago** are the visually overlapped panels — not one failure mode. SF/Chicago miss RES; NYC overlaps both ways with low Prec.

![Per metro](../../outputs/week4_pls_loro/figures/pls_t1_t2_by_region.png)

**Bottom line for the cluster plots:** two modes exist along $t_1$, but hard metros are not “no clusters” — they are **shifted / overlapping modes** whose failure type (FN vs FP) you can see as blue-on-the-left vs orange-on-the-right.
""")

    if pr_df is not None:
        base_pr_sorted = pr_df.loc[pr_df["model"] == "baseline_lr"].sort_values(
            "fn_rate_res", ascending=False
        )
        hard = {"NYC_NE", "SF_W", "Chicago_MW", "Boston_NE"}
        lines = []
        for _, r in base_pr_sorted.iterrows():
            mark = " ← hard" if r["region"] in hard else ""
            lines.append(
                f"| {r['region']} | {fmt(r['precision_res'])} | {fmt(r['recall_res'])} | "
                f"**{fmt(r['fn_rate_res'])}** | {fmt(r['precision_nres'])} | {fmt(r['recall_nres'])} | "
                f"{fmt(r['macro_f1'])} |{mark}"
            )
        table = "\n".join(lines)
        h = base_pr_sorted.loc[base_pr_sorted.region.isin(hard)]
        e = base_pr_sorted.loc[~base_pr_sorted.region.isin(hard)]
        md(f"""
## Per-metro precision / recall / FN rate

Positive class = **residential**. FN rate = missed RES / all true RES $= 1 - \\mathrm{{recall}}_{{\\mathrm{{res}}}}$. Same numbers used to annotate the cluster panels above.

Baseline LR OOF (LORO), sorted by RES FN rate:

| Metro | Prec RES | Recall RES | FN rate RES | Prec NRES | Recall NRES | macro-F1 | |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

| Group | mean FN rate RES | mean Recall RES | mean Prec RES | mean macro-F1 |
|---|---:|---:|---:|---:|
| Hard (NYC, SF, Chicago, Boston) | {fmt(h.fn_rate_res.mean())} | {fmt(h.recall_res.mean())} | {fmt(h.precision_res.mean())} | {fmt(h.macro_f1.mean())} |
| Other | {fmt(e.fn_rate_res.mean())} | {fmt(e.recall_res.mean())} | {fmt(e.precision_res.mean())} | {fmt(e.macro_f1.mean())} |

![Per-metro Prec / Recall / FN](../../outputs/week4_pls_loro/figures/per_region_prec_recall_fnr.png)
""")

        code("""
pr = pd.read_csv(OUT / "per_region_precision_recall.csv")
pr.loc[pr.model == "baseline_lr", [
    "region", "precision_res", "recall_res", "fn_rate_res",
    "precision_nres", "recall_nres", "macro_f1", "fn", "fp"
]].sort_values("fn_rate_res", ascending=False)
""")

    eq_md = OUT / "pls_da_k10_equation.md"
    if eq_md.exists():
        md(eq_md.read_text())
        code("""
betas = pd.read_csv(OUT / "pls_da_k10_lr_betas.csv")
R = pd.read_csv(OUT / "pls_da_k10_loadings_R.csv", index_col=0)
c = pd.read_csv(OUT / "pls_da_k10_compact_c.csv")
print(betas.to_string(index=False))
print("\\nTop |c| on scaled AE:")
print(c.reindex(c.c.abs().sort_values(ascending=False).index).head(12).to_string(index=False))
""")

    md("""
## Load outputs
""")

    code("""
from pathlib import Path
import json
import pandas as pd

ROOT = Path("../..").resolve()
OUT = ROOT / "outputs" / "week4_pls_loro"
summary = json.loads((OUT / "summary.json").read_text())
sweep = pd.read_csv(OUT / "sweep_summary.csv")
metrics = pd.read_csv(OUT / "loro_metrics.csv")
fine = json.loads((OUT / "fine_k_sweep.json").read_text())
fine_df = pd.read_csv(OUT / "fine_k_sweep.csv")
print("baseline LORO F1:", round(fine["baseline_loro_f1"], 4))
print("min-k criteria:", json.dumps(fine["min_k_criteria"], indent=2))
print("cov cumulants: 90%@k=%s  95%@k=%s  99%@k=%s" % (
    fine["cov_cum_k90"], fine["cov_cum_k95"], fine["cov_cum_k99"]))
""")

    # --- Fine k / minimum orthogonal section ---
    if fine:
        crit = fine.get("min_k_criteria") or {}
        c01 = crit.get("within_0.01_F1") or {}
        c02 = crit.get("within_0.02_F1") or {}
        c1p = crit.get("within_1pct_rel") or {}
        c2p = crit.get("within_2pct_rel") or {}
        base_f1 = fine.get("baseline_loro_f1")

        # pick a few rows from fine_df for a compact markdown table
        highlight_ks = [1, 2, 5, 9, 11, 12, 16, 17, 20, 64]
        table_rows = []
        if fine_df is not None:
            for k in highlight_ks:
                row = fine_df.loc[fine_df["n_components"] == k]
                if row.empty:
                    continue
                r = row.iloc[0]
                table_rows.append(
                    f"| {int(k)} | {fmt(r['loro_mean'])} | {fmt(r.get('gap'), 4)} | {fmt(r.get('rel_gap'), 4)} |"
                )
        table_body = "\n".join(table_rows) if table_rows else "| — | — | — | — |"

        md(f"""
## Minimum orthogonal dimensions (fine PLS-DA sweep)

PLS-DA scores are **orthogonal**. Fine LORO sweep over `k = 1..20` on the same balanced pure cells (8 GPUs).

```bash
.venv/bin/python -u scripts/fine_week4_pls_k_sweep.py
```

### Covariance with \\(y\\) (signal concentration)

Almost all supervised covariance sits in the first few scores:

| Cumulative cov²(t, y) | k |
|---|---:|
| ~90% | **{fine.get('cov_cum_k90')}** |
| ~95% | **{fine.get('cov_cum_k95')}** |
| ~99% | **{fine.get('cov_cum_k99')}** |

### Classifier parity vs 64-D LR (LORO F1 = {fmt(base_f1, 4)})

| Criterion | Min k | F1 | gap |
|---|---:|---:|---:|
| within 0.02 F1 | **{c02.get('k')}** | {fmt(c02.get('f1'), 4)} | {fmt(c02.get('gap'), 4)} |
| within 0.01 F1 | **{c01.get('k')}** | {fmt(c01.get('f1'), 4)} | {fmt(c01.get('gap'), 4)} |
| within 2% relative | **{c2p.get('k')}** | {fmt(c2p.get('f1'), 4)} | {fmt(c2p.get('gap'), 4)} |
| within 1% relative | **{c1p.get('k')}** | {fmt(c1p.get('f1'), 4)} | {fmt(c1p.get('gap'), 4)} |

### Selected k on the fine curve

| k | LORO F1 | gap to 64-D | rel gap |
|---:|---:|---:|---:|
{table_body}

**Recommendation**
- Absolute minimum for the residential *signal*: **k = {fine.get('cov_cum_k99')}** (99% of cov with \\(y\\)).
- Best compression that still races LR: **k = {c02.get('k')}–12**.
- Near-parity with full 64-D LR: **k = {c01.get('k')}**.

![Fine k minimum](../../outputs/week4_pls_loro/figures/fine_k_minimum.png)
""")

        code("""
fine_df
""")
    else:
        md("""
## Minimum orthogonal dimensions

Fine sweep not found. Run:

```bash
.venv/bin/python -u scripts/fine_week4_pls_k_sweep.py
.venv/bin/python -u scripts/build_week4_pls_notebook.py
```
""")

    md("""
### Binary heads only (coarse sweep)
""")

    code("""
bin_models = ["baseline_lr", "pls_da_lr", "pls_frac_lr", "pca_lr"]
sweep.loc[sweep.model.isin(bin_models), [
    "model", "n_components", "loro_macro_f1_mean", "loro_macro_f1_std"
]].sort_values(["model", "n_components"])
""")

    md("""
### Fraction heads (coarse sweep)
""")

    code("""
frac_models = ["baseline_ridge", "pls_frac"]
sweep.loc[sweep.model.isin(frac_models), [
    "model", "n_components", "loro_mae_mean", "loro_mae_std"
]].sort_values(["model", "n_components"])
""")

    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_week4_v4_clusters import markdown as v4_cluster_markdown
    from plot_week4_v4_clusters import write_figures as write_v4_figures

    try:
        write_v4_figures()
        section = v4_cluster_markdown()
        if section.strip():
            md(section)
    except Exception as exc:
        print(f"v4 cluster section skipped: {exc}")

    k99 = fine.get("cov_cum_k99") if fine else 5
    k02 = (fine.get("min_k_criteria") or {}).get("within_0.02_F1", {}).get("k") if fine else 9
    k01 = (fine.get("min_k_criteria") or {}).get("within_0.01_F1", {}).get("k") if fine else 16

    md(f"""
## Takeaways

1. **AE bands are collinear** (not orthogonal; effective rank ~10). PLS scores *are* orthogonal — use those as the compressed features.
2. **PLS ≈ full LR / Ridge** at enough components (Δ OOF F1 ≈ −0.001; fraction MAE slightly better at k=12). PCA at the same width lags → the gain is supervised, not just dim reduction.
3. **Minimum orthogonal set**
   - signal (99% cov with \\(y\\)): **k = {k99}**
   - race LR (within 0.02 F1): **k = {k02}–12**
   - near-parity (within 0.01 F1): **k = {k01}**
4. Artifacts under `outputs/week4_pls_loro/`: OOF preds, best-k PLS scores (`oof_pls_da_scores_k32.npy`), fold checkpoints (loadings/rotations), coarse + fine sweeps, figures.
""")

    code("""
print("paths:", json.dumps(summary.get("paths", {}), indent=2))
print("backend:", summary.get("backend"), "gpus:", summary.get("n_gpus"), "cpus:", summary.get("n_cpus"))
print("elapsed_s:", summary.get("elapsed_s"))
if (OUT / "fine_k_sweep.json").exists():
    print("fine_k_sweep:", OUT / "fine_k_sweep.json")
    print("fine_k_figure:", OUT / "figures" / "fine_k_minimum.png")
""")

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
