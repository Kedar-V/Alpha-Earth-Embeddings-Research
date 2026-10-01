#!/usr/bin/env python3
"""Build notebooks/week4/08_pls_comparison.ipynb from PLS LORO outputs."""

from __future__ import annotations

import json
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
    if viz:
        md(f"""
## Do RES / NRES form two clusters in $(t_1, t_2)$?

Yes — mostly along **$t_1$**. Orange = non-residential (left), blue = residential (right). The decision boundary is nearly vertical, so $t_2$ adds little.

| Probe on pooled OOF scores | macro-F1 |
|---|---:|
| 1D LR on $t_1$ only | {fmt(viz.get('oof_f1_1d_t1'))} |
| 2D LR on $(t_1, t_2)$ | {fmt(viz.get('oof_f1_2d_t1_t2'))} |

These pooled-OOF numbers are descriptive (refit on all metros’ OOF scores). Honest LORO at k=2 is lower (~0.74 in the fine sweep) because fold transfer is harder than a global 2D cut.

```bash
.venv/bin/python -u scripts/plot_week4_pls_t1_t2.py
```

![Global t1 vs t2](../../outputs/week4_pls_loro/figures/pls_t1_t2_global.png)

![Marginals](../../outputs/week4_pls_loro/figures/pls_t1_t2_marginals.png)

![Per metro](../../outputs/week4_pls_loro/figures/pls_t1_t2_by_region.png)

Clear two-blob structure in Phoenix / Atlanta / Denver / LA / Seattle; more overlap in NYC / SF / Chicago — same pattern as the week-4 UMAP hard cities.
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
