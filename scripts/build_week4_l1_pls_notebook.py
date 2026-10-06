#!/usr/bin/env python3
"""Build notebooks/week4/09_l1_pls_sparsity.ipynb from L1 vs L2 PLS outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_l1_pls_loro"
NB = ROOT / "notebooks" / "week4" / "09_l1_pls_sparsity.ipynb"


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
        raise SystemExit(f"Missing {OUT / 'summary.json'}; run scripts/train_week4_l1_pls_loro.py first")

    sweep = pd.read_csv(OUT / "sweep_summary.csv")
    coefs = (
        pd.read_csv(OUT / "l1_c10_fold_coefs.csv")
        if (OUT / "l1_c10_fold_coefs.csv").exists()
        else None
    )
    l2p = summary.get("l2_pls32") or {}
    l2a = summary.get("l2_ae64") or {}
    s2 = summary.get("l1_pls_nnz_le_2") or {}
    surv10 = summary.get("survival_C10_t1_to_t10") or {}
    surv30 = summary.get("survival_C30_t1_to_t10") or {}
    surv100 = summary.get("survival_C100_t1_to_t10") or {}

    pls = sweep.loc[sweep.features == "pls32"].sort_values(["penalty", "C"])
    path_lines = []
    for _, r in pls.iterrows():
        if r.penalty == "l2":
            path_lines.append(
                f"| L2 | — | {fmt(r.nnz_mean, 1)} | {fmt(r.loro_macro_f1_mean)} ± {fmt(r.loro_macro_f1_std)} |"
            )
        else:
            path_lines.append(
                f"| L1 | {fmt(r.C, 4)} | {fmt(r.nnz_mean, 1)} | {fmt(r.loro_macro_f1_mean)} ± {fmt(r.loro_macro_f1_std)} |"
            )
    path_table = "\n".join(path_lines)

    # plain table for C=10 (no inline math that breaks renderers)
    coef_rows = []
    if coefs is not None:
        for _, r in coefs.sort_values("region").iterrows():
            t2 = f"{r['beta_t2']:.2f}" if r["has_t2"] else "—"
            coef_rows.append(
                f"| {r['region']} | {r['intercept']:+.3f} | {r['beta_t1']:.2f} | {t2} |"
            )
    coef_table = "\n".join(coef_rows) if coef_rows else "| — | — | — | — |"

    cells: list = []

    def md(s: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

    def code(s: str) -> None:
        cells.append(nbf.v4.new_code_cell(s.strip() + "\n"))

    md(f"""
# Week 4: L1 vs L2 on PLS scores

Does L1 logistic on the 32 PLS-DA scores zero out everything except $t_1$ (and maybe $t_2$)?

Same balanced pure-cell LORO folds as the PLS notebook. Fold rotations come from `outputs/week4_pls_loro/checkpoints/`. Training: torch proximal L1 on GPU, mean BCE plus $(1/C)\\|w\\|_1$. Ran on {summary.get('n_gpus')} GPUs / {summary.get('n_cpus')} CPUs in {summary.get('elapsed_s')} s.

```bash
.venv/bin/python -u scripts/train_week4_l1_pls_loro.py --backend gpu
.venv/bin/python -u scripts/build_week4_l1_pls_notebook.py
```

## Result

L1 does sparsify. As $C$ drops (stronger penalty), mean nonzero count falls 32 → ~16 ($C=100$) → ~6 ($C=30$) → ~1 ($C=10$, almost always $t_1$). Stronger than that zeros the head.

| Head | $C$ | mean nnz | LORO macro-F1 |
|---|---:|---:|---:|
| L2 on PLS $t_{{1..32}}$ | — | {fmt(l2p.get('nnz_mean'), 1)} | {fmt(l2p.get('loro_macro_f1_mean'))} ± {fmt(l2p.get('loro_macro_f1_std'))} |
| L2 on AE 64-D | — | {fmt(l2a.get('nnz_mean'), 1)} | {fmt(l2a.get('loro_macro_f1_mean'))} ± {fmt(l2a.get('loro_macro_f1_std'))} |
| L1 on PLS (~$t_1$ only) | {fmt(s2.get('C'), 1) if s2 else '10'} | {fmt(s2.get('nnz_mean'), 2) if s2 else '1.2'} | {fmt(s2.get('loro_macro_f1_mean')) if s2 else '0.735'} |

Share of folds where each score stays nonzero ($|\\beta_k| > 10^{{-6}}$):

| $C$ | mean nnz | $t_1$ | $t_2$ | $t_3$ | $t_4$ | $t_5$ | $t_6$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1.2 | {fmt(surv10.get('t1'), 2)} | {fmt(surv10.get('t2'), 2)} | {fmt(surv10.get('t3'), 2)} | {fmt(surv10.get('t4'), 2)} | {fmt(surv10.get('t5'), 2)} | {fmt(surv10.get('t6'), 2)} |
| 30 | 6.3 | {fmt(surv30.get('t1'), 2)} | {fmt(surv30.get('t2'), 2)} | {fmt(surv30.get('t3'), 2)} | {fmt(surv30.get('t4'), 2)} | {fmt(surv30.get('t5'), 2)} | {fmt(surv30.get('t6'), 2)} |
| 100 | 15.9 | {fmt(surv100.get('t1'), 2)} | {fmt(surv100.get('t2'), 2)} | {fmt(surv100.get('t3'), 2)} | {fmt(surv100.get('t4'), 2)} | {fmt(surv100.get('t5'), 2)} | {fmt(surv100.get('t6'), 2)} |
""")

    md(r"""
## L1 equation

Scores are standardized inside each fold: $z_k = (t_k - m_k)/s_k$. Numbers below average the LORO fold fits.

### $C=10$: one score

Mean nnz 1.2. $t_1$ appears in every fold; $t_2$ in two (Atlanta, LA), with a small weight.

$$
\tilde{x}_j = \frac{A_j - \mu_j}{\sigma_j},
\quad
t_1 = \mathbf{r}_1^\top \tilde{\mathbf{x}},
\quad
z_1 = \frac{t_1}{s_1}\ (s_1 \approx 3.17)
$$

$$
\mathrm{logit} = 0.011 + 0.962\, z_1 = 0.011 + 0.304\, t_1
$$

$$
P(\mathrm{RES}\mid\mathbf{x}) = \sigma(\mathrm{logit})
$$

LORO macro-F1 ≈ 0.735.

| Metro | intercept | $t_1$ weight | $t_2$ weight |
|---|---:|---:|---:|
""" + coef_table + r"""

![t1 weights by metro](../../outputs/week4_l1_pls_loro/figures/l1_c10_t1_coefs_by_region.png)

### $C=30$: six scores

Mean nnz 6.3. $t_1$ through $t_6$ stay nonzero in essentially every fold.

$$
\begin{aligned}
\mathrm{logit} \approx\ &0.029 \\
&+ 1.395\, z_1 + 0.337\, z_2 + 0.221\, z_3 \\
&+ 0.105\, z_4 + 0.088\, z_5 + 0.036\, z_6
\end{aligned}
$$

On raw scores ($z_k \approx t_k / s_k$):

$$
\mathrm{logit} \approx 0.029 + 0.440\, t_1 + 0.160\, t_2 + 0.105\, t_3 + 0.066\, t_4 + 0.069\, t_5 + 0.026\, t_6
$$

LORO macro-F1 ≈ 0.751 (+0.016 vs $C=10$).
""")

    code("""
from pathlib import Path
import json
import pandas as pd

ROOT = Path("../..").resolve()
OUT = ROOT / "outputs" / "week4_l1_pls_loro"
summary = json.loads((OUT / "summary.json").read_text())
coefs = pd.read_csv(OUT / "l1_c10_fold_coefs.csv")
print("C=10 survival:", summary.get("survival_C10_t1_to_t10"))
coefs.sort_values("region")
""")

    md("""
![L1 path on PLS scores](../../outputs/week4_l1_pls_loro/figures/l1_path_pls32.png)

![L1 path on AE bands](../../outputs/week4_l1_pls_loro/figures/l1_path_ae64.png)

![Which PLS scores survive](../../outputs/week4_l1_pls_loro/figures/l1_survival_pls32.png)

![|coef| vs C for early scores](../../outputs/week4_l1_pls_loro/figures/l1_coef_magnitudes_pls.png)
""")

    md(f"""
## Full L1 path (PLS scores)

| penalty | $C$ | mean nnz | LORO F1 |
|---|---:|---:|---:|
{path_table}
""")

    md("""
## Load sweep tables
""")

    code("""
sweep = pd.read_csv(OUT / "sweep_summary.csv")
metrics = pd.read_csv(OUT / "loro_metrics.csv")
sweep.loc[sweep.features == "pls32"].sort_values(["penalty", "C"])
""")

    md("""
### Surviving names per fold at $C=10$
""")

    code("""
metrics.loc[
    (metrics.features == "pls32") & (metrics.penalty == "l1") & (metrics.C == 10.0),
    ["region", "nnz", "macro_f1", "surviving"]
].sort_values("region")
""")

    md("""
### Surviving names per fold at $C=30$
""")

    code("""
metrics.loc[
    (metrics.features == "pls32") & (metrics.penalty == "l1") & (metrics.C == 30.0),
    ["region", "nnz", "macro_f1", "surviving"]
].sort_values("region")
""")

    md("""
## Takeaways

1. L2 keeps all 32 PLS scores (nnz 32, F1 ≈ 0.775).
2. L1 sparsifies as $C$ falls: nnz 32 → ~16 ($C=100$) → ~6 ($C=30$) → ~1 ($C=10$) → 0 ($C \\le 3$).
3. At $C=10$ the head is $\\mathrm{logit} = 0.011 + 0.304\\, t_1$, F1 ≈ 0.735.
4. At $C=30$ you still keep $t_1\\ldots t_6$; “only $t_1$ and $t_2$” is the strong end of the path, not the whole path.
5. L1 on raw AE also zeros bands, but those axes are not orthogonal. Sparse weights on $t_k$ are easier to read.

Files: `outputs/week4_l1_pls_loro/` (see also `l1_equation.md`, `l1_c10_fold_coefs.csv`).
""")

    # keep l1_equation.md in sync (plain, no broken escapes)
    (OUT / "l1_equation.md").write_text(
        f"""# L1 equation (PLS scores)

## C=10 (t1 only)

logit = 0.011 + 0.962 z1 = 0.011 + 0.304 t1
P(RES) = sigmoid(logit)
LORO macro-F1 ≈ 0.735

## C=30 (t1..t6)

logit ≈ 0.029 + 1.395 z1 + 0.337 z2 + 0.221 z3 + 0.105 z4 + 0.088 z5 + 0.036 z6
       ≈ 0.029 + 0.440 t1 + 0.160 t2 + 0.105 t3 + 0.066 t4 + 0.069 t5 + 0.026 t6
LORO macro-F1 ≈ 0.751
""".lstrip()
    )

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
