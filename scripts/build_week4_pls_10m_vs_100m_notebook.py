#!/usr/bin/env python3
"""Build notebooks/week4/10_pls_10m_vs_100m.ipynb from comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "week4_pls_10m_vs_100m"
NB = ROOT / "notebooks" / "week4" / "10_pls_10m_vs_100m.ipynb"


def fmt(x, nd=3):
    if x is None:
        return "—"
    try:
        if x != x:
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def main() -> None:
    if not (OUT / "summary.json").exists():
        raise SystemExit(f"Missing {OUT / 'summary.json'}; run scripts/train_week4_pls_10m_vs_100m.py first")

    s = json.loads((OUT / "summary.json").read_text())
    sweep = pd.read_csv(OUT / "sweep_summary.csv")
    delta = pd.read_csv(OUT / "delta_f1_by_metro.csv") if (OUT / "delta_f1_by_metro.csv").exists() else None

    s10, s100 = s["10m"], s["100m"]
    cross = s.get("cross_scale", {})

    def row(scale_sum, key, label):
        v = scale_sum.get(key, {}) or {}
        return (
            label,
            fmt(v.get("loro_macro_f1_mean")),
            str(v.get("n_components") or "—"),
            fmt(v.get("mean_nnz"), 1) if v.get("mean_nnz") is not None else "—",
        )

    families = [
        ("baseline_lr", "64-D LR"),
        ("pls_da_best", "PLS-DA best-k"),
        ("t1_only", "t₁ only"),
        ("l1_c10", "L1 on PLS (C=10)"),
        ("l1_c30", "L1 on PLS (C=30)"),
        ("pca_best", "PCA best-k"),
    ]

    table_rows = []
    for key, lab in families:
        a = s10.get(key, {}) or {}
        b = s100.get(key, {}) or {}
        f10 = a.get("loro_macro_f1_mean")
        f100 = b.get("loro_macro_f1_mean")
        d = (float(f10) - float(f100)) if f10 is not None and f100 is not None else None
        table_rows.append(
            f"| {lab} | {fmt(f10)} | {fmt(f100)} | {fmt(d, 3) if d is not None else '—'} |"
        )

    # k=1 numbers
    k1_10 = sweep.loc[(sweep.scale == "10m") & (sweep.model == "pls_da_lr") & (sweep.n_components == 1)]
    k1_100 = sweep.loc[(sweep.scale == "100m") & (sweep.model == "pls_da_lr") & (sweep.n_components == 1)]
    f1_k1_10 = float(k1_10["loro_macro_f1_mean"].iloc[0]) if not k1_10.empty else None
    f1_k1_100 = float(k1_100["loro_macro_f1_mean"].iloc[0]) if not k1_100.empty else None

    mae10 = (s10.get("pls_frac_best") or {}).get("loro_mae_mean")
    mae100 = (s100.get("pls_frac_best") or {}).get("loro_mae_mean")
    ridge10 = (s10.get("baseline_ridge") or {}).get("loro_mae_mean")
    ridge100 = (s100.get("baseline_ridge") or {}).get("loro_mae_mean")

    mean_delta_metro = float(delta["delta"].mean()) if delta is not None else cross.get("delta_f1_pls_da_best_10m_minus_100m")

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""# PLS: 10 m cells vs 100 m mean-pooled tiles

LORO over the same 10 metros. Features are 64-D AlphaEarth at both scales — at 100 m each tile is the **mean** of its 10 m cell embeddings (not the 10×10 grid model).

| | 10 m | 100 m |
|---|---:|---:|
| Units | {s10['n']:,} pure cells | {s100['n']:,} tiles (min 5 cells) |
| Balanced RES / NRES | {s10['n_res']:,} / {s10['n_nres']:,} | {s100['n_res']:,} / {s100['n_nres']:,} |
| Best PLS-DA F1 | {fmt((s10.get('pls_da_best') or {}).get('loro_macro_f1_mean'))} (k={ (s10.get('pls_da_best') or {}).get('n_components') }) | {fmt((s100.get('pls_da_best') or {}).get('loro_macro_f1_mean'))} (k={ (s100.get('pls_da_best') or {}).get('n_components') }) |
| 64-D LR F1 | {fmt((s10.get('baseline_lr') or {}).get('loro_macro_f1_mean'))} | {fmt((s100.get('baseline_lr') or {}).get('loro_macro_f1_mean'))} |
| t₁-only F1 | {fmt((s10.get('t1_only') or {}).get('loro_macro_f1_mean'))} | {fmt((s100.get('t1_only') or {}).get('loro_macro_f1_mean'))} |

**Takeaway.** Mean-pooling to 100 m costs about **{fmt(cross.get('delta_f1_pls_da_best_10m_minus_100m'), 3)}** macro-F1 on the best PLS-DA head. The sparsity story is the same at both scales: strong L1 (C=10) leaves ~1 nonzero score ($t_1$), with F1 matching the explicit $t_1$-only head.

```bash
.venv/bin/python -u scripts/train_week4_pls_10m_vs_100m.py --backend gpu
.venv/bin/python -u scripts/build_week4_pls_10m_vs_100m_notebook.py
```
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Result table

LORO mean macro-F1. Δ = 10 m − 100 m.

| Head | 10 m | 100 m | Δ |
|---|---:|---:|---:|
{chr(10).join(table_rows)}

Pooled OOF F1 (all metros concatenated): 10 m baseline {fmt(s10.get('oof_macro_f1_baseline'))}, PLS {fmt(s10.get('oof_macro_f1_pls'))}; 100 m baseline {fmt(s100.get('oof_macro_f1_baseline'))}, PLS {fmt(s100.get('oof_macro_f1_pls'))}.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## F1 vs number of PLS components

![F1 vs k](../../outputs/week4_pls_10m_vs_100m/figures/f1_vs_k_by_scale.png)

"""
            + f"At $k=1$: 10 m F1 ≈ {fmt(f1_k1_10)}, 100 m ≈ {fmt(f1_k1_100)}. "
            + f"Best 10 m at k={(s10.get('pls_da_best') or {}).get('n_components')}; "
            + f"best 100 m at k={(s100.get('pls_da_best') or {}).get('n_components')}."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Heads side by side

![Family compare](../../outputs/week4_pls_10m_vs_100m/figures/f1_family_10m_vs_100m.png)
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Per-metro gap

![Delta by metro](../../outputs/week4_pls_10m_vs_100m/figures/delta_f1_by_metro.png)

Mean metro Δ (best PLS-DA) ≈ {fmt(mean_delta_metro, 3)}. 10 m is higher in every fold.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## L1 sparsity (same pattern at both scales)

![L1 nnz](../../outputs/week4_pls_10m_vs_100m/figures/l1_nnz_by_scale.png)

| Scale | C=10 mean nnz | C=10 F1 | t₁-only F1 | C=30 mean nnz | C=30 F1 |
|---|---:|---:|---:|---:|---:|
| 10 m | {fmt((s10.get('l1_c10') or {}).get('mean_nnz'), 1)} | {fmt((s10.get('l1_c10') or {}).get('loro_macro_f1_mean'))} | {fmt((s10.get('t1_only') or {}).get('loro_macro_f1_mean'))} | {fmt((s10.get('l1_c30') or {}).get('mean_nnz'), 1)} | {fmt((s10.get('l1_c30') or {}).get('loro_macro_f1_mean'))} |
| 100 m | {fmt((s100.get('l1_c10') or {}).get('mean_nnz'), 1)} | {fmt((s100.get('l1_c10') or {}).get('loro_macro_f1_mean'))} | {fmt((s100.get('t1_only') or {}).get('loro_macro_f1_mean'))} | {fmt((s100.get('l1_c30') or {}).get('mean_nnz'), 1)} | {fmt((s100.get('l1_c30') or {}).get('loro_macro_f1_mean'))} |

$C=10$ ≈ $t_1$ only. $C=30$ keeps ~6–7 scores and picks up ~0.015–0.016 F1 at both scales.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Fraction MAE

![MAE vs k](../../outputs/week4_pls_10m_vs_100m/figures/mae_vs_k_by_scale.png)

Best PLS fraction MAE: 10 m {fmt(mae10)} (k={(s10.get('pls_frac_best') or {}).get('n_components')}), Ridge {fmt(ridge10)}.  
100 m {fmt(mae100)} (k={(s100.get('pls_frac_best') or {}).get('n_components')}), Ridge {fmt(ridge100)}.

MAE is not a clean scale comparison: 10 m uses pure balanced cells (often extreme fractions), 100 m uses tile-mean fractions over mixed cells. Treat F1 as the primary scale metric.
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## What this does *not* compare

- **10×10 grid LR** on 100 m tiles (`outputs/week4_100m_tile_train/`) — 6500-D spatial input, different feature space.
- **Pooling 10 m predictions** onto tiles (`outputs/week4_100m_fraction_eval/`) — aggregation of cell scores, not a new PLS fit.

Here the question is: if you keep a single 64-D AE vector and only change the spatial unit (cell vs mean-pooled tile), how does PLS / $t_1$ / L1 behave?
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"""## Artifacts

| Path | Contents |
|---|---|
| `outputs/week4_pls_10m_vs_100m/summary.json` | Cross-scale numbers |
| `outputs/week4_pls_10m_vs_100m/sweep_summary.csv` | F1 / MAE / nnz by scale × model × k |
| `outputs/week4_pls_10m_vs_100m/10m/` · `100m/` | Per-scale OOF, fold checkpoints, L1 coefs |
| `outputs/week4_pls_10m_vs_100m/figures/` | Claim-first plots |

Elapsed ≈ {s.get('elapsed_s')} s on {s.get('n_gpus')} GPUs.
"""
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Wrote {NB}")


if __name__ == "__main__":
    main()
