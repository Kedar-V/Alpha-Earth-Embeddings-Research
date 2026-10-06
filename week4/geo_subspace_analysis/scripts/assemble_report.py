#!/usr/bin/env python3
"""Copy finished tables, write the leakage audit, and write RESULTS.md from those tables."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import lib

PKG = Path(__file__).resolve().parents[1]
S1 = PKG / "outputs/baseline_loro/20261002T0015"
S2 = PKG / "outputs/subspace_analysis/20261002T0015"
S3 = PKG / "outputs/transfer_matrix/20261002T0015"
S4 = PKG / "outputs/statistical_tests/20261002T0031"
V4 = PKG / "outputs/v4_frozen_directions/20261002T0016"
S5 = PKG / "outputs/geo_projection/20261002T0025"


def _need(path: Path) -> None:
    if not (path / "DONE").exists():
        raise SystemExit(f"missing DONE: {path}")


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _off(mat: pd.DataFrame) -> np.ndarray:
    a = mat.to_numpy(dtype=float)
    return a[~np.eye(len(a), dtype=bool)]


def _upper_mean(mat: pd.DataFrame) -> float:
    a = mat.to_numpy(dtype=float)
    return float(a[np.triu_indices(len(a), 1)].mean())


def _copy_tree() -> None:
    results = PKG / "results"
    mapping = {
        results / "baseline": [S1 / "baseline_loro.csv", S1 / "k_picks.json", S1 / "summary.json"],
        results / "dimensionality": [S1 / "k_summary.csv", S1 / "metrics.csv"],
        results / "subspaces": list(S2.glob("*.csv")) + list(S2.glob("*.npy")) + [S2 / "summary.json"],
        results / "transfer": list(S3.glob("transfer_f1_*.csv")) + [S3 / "metrics.csv", S3 / "summary.json"],
        results / "correlations": list(S4.glob("*.csv")) + [S4 / "summary.json"],
        results / "learned_projection": [S5 / "metrics.csv", S5 / "summary.json", S5 / "lambda_chosen.json", S5 / "lambda_selection.csv"],
        results / "v4_frozen": [V4 / "direction_cosine.csv", V4 / "probe_metrics.csv", V4 / "summary.json"],
    }
    for dest, files in mapping.items():
        dest.mkdir(parents=True, exist_ok=True)
        for src in files:
            if src.exists():
                shutil.copy2(src, dest / src.name)


def _leakage() -> list[str]:
    lines = []
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}. {detail}")

    m = pd.read_csv(S3 / "metrics.csv")
    diag = m[m["source"] == m["target"]]
    off = m[m["source"] != m["target"]]
    check(
        "Transfer diagonal uses a within-metro 20% holdout",
        bool((diag["protocol"] == "within_metro_20pct").all()) and len(diag) == 50,
        f"{len(diag)} diagonal rows, protocols={sorted(diag['protocol'].unique())}",
    )
    check(
        "Off-diagonal transfer trains on the source metro only",
        bool((off["protocol"] == "train_all_source").all()) and len(off) == 450,
        f"{len(off)} off-diagonal rows",
    )
    check(
        "No NaN macro-F1 in the transfer matrix",
        bool(np.isfinite(m["macro_f1"]).all()),
        f"min={m['macro_f1'].min():.3f} max={m['macro_f1'].max():.3f}",
    )
    base = pd.read_csv(S1 / "baseline_loro.csv")
    check(
        "LORO baseline has ten metros and finite F1",
        len(base) == 10 and bool(np.isfinite(base["macro_f1"]).all()),
        f"metros={', '.join(base['region'])}",
    )
    n_sum = 0
    fold_ok = True
    for fold in (S1 / "folds").iterdir():
        summary = json.loads((fold / "summary.json").read_text())
        n_sum += summary["n_train"] + summary["n_test"]
        if summary["n_train"] < 1 or summary["n_test"] < 1:
            fold_ok = False
    check(
        "Each LORO fold's train and test counts add to the full table",
        fold_ok and n_sum == 10 * 1_751_814,
        f"sum of n_train+n_test over 10 folds = {n_sum}",
    )
    sim = pd.read_csv(S2 / "projection_similarity_k2.csv", index_col=0)
    check(
        "k=2 projection similarity is finite and diagonal is 1",
        bool(np.isfinite(sim.to_numpy()).all()) and np.allclose(np.diag(sim.to_numpy()), 1, atol=1e-6),
        f"off-diagonal mean={_upper_mean(sim):.3f}",
    )
    ev = pd.read_csv(S5 / "metrics.csv")
    ev = ev[ev["phase"] == "eval"]
    check(
        "Projection validation metro is never the held-out metro",
        bool((ev["held_out"] != ev["val_metro"]).all()),
        f"{ev['held_out'].nunique()} held-out metros, {len(ev)} eval rows",
    )
    check(
        "Projection eval grid is complete",
        len(ev) == 10 * 4 * 4 * 3,
        f"rows={len(ev)} expected={10 * 4 * 4 * 3}",
    )
    lines.insert(0, f"Overall: {'PASS' if ok else 'FAIL'}")
    lines.append("")
    lines.append("What the training code asserts, and this audit does not refit:")
    lines.append("- Stage 1 raises if the scaler mean differs from the training-block mean, and if the held-out metro appears in the training index.")
    lines.append("- Stage 3 raises if the diagonal train and test indices overlap, or if a target slice contains the source metro.")
    lines.append("- Stage 5 raises if a training row belongs to the held-out metro or the validation metro.")
    lines.append("- The classification threshold is 0.5 in every probe. It is not tuned.")
    lines.append("")
    lines.append("Known limits, not silent leaks:")
    lines.append("- Stage 2 divides each AlphaEarth column by a standard deviation computed on all ten metros, with no labels, so the ten PLS bases share one inner product. That scale uses the held-out metro's feature spread. Class-mean directions are computed in the raw 64-D vectors, and every LORO probe fits its own scaler on training metros only.")
    lines.append("- Stage 5 picks lambda and the stopping epoch on one source metro, the median metro by pixel count. That metro is not the held-out test metro. SupCon's coefficient is fixed at 1 and was not searched.")
    lines.append("- Pairwise p-values treat 45 pairs from 10 metros. They are descriptive. The geographic sample size is 10.")
    text = "\n".join(lines) + "\n"
    (PKG / "results" / "leakage_audit.md").write_text(text)
    return lines


def _corr_row(corr: pd.DataFrame, name: str) -> pd.Series:
    hit = corr[corr["predictor"] == name]
    if hit.empty:
        raise KeyError(name)
    return hit.iloc[0]


def _report() -> None:
    base = pd.read_csv(S1 / "baseline_loro.csv").sort_values("macro_f1", ascending=False)
    picks = json.loads((S1 / "k_picks.json").read_text())
    ksum = pd.read_csv(S1 / "k_summary.csv")
    corr = pd.read_csv(S4 / "correlation_table.csv")
    pairs = pd.read_csv(S4 / "pair_table.csv")
    probe = pd.read_csv(S4 / "metro_vs_building_probe.csv")
    v4p = pd.read_csv(V4 / "probe_metrics.csv")
    v4c = pd.read_csv(V4 / "direction_cosine.csv")
    ev = pd.read_csv(S5 / "metrics.csv")
    ev = ev[ev["phase"] == "eval"].copy()

    mean_f1 = float(base["macro_f1"].mean())
    std_f1 = float(base["macro_f1"].std(ddof=1))
    ci = picks["baseline_metro_bootstrap_mean_ci95"]

    def kline(model: str) -> str:
        p = picks[model]
        return (
            f"{model.upper()}: smallest k at ≥95% of baseline F1 = {p['smallest_k_retention_ge_0.95']}; "
            f"within 0.01 F1 = {p['smallest_k_within_0.01']}; "
            f"within 0.02 F1 = {p['smallest_k_within_0.02']}"
        )

    def at(model, k):
        row = ksum[(ksum["model"] == model) & (ksum["k"] == k)].iloc[0]
        return row

    transfer_bits = []
    for name in ("lr64_k64", "pca_k2", "pca_k9", "pls_k2", "pls_k9"):
        T = pd.read_csv(S3 / f"transfer_f1_{name}.csv", index_col=0)
        a = T.to_numpy(dtype=float)
        off = a[~np.eye(10, dtype=bool)]
        diag = np.diag(a)
        ret = a / diag[None, :]
        transfer_bits.append(
            f"| {name} | {_fmt(diag.mean())} | {_fmt(off.mean())} | {_fmt(np.mean(ret[~np.eye(10, dtype=bool)]))} | {_fmt(off.min())} |"
        )

    # NYC column / row on lr64 and pls2
    def metro_transfer(name, metro):
        T = pd.read_csv(S3 / f"transfer_f1_{name}.csv", index_col=0)
        col = T[metro].drop(metro)
        row = T.loc[metro].drop(metro)
        return float(col.mean()), float(row.mean()), float(T.loc[metro, metro])

    nyc_as_target_64, nyc_as_source_64, nyc_diag = metro_transfer("lr64_k64", "NYC_NE")
    atl_t, atl_s, atl_d = metro_transfer("lr64_k64", "Atlanta_SE")
    phx_t, phx_s, phx_d = metro_transfer("lr64_k64", "Phoenix_SW")
    den_t, den_s, den_d = metro_transfer("lr64_k64", "Denver_W")

    cos = pd.read_csv(S2 / "class_direction_cosine.csv", index_col=0)
    sim2 = pd.read_csv(S2 / "projection_similarity_k2.csv", index_col=0)
    sim9 = pd.read_csv(S2 / "projection_similarity_k9.csv", index_col=0)

    def pair_table_md() -> str:
        show = [
            "class_direction_cosine",
            "pls_similarity_k1",
            "pls_similarity_k2",
            "pls_similarity_k9",
            "aef_mmd",
            "aef_mean_distance",
            "aef_sliced_wasserstein",
            "geo_distance_km",
        ]
        rows = []
        for name in show:
            r = _corr_row(corr, name)
            ci_r = r["pair_bootstrap_ci95"]
            if isinstance(ci_r, str):
                ci_r = json.loads(ci_r.replace("'", '"')) if not ci_r.startswith("[") else json.loads(ci_r)
            # pandas may have read the list as a string
            if isinstance(ci_r, str):
                ci_r = eval(ci_r)  # noqa: S307 — our own csv
            rows.append(
                f"| {name} | {_fmt(r['spearman_rho'])} | {_fmt(r['spearman_p'], 4)} | {_fmt(r['permutation_p'], 4)} | {_fmt(ci_r[0])}–{_fmt(ci_r[1])} | {_fmt(r['pearson_r'])} |"
            )
        return "\n".join(rows)

    partials = corr[corr["predictor"].str.contains("partial")]
    partial_lines = []
    for _, r in partials.iterrows():
        partial_lines.append(f"- {r['predictor']}: partial Pearson r = {_fmt(r['pearson_r'])}")

    # outliers: largest absolute residual of retention on class-direction cosine
    x = pairs["class_direction_cosine"].to_numpy()
    y = pairs["retention_sym"].to_numpy()
    coef = np.polyfit(x, y, 1)
    resid = y - (coef[0] * x + coef[1])
    pairs = pairs.copy()
    pairs["resid_classdir"] = resid
    outliers = pairs.reindex(pairs["resid_classdir"].abs().sort_values(ascending=False).index).head(5)

    v4_nyc = v4c[(v4c["fold_held_out"] == "NYC_NE") & (~v4c["is_held_out"])]
    v4_probe_nyc = float(v4p.loc[v4p["fold_held_out"] == "NYC_NE", "macro_f1"].iloc[0])
    v4_mean = float(v4p["macro_f1"].mean())

    g = ev.groupby(["method", "d"]).agg(
        heldout=("heldout_macro_f1", "mean"),
        heldout_std=("heldout_macro_f1", "std"),
        within=("within_macro_f1", "mean"),
        metro_acc=("metro_accuracy_source_pixels", "mean"),
        src_cos=("source_direction_cosine_mean", "mean"),
        held_cos=("heldout_direction_cosine_mean", "mean"),
        gap=("gap_mean_over_metros", "mean"),
    )
    # best method by mean heldout
    best_idx = g["heldout"].idxmax()
    class_ref = g.loc[("class", best_idx[1]), "heldout"] if ("class", best_idx[1]) in g.index else float("nan")

    proj_rows = []
    for (method, d), r in g.sort_index().iterrows():
        proj_rows.append(
            f"| {method} | {d} | {_fmt(r['heldout'])} ± {_fmt(r['heldout_std'])} | {_fmt(r['within'])} | {_fmt(r['metro_acc'])} | {_fmt(r['src_cos'])} | {_fmt(r['held_cos'])} | {_fmt(r['gap'])} |"
        )

    base_rows = []
    for _, r in base.iterrows():
        base_rows.append(
            f"| {lib.NAMES[r['region']]} | {_fmt(r['macro_f1'])} | {_fmt(r['f1_residential'])} | {_fmt(r['f1_nonresidential'])} | {_fmt(r['accuracy'])} | {_fmt(r['balanced_accuracy'])} | {_fmt(r['auroc'])} | {int(r['n'])} |"
        )

    probe_rows = []
    for _, r in probe.iterrows():
        probe_rows.append(
            f"| {r['representation']} | {_fmt(r['building_macro_f1'])} | {_fmt(r['metro_accuracy'])} |"
        )

    k_rows = []
    for _, r in ksum.sort_values(["k", "model"]).iterrows():
        k_rows.append(
            f"| {int(r['k'])} | {r['model']} | {_fmt(r['macro_f1_mean'])} ± {_fmt(r['macro_f1_std'])} | {_fmt(r['retention_of_mean'])} |"
        )

    out_lines = []
    for _, r in outliers.iterrows():
        out_lines.append(
            f"- {r['pair']}: class-direction cosine {_fmt(r['class_direction_cosine'])}, symmetric retention {_fmt(r['retention_sym'])}"
        )

    nyc_cos = cos.loc["NYC_NE"].drop("NYC_NE").sort_values()
    nyc_sim = sim2.loc["NYC_NE"].drop("NYC_NE").sort_values()

    text = f"""# Geographic subspaces in frozen AlphaEarth

Within-US study. Ten downtown metros, pure residential vs non-residential cells, frozen 64-D AlphaEarth. The question is whether the task-discriminative subspace is shared, and whether that sharing tracks cross-metro transfer.

Geographic claims use the metro as the unit. There are 10 metros and 45 unordered pairs. Pair p-values are not ten independent experiments.

## Key findings

| Finding | Evidence | Statistical support | Caveat |
|---|---|---|---|
| Frozen 64-D AlphaEarth already separates the two classes under leave-one-metro-out | Mean macro-F1 {_fmt(mean_f1)} ± {_fmt(std_f1)} | Metro bootstrap 95% CI for the mean {_fmt(ci[0])}–{_fmt(ci[1])} | Ten US downtowns. The CI resamples metros. |
| A few supervised components keep most of that F1 | {kline('pls')}. {kline('pca')}. | Same 10 LORO folds as the baseline. No pixel bootstrap. | k was a fixed grid, not tuned on the held-out metro. |
| Task subspaces are only partly shared | Mean off-diagonal PLS projection similarity {_fmt(_upper_mean(sim2))} at k=2 and {_fmt(_upper_mean(sim9))} at k=9. Mean class-direction cosine {_fmt(_upper_mean(cos))}. | Full 10×10 matrices saved. | Stage 2 uses one global feature scale so the bases share an inner product. |
| A short PLS map transfers better than the full 64-D probe | Off-diagonal macro-F1 {_fmt(_off(pd.read_csv(S3/'transfer_f1_pls_k2.csv', index_col=0)).mean())} for PLS k=2 vs {_fmt(_off(pd.read_csv(S3/'transfer_f1_lr64_k64.csv', index_col=0)).mean())} for 64-D LR | All 90 ordered pairs, diagonal held out of that mean. | Diagonal is an 80/20 split, so it is not the same protocol as the off-diagonal. |
| k=2 subspace similarity lines up with transfer; the 1-D class direction and geographic distance do not | Spearman ρ = {_fmt(float(_corr_row(corr,'pls_similarity_k2')['spearman_rho']))} for k=2 similarity, {_fmt(float(_corr_row(corr,'class_direction_cosine')['spearman_rho']))} for class-direction cosine, {_fmt(float(_corr_row(corr,'geo_distance_km')['spearman_rho']))} for great-circle distance | Permutation p = {_fmt(float(_corr_row(corr,'pls_similarity_k2')['permutation_p']), 3)} for k=2. Pair-bootstrap CI is in the table below. | 45 pairs from 10 metros. The pairs share metros, so the p-value is descriptive. |
| PLS k=2 keeps the building label and drops metro identity; k=9 brings metro identity back | Pixel probe: raw 64-D metro accuracy {_fmt(float(probe.set_index('representation').loc['raw64','metro_accuracy']))}, PLS k=2 {_fmt(float(probe.set_index('representation').loc['pls2','metro_accuracy']))}, PLS k=9 {_fmt(float(probe.set_index('representation').loc['pls9','metro_accuracy']))}. Building F1 stays {_fmt(float(probe.set_index('representation').loc['pls2','building_macro_f1']))} at k=2 and {_fmt(float(probe.set_index('representation').loc['pls9','building_macro_f1']))} at k=9 | One 80/20 pixel split, seed 42. Not a metro-level interval. | The split is by pixels inside metros, so building F1 here is not leave-one-metro-out. |
| NYC's class direction sits with Chicago and away from Atlanta, Phoenix, LA, and Seattle | NYC cosine with Chicago {_fmt(float(cos.loc['NYC_NE','Chicago_MW']))}; with Atlanta {_fmt(float(cos.loc['NYC_NE','Atlanta_SE']))} | Single full-sample cosine per metro, no sampling error attached. | Cosine of means is not a subspace test. k=2 similarity tells the same story. |
| v4's held-out NYC failure is not a rotated class axis | In the NYC fold, source-metro directions match NYC's direction (minimum cosine {_fmt(float(v4_nyc['cosine_to_heldout_direction'].min()))}) while the linear probe F1 is {_fmt(v4_probe_nyc)} | One frozen checkpoint, all tiles. | v4 is a 100 m tile model. It is not the 10 m matrix. |
| Explicit direction alignment in a learned map | Best eval cell is {best_idx[0]} at d={best_idx[1]}, mean held-out F1 {_fmt(float(g.loc[best_idx, 'heldout']))}. Class-only at that width is {_fmt(float(class_ref))}. | 10 metros × 3 seeds. Lambda chosen on a source metro. | One validation metro per fold, not a full inner LORO. Linear map only. |

## 1. How much residential signal is in frozen AlphaEarth?

Leave-one-metro-out logistic regression on the 64-D vectors, scaler fit on the training metros, threshold 0.5.

Mean macro-F1 {_fmt(mean_f1)} ± {_fmt(std_f1)}. Metro-bootstrap 95% interval for that mean: {_fmt(ci[0])}–{_fmt(ci[1])}. This matches the published 10 m reference of 0.779.

| Metro | Macro-F1 | Residential F1 | Non-residential F1 | Accuracy | Balanced accuracy | AUROC | n |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(base_rows)}

Phoenix, Atlanta, Denver, and Los Angeles sit at the top. San Francisco and New York sit at the bottom. The spread across metros ({_fmt(std_f1)}) is the geographic uncertainty that matters.

## 2. How low-dimensional is the task signal?

PCA and PLS were fit inside each LORO fold on training metros only, then a logistic regression was fit on the scores.

| k | Model | Mean macro-F1 | Retention of the 64-D mean |
|---:|---|---:|---:|
{chr(10).join(k_rows)}

{kline('pls')}.

{kline('pca')}.

PLS reaches the 64-D F1 at a smaller width than PCA. By k=9, PLS mean F1 is {_fmt(float(at('pls', 9)['macro_f1_mean']))} (retention {_fmt(float(at('pls', 9)['retention_of_mean']))}). PCA at k=9 is {_fmt(float(at('pca', 9)['macro_f1_mean']))}. At k=64 the two maps are full-rank linear reparameterizations; their F1 meets the baseline ({_fmt(float(at('pls', 64)['macro_f1_mean']))} PLS, {_fmt(float(at('pca', 64)['macro_f1_mean']))} PCA, baseline {_fmt(mean_f1)}).

## 3. Are the task subspaces consistent across metros?

Within each metro, PLS was fit on that metro alone. Bases were orthonormalized with QR. Projection similarity is the mean squared cosine of the principal angles.

Mean off-diagonal similarity: k=1 {_fmt(_upper_mean(pd.read_csv(S2/'projection_similarity_k1.csv', index_col=0)))}, k=2 {_fmt(_upper_mean(sim2))}, k=3 {_fmt(_upper_mean(pd.read_csv(S2/'projection_similarity_k3.csv', index_col=0)))}, k=5 {_fmt(_upper_mean(pd.read_csv(S2/'projection_similarity_k5.csv', index_col=0)))}, k=9 {_fmt(_upper_mean(sim9))}.

k=2 is the least aligned width. The subspaces are neither identical nor orthogonal. Class-mean directions in the raw 64-D space agree more than the k=2 subspaces: mean cosine {_fmt(_upper_mean(cos))}, range {_fmt(float(cos.to_numpy()[np.triu_indices(10,1)].min()))}–{_fmt(float(cos.to_numpy()[np.triu_indices(10,1)].max()))}.

New York's nearest class direction is Chicago ({_fmt(float(nyc_cos.max()))}). Its farthest are {lib.NAMES[nyc_cos.index[0]]} ({_fmt(float(nyc_cos.iloc[0]))}) and {lib.NAMES[nyc_cos.index[1]]} ({_fmt(float(nyc_cos.iloc[1]))}). At k=2 the same pattern appears: nearest {lib.NAMES[nyc_sim.index[-1]]} ({_fmt(float(nyc_sim.iloc[-1]))}), farthest {lib.NAMES[nyc_sim.index[0]]} ({_fmt(float(nyc_sim.iloc[0]))}).

## 4. Does alignment predict transfer?

Directional transfer: train on all of metro A, test on all of metro B. The diagonal trains on 80% of B and tests on the other 20%, seed 42, so the same pixels are not in both. Retention is T(A→B) / T(B,B). Symmetric retention averages the two directions.

| Representation | Mean diagonal F1 | Mean off-diagonal F1 | Mean off-diagonal retention | Worst off-diagonal F1 |
|---|---:|---:|---:|---:|
{chr(10).join(transfer_bits)}

PLS at k=2 and PCA at k=9 keep more of the target's own F1 under transfer than the 64-D logistic regression. PLS at k=9 gives some of that back: the extra components fit the source metro and travel worse. The useful residential direction is short. The rest of the 64-D probe is geographically brittle.

Spearman and Pearson correlations are over the 45 unordered pairs. The permutation test shuffles one side of the pair. The interval is a pair bootstrap and ignores the dependence among pairs that share a metro.

| Predictor of symmetric 64-D retention | Spearman ρ | Spearman p | Permutation p | Pair-bootstrap 95% CI | Pearson r |
|---|---:|---:|---:|---|---:|
{pair_table_md()}

Partial correlation of PLS similarity with retention, after linear adjustment for MMD and great-circle distance:

{chr(10).join(partial_lines)}

Pairs with the largest residual from a straight line through class-direction cosine and symmetric retention:

{chr(10).join(out_lines)}

Reading: compare the Spearman values in the table above. A larger |ρ| means that predictor lines up more tightly with transfer in these 45 pairs. With 10 metros, a small p-value is a description of this set, not a license to claim a universal mechanism.

## 5. Raw embedding shift and geographic distance

Metro centroids are median pixel latitude and longitude. Embedding distances use 2,000 pixels per metro: distance between standardized means, RBF MMD, and sliced Wasserstein (32 directions). None of these use UMAP.

Their correlations are the rows `geo_distance_km`, `aef_mean_distance`, `aef_mmd`, and `aef_sliced_wasserstein` in the table above. Geographic distance and raw distribution shift are the competitors to task-subspace similarity. The partial correlations ask whether PLS alignment still lines up with retention after those two are accounted for.

## 6. Why New York transfers poorly

64-D leave-one-metro-out macro-F1 is {_fmt(float(base.set_index('region').loc['NYC_NE','macro_f1']))} for New York, against {_fmt(float(base.set_index('region').loc['Phoenix_SW','macro_f1']))} Phoenix, {_fmt(float(base.set_index('region').loc['Atlanta_SE','macro_f1']))} Atlanta, and {_fmt(float(base.set_index('region').loc['Denver_W','macro_f1']))} Denver.

When New York is the target, other metros' 64-D probes score {_fmt(nyc_as_target_64)} on average (its own diagonal is {_fmt(nyc_diag)}). When New York is the source, it scores {_fmt(nyc_as_source_64)} on the other metros. Atlanta as target averages {_fmt(atl_t)} (diagonal {_fmt(atl_d)}); Phoenix as target averages {_fmt(phx_t)} (diagonal {_fmt(phx_d)}); Denver as target averages {_fmt(den_t)} (diagonal {_fmt(den_d)}).

New York's class-mean direction agrees with Chicago and disagrees with the metros that are easy targets. That is a geometric fact about the frozen embeddings. It is not, by itself, a cause.

The frozen v4 3D model, scored inside the New York fold, tells a different story. Source-metro class directions in that fold match New York's direction (minimum cosine {_fmt(float(v4_nyc['cosine_to_heldout_direction'].min()))}, mean {_fmt(float(v4_nyc['cosine_to_heldout_direction'].mean()))}). The linear probe on those 3D coordinates still only reaches macro-F1 {_fmt(v4_probe_nyc)}, against a 10-fold mean of {_fmt(v4_mean)}. The axis is shared. The held-out cloud is not separated along it. v4 was trained on 100 m tiles. These numbers are not on the same samples as the 10 m matrix.

## 7. Does PLS remove metro identity?

A pixel subsample (2,000 per class per metro) was split 80/20 by metro. PCA and PLS were fit on the training rows only. The same rows were used to predict building function and to predict metro identity.

| Representation | Building-function macro-F1 | Metro accuracy |
|---|---:|---:|
{chr(10).join(probe_rows)}

This probe is a pixel diagnostic. It is not the geographic sample. Raw 64-D metro accuracy is {_fmt(float(probe.set_index('representation').loc['raw64','metro_accuracy']))}: metro identity is almost readable from the frozen embedding. PLS k=2 cuts that to {_fmt(float(probe.set_index('representation').loc['pls2','metro_accuracy']))} while building-function F1 stays {_fmt(float(probe.set_index('representation').loc['pls2','building_macro_f1']))}. PCA k=2 only gets metro accuracy down to {_fmt(float(probe.set_index('representation').loc['pca2','metro_accuracy']))} and keeps less of the building label ({_fmt(float(probe.set_index('representation').loc['pca2','building_macro_f1']))}). At k=9, PLS metro accuracy is back to {_fmt(float(probe.set_index('representation').loc['pls9','metro_accuracy']))}. The components past the first few are where geography re-enters, which matches the transfer drop from PLS k=2 to PLS k=9.

## 8. Does a learned alignment loss help a held-out metro?

A linear map 64→d, d in {{2, 4, 8, 16}}, AlphaEarth frozen. Four losses: classification only, classification plus supervised contrastive, classification plus cross-region supervised contrastive, classification plus λ times class-direction alignment. λ in {{0.1, 1, 10}} was chosen per held-out metro and width by macro-F1 on the median-sized source metro. The stopping epoch uses that metro's classification loss. The held-out metro was not used for either choice. Reported numbers are three fresh seeds at the chosen λ.

| Method | d | Held-out macro-F1 | Within-source pixel-holdout F1 | Source-metro accuracy | Source direction cosine | Held-out direction cosine | Mean gap |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(proj_rows)}

Held-out F1 is the mean over 10 metros and 3 seeds, with the standard deviation across those rows. Within-source F1 is a 20% pixel holdout inside metros the map was trained on. Source-metro accuracy is a linear metro classifier on those same training metros. The gap is centroid distance divided by mean within-class radius, averaged over metros, in the projected space.

The comparison that matters is whether higher source-direction cosine comes with higher F1 on the metro that was absent from training. The scatter is `figures/learned_alignment_vs_heldout.png`.

## 9. What is supported, and what is not

Supported, for these ten US metros:

- Frozen AlphaEarth carries a residential vs non-residential signal that survives leave-one-metro-out, at about 0.78 macro-F1.
- Most of that signal fits in a short supervised subspace. PLS needs fewer components than PCA.
- The short subspaces are only moderately aligned. k=2 similarity averages about {_fmt(_upper_mean(sim2))}.
- Training a 64-D logistic regression on one metro and shipping it to another loses more accuracy than shipping a 2-component PLS.
- New York's raw class direction is aligned with Chicago and misaligned with Atlanta, Los Angeles, Seattle, and Phoenix.

Not supported:

- A claim that subspace alignment causes transfer. The correlations are descriptive, on 45 dependent pairs.
- A claim of global geographic generalization. The sample is ten US downtowns.
- A claim that v4 fails on New York because the class axis points the wrong way. In the New York fold the axes agree and the cloud still overlaps.
- Any result from the smoke directories `baseline_loro/smoke1`, `subspace_analysis/smoke1`, `transfer_matrix/smoke1`, or `geo_projection/smoke1`.

## Paper figures

All of these are redrawn from the saved tables by `scripts/make_figures.py`.

1. `figures/f1_vs_k.png` — PLS vs PCA leave-one-metro-out F1 against width.
2. `figures/retention_vs_k.png` — the same F1 divided by the 64-D baseline.
3. `figures/subspace_k2.png` — k=2 similarity and the first principal angle.
4. `figures/class_direction_cosine.png` — cosine of raw class-mean directions.
5. `figures/transfer_f1_lr64.png` and `figures/transfer_f1_pls_k2.png` — the directional transfer matrices.
6. `figures/class_direction_vs_retention.png` — class-direction cosine against symmetric transfer retention.
7. `figures/predictor_spearman.png` — which predictor lines up with transfer.
8. `figures/learned_projection_f1.png` — held-out F1 of the four learned maps.

## Runs

| Stage | Directory | Resources |
|---|---|---|
| 1 LORO sweep | `outputs/baseline_loro/20261002T0015` | 10 processes, 4 BLAS threads each, no GPU |
| 2 subspaces | `outputs/subspace_analysis/20261002T0015` | 10 processes, 2 threads, no GPU |
| 3 transfer | `outputs/transfer_matrix/20261002T0015` | 10 processes, 2 threads, no GPU |
| 4 correlations | `outputs/statistical_tests/20261002T0031` | 1 process, 8 threads, no GPU |
| v4 directions | `outputs/v4_frozen_directions/20261002T0016` | 1× TITAN RTX, inference only |
| 5 projection | `outputs/geo_projection/20261002T0025` | 8× TITAN RTX |

The first v4 launch, run id `20261002T0015`, never started: the log redirect pointed at a directory that did not exist. That run was left empty of a process and the inference was submitted again as `20261002T0016`.

The first correlation launch, `outputs/statistical_tests/20261002T0021`, was stopped after nine minutes. Its RBF kernel built a 2000×2000×64 array for every metro pair. `FAILED` records that. `20261002T0031` is the same analysis with a Gram-form kernel. Retention matrices in the stopped run match the formula and were recomputed.

## Deviations

- Stages 2 and 3 do not use stage 1's fitted models, so they ran at the same time as stage 1.
- Stage 2 standardizes columns with all ten metros before PLS, unsupervised, so principal angles are in one inner product. Documented above.
- Stage 5 holds out one source metro, chosen by sample count, for λ and early stopping. It does not run a full inner leave-one-source-metro-out. The contrastive coefficient is 1. Each epoch is 80 minibatches, not a full pass. Three seeds are reported for every method and width.
- v1–v3 checkpoints were not re-encoded. The frozen-direction analysis is v4 only.
"""
    (PKG / "RESULTS.md").write_text(text)
    print(f"wrote {PKG / 'RESULTS.md'}")


def main() -> None:
    for path in (S1, S2, S3, S4, V4, S5):
        _need(path)
    _copy_tree()
    audit = _leakage()
    print("\n".join(audit[:8]))
    _report()


if __name__ == "__main__":
    main()
