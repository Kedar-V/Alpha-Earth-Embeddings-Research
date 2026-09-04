#!/usr/bin/env python3
"""Generate notebooks/05_us_loro_building_type_baseline.ipynb.

Protocol B: locked leave-one-region-out + macro-F1 on the US
sota_building_type dataset (AlphaEarth-only baseline; not a modality-matched SOTA claim).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "05_us_loro_building_type_baseline.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    cells: list[nbf.NotebookNode] = [
        md(
            """
# 05 — US AlphaEarth baseline (locked LORO + macro-F1)

**Protocol B.** Evaluate **AlphaEarth-only** building-type classification on the US dataset from
notebook 04 under a **locked evaluation protocol**. This is a lower-modality baseline on
high-precision OSM/Overture labels — **not** a modality-matched comparison to multi-modal
SOTA (street-view / VHR / HeteroGNN ~86–96% accuracy).

**Primary metric:** macro-F1 under **leave-one-region-out (LORO)**.  
**Secondary:** pooled stratified holdout (optimistic upper bound); per-class F1; confusion matrix.

**Data.** `outputs/sota_building_type/dataset.parquet` (10 metros × 10 labels × 64-D centroid embeddings).

**Models.** Logistic regression and HistGradientBoosting on standardized `A00…A63`.
"""
        ),
        md(
            """
## Claim language (locked)

| Allowed | Not allowed |
|---|---|
| “AlphaEarth + simple classifier on OSM high-precision 10-class GT” | “Matches / beats SOTA 86–96%” |
| LORO macro-F1 / accuracy / per-class F1 | Comparing modalities as if equal |
| Within-dataset ablations later (+SVI/+VHR) | Citing HeteroGNN numbers as peer scores without matched cities/splits/modalities |

Reference SOTA for later Protocol A only: Liang et al., ISPRS 2026 (HeteroGNN) —
Amsterdam / Washington D.C. / Berlin; mean F1 0.70–0.73.
"""
        ),
        code(
            """
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_rows", 40)
pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

ROOT = Path(".").resolve()
if not (ROOT / "requirements.txt").exists() and (ROOT.parent / "requirements.txt").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "outputs" / "sota_building_type" / "dataset.parquet"
OUT_DIR = ROOT / "outputs" / "us_loro_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = [f"A{i:02d}" for i in range(64)]
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2

LABEL_ORDER = [
    "single_family",
    "multi_family",
    "retail",
    "hotel",
    "office",
    "industrial",
    "education",
    "medical",
    "religious",
    "civic",
]

print("ROOT:", ROOT)
print("Data:", DATA_PATH)
print("Outputs:", OUT_DIR)
"""
        ),
        md(
            """
## 1. Load locked dataset

Require full 10-class coverage, no null embedding bands, and unique building ids.
"""
        ),
        code(
            """
assert DATA_PATH.exists(), f"Missing {DATA_PATH} — run notebook 04 first."

df = pd.read_parquet(DATA_PATH)
df["id"] = df["id"].astype(str)
df["label"] = pd.Categorical(df["label"], categories=LABEL_ORDER, ordered=True)

assert df["id"].is_unique
assert df[BANDS].notna().all().all()
assert set(df["label"].dropna().unique()) == set(LABEL_ORDER)
assert df.groupby("region")["label"].nunique().min() == len(LABEL_ORDER)

# Fixed integer codes matching LABEL_ORDER (do NOT use LabelEncoder — it sorts alphabetically).
label_to_y = {lab: i for i, lab in enumerate(LABEL_ORDER)}
df["y"] = df["label"].astype(str).map(label_to_y).astype(int)
assert df["y"].between(0, len(LABEL_ORDER) - 1).all()

X = df[BANDS].to_numpy(dtype=np.float64)
y = df["y"].to_numpy(dtype=int)
regions = df["region"].to_numpy()
region_list = sorted(df["region"].unique())

print(f"n={len(df):,} · regions={len(region_list)} · classes={len(LABEL_ORDER)}")
print(df["label"].value_counts().reindex(LABEL_ORDER))
print(df.groupby("region").size().rename("n"))
"""
        ),
        md(
            """
## 2. Locked split protocol

1. **Pooled stratified holdout** — 80/20 by `label` (same seed always). Written to
   `split_pooled ∈ {train, test}`.
2. **Leave-one-region-out** — for each metro \(R\): train on all \(\\neq R\), test on \(R\).
   Primary reporting metric is **macro-F1** on each holdout (and the mean across regions).

Random `split_seed` from notebook 04 is **not** used as an evaluation split.
"""
        ),
        code(
            """
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(
    idx,
    test_size=HOLDOUT_FRACTION,
    stratify=y,
    random_state=RANDOM_STATE,
)

split_pooled = np.full(len(df), "train", dtype=object)
split_pooled[idx_te] = "test"
df["split_pooled"] = split_pooled

# Persist locked assignments for later ablations (+SVI / +VHR) on the same rows
splits_path = OUT_DIR / "locked_splits.parquet"
df[["id", "region", "label", "y", "split_pooled"]].to_parquet(splits_path, index=False)
print("Wrote", splits_path)
print(df["split_pooled"].value_counts())
print("Test label mix:\\n", df.loc[df["split_pooled"] == "test", "label"].value_counts().reindex(LABEL_ORDER))
"""
        ),
        md(
            """
## 3. Model zoo (AlphaEarth-only)

Both models use `class_weight` / imbalance-aware defaults where available.
Features are always the same 64-D annual centroid embeddings.
"""
        ),
        code(
            """
def make_lr() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_hgb() -> HistGradientBoostingClassifier:
    # HGB has no class_weight; use balanced sample weights at fit time.
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=RANDOM_STATE,
    )


def balanced_sample_weight(y_fit: np.ndarray) -> np.ndarray:
    _, counts = np.unique(y_fit, return_counts=True)
    n = len(y_fit)
    k = len(counts)
    inv = {c: n / (k * cnt) for c, cnt in zip(*np.unique(y_fit, return_counts=True))}
    return np.array([inv[yi] for yi in y_fit], dtype=np.float64)


def fit_predict(model_name: str, X_tr, y_tr, X_te):
    if model_name == "logreg":
        model = make_lr()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        return model, pred
    if model_name == "hgb":
        model = make_hgb()
        model.fit(X_tr, y_tr, sample_weight=balanced_sample_weight(y_tr))
        pred = model.predict(X_te)
        return model, pred
    raise ValueError(model_name)


MODELS = ["logreg", "hgb"]
print("Models:", MODELS)
"""
        ),
        md(
            """
## 4. Pooled stratified holdout (secondary)
"""
        ),
        code(
            """
pooled_rows = []
pooled_preds = {}

for name in MODELS:
    _, pred = fit_predict(name, X[idx_tr], y[idx_tr], X[idx_te])
    pooled_preds[name] = pred
    pooled_rows.append(
        {
            "protocol": "pooled_holdout",
            "model": name,
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
            "accuracy": float(accuracy_score(y[idx_te], pred)),
            "f1_macro": float(f1_score(y[idx_te], pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y[idx_te], pred, average="weighted", zero_division=0)),
        }
    )
    print("\\n=== Pooled ·", name, "===")
    print(
        classification_report(
            y[idx_te],
            pred,
            labels=list(range(len(LABEL_ORDER))),
            target_names=LABEL_ORDER,
            digits=3,
            zero_division=0,
        )
    )

pooled = pd.DataFrame(pooled_rows)
pooled
"""
        ),
        md(
            """
## 5. Leave-one-region-out (primary)

For each holdout metro, train on the other nine. Aggregate **mean ± std macro-F1** across
regions. Also store out-of-fold predictions for a pooled LORO confusion matrix.
"""
        ),
        code(
            """
loro_rows = []
oof = {name: np.full(len(df), -1, dtype=int) for name in MODELS}

for holdout in region_list:
    mask_te = regions == holdout
    mask_tr = ~mask_te
    n_te = int(mask_te.sum())
    n_tr = int(mask_tr.sum())
    # Every region has all 10 labels in this dataset; still guard.
    if n_te < 50 or n_tr < 200:
        print(f"Skip {holdout}: n_te={n_te} n_tr={n_tr}")
        continue
    if len(np.unique(y[mask_te])) < len(LABEL_ORDER):
        print(f"Warn {holdout}: missing classes in test")

    for name in MODELS:
        _, pred = fit_predict(name, X[mask_tr], y[mask_tr], X[mask_te])
        oof[name][mask_te] = pred
        loro_rows.append(
            {
                "protocol": "loro",
                "model": name,
                "region": holdout,
                "n_train": n_tr,
                "n_test": n_te,
                "accuracy": float(accuracy_score(y[mask_te], pred)),
                "f1_macro": float(f1_score(y[mask_te], pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y[mask_te], pred, average="weighted", zero_division=0)),
            }
        )
    print(f"LORO done · {holdout} · n_test={n_te}")

loro = pd.DataFrame(loro_rows)
loro_path = OUT_DIR / "loro_metrics.csv"
loro.to_csv(loro_path, index=False)
print("Wrote", loro_path)

summary = (
    loro.groupby("model")
    .agg(
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_weighted_mean=("f1_weighted", "mean"),
    )
    .reset_index()
)
summary_path = OUT_DIR / "loro_summary.csv"
summary.to_csv(summary_path, index=False)
print("Wrote", summary_path)
print("\\nPrimary result — LORO macro-F1 (mean ± std):")
for _, r in summary.iterrows():
    print(
        f"  {r['model']:8s}  macro-F1 {r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}"
        f"  · acc {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}"
    )
summary
"""
        ),
        code(
            """
# Per-class F1 from LORO out-of-fold predictions (concatenated holdouts)
per_class_rows = []
for name in MODELS:
    pred = oof[name]
    assert (pred >= 0).all(), f"Incomplete OOF for {name}"
    f1s = f1_score(y, pred, average=None, labels=list(range(len(LABEL_ORDER))), zero_division=0)
    for lab, f1 in zip(LABEL_ORDER, f1s):
        per_class_rows.append({"model": name, "label": lab, "f1": float(f1)})
    print("\\n=== LORO OOF ·", name, "===")
    print(
        classification_report(
            y,
            pred,
            labels=list(range(len(LABEL_ORDER))),
            target_names=LABEL_ORDER,
            digits=3,
            zero_division=0,
        )
    )

per_class = pd.DataFrame(per_class_rows)
per_class_path = OUT_DIR / "loro_per_class_f1.csv"
per_class.to_csv(per_class_path, index=False)
print("Wrote", per_class_path)
per_class.pivot(index="label", columns="model", values="f1").reindex(LABEL_ORDER)
"""
        ),
        md(
            """
## 6. Figures
"""
        ),
        code(
            """
# LORO macro-F1 by region (both models)
fig, ax = plt.subplots(figsize=(10, 4.8))
width = 0.38
x = np.arange(len(region_list))
for i, name in enumerate(MODELS):
    sub = loro[loro["model"] == name].set_index("region").reindex(region_list)
    ax.bar(x + (i - 0.5) * width, sub["f1_macro"], width=width, label=name)
ax.set_xticks(x)
ax.set_xticklabels(region_list, rotation=35, ha="right")
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1)
ax.set_title("Leave-one-region-out macro-F1 (AlphaEarth-only)")
ax.legend()
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
fig.tight_layout()
bar_path = OUT_DIR / "loro_macro_f1_by_region.png"
fig.savefig(bar_path, dpi=150)
print("Saved", bar_path)
plt.show()
"""
        ),
        md(
            """
### Regional variance

How stable is transfer across metros? Rank regions, summarize spread (mean / std / CV / range),
and plot macro-F1 + accuracy jointly.
"""
        ),
        code(
            r'''
# --- Regional variance table ---
region_var_rows = []
for name in MODELS:
    sub = loro[loro["model"] == name].copy()
    mu = sub["f1_macro"].mean()
    sd = sub["f1_macro"].std(ddof=1)
    region_var_rows.append(
        {
            "model": name,
            "n_regions": int(len(sub)),
            "f1_macro_mean": float(mu),
            "f1_macro_std": float(sd),
            "f1_macro_cv": float(sd / mu) if mu > 0 else np.nan,
            "f1_macro_min": float(sub["f1_macro"].min()),
            "f1_macro_max": float(sub["f1_macro"].max()),
            "f1_macro_range": float(sub["f1_macro"].max() - sub["f1_macro"].min()),
            "accuracy_mean": float(sub["accuracy"].mean()),
            "accuracy_std": float(sub["accuracy"].std(ddof=1)),
            "best_region": sub.loc[sub["f1_macro"].idxmax(), "region"],
            "worst_region": sub.loc[sub["f1_macro"].idxmin(), "region"],
        }
    )

region_var = pd.DataFrame(region_var_rows)
region_var_path = OUT_DIR / "loro_region_variance.csv"
region_var.to_csv(region_var_path, index=False)
print("Wrote", region_var_path)
print(region_var.to_string(index=False))

# Ranked region table for best model
best_model = region_var.sort_values("f1_macro_mean", ascending=False).iloc[0]["model"]
ranked = (
    loro[loro["model"] == best_model]
    .sort_values("f1_macro", ascending=False)
    .reset_index(drop=True)
)
ranked["rank"] = np.arange(1, len(ranked) + 1)
ranked_path = OUT_DIR / f"loro_region_rank_{best_model}.csv"
ranked.to_csv(ranked_path, index=False)
print("\nRanked regions ·", best_model)
print(ranked[["rank", "region", "n_test", "accuracy", "f1_macro", "f1_weighted"]].to_string(index=False))
'''
        ),
        code(
            r'''
# --- Plot A: ranked horizontal bars + mean/±1 std band ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)

for ax, name, color in zip(axes, MODELS, ["#4C78A8", "#F58518"]):
    sub = loro[loro["model"] == name].sort_values("f1_macro", ascending=True)
    mu = sub["f1_macro"].mean()
    sd = sub["f1_macro"].std(ddof=1)
    ax.barh(sub["region"], sub["f1_macro"], color=color, edgecolor="k", linewidth=0.3)
    ax.axvline(mu, color="k", lw=1.5, label=f"mean={mu:.3f}")
    ax.axvspan(mu - sd, mu + sd, color="k", alpha=0.12, label=f"±1 std ({sd:.3f})")
    ax.set_xlim(0, max(0.45, float(sub["f1_macro"].max()) + 0.05))
    ax.set_xlabel("Macro-F1")
    ax.set_title(f"LORO by region · {name}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)

fig.suptitle("Regional variance of leave-one-region-out macro-F1", y=1.02)
fig.tight_layout()
rank_path = OUT_DIR / "loro_region_variance_ranked.png"
fig.savefig(rank_path, dpi=150, bbox_inches="tight")
print("Saved", rank_path)
plt.show()
'''
        ),
        code(
            r'''
# --- Plot B: scatter accuracy vs macro-F1; heatmap of metrics ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

ax = axes[0]
for name, marker, color in zip(MODELS, ["o", "s"], ["#4C78A8", "#F58518"]):
    sub = loro[loro["model"] == name]
    ax.scatter(
        sub["accuracy"],
        sub["f1_macro"],
        s=70,
        marker=marker,
        c=color,
        edgecolors="k",
        linewidths=0.4,
        label=name,
        zorder=3,
    )
    for _, r in sub.iterrows():
        ax.annotate(
            r["region"].replace("_", "\n"),
            (r["accuracy"], r["f1_macro"]),
            fontsize=6.5,
            textcoords="offset points",
            xytext=(4, 3),
        )
ax.set_xlabel("Accuracy")
ax.set_ylabel("Macro-F1")
ax.set_title("Region performance cloud")
ax.grid(True, linestyle=":", alpha=0.4)
ax.legend()
ax.set_xlim(0, max(0.5, float(loro["accuracy"].max()) + 0.05))
ax.set_ylim(0, max(0.4, float(loro["f1_macro"].max()) + 0.05))

# Heatmap: regions × (model_metric)
ax = axes[1]
heat = (
    loro.pivot(index="region", columns="model", values="f1_macro")
    .reindex(index=sorted(region_list))
)
# order rows by best-model F1
heat = heat.loc[heat[best_model].sort_values(ascending=False).index]
im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=0.4)
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_title("Macro-F1 heatmap (regions × model)")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.to_numpy()[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1")

fig.tight_layout()
cloud_path = OUT_DIR / "loro_region_variance_cloud_heatmap.png"
fig.savefig(cloud_path, dpi=150)
print("Saved", cloud_path)
plt.show()
'''
        ),
        code(
            r'''
# --- Plot C: paired model comparison + deviation from mean ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

wide = loro.pivot(index="region", columns="model", values="f1_macro").reindex(region_list)
ax = axes[0]
ax.scatter(wide["logreg"], wide["hgb"], s=80, c="#54A24B", edgecolors="k", linewidths=0.4, zorder=3)
lim_hi = max(0.35, float(wide.max().max()) + 0.05)
ax.plot([0, lim_hi], [0, lim_hi], "k--", lw=1, alpha=0.6, label="y = x")
for reg, row in wide.iterrows():
    ax.annotate(reg, (row["logreg"], row["hgb"]), fontsize=7, textcoords="offset points", xytext=(4, 3))
ax.set_xlabel("LogReg macro-F1")
ax.set_ylabel("HGB macro-F1")
ax.set_title("Per-region model agreement")
ax.set_xlim(0, lim_hi)
ax.set_ylim(0, lim_hi)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)
ax.legend(loc="upper left")

ax = axes[1]
sub = loro[loro["model"] == best_model].copy()
mu = sub["f1_macro"].mean()
sub = sub.sort_values("f1_macro")
colors = ["#E45756" if v < mu else "#54A24B" for v in sub["f1_macro"]]
ax.barh(sub["region"], sub["f1_macro"] - mu, color=colors, edgecolor="k", linewidth=0.3)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel(f"Macro-F1 − mean ({mu:.3f})")
ax.set_title(f"Deviation from mean · {best_model}")
ax.grid(True, axis="x", linestyle=":", alpha=0.4)

fig.tight_layout()
dev_path = OUT_DIR / "loro_region_variance_deviation.png"
fig.savefig(dev_path, dpi=150)
print("Saved", dev_path)
plt.show()
'''
        ),
        code(
            """
# US bubble map for best LORO model by mean macro-F1
best_model = summary.sort_values("f1_macro_mean", ascending=False).iloc[0]["model"]
map_df = loro[loro["model"] == best_model].copy()

# Region centroids from notebook 04 AOIs (approximate downtown centers)
REGION_XY = {
    "Boston_NE": (-71.075, 42.35),
    "NYC_NE": (-73.975, 40.75),
    "Chicago_MW": (-87.65, 41.875),
    "Atlanta_SE": (-84.375, 33.775),
    "Houston_S": (-95.375, 29.775),
    "Denver_W": (-104.975, 39.74),
    "Phoenix_SW": (-112.05, 33.475),
    "Seattle_NW": (-122.325, 47.615),
    "LA_W": (-118.275, 34.06),
    "SF_W": (-122.44, 37.77),
}
map_df["lon"] = map_df["region"].map(lambda r: REGION_XY[r][0])
map_df["lat"] = map_df["region"].map(lambda r: REGION_XY[r][1])

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(-128, -66)
ax.set_ylim(23, 50)
ax.set_aspect(1.3)
ax.set_facecolor("#f4f7fb")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"LORO macro-F1 · model={best_model} (AlphaEarth-only baseline)")

sc = ax.scatter(
    map_df["lon"],
    map_df["lat"],
    c=map_df["f1_macro"],
    s=map_df["n_test"] * 0.35,
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    edgecolors="k",
    linewidths=0.6,
    zorder=3,
)
cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Macro-F1")

for _, r in map_df.iterrows():
    ax.annotate(
        f"{r['region']}\\n{r['f1_macro']:.2f}",
        (r["lon"], r["lat"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8,
        ha="left",
    )

fig.tight_layout()
map_path = OUT_DIR / "loro_us_macro_f1_map.png"
fig.savefig(map_path, dpi=150)
print("Saved", map_path)
plt.show()
"""
        ),
        code(
            """
# Confusion matrix from LORO OOF (best model)
pred = oof[best_model]
fig, ax = plt.subplots(figsize=(8.5, 7.5))
ConfusionMatrixDisplay.from_predictions(
    y,
    pred,
    labels=list(range(len(LABEL_ORDER))),
    display_labels=LABEL_ORDER,
    cmap="Blues",
    colorbar=True,
    xticks_rotation=45,
    ax=ax,
    values_format="d",
)
ax.set_title(f"LORO out-of-fold confusion · {best_model}")
fig.tight_layout()
cm_path = OUT_DIR / f"loro_confusion_{best_model}.png"
fig.savefig(cm_path, dpi=150)
print("Saved", cm_path)
plt.show()
"""
        ),
        code(
            """
# Per-class F1 bars
pivot = per_class.pivot(index="label", columns="model", values="f1").reindex(LABEL_ORDER)
fig, ax = plt.subplots(figsize=(10, 4.5))
pivot.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("F1")
ax.set_ylim(0, 1)
ax.set_title("LORO OOF per-class F1")
ax.legend(title="model")
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
plt.xticks(rotation=35, ha="right")
fig.tight_layout()
pc_path = OUT_DIR / "loro_per_class_f1.png"
fig.savefig(pc_path, dpi=150)
print("Saved", pc_path)
plt.show()
"""
        ),
        md(
            """
## 7. Snapshot

Re-state: these numbers are an **AlphaEarth-only US baseline** under locked LORO.
Do not juxtapose them with HeteroGNN 86–96% as if modalities and geographies matched.
"""
        ),
        code(
            """
metrics_all = pd.concat([pooled, loro], ignore_index=True)
metrics_path = OUT_DIR / "all_metrics.csv"
metrics_all.to_csv(metrics_path, index=False)

snapshot = {
    "protocol": "B_us_alphaearth_only_loro",
    "claim": (
        "AlphaEarth + simple classifier on OSM/Overture high-precision 10-class GT; "
        "not modality-matched to multi-modal SOTA."
    ),
    "data": str(DATA_PATH.relative_to(ROOT)),
    "n_buildings": int(len(df)),
    "n_regions": int(len(region_list)),
    "labels": LABEL_ORDER,
    "regions": region_list,
    "primary_metric": "loro_macro_f1_mean",
    "models": MODELS,
    "best_model_by_loro_macro_f1": best_model,
    "pooled_holdout": pooled.to_dict(orient="records"),
    "loro_summary": summary.to_dict(orient="records"),
    "loro_by_region": loro.to_dict(orient="records"),
    "random_state": RANDOM_STATE,
    "holdout_fraction": HOLDOUT_FRACTION,
    "outputs": sorted(p.name for p in OUT_DIR.iterdir()),
}
snap_path = OUT_DIR / "baseline_snapshot.json"
snap_path.write_text(json.dumps(snapshot, indent=2))
print("Wrote", snap_path)
print(
    json.dumps(
        {
            "best_model": best_model,
            "loro_summary": snapshot["loro_summary"],
            "pooled_holdout": snapshot["pooled_holdout"],
        },
        indent=2,
    )
)
"""
        ),
    ]

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "AlphaEarth (.venv)",
                "language": "python",
                "name": "alphaearth",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
