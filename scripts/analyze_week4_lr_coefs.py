#!/usr/bin/env python3
"""Week 4 LR coefficient audit — which AE axes matter?

Uses *existing* LORO fold checkpoints (no refit):
  1) mean |coef| / sign consistency across folds
  2) inference knockout: zero scaled dims and rescore OOF F1

Usage:
  .venv/bin/python -u scripts/analyze_week4_lr_coefs.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "week4_pixel_balanced_train" / "checkpoints"
BAL = ROOT / "datasets" / "week4_pixel_gt_balanced" / "pixels.parquet"
OUT = ROOT / "outputs" / "week4_lr_coef_audit"
FIG = OUT / "figures"
BANDS = [f"A{i:02d}" for i in range(64)]
WEAK_FRAC_OF_MEDIAN = 0.25
TOP_K = 16
BOTTOM_K = 16


def load_coef_matrix() -> tuple[pd.DataFrame, list[str]]:
    rows, folds = [], []
    paths = sorted(CKPT.glob("fold_*.joblib"))
    for path in tqdm(paths, desc="Load fold coefs", unit="fold"):
        bundle = joblib.load(path)
        region = bundle["region_held_out"]
        lr = bundle["classifier"].named_steps["lr"]
        coef = lr.coef_.ravel().astype(np.float64)
        assert lr.classes_.tolist() == [0, 1], lr.classes_
        assert len(coef) == 64, len(coef)
        rows.append(coef)
        folds.append(region)
    return pd.DataFrame(np.vstack(rows), index=folds, columns=BANDS), folds


def summarize(coefs: pd.DataFrame) -> pd.DataFrame:
    abs_m = coefs.abs()
    mean_abs = abs_m.mean(axis=0)
    mean_signed = coefs.mean(axis=0)
    signs = np.sign(coefs.to_numpy())
    mean_sign = np.sign(mean_signed.to_numpy())
    cons = []
    for j in range(64):
        ms = mean_sign[j]
        cons.append(float("nan") if ms == 0 else float((signs[:, j] == ms).mean()))
    med = float(mean_abs.median())
    out = pd.DataFrame(
        {
            "band": BANDS,
            "mean_abs_coef": mean_abs.to_numpy(),
            "std_abs_coef": abs_m.std(axis=0).to_numpy(),
            "mean_signed_coef": mean_signed.to_numpy(),
            "sign_consistency": cons,
            "rank_by_mean_abs": mean_abs.rank(ascending=False, method="min")
            .astype(int)
            .to_numpy(),
            "weak": (mean_abs < WEAK_FRAC_OF_MEDIAN * med).to_numpy(),
        }
    )
    return out.sort_values("mean_abs_coef", ascending=False).reset_index(drop=True)


def plot_importance(summary: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = summary.sort_values("mean_abs_coef", ascending=True)
    colors = ["#B0B0B0" if w else "#4C78A8" for w in d["weak"]]
    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    y = np.arange(len(d))
    ax.barh(y, d["mean_abs_coef"], color=colors, height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["band"].tolist(), fontsize=7)
    ax.set_xlabel("Mean |coef| across LORO folds (per 1 SD of band)")
    n_weak = int(summary["weak"].sum())
    ax.set_title(
        f"AE band importance in W4 LR — {n_weak} axes stay weak "
        f"(<{WEAK_FRAC_OF_MEDIAN:.0%} of median |coef|)",
        fontsize=12,
        loc="left",
        pad=10,
    )
    ax.axvline(
        WEAK_FRAC_OF_MEDIAN * float(summary["mean_abs_coef"].median()),
        color="#D55E00",
        ls="--",
        lw=1.2,
        label=f"weak threshold ({WEAK_FRAC_OF_MEDIAN:.0%}× median)",
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "lr_coef_mean_abs_by_band.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG / 'lr_coef_mean_abs_by_band.png'}")


def plot_heatmap(coefs: pd.DataFrame) -> None:
    order = coefs.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    M = coefs[order].to_numpy()
    vmax = float(np.percentile(np.abs(M), 98))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(coefs.index)))
    ax.set_yticklabels(coefs.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(64))
    ax.set_xticklabels(order, fontsize=6, rotation=90)
    ax.set_title(
        "LORO fold LR coefficients (bands ordered by mean |coef|)",
        fontsize=12,
        loc="left",
        pad=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("coef (log-odds RES per 1 SD)")
    fig.tight_layout()
    fig.savefig(FIG / "lr_coef_heatmap_folds_x_bands.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG / 'lr_coef_heatmap_folds_x_bands.png'}")


def _predict_with_mask(pipe, X: np.ndarray, zero_idx: list[int]) -> np.ndarray:
    """Existing model: scale, optionally zero dims (= mean after StandardScaler), predict."""
    Xs = pipe.named_steps["scaler"].transform(X)
    if zero_idx:
        Xs = np.array(Xs, copy=True)
        Xs[:, zero_idx] = 0.0
    return pipe.named_steps["lr"].predict(Xs)


def run_knockout(summary: pd.DataFrame) -> pd.DataFrame:
    """Inference-only checks on saved fold models — no refit, no GPU needed."""
    print("Knockout on existing checkpoints (inference only) ...", flush=True)
    df = pd.read_parquet(BAL, columns=["region", "y"] + BANDS)
    y = df["y"].to_numpy(dtype=np.int64)
    regions = df["region"].astype(str).to_numpy()

    top = summary.sort_values("mean_abs_coef", ascending=False).head(TOP_K)["band"].tolist()
    bottom = summary.sort_values("mean_abs_coef", ascending=True).head(BOTTOM_K)["band"].tolist()
    weak = summary.loc[summary["weak"], "band"].tolist()

    plans = [
        ("full_64", []),
        (f"zero_weak_{len(weak)}", [BANDS.index(b) for b in weak]),
        (f"zero_bottom_{BOTTOM_K}", [BANDS.index(b) for b in bottom]),
        (f"keep_top_{TOP_K}_only", [i for i in range(64) if BANDS[i] not in set(top)]),
    ]

    rows = []
    t0 = time.time()
    paths = sorted(CKPT.glob("fold_*.joblib"))
    for path in tqdm(paths, desc="Knockout OOF", unit="fold"):
        bundle = joblib.load(path)
        region = bundle["region_held_out"]
        pipe = bundle["classifier"]
        mask = regions == region
        X_te = df.loc[mask, BANDS].to_numpy(dtype=np.float64)
        y_te = y[mask]
        for label, zero_idx in plans:
            pred = _predict_with_mask(pipe, X_te, zero_idx)
            rows.append(
                {
                    "label": label,
                    "region": region,
                    "n_zeroed": int(len(zero_idx)),
                    "n_active": 64 - int(len(zero_idx)),
                    "macro_f1": float(
                        f1_score(y_te, pred, average="macro", zero_division=0)
                    ),
                    "n_test": int(mask.sum()),
                }
            )
    print(f"  knockout done in {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def plot_knockout(abl: pd.DataFrame) -> pd.DataFrame:
    g = abl.groupby("label", as_index=False).agg(
        n_active=("n_active", "first"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    prefer = ["full_64", "zero_weak", "zero_bottom", "keep_top"]

    def key(lab: str) -> int:
        for i, p in enumerate(prefer):
            if lab.startswith(p):
                return i
        return 99

    g = g.assign(_k=g["label"].map(key)).sort_values("_k").drop(columns="_k")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(g))
    ax.bar(
        x,
        g["macro_f1_mean"],
        yerr=g["macro_f1_std"].fillna(0),
        color=["#4C78A8", "#54A24B", "#F58518", "#B0B0B0"][: len(g)],
        width=0.65,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lab}\n({int(n)} active)" for lab, n in zip(g["label"], g["n_active"])],
        fontsize=8,
    )
    ax.set_ylabel("OOF macro-F1 (mean ± std)")
    ax.set_ylim(0, 1.0)
    for i, r in enumerate(g.itertuples()):
        ax.text(i, r.macro_f1_mean + 0.03, f"{r.macro_f1_mean:.3f}", ha="center", fontsize=9)
    ax.set_title(
        "Existing fold models: zero weak axes; silence all but top-16",
        fontsize=11,
        loc="left",
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "lr_coef_knockout_oof_f1.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return g


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    coefs, folds = load_coef_matrix()
    coefs.to_csv(OUT / "lr_coefs_by_fold.csv")
    summary = summarize(coefs)
    summary.to_csv(OUT / "lr_coef_summary.csv", index=False)

    weak_bands = summary.loc[summary["weak"], "band"].tolist()
    top_bands = summary.head(TOP_K)["band"].tolist()
    bottom_bands = summary.sort_values("mean_abs_coef").head(BOTTOM_K)["band"].tolist()
    print(
        f"folds={len(folds)}  weak={len(weak_bands)}: {', '.join(weak_bands) or '(none)'}",
        flush=True,
    )
    print(f"top-{TOP_K}: {', '.join(top_bands)}", flush=True)
    print(f"bottom-{BOTTOM_K}: {', '.join(bottom_bands)}", flush=True)

    plot_importance(summary)
    plot_heatmap(coefs)

    knock = run_knockout(summary)
    knock.to_csv(OUT / "knockout_oof_by_fold.csv", index=False)
    knock_sum = plot_knockout(knock)
    knock_sum.to_csv(OUT / "knockout_oof_summary.csv", index=False)

    payload = {
        "name": "week4_lr_coef_audit",
        "method": "coef ranking + inference knockout on saved fold models (no refit)",
        "n_folds": len(folds),
        "folds": folds,
        "weak_frac_of_median": WEAK_FRAC_OF_MEDIAN,
        "n_weak": len(weak_bands),
        "weak_bands": weak_bands,
        "top_k": TOP_K,
        "top_bands": top_bands,
        "bottom_bands": bottom_bands,
        "median_mean_abs_coef": float(summary["mean_abs_coef"].median()),
        "knockout": knock_sum.to_dict(orient="records"),
        "figures": {
            "importance": str((FIG / "lr_coef_mean_abs_by_band.png").relative_to(ROOT)),
            "heatmap": str((FIG / "lr_coef_heatmap_folds_x_bands.png").relative_to(ROOT)),
            "knockout": str((FIG / "lr_coef_knockout_oof_f1.png").relative_to(ROOT)),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(knock_sum.round(3).to_string(index=False))
    print(f"Wrote {OUT / 'summary.json'} ({payload['elapsed_s']}s)")


if __name__ == "__main__":
    main()
