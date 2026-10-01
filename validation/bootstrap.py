"""Cluster bootstrap over AOIs."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def cluster_bootstrap(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    cluster_col: str,
    cfg: dict[str, Any],
    metric_fn: Callable | None = None,
    iterations: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Resample AOI clusters with replacement; return metric distributions."""
    metric_fn = metric_fn or (lambda yt, yp: compute_metrics(yt, yp, cfg))
    iterations = int(iterations if iterations is not None else cfg["bootstrap"]["iterations"])
    seed = int(seed if seed is not None else cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].astype(str).unique()
    rows = []
    for i in range(iterations):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [df[df[cluster_col].astype(str) == c] for c in draw]
        sample = pd.concat(parts, ignore_index=True)
        m = metric_fn(sample[y_true_col].to_numpy(), sample[y_pred_col].to_numpy())
        m["iteration"] = i
        rows.append(m)
    return pd.DataFrame(rows)


def summarize_bootstrap(boot: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    metric_cols = [c for c in boot.columns if c != "iteration"]
    rows = []
    for col in metric_cols:
        vals = boot[col].astype(float)
        rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": float(np.nanpercentile(vals, lo)),
                "ci_high": float(np.nanpercentile(vals, hi)),
            }
        )
    return pd.DataFrame(rows)
