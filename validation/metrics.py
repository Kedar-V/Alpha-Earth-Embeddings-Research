"""Classification metrics for RES / NRES (identical for model and GHSL)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _binary_arrays(y_true: np.ndarray, y_pred: np.ndarray, res: str = "RES") -> tuple[np.ndarray, np.ndarray]:
    yt = (np.asarray(y_true).astype(str) == res).astype(int)
    yp = (np.asarray(y_pred).astype(str) == res).astype(int)
    return yt, yp


def iou_per_class(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> float:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    inter = np.sum((yt == positive) & (yp == positive))
    union = np.sum((yt == positive) | (yp == positive))
    return float(inter / union) if union else float("nan")


def compute_metrics(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    res = (cfg or {}).get("labels", {}).get("residential", "RES")
    yt, yp = _binary_arrays(y_true, y_pred, res=res)
    if len(yt) == 0:
        return {k: float("nan") for k in [
            "n", "accuracy", "balanced_accuracy", "macro_f1", "residential_f1",
            "non_residential_f1", "residential_iou", "non_residential_iou", "mean_iou",
            "precision", "recall", "residential_precision", "residential_recall",
            "non_residential_precision", "non_residential_recall",
        ]}

    res_f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    nres_f1 = f1_score(yt, yp, pos_label=0, zero_division=0)
    res_iou = iou_per_class(yt, yp, positive=1)
    nres_iou = iou_per_class(yt, yp, positive=0)
    return {
        "n": float(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "residential_f1": float(res_f1),
        "non_residential_f1": float(nres_f1),
        "residential_iou": float(res_iou),
        "non_residential_iou": float(nres_iou),
        "mean_iou": float(np.nanmean([res_iou, nres_iou])),
        "precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "residential_precision": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "residential_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "non_residential_precision": float(precision_score(yt, yp, pos_label=0, zero_division=0)),
        "non_residential_recall": float(recall_score(yt, yp, pos_label=0, zero_division=0)),
    }


def confusion(y_true, y_pred, labels=("RES", "NRES")) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(cm, index=[f"gt_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def coverage_and_selective(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    covered_mask: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """End-to-end metrics treating uncovered predictions as abstentions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    covered = np.asarray(covered_mask).astype(bool)
    n = len(y_true)
    n_cov = int(covered.sum())
    coverage = float(n_cov / n) if n else float("nan")
    if n_cov:
        selective = compute_metrics(y_true[covered], y_pred[covered], cfg)
        correct = (y_true[covered].astype(str) == y_pred[covered].astype(str)).sum()
        effective_acc = float(correct / n)  # abstentions count as errors
    else:
        selective = compute_metrics([], [], cfg)
        effective_acc = 0.0
    return {
        "n_total": float(n),
        "n_covered": float(n_cov),
        "coverage": coverage,
        "selective_macro_f1": selective["macro_f1"],
        "selective_residential_f1": selective["residential_f1"],
        "selective_precision": selective["precision"],
        "selective_recall": selective["recall"],
        "selective_accuracy": selective["accuracy"],
        "selective_balanced_accuracy": selective["balanced_accuracy"],
        "effective_accuracy": effective_acc,
    }
