"""Unit tests for validation metrics and GHSL mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from validation.ghsl import map_ghsl_value
from validation.metrics import compute_metrics, coverage_and_selective, iou_per_class
from validation.bootstrap import summarize_bootstrap


CFG = {
    "labels": {"residential": "RES", "non_residential": "NRES", "exclude": ["MIXED", "UNKNOWN"]},
    "ghsl": {
        "residential_value": 1,
        "non_residential_value": 2,
        "nonbuilt_value": 0,
        "nodata_value": 255,
    },
}


def test_map_ghsl_value():
    assert map_ghsl_value(1, CFG) == "RES"
    assert map_ghsl_value(2, CFG) == "NRES"
    assert map_ghsl_value(0, CFG) == "NONBUILT"
    assert map_ghsl_value(255, CFG) == "NODATA"


def test_perfect_metrics():
    y = ["RES", "RES", "NRES", "NRES"]
    m = compute_metrics(y, y, CFG)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mean_iou"] == 1.0


def test_iou():
    yt = np.array([1, 1, 0, 0])
    yp = np.array([1, 0, 0, 0])
    assert abs(iou_per_class(yt, yp, 1) - 0.5) < 1e-9


def test_coverage_abstention():
    y_true = np.array(["RES", "NRES", "RES", "NRES"])
    y_pred = np.array(["RES", "NRES", "NONBUILT", "NODATA"])
    covered = np.array([True, True, False, False])
    out = coverage_and_selective(y_true, y_pred, covered, CFG)
    assert abs(out["coverage"] - 0.5) < 1e-9
    assert abs(out["selective_accuracy"] - 1.0) < 1e-9
    assert abs(out["effective_accuracy"] - 0.5) < 1e-9


def test_summarize_bootstrap():
    boot = pd.DataFrame({"macro_f1": [0.7, 0.8, 0.75], "iteration": [0, 1, 2]})
    s = summarize_bootstrap(boot, 0.95)
    assert "macro_f1" in set(s["metric"])
    row = s[s["metric"] == "macro_f1"].iloc[0]
    assert 0.7 <= row["mean"] <= 0.8
