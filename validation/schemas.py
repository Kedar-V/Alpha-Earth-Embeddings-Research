"""Column contracts for validation inputs."""

from __future__ import annotations

GT_REQUIRED = ("building_id", "aoi_id", "geometry", "label")
GT_OPTIONAL = ("confidence", "source")

PRED_REQUIRED = ("building_id", "prediction")
PRED_OPTIONAL = ("p_res", "p_nres")

BINARY_LABELS = ("RES", "NRES")
EXCLUDED_LABELS = ("MIXED", "UNKNOWN")
PRED_LABELS = ("RES", "NRES")

GHSL_TO_LABEL = {
    1: "RES",
    2: "NRES",
    0: "NONBUILT",
    255: "NODATA",
}
