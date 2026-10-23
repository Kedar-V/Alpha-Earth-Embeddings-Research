"""GHS-SMOD Degree of Urbanisation codes (R2023A, L2).

Native codes match JRC/GHSL/P2023A/GHS_SMOD_V2-0 `smod_code`.
`dou` is the study rollup used since week 2 (urban / peri-urban / rural).
Water and no-data stay on `smod_label` and do not get a `dou` value.
"""

from __future__ import annotations

# Official L2 settlement classes.
SMOD_LABEL = {
    -200: "nodata",
    10: "water",
    11: "very_low_density_rural",
    12: "low_density_rural",
    13: "rural_cluster",
    21: "suburban_peri_urban",
    22: "semi_dense_urban_cluster",
    23: "dense_urban_cluster",
    30: "urban_centre",
}

# Week-2 study rollup. Not the official UN L1 aggregation.
DOU_ROLLUP = {
    30: "urban",
    23: "urban",
    22: "urban",
    21: "peri_urban",
    13: "rural",
    12: "rural",
    11: "rural",
}

SMOD_LABEL_NAMES = tuple(SMOD_LABEL.values())
DOU_NAMES = ("urban", "peri_urban", "rural")


def smod_label(code: int) -> str:
    return SMOD_LABEL.get(int(code), "unclassified")


def dou_rollup(code: int) -> str | None:
    return DOU_ROLLUP.get(int(code))
