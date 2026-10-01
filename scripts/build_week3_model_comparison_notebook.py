#!/usr/bin/env python3
"""DEPRECATED generator for notebooks/week3/04_model_comparison.ipynb.

Notebook 04 is now the Kyle Week-3 briefing (hand-authored story + embedded
figures). Do not regenerate it from this script.

To refresh the ladder charts only:
  .venv/bin/python -c "exec(open('scripts/compare_week3_four_models.py').read())"  # existing four-model figs
  # ladder figs: see outputs/week3_model_comparison/figures/08_*.png 09_*.png
"""
from __future__ import annotations

import sys

print(
    "Refusing to overwrite notebooks/week3/04_model_comparison.ipynb.\n"
    "That notebook is the Kyle Week-3 briefing (story + embedded plots).\n"
    "Edit the .ipynb directly, or ask the agent to update it.",
    file=sys.stderr,
)
raise SystemExit(1)
