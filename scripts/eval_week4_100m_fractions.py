#!/usr/bin/env python3
"""Evaluate RES/NRES fraction MAE at 100 m (wrapper).

Requires datasets/week4_100m_tiles/ from build_week4_100m_tiles.py.
Rebuilds tiles if missing, then scores.

Usage:
  .venv/bin/python -u scripts/eval_week4_100m_fractions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_week4_100m_tiles import OUT_DATA, run_build, run_eval  # noqa: E402


def main() -> None:
    if not (OUT_DATA / "tiles.parquet").exists():
        print("Tiles missing — building first …", flush=True)
        run_build()
    run_eval()


if __name__ == "__main__":
    main()
