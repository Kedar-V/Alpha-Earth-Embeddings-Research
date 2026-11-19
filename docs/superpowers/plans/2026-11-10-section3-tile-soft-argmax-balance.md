# Section 3 Tile Soft-Argmax 1:1 Balance — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Section 3’s 100 m panel on soft-argmax 1:1 balanced tiles from the unbalanced soft-covered building universe; leave the building panel unchanged.

**Architecture:** Extend `eval_week6_overture_polygon_mae_100m.py` to (1) load unbalanced soft∩Ridge∩PLS∩BUILT-S buildings, (2) pool to 100 m tiles, (3) soft-argmax label + per-metro 1:1 undersample, (4) emit metrics. Rebuild Section 3 board 100 m blocks and reinject the canvas. Building metrics stay from `20261110_041557`.

**Tech Stack:** Python, pandas/pyarrow, existing week6 eval + submit-job pattern, canvas gzip/base64 inject.

---

### Task 1: Unbalanced soft-covered building universe

**Files:**
- Modify: `scripts/eval_week6_overture_polygon_mae_100m.py` (or produce via `eval_week6_overture_polygon_mae.py --no-balanced`)
- Artifacts under `outputs/week6_overture_polygon_mae*/`

- [ ] Ensure a parquet of Track-A buildings with soft∩Ridge∩PLS∩BUILT-S **without** `y_res` 1:1
- [ ] Point the 100 m script at that universe (not balanced `shared_soft` alone)

### Task 2: Soft-argmax tile balance + score

**Files:**
- Modify: `scripts/eval_week6_overture_polygon_mae_100m.py`

- [ ] After min≥20 filter: `tile_y = (soft_gt_frac_res >= soft_gt_frac_nres).astype(int)`
- [ ] Per-metro undersample to 1:1 (seed 42); write `tiles_100m.parquet` = balanced, keep pre-balance as `tiles_100m_all.parquet` or `tiles_100m_prebalance.parquet`
- [ ] Score + write `section3_100m_metrics.json` with balance metadata

### Task 3: Rebuild Section 3 board 100 m + canvas

**Files:**
- Modify: board HTML under 100 m run artifacts
- Modify: `notebooks/ghsl_built_s_vs_ae/AlphaEarth Weeks 1–5 Canvas.html`
- Modify: `notebooks/week6/01_ae_vs_ghsl_overture_polygons.ipynb` (note dual balance rules)

- [ ] Keep building panel numbers from balanced building run
- [ ] Replace 100 m gap bars / per-metro with new metrics
- [ ] Copy: buildings = y_res 1:1; tiles = soft-argmax 1:1
- [ ] Reinject canvas; smoke-check live §3

### Task 4: Submit & verify

- [ ] Launch via submit pattern / nohup with progress.json
- [ ] Confirm per-metro tile RES/NRES counts are 1:1
- [ ] Confirm building panel unchanged vs prior board
