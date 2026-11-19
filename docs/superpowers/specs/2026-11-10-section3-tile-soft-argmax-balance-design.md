# Section 3 — 100 m tile soft-argmax 1:1 balance

Date: 2026-11-10  
Status: approved (design conversation); awaiting spec file review before implementation plan

## Goal

Section 3 keeps the **building** soft-GT panel as today (buildings balanced 1:1 on `y_res`). The **100 m** panel must **not** inherit that building balance. Instead, form tiles from the unbalanced soft-covered building universe, label each tile by soft-fraction argmax, then undersample to 1:1 RES/NRES per metro.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Tile class for balance | Soft argmax: RES iff `soft_gt_frac_res ≥ soft_gt_frac_nres`, else NRES |
| Balance grain | Per-metro 1:1 undersample (seed 42), same spirit as Section 2 cell balance |
| Building panel | Unchanged — current building-balanced soft GT |
| Hard GT | Out of scope for Section 3 display |
| Section 2 | Unchanged |

## Protocol

### Building panel (unchanged)

- Cohort: Track-A buildings, per-metro 1:1 on `y_res` (seed 42).
- GT: mean week-4 `frac_*_full` on inside-polygon cells.
- Models: GHS-BUILT-S (building-zonal), AE Ridge OOF, AE PLS OOF.
- Artifact source: existing week-6 balanced run  
  `outputs/week6_overture_polygon_mae/20261110_041557/`.

### 100 m panel (new)

1. **Building universe (no 1:1)**  
   All Track-A buildings with soft ∩ Ridge ∩ PLS ∩ BUILT-S coverage.  
   Do **not** use the building-balanced `shared_soft` set from the balanced week-6 parquet as the sole universe (that file is already 1:1 on `y_res`).  
   Obtain via week-6 polygon eval with `--balanced false` (new run) or an equivalent join that yields the same coverage flags on the full Track-A set.

2. **Cells**  
   Inside-polygon AE cells for those buildings; join soft GT, Ridge OOF, PLS OOF; broadcast building-zonal BUILT-S onto cells (polygon path, not AE-centre sampling).

3. **Tiles**  
   Mean-pool into per-metro UTM 100 m tiles (`tile_id` from week4 map, else `floor(utm/100)`). Keep tiles with `n_cells ≥ 20`.

4. **Balance**  
   `tile_y = 1` if `soft_gt_frac_res ≥ soft_gt_frac_nres`, else `0`.  
   Per `region`, undersample majority class to minority count (seed 42). Persist kept `tile_id`s.

5. **Score**  
   Soft MAE on balanced tiles for BUILT-S / Ridge / PLS (pooled + per-metro). Winner = lowest mean(res, nres).

6. **Canvas**  
   Rebuild Section 3 **100 m** gap bars + per-metro dumbbells/table only.  
   Copy must state: buildings = `y_res` 1:1; tiles = soft-argmax 1:1.  
   Building block text/numbers stay from the balanced building run.

## Why a separate building universe for tiles

`per_building_with_pls.parquet` in the balanced week-6 run only contains buildings after `y_res` 1:1. Pooling those cells into tiles still carries building balance. Soft-argmax tile 1:1 on top of that would double-filter and shrink the cohort for the wrong reason. Tiles must start from the unbalanced soft-covered set.

## Deliverables

- New (or extended) 100 m eval run under `outputs/week6_overture_polygon_mae_100m/<timestamp>/` with:
  - `tiles_100m_all.parquet` (min≥20, pre-balance)
  - `tiles_100m.parquet` (soft-argmax 1:1 balanced)
  - `section3_100m_metrics.json`, MAE CSVs, progress/summary logs
- Unbalanced soft-covered building artifact if produced by a `--balanced false` week-6 run
- Updated `section3_board.html` + reinjected canvas Section 3 (100 m panels + clarifying copy)
- Short notebook note on the dual balance rules

## Non-goals

- Replacing or dropping the building soft panel
- Hard-GT Section 3 panels
- Changing Section 2 cell/tile protocol
- AE-centre BUILT-S sampling for this path

## Success criteria

- Section 3 100 m `n` and metrics come from soft-argmax 1:1 tiles, not building-balanced inheritance
- Per-metro tile counts are 1:1 RES/NRES (up to odd minority)
- Building panel metrics match the existing balanced building run
- Canvas copy states both balance rules explicitly
