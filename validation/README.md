# GHSL vs model validation

Publication-oriented comparison of our **RES / NRES** building classifier against **GHS-BUILT-C FUN** (R2023A, 10 m, ESRI:54009).

Scope (per Kyle meeting + validation strategy):

- Binary **residential vs non-residential only** (`MIXED` / `UNKNOWN` excluded, never silently remapped)
- Same GT locations, labels, spatial support, and metrics for both systems
- Leave-one-region-out CNN predictions (Week-2 bbox CNN)
- No bilinear resampling of the categorical GHSL raster

## Quick start

```bash
# 0) Pre-model EDA (what the raster/GT/benchmarks mean)
#    Regenerator: scripts/build_ghsl_validation_eda_notebook.py
#    Open: notebooks/10_ghsl_built_c_validation_eda.ipynb  → outputs/ghsl_eda/

# 1) Build input tables from existing CNN OOF + footprints
.venv/bin/python scripts/prepare_ghsl_validation_inputs.py

# 2) Run building + pixel validation (downloads any missing GHSL tiles)
#    AOIs are evaluated in parallel via ProcessPoolExecutor (see config workers).
.venv/bin/python validation/run_validation.py
```

Outputs land in `outputs/validation/`.

## Inputs

| Path | Role |
|------|------|
| `data/validation/gt_buildings.parquet` | GT footprints (`RES`/`NRES`) |
| `data/predictions/model_predictions.parquet` | LORO model preds + probs |
| `data/ghsl/tiles/*.tif` | GHS-BUILT-C FUN tiles |

GHSL FUN codes: `0` non-built, `1` residential, `2` non-residential, `255` nodata.

## Two benchmarks

1. **Functional** — only pixels/buildings where GHSL ∈ {RES, NRES}; apples-to-apples classification.
2. **End-to-end** — GHSL `0`/`255` count as abstentions; report coverage + selective + effective accuracy.

## Config

All knobs live in `validation/config.yaml`.
