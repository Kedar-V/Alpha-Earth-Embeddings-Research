# Week 7 XGBoost Section 3 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop-in XGBoost LORO soft-fraction OOF replacing AE Ridge on the locked Section 3 MAE board; update READMEs; notebooks/week7.

**Architecture:** Train XGB multi-output LORO on week4 balanced pixels (GPU per fold) → full-GT OOF parquet → Section 3 building + 100 m MAE with models `{ae_xgb, ae_ridge, ae_pls, ghsl_built_s}` reusing existing GHSL/soft aggregation where possible.

**Tech stack:** xgboost 3.x CUDA hist, pandas/parquet, ProcessPoolExecutor, submit-job observability.

## File structure

| File | Responsibility |
|---|---|
| `scripts/train_week7_xgb_loro.py` | LORO XGB train + full-GT OOF + progress |
| `scripts/submit_week7_xgb_loro.sh` | nohup launcher |
| `scripts/eval_week7_section3_xgb_mae.py` | Building + 100 m Section 3 MAE with XGB |
| `scripts/submit_week7_section3_xgb_mae.sh` | nohup launcher |
| `notebooks/week7/*` | briefing + results |
| READMEs | lock Section 3 baseline |

## Tasks

### Task 1: Train XGB LORO
- [ ] Implement train script with fold checkpoints, progress.json, OOF + full_gt OOF
- [ ] Submit via submit_week7_xgb_loro.sh

### Task 2: Section 3 MAE
- [ ] Eval script: rebuild AE XGB building means; reuse Ridge/PLS/GHSL from week6 artifacts when present
- [ ] Building soft+1:1 y_res + 100 m soft-argmax min≥10
- [ ] Submit after train completes

### Task 3: Notebooks + READMEs
- [ ] notebooks/week7 + update root/week6 READMEs
