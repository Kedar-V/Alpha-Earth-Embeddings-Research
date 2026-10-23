#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kv133/Alpha-Earth-Embeddings-Research
DIR="$ROOT/week4/international_geo_generalization"
LOGDIR="$DIR/viz/usa_embedding_globe/logs/dashboard_dou_sample"
cd "$DIR"
exec "$ROOT/.venv/bin/python" -u scripts/sample_dashboard_dou_balanced.py \
  --n-per-dou-class 200 \
  --min-k 50 \
  --workers 12 \
  --seed 42 \
  --countries USA IND JPN \
  >>"$LOGDIR/job3.log" 2>&1
