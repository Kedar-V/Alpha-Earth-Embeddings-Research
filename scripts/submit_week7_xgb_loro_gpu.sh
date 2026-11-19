#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="$ROOT/outputs/week7_xgb_loro_gpu"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/launch_${STAMP}.log"
# Separate out-root from CPU train (outputs/week7_xgb_loro/) — do not touch that job
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/train_week7_xgb_loro_gpu.py" \
  --out-root "$OUT_ROOT" \
  "$@" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
