#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="$ROOT/outputs/mixed_cells_expanded_labels"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/launch_${STAMP}.log"
# Fetch uses all CPUs; scoring caps inside the script to avoid GeoDataFrame OOM
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/eval_mixed_cells_expanded_labels.py" \
  "$@" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
