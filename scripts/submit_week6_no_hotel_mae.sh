#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="$ROOT/outputs/week6_no_hotel_mae"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/launch_${STAMP}.log"
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/eval_week6_no_hotel_mae.py" \
  "$@" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
