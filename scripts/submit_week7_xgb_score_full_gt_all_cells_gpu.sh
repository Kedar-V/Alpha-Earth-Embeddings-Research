#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="$ROOT/outputs/week7_xgb_score_full_gt_all_cells"
CKPT_DEFAULT="$ROOT/outputs/week7_xgb_loro_gpu/20261113_014241"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/launch_${STAMP}.log"

# Default ckpt run; allow override via --ckpt-run ...
ARGS=("$@")
if [[ "$*" != *"--ckpt-run"* ]]; then
  ARGS=(--ckpt-run "$CKPT_DEFAULT" "${ARGS[@]}")
fi

nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/score_week7_xgb_full_gt_all_cells_gpu.py" \
  "${ARGS[@]}" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
