#!/usr/bin/env bash
# Submit Week 14: AE-grid ViT + AE mean + S2 late fusion LORO (locked Section 3/4).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="$ROOT/outputs/week14_ae_grid_vit_s2_loro"
mkdir -p "$OUT_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/launch_${STAMP}.log"
FOLD_WORKERS="${FOLD_WORKERS:-2}"
EPOCHS="${EPOCHS:-12}"
GRID_WORKERS="${GRID_WORKERS:-8}"

EXTRA_ARGS=()
if [[ -n "${RESUME_DIR:-}" ]]; then
  EXTRA_ARGS+=(--resume-dir "$RESUME_DIR")
fi

WRAPPER="$OUT_ROOT/wrapper_${STAMP}.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "cd \"$ROOT\""
  printf 'exec "%s" -u "%s" --fold-workers %q --epochs %q --grid-workers %q' \
    "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/run_week14_ae_grid_vit_s2_loro.py" \
    "$FOLD_WORKERS" "$EPOCHS" "$GRID_WORKERS"
  if ((${#EXTRA_ARGS[@]})); then
    printf ' %q' "${EXTRA_ARGS[@]}"
  fi
  echo
} >"$WRAPPER"
chmod +x "$WRAPPER"

nohup bash "$WRAPPER" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
echo "Monitor: tail -f $LOG"
echo "Progress: $OUT_ROOT/latest/progress.json"
