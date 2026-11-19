#!/usr/bin/env bash
# Submit Week 11: AE+S2 XGB LORO with frac_empty head (locked Section 3/4 eval).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="$ROOT/outputs/week11_ae_s2_empty_head_loro"
mkdir -p "$OUT_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/launch_${STAMP}.log"
FOLD_WORKERS="${FOLD_WORKERS:-2}"

WRAPPER="$OUT_ROOT/wrapper_${STAMP}.sh"
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_week11_ae_s2_empty_head_loro.py" --fold-workers "$FOLD_WORKERS"
EOF
chmod +x "$WRAPPER"

nohup bash "$WRAPPER" >"$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "launch_log=$LOG"
echo "$PID" >"$OUT_ROOT/latest_launch.pid"
echo "$LOG" >"$OUT_ROOT/latest_launch.logpath"
disown "$PID" || true
echo "Monitor: tail -f $LOG"
echo "Progress: $OUT_ROOT/latest/progress.json (after run dir appears)"
