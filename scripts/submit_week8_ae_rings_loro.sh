#!/usr/bin/env bash
# Submit Week 8 AE multi-scale rings + train-where-you-score LORO, then Section 5 canvas.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="$ROOT/outputs/week8_ae_rings_loro"
mkdir -p "$OUT_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/launch_${STAMP}.log"
WORKERS="${WORKERS:-10}"
ALPHA="${ALPHA:-10}"

WRAPPER="$OUT_ROOT/wrapper_${STAMP}.sh"
# Extra CLI args for the python job (do not expand empty "$@" into a literal "").
EXTRA_ARGS=()
if (($# > 0)); then
  EXTRA_ARGS=("$@")
fi
# Serialize extra args for the wrapper (quoted).
EXTRA_Q=""
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
  EXTRA_Q+=" $(printf '%q' "$a")"
done
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
"$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_week8_ae_rings_loro.py" \\
  --workers "$WORKERS" --alpha "$ALPHA"${EXTRA_Q}
METRICS="\$("$ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
p = Path("$OUT_ROOT") / "latest" / "artifacts" / "section5_ring_metrics.json"
print(p.resolve() if p.exists() else "")
PY
)"
if [[ -n "\$METRICS" && -f "\$METRICS" ]]; then
  "$ROOT/.venv/bin/python" -u "$ROOT/scripts/rebuild_section5_ring_board.py" --metrics "\$METRICS"
fi
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
