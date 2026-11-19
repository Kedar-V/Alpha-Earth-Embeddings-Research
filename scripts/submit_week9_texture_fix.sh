#!/usr/bin/env bash
# Week 9 texture fix: reuse cached mosaics from 20261113_055624 → new timestamped run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="$ROOT/outputs/week9_ae_s2_nres_loro"
OLD="$OUT_ROOT/20261113_055624"
MOSAIC_DIR="$OLD/artifacts/s2_mosaics"
AE_MEANS="$OLD/artifacts/ae_means.parquet"
mkdir -p "$OUT_ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/launch_texture_fix_${STAMP}.log"
WORKERS="${WORKERS:-10}"

WRAPPER="$OUT_ROOT/wrapper_texture_fix_${STAMP}.sh"
cat >"$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_week9_ae_s2_nres_loro.py" \
  --workers "$WORKERS" \
  --mosaic-dir "$MOSAIC_DIR" \
  --no-download \
  --copy-ae-means-from "$AE_MEANS"
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
