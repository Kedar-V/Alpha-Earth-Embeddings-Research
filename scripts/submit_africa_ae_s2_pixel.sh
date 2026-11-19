#!/usr/bin/env bash
# Submit Africa-5 AE vs AE+S2 vs GHSL pixel MAE job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${1:-$ROOT/outputs/africa_ae_s2_pixel/${TS}}"
mkdir -p "$RUN_DIR"/{checkpoints,artifacts,figures,failures,s2_mosaics,s2_feats,logs}

WORKERS="${WORKERS:-5}"  # one process per Africa city
BALANCED="${BALANCED:-$ROOT/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/balanced_pure.parquet}"
SCORED="${SCORED:-$ROOT/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/scored.parquet}"

{
  echo "job: africa_ae_s2_pixel"
  echo "run_dir: $RUN_DIR"
  echo "balanced: $BALANCED"
  echo "scored: $SCORED"
  echo "workers: $WORKERS"
  echo "cpus: $(nproc)"
  echo "started: $(date -Iseconds)"
} >"$RUN_DIR/launch.txt"

echo "[submit] run_dir=$RUN_DIR workers=$WORKERS cpus=$(nproc)"
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_africa_ae_s2_pixel.py" \
  --run-dir "$RUN_DIR" \
  --balanced "$BALANCED" \
  --scored "$SCORED" \
  --workers "$WORKERS" \
  >"$RUN_DIR/stdout.log" 2>&1 &
PID=$!
echo "$PID" >"$RUN_DIR/job.pid"
echo "pid=$PID" >>"$RUN_DIR/launch.txt"
echo "[submit] PID=$PID"
echo "[submit] log=$RUN_DIR/job.log"
echo "[submit] progress=$RUN_DIR/progress.json"
sleep 4
if kill -0 "$PID" 2>/dev/null; then
  echo "[submit] process alive"
  tail -n 20 "$RUN_DIR/stdout.log" || true
else
  echo "[submit] process died — see $RUN_DIR/stdout.log" >&2
  tail -n 40 "$RUN_DIR/stdout.log" || true
  exit 1
fi
