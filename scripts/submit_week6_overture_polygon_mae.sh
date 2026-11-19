#!/usr/bin/env bash
# Submit Week 6 Overture-polygon AE vs GHS-BUILT-S MAE (submit-job style).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${1:-$ROOT/outputs/week6_overture_polygon_mae/${TS}}"
mkdir -p "$RUN_DIR"/{checkpoints,artifacts,figures,failures,logs}

# Cap workers: each GHSL process opens all BUILT-S + NRES tiles.
NPROC="$(nproc)"
WORKERS="${WORKERS:-$(( NPROC > 16 ? 16 : NPROC ))}"
CHUNK_SIZE="${CHUNK_SIZE:-1500}"
# Default: Section-2 style 1:1 balance + soft+hard GT; reuse prior GHSL zonal.
REUSE_GHSL="${REUSE_GHSL:-$ROOT/outputs/week6_overture_polygon_mae/20261110_034500/checkpoints/ghsl_by_building.parquet}"
EXTRA_ARGS=("${@:2}")

CMD=(
  "$ROOT/.venv/bin/python" -u "$ROOT/scripts/eval_week6_overture_polygon_mae.py"
  --run-dir "$RUN_DIR"
  --workers "$WORKERS"
  --chunk-size "$CHUNK_SIZE"
  --balanced
)
if [[ -n "$REUSE_GHSL" && -f "$REUSE_GHSL" ]]; then
  CMD+=(--reuse-ghsl-from "$REUSE_GHSL")
fi
CMD+=("${EXTRA_ARGS[@]}")

{
  echo "cmd=${CMD[*]}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cpus=$(nproc)"
  echo "workers=$WORKERS"
  echo "chunk_size=$CHUNK_SIZE"
  echo "balanced=1"
  echo "soft_and_hard_gt=1"
  echo "reuse_ghsl=$REUSE_GHSL"
  echo "submit_job=1"
  echo "host=$(hostname)"
} >"$RUN_DIR/launch.txt"

echo "[submit] run_dir=$RUN_DIR workers=$WORKERS cpus=$(nproc) balanced=1 soft+hard"
nohup "${CMD[@]}" >"$RUN_DIR/stdout.log" 2>&1 &
PID=$!
echo "$PID" >"$RUN_DIR/job.pid"
echo "pid=$PID" >>"$RUN_DIR/launch.txt"
echo "[submit] PID=$PID"
echo "[submit] log=$RUN_DIR/job.log"
echo "[submit] stdout=$RUN_DIR/stdout.log"
echo "[submit] progress=$RUN_DIR/progress.json"

sleep 5
if kill -0 "$PID" 2>/dev/null; then
  echo "[submit] process alive"
  tail -n 25 "$RUN_DIR/stdout.log" || true
  if [[ -f "$RUN_DIR/progress.json" ]]; then
    echo "[submit] progress:"
    cat "$RUN_DIR/progress.json"
  fi
else
  echo "[submit] process died — see $RUN_DIR/stdout.log" >&2
  tail -n 50 "$RUN_DIR/stdout.log" || true
  exit 1
fi
