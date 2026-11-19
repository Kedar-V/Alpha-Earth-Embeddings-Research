#!/usr/bin/env bash
# Submit Africa city-LORO (Ridge + GPU XGB) with submit-job observability.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
# shellcheck disable=SC1091
source "$ROOT/week4/international_geo_generalization/db/.env"
set +a
export PGDATA_DIR="${PGDATA_DIR:-$ROOT/pgdata}"

if ! "$ROOT/.tools/postgis/bin/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1; then
  echo "[submit] starting postgres from $PGDATA_DIR"
  DB_RUNTIME=local "$ROOT/week4/international_geo_generalization/scripts/init_db.sh" >/tmp/init_db_africa_loro.submit.log 2>&1 || true
  "$ROOT/.tools/postgis/bin/pg_ctl" -D "$PGDATA_DIR" status
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-$ROOT/outputs/africa_city_loro_gpu/${TS}}"
mkdir -p "$RUN_DIR"/{checkpoints,artifacts,figures,failures,logs,_cache,exports}

COUNTRIES="${COUNTRIES:-EGY,NGA}"
N_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

{
  echo "job: africa_city_loro_gpu"
  echo "countries: ${COUNTRIES}"
  echo "run_dir: ${RUN_DIR}"
  echo "gpus: ${N_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  echo "cpus: $(nproc)"
  echo "started: $(date -Iseconds)"
  echo "protocol: city-LORO soft-frac MAE; AE Ridge + AE XGB GPU"
} > "${RUN_DIR}/launch.txt"

echo "[submit] run_dir=$RUN_DIR countries=$COUNTRIES gpus=$N_GPUS"
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "$ROOT/.venv/bin/python" -u "$ROOT/scripts/run_africa_city_loro_gpu.py" \
  --run-dir "$RUN_DIR" \
  --countries "$COUNTRIES" \
  --workers "${N_GPUS}" \
  >"$RUN_DIR/stdout.log" 2>&1 &
PID=$!
echo "$PID" >"$RUN_DIR/job.pid"
echo "pid=$PID" >>"$RUN_DIR/launch.txt"
echo "[submit] PID=$PID"
echo "[submit] log=$RUN_DIR/job.log"
echo "[submit] progress=$RUN_DIR/progress.json"
echo "[submit] stdout=$RUN_DIR/stdout.log"
sleep 4
if kill -0 "$PID" 2>/dev/null; then
  echo "[submit] process alive"
  tail -n 20 "$RUN_DIR/stdout.log" || true
  cat "$RUN_DIR/progress.json" 2>/dev/null || true
else
  echo "[submit] process died — see $RUN_DIR/stdout.log" >&2
  tail -n 40 "$RUN_DIR/stdout.log" || true
  exit 1
fi
