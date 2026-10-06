#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/japan_ingest_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR
export CPUS="${CPUS:-70}"
export GPUS="${GPUS:-all}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export DISCOVERY_RADIUS_M="${DISCOVERY_RADIUS_M:-30000}"
export RING_KM="${RING_KM:-5}"
export AREA_HEADROOM="${AREA_HEADROOM:-1.10}"
export TARGET_AREA_M2="${TARGET_AREA_M2:-24604226.98057363}"
export AEF_SHARD_SIZE="${AEF_SHARD_SIZE:-1000}"
export PARALLEL_DISCOVERY="${PARALLEL_DISCOVERY:-8}"
export PARALLEL_AEF="${PARALLEL_AEF:-2}"
export WORKERS="${WORKERS:-$(( CPUS - 2 ))}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
fi

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

# Optional: pass --city-id JPN_X JPN_Y ... to ingest a subset (new cities only).
EXTRA=()
if [[ "${1:-}" == "--city-id" ]]; then
  shift
  while [[ $# -gt 0 ]]; do
    EXTRA+=(--city-id "$1")
    shift
  done
fi

{
  echo "job: japan_ingest_to_postgres"
  echo "country_id: JPN"
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    echo "cities: ${EXTRA[*]}"
  else
    echo "cities: all selected JPN"
  fi
  echo "cpus: ${CPUS}"
  echo "gpus: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "workers: ${WORKERS}"
  echo "discovery_radius_m: ${DISCOVERY_RADIUS_M}"
  echo "ring_km: ${RING_KM}"
  echo "area_headroom: ${AREA_HEADROOM}"
  echo "target_area_m2: ${TARGET_AREA_M2}"
  echo "aef_shard_size: ${AEF_SHARD_SIZE}"
  echo "parallel_discovery: ${PARALLEL_DISCOVERY}"
  echo "parallel_aef: ${PARALLEL_AEF}"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
} > "${RUN_DIR}/launch.txt"

python3 - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, subprocess
print("host:", platform.node())
print("cpus:", os.cpu_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    print(subprocess.check_output(["nvidia-smi","-L"], text=True))
except Exception as e:
    print("nvidia-smi:", e)
PY

PYBIN="${REPO}/.venv/bin/python"
nohup env RUN_DIR="${RUN_DIR}" CPUS="${CPUS}" GPUS="${GPUS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  "${PYBIN}" -u "${ROOT}/scripts/ingest_usa_to_postgres.py" \
    --country-id JPN \
    --discovery-radius-m "${DISCOVERY_RADIUS_M}" \
    --ring-km "${RING_KM}" \
    --area-headroom "${AREA_HEADROOM}" \
    --target-area-m2 "${TARGET_AREA_M2}" \
    --aef-shard-size "${AEF_SHARD_SIZE}" \
    --parallel-discovery "${PARALLEL_DISCOVERY}" \
    --parallel-aef "${PARALLEL_AEF}" \
    --workers "${WORKERS}" \
    "${EXTRA[@]}" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"

cp "${ROOT}/scripts/monitor_15m.py" "${RUN_DIR}/monitor_15m.py"
nohup env RUN_DIR="${RUN_DIR}" "${PYBIN}" -u "${RUN_DIR}/monitor_15m.py" \
  >> "${RUN_DIR}/monitor_15m.stdout" 2>&1 &
echo $! | tee "${RUN_DIR}/monitor_15m.pid"
echo "Monitor: ${RUN_DIR}/monitor_15m.log"
