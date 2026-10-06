#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/usa_ingest_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR
export CPUS="${CPUS:-70}"
export GPUS="${GPUS:-all}"
# Keep BLAS single-threaded inside process-pool workers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export DISCOVERY_RADIUS_M="${DISCOVERY_RADIUS_M:-30000}"
export RING_KM="${RING_KM:-5}"
export AREA_HEADROOM="${AREA_HEADROOM:-1.10}"
export AEF_SHARD_SIZE="${AEF_SHARD_SIZE:-1000}"
export PARALLEL_DISCOVERY="${PARALLEL_DISCOVERY:-8}"
export PARALLEL_AEF="${PARALLEL_AEF:-2}"
export WORKERS="${WORKERS:-$(( CPUS - 2 ))}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
fi

# Load DB env (do not echo password)
set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: usa_ingest_to_postgres"
  echo "cpus: ${CPUS}"
  echo "gpus: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "workers: ${WORKERS}"
  echo "discovery_radius_m: ${DISCOVERY_RADIUS_M}"
  echo "ring_km: ${RING_KM}"
  echo "area_headroom: ${AREA_HEADROOM}"
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
EXTRA=()
if [[ "${1:-}" == "--city-id" ]]; then
  shift
  while [[ $# -gt 0 ]]; do EXTRA+=(--city-id "$1"); shift; done
fi

nohup env RUN_DIR="${RUN_DIR}" CPUS="${CPUS}" GPUS="${GPUS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  "${PYBIN}" -u "${ROOT}/scripts/ingest_usa_to_postgres.py" \
    --discovery-radius-m "${DISCOVERY_RADIUS_M}" \
    --ring-km "${RING_KM}" \
    --area-headroom "${AREA_HEADROOM}" \
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
