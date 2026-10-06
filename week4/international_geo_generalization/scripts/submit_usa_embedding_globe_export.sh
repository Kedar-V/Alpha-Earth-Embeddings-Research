#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/viz/usa_embedding_globe/runs/export_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

CPUS="$(nproc)"
# I/O-bound tile fetch: 2× CPUs. GPUs unused (HTTP imagery + tiny UMAP).
WORKERS="${WORKERS:-$(( CPUS * 2 ))}"
export RUN_DIR CPUS WORKERS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$CPUS}"

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: export_usa_embedding_globe_sample"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS} (I/O tile downloads)"
  echo "gpus: all visible but unused (not GPU-bound)"
  echo "n_per_city: 100"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "output: ${ROOT}/viz/usa_embedding_globe/data"
} > "${RUN_DIR}/launch.txt"

{
  echo "host: $(hostname)"
  echo "cpus: ${CPUS}"
  nvidia-smi -L 2>/dev/null || echo "no nvidia"
} > "${RUN_DIR}/environment.txt"

PYBIN="${REPO}/.venv/bin/python"
nohup env RUN_DIR="${RUN_DIR}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS}" \
  "${PYBIN}" -u "${ROOT}/scripts/export_usa_embedding_globe_sample.py" \
    --n-per-city 100 \
    --seed 42 \
    --workers "${WORKERS}" \
    --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
