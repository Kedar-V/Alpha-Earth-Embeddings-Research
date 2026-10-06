#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/viz/usa_embedding_globe/runs/boundary_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

CPUS="$(nproc)"
export RUN_DIR CPUS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: build_boundary_walks"
  echo "cpus: ${CPUS}"
  echo "gpus: unused (CPU fold scoring + DB read)"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "output: ${ROOT}/viz/usa_embedding_globe/data/boundary_walks.json"
} > "${RUN_DIR}/launch.txt"

{
  echo "host: $(hostname)"
  echo "cpus: ${CPUS}"
  nvidia-smi -L 2>/dev/null || echo "no nvidia"
} > "${RUN_DIR}/environment.txt"

PYBIN="${REPO}/.venv/bin/python"
nohup env RUN_DIR="${RUN_DIR}" \
  "${PYBIN}" -u "${ROOT}/scripts/build_boundary_walks.py" \
    --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
