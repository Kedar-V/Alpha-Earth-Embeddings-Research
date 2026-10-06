#!/usr/bin/env bash
# Expand globe viewer to USA + India + Japan (export → boundary walks).
# Parallel: fold fits (ProcessPool), imagery workers (2× CPUs), city walks (threads).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/viz/usa_embedding_globe/runs/intl_globe_${STAMP}"
EXPORT_RUN="${RUN_DIR}/export"
WALK_RUN="${RUN_DIR}/boundary"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}
mkdir -p "${EXPORT_RUN}"/{checkpoints,logs,artifacts,failures}
mkdir -p "${WALK_RUN}"/{checkpoints,logs,artifacts,failures}

CPUS="$(nproc)"
WORKERS="${WORKERS:-$(( CPUS * 2 ))}"
FOLD_WORKERS="${FOLD_WORKERS:-10}"
WALK_WORKERS="${WALK_WORKERS:-${CPUS}}"
COUNTRIES="${COUNTRIES:-USA IND JPN}"
N_PER_CITY="${N_PER_CITY:-100}"
N_PER_CITY_WEAK="${N_PER_CITY_WEAK:-500}"

export RUN_DIR EXPORT_RUN WALK_RUN CPUS WORKERS FOLD_WORKERS WALK_WORKERS
export COUNTRIES N_PER_CITY N_PER_CITY_WEAK
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$CPUS}"

{
  echo "job: intl_embedding_globe (export + boundary walks)"
  echo "countries: ${COUNTRIES}"
  echo "n_per_city: ${N_PER_CITY}"
  echo "n_per_city_weak: ${N_PER_CITY_WEAK}"
  echo "cpus: ${CPUS}"
  echo "imagery_workers: ${WORKERS}"
  echo "fold_workers: ${FOLD_WORKERS}"
  echo "walk_workers: ${WALK_WORKERS}"
  echo "gpus: visible but unused (CPU/DB/HTTP workload)"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "output: ${ROOT}/viz/usa_embedding_globe/data"
} > "${RUN_DIR}/launch.txt"

{
  echo "host: $(hostname)"
  echo "cpus: ${CPUS}"
  nvidia-smi -L 2>/dev/null || echo "no nvidia"
} > "${RUN_DIR}/environment.txt"

cat > "${RUN_DIR}/progress.json" <<EOF
{
  "status": "running",
  "stage": "launching",
  "started_at": "$(date +%Y-%m-%dT%H:%M:%S)",
  "updated_at": "$(date +%Y-%m-%dT%H:%M:%S)",
  "countries": "${COUNTRIES}",
  "resource_allocation": {
    "cpus": ${CPUS},
    "imagery_workers": ${WORKERS},
    "fold_workers": ${FOLD_WORKERS},
    "walk_workers": ${WALK_WORKERS},
    "gpu_used": false
  },
  "output": "${ROOT}/viz/usa_embedding_globe/data",
  "job_run_dir": "${RUN_DIR}"
}
EOF

nohup env \
  RUN_DIR="${RUN_DIR}" \
  EXPORT_RUN="${EXPORT_RUN}" \
  WALK_RUN="${WALK_RUN}" \
  WORKERS="${WORKERS}" \
  FOLD_WORKERS="${FOLD_WORKERS}" \
  WALK_WORKERS="${WALK_WORKERS}" \
  N_PER_CITY="${N_PER_CITY}" \
  N_PER_CITY_WEAK="${N_PER_CITY_WEAK}" \
  COUNTRIES="${COUNTRIES}" \
  bash "${ROOT}/scripts/run_intl_embedding_globe_pipeline.sh" \
  > "${RUN_DIR}/wrapper.log" 2>&1 &

echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
echo "PID: $(cat "${RUN_DIR}/job.pid")"
