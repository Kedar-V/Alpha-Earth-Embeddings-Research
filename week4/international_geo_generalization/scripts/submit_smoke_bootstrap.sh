#!/usr/bin/env bash
# Submit observable DB smoke bootstrap (CPUS=70, GPUS=all).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/smoke_bootstrap_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR
export CPUS="${CPUS:-70}"
export GPUS="${GPUS:-all}"
export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"

# Expose GPUs to any later CUDA stages (idle during DB smoke)
if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
fi

{
  echo "job: smoke_db_bootstrap"
  echo "cpus: ${CPUS}"
  echo "gpus: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
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

PYBIN="${REPO}/.tools/postgis/bin/python"
[[ -x "${PYBIN}" ]] || PYBIN=python3

nohup env RUN_DIR="${RUN_DIR}" CPUS="${CPUS}" GPUS="${GPUS}" \
  "${PYBIN}" "${ROOT}/scripts/smoke_db_bootstrap.py" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
