#!/usr/bin/env bash
# Rerun week-5 experiments on a country-level DoU-balanced extract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${REPO}/week5/runs/dou_balanced_week5_${STAMP}}"

mkdir -p "${RUN_DIR}"/{checkpoints,logs,failures,extract}

export REPO RUN_DIR
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export WORKERS="${WORKERS:-${CPUS}}"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export N_BOOTSTRAP="${N_BOOTSTRAP:-50}"
export N_SHUFFLE="${N_SHUFFLE:-5}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
fi

PYBIN="${REPO}/.venv/bin/python"
if [[ ! -x "${PYBIN}" ]]; then
  PYBIN="$(command -v python3)"
fi

{
  echo "job: dou_balanced_week5"
  echo "protocol: country-level urban=peri_urban=rural, 1:1 res/nres within each dou"
  echo "max_per_class: 0 (no extra per-city cap; extract balance is the training set)"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS}"
  echo "gpus: ${CUDA_VISIBLE_DEVICES:-none} (task_invariant stage)"
  echo "seeds: ${SEEDS}"
  echo "n_bootstrap: ${N_BOOTSTRAP}"
  echo "n_shuffle: ${N_SHUFFLE}"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "stages: extract -> global_local_subspace -> three_country_falsification -> task_invariant_subspace"
  echo "priors: this run only; October 5 runs are not reused"
} > "${RUN_DIR}/launch.txt"

"${PYBIN}" - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, subprocess
print("host:", platform.node())
print("python:", platform.python_version())
print("cpus:", os.cpu_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "ngpu:", torch.cuda.device_count())
except Exception as e:
    print("torch:", e)
print("free:")
print(subprocess.check_output(["free", "-h"], text=True))
PY

# shellcheck disable=SC2086
nohup env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYBIN}" -u "${SCRIPT_DIR}/run_dou_balanced_week5.py" \
    --run-dir "${RUN_DIR}" \
    --workers "${WORKERS}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --n-shuffle "${N_SHUFFLE}" \
    --seeds ${SEEDS} \
  >> "${RUN_DIR}/job.log" 2>&1 &

echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
