#!/usr/bin/env bash
# Submit Experiment 05: task-invariant vs geography-specific subspaces (submit-job).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO}/week5/runs/task_invariant_subspace_${STAMP}"
PRIOR_RUN="${PRIOR_RUN:-${REPO}/week5/runs/global_local_subspace_20261005_173516}"
EXP04_RUN="${EXP04_RUN:-${REPO}/week5/runs/three_country_falsification_20261005_185246}"
EXTRACT_DIR="${EXTRACT_DIR:-${REPO}/week4/international_geo_generalization/runs/intl_transfer_20261005_142709/extract}"

mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures,figures,artifacts/subspaces,artifacts/bootstrap,artifacts/seeds}

export RUN_DIR PRIOR_RUN EXP04_RUN EXTRACT_DIR REPO
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export WORKERS="${WORKERS:-${CPUS}}"
export MAX_PER_CLASS="${MAX_PER_CLASS:-5000}"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export N_BOOTSTRAP="${N_BOOTSTRAP:-50}"
export N_SHUFFLE="${N_SHUFFLE:-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Use all GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
fi

PYBIN="${REPO}/.venv/bin/python"
if [[ ! -x "${PYBIN}" ]]; then
  PYBIN="$(command -v python3)"
fi
export PYBIN

{
  echo "job: task_invariant_subspace_exp05"
  echo "hypothesis: S_c ≈ S_G_task + S_L,c (predictive invariance)"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS}"
  echo "gpus: all (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "max_per_class: ${MAX_PER_CLASS}"
  echo "seeds: ${SEEDS}"
  echo "n_bootstrap: ${N_BOOTSTRAP}"
  echo "n_shuffle: ${N_SHUFFLE}"
  echo "prior_run: ${PRIOR_RUN}"
  echo "exp04_run: ${EXP04_RUN}"
  echo "extract_dir: ${EXTRACT_DIR}"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
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
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(" ", i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch:", e)
try:
    print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
except Exception as e:
    print("nvidia-smi:", e)
print("free:")
print(subprocess.check_output(["free", "-h"], text=True))
print("prior exists:", os.path.isdir(os.environ["PRIOR_RUN"]))
print("exp04 exists:", os.path.isdir(os.environ["EXP04_RUN"]))
print("extract exists:", os.path.isdir(os.environ["EXTRACT_DIR"]))
PY

# shellcheck disable=SC2086
nohup env \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${PYBIN}" -u "${SCRIPT_DIR}/run_experiment.py" \
    --run-dir "${RUN_DIR}" \
    --prior-run "${PRIOR_RUN}" \
    --exp04-run "${EXP04_RUN}" \
    --extract-dir "${EXTRACT_DIR}" \
    --workers "${WORKERS}" \
    --max-per-class "${MAX_PER_CLASS}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --n-shuffle "${N_SHUFFLE}" \
    --seeds ${SEEDS} \
  >> "${RUN_DIR}/job.log" 2>&1 &

echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
echo "Report: ${RUN_DIR}/REPORT.md"
echo "Notebook: ${REPO}/week5/notebooks/05_task_invariant_subspace.ipynb"
