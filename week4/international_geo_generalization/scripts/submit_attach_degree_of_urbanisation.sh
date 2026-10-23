#!/usr/bin/env bash
# Attach GHS-SMOD Degree of Urbanisation to every AlphaEarth cell.
# Idempotent: cities already fully labeled are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/dou_attach_${STAMP}}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR
export CPUS="${CPUS:-$(nproc)}"
export WORKERS="${WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

RASTER="${RASTER:-${REPO}/data/ghsl/smod/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.tif}"
PYBIN="${REPO}/.venv/bin/python"

{
  echo "job: attach_degree_of_urbanisation"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS}"
  echo "raster: ${RASTER}"
  echo "epoch: 2025"
  echo "product: GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "resume: cities with a complete cell_degree_of_urbanisation row set are skipped"
} > "${RUN_DIR}/launch.txt"

"${PYBIN}" - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, shutil
print("host:", platform.node())
print("python:", platform.python_version())
print("cpus:", os.cpu_count())
print("workers:", os.environ.get("WORKERS"))
print("raster:", os.environ.get("RASTER", """${RASTER}"""))
usage = shutil.disk_usage("""${REPO}""")
print(f"disk_free_gb: {usage.free / 1e9:.1f}")
PY

nohup env RUN_DIR="${RUN_DIR}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  "${PYBIN}" -u "${ROOT}/scripts/attach_degree_of_urbanisation.py" \
    --raster "${RASTER}" \
    --run-dir "${RUN_DIR}" \
    --workers "${WORKERS}" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
