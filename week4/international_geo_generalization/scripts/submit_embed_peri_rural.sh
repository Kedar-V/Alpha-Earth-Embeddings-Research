#!/usr/bin/env bash
# Embed AlphaEarth cells for suburban (peri-urban) and rural buildings that
# are already in `buildings` but have no cell intersections.
# Urban leftovers are not embedded.
# Idempotent: a city with CITY.DONE is skipped. AEF shards resume per tile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/embed_peri_rural_${STAMP}}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR
export CPUS="${CPUS:-$(nproc)}"
# One city at a time. Shard workers use the CPUs; Postgres writers stay at 4.
export WORKERS="${WORKERS:-$(( CPUS - 4 ))}"
export DB_WRITERS="${DB_WRITERS:-4}"
export PARALLEL_CITIES="${PARALLEL_CITIES:-8}"
export AEF_SHARD_SIZE="${AEF_SHARD_SIZE:-1000}"
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
  echo "job: embed_peri_rural_buildings"
  echo "cpus: ${CPUS}"
  echo "aef_workers: ${WORKERS}"
  echo "db_writers: ${DB_WRITERS}"
  echo "cities_parallel: ${PARALLEL_CITIES}"
  echo "aef_shard_size: ${AEF_SHARD_SIZE}"
  echo "raster: ${RASTER}"
  echo "keep: smod 21 peri_urban; smod 11,12,13 rural"
  echo "exclude: urban leftovers; buildings that already have cells"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "resume: checkpoints/<city_id>/{SELECT,AEF,LABELS,DOU,CITY}.DONE"
} > "${RUN_DIR}/launch.txt"

"${PYBIN}" - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, shutil
print("host:", platform.node())
print("python:", platform.python_version())
print("cpus:", os.cpu_count())
print("aef_workers:", os.environ.get("WORKERS"))
print("db_writers:", os.environ.get("DB_WRITERS"))
print("raster:", """${RASTER}""")
usage = shutil.disk_usage("""${REPO}""")
print(f"disk_free_gb: {usage.free / 1e9:.1f}")
PY

nohup env RUN_DIR="${RUN_DIR}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  "${PYBIN}" -u "${ROOT}/scripts/embed_peri_rural_buildings.py" \
    --raster "${RASTER}" \
    --run-dir "${RUN_DIR}" \
    --workers "${WORKERS}" \
    --db-writers "${DB_WRITERS}" \
    --parallel-cities "${PARALLEL_CITIES}" \
    --aef-shard-size "${AEF_SHARD_SIZE}" \
  > "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
