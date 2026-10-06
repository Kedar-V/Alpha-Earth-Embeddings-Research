#!/usr/bin/env bash
# Wait for a japan_ingest RUN_DIR to finish, then LORO + PLS3 walks + report/merge.
set -euo pipefail

INGEST_DIR="${1:?usage: run_japan10_after_ingest.sh <japan_ingest_run_dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
PY="${REPO}/.venv/bin/python"
CHAIN_DIR="${ROOT}/runs/japan10_chain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CHAIN_DIR}"
echo "$$" > "${CHAIN_DIR}/job.pid"
echo "ingest=${INGEST_DIR}" > "${CHAIN_DIR}/launch.txt"
echo "chain_dir=${CHAIN_DIR}" >> "${CHAIN_DIR}/launch.txt"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${CHAIN_DIR}/job.log"; }

prog() {
  "$PY" - <<PY
import json, time
from pathlib import Path
p = Path("${CHAIN_DIR}") / "progress.json"
d = json.loads(p.read_text()) if p.exists() else {}
d.update(status="running", stage="$1", percent_complete=float($2),
         updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
         last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
         ingest_dir="${INGEST_DIR}")
if "started_at" not in d:
    d["started_at"] = d["updated_at"]
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

prog wait_ingest 5
log "waiting for ingest DONE: ${INGEST_DIR}"
while true; do
  if [[ -f "${INGEST_DIR}/DONE" ]]; then
    log "ingest DONE"
    break
  fi
  if [[ -f "${INGEST_DIR}/FAILED" ]]; then
    log "ingest FAILED"
    echo failed > "${CHAIN_DIR}/FAILED"
    prog failed 100
    exit 1
  fi
  # also treat progress status=completed
  if [[ -f "${INGEST_DIR}/progress.json" ]]; then
    st=$("$PY" -c "import json; print(json.load(open('${INGEST_DIR}/progress.json')).get('status',''))")
    if [[ "$st" == "completed" ]]; then
      log "ingest progress status=completed"
      break
    fi
  fi
  sleep 30
done

# Launch LORO (foreground via its pipeline, capture RUN_DIR)
prog loro 20
log "STAGE japan LORO"
set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a
# submit_japan_loro backgrounds; wait on its DONE
LORO_OUT=$("${ROOT}/scripts/submit_japan_loro.sh" | tee -a "${CHAIN_DIR}/job.log")
LORO_DIR=$(echo "$LORO_OUT" | sed -n 's/^RUN_DIR=//p' | tail -1)
echo "loro_dir=${LORO_DIR}" >> "${CHAIN_DIR}/launch.txt"
log "waiting for LORO ${LORO_DIR}"
while true; do
  if [[ -f "${LORO_DIR}/DONE" ]] || [[ -f "${LORO_DIR}/RESULTS.md" ]]; then
    # RESULTS may appear slightly before DONE
    if [[ -f "${LORO_DIR}/DONE" ]] || [[ -f "${LORO_DIR}/summary.json" ]]; then
      break
    fi
  fi
  if [[ -f "${LORO_DIR}/FAILED" ]]; then
    log "LORO FAILED"; echo failed > "${CHAIN_DIR}/FAILED"; exit 1
  fi
  # poll progress
  if [[ -f "${LORO_DIR}/progress.json" ]]; then
    st=$("$PY" -c "import json; print(json.load(open('${LORO_DIR}/progress.json')).get('status',''))")
    if [[ "$st" == "completed" ]]; then
      break
    fi
  fi
  sleep 20
done
log "LORO complete: ${LORO_DIR}"

# Folds + export + walks (Japan-only)
prog folds_walks 55
STAMP=$(date +%Y%m%d_%H%M%S)
WALK_RUN="${ROOT}/viz/usa_embedding_globe/runs/japan10_walks_${STAMP}"
mkdir -p "${WALK_RUN}"/{checkpoints,logs,artifacts,failures}
echo "walk_run=${WALK_RUN}" >> "${CHAIN_DIR}/launch.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPUS=$(nproc)

log "STAGE fit JPN×10 folds"
"$PY" -u "${ROOT}/scripts/intl_globe_folds.py" \
  --countries JPN --force --workers 10 --n-pls 32 \
  --extract-dir "${LORO_DIR}/extract" \
  --out-dir "${ROOT}/viz/usa_embedding_globe/data/intl_folds" \
  >> "${WALK_RUN}/job.log" 2>&1

log "STAGE export JPN points"
POINTS_OUT="${WALK_RUN}/artifacts/points_japan10.json"
"$PY" -u "${ROOT}/scripts/export_usa_embedding_globe_sample.py" \
  --countries JPN --n-per-city 200 --n-per-city-weak 500 --seed 42 \
  --skip-imagery \
  --extract-dir "${LORO_DIR}/extract" \
  --intl-fold-dir "${ROOT}/viz/usa_embedding_globe/data/intl_folds" \
  --fold-workers 10 \
  --out-points "${POINTS_OUT}" \
  --run-dir "${WALK_RUN}/export" \
  >> "${WALK_RUN}/job.log" 2>&1

log "STAGE boundary walks"
WALKS_OUT="${WALK_RUN}/artifacts/boundary_walks_japan10.json"
mkdir -p "${WALK_RUN}/boundary"/{checkpoints,logs,artifacts,failures}
"$PY" -u "${ROOT}/scripts/build_boundary_walks.py" \
  --run-dir "${WALK_RUN}/boundary" --workers "${CPUS}" \
  --points "${POINTS_OUT}" \
  --out "${WALKS_OUT}" \
  --intl-fold-dir "${ROOT}/viz/usa_embedding_globe/data/intl_folds" \
  >> "${WALK_RUN}/job.log" 2>&1

prog report 90
log "STAGE report+merge"
"$PY" -u "${ROOT}/scripts/report_country10_boundary_walks.py" \
  --run-dir "${WALK_RUN}" \
  --loro-dir "${LORO_DIR}" \
  --data-dir "${ROOT}/viz/usa_embedding_globe/data" \
  --country JPN \
  --merge-globe \
  >> "${WALK_RUN}/job.log" 2>&1

# copy report pointer
cp -f "${WALK_RUN}/REPORT.md" "${CHAIN_DIR}/REPORT.md" 2>/dev/null || true
cp -f "${LORO_DIR}/RESULTS.md" "${CHAIN_DIR}/LORO_RESULTS.md" 2>/dev/null || true
"$PY" - <<PY
import json, time
from pathlib import Path
p = Path("${CHAIN_DIR}") / "progress.json"
p.write_text(json.dumps({
  "status":"completed","stage":"done","percent_complete":100.0,
  "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
  "ingest_dir":"${INGEST_DIR}",
  "loro_dir":"${LORO_DIR}",
  "walk_run":"${WALK_RUN}",
}, indent=2)+"\n")
(Path("${CHAIN_DIR}")/"DONE").write_text("ok\n")
PY
log "DONE chain"
echo "CHAIN_DIR=${CHAIN_DIR}"
echo "LORO_DIR=${LORO_DIR}"
echo "WALK_RUN=${WALK_RUN}"
