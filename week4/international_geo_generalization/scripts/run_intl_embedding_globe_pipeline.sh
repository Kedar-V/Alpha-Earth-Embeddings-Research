#!/usr/bin/env bash
# Inner pipeline for submit_intl_embedding_globe.sh (export → boundary walks).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
PYBIN="${REPO}/.venv/bin/python"

: "${RUN_DIR:?RUN_DIR required}"
: "${EXPORT_RUN:?EXPORT_RUN required}"
: "${WALK_RUN:?WALK_RUN required}"
WORKERS="${WORKERS:-160}"
FOLD_WORKERS="${FOLD_WORKERS:-10}"
WALK_WORKERS="${WALK_WORKERS:-$(nproc)}"
N_PER_CITY="${N_PER_CITY:-100}"
N_PER_CITY_WEAK="${N_PER_CITY_WEAK:-500}"
COUNTRIES="${COUNTRIES:-USA IND JPN}"
export COUNTRIES N_PER_CITY_WEAK

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

write_progress() {
  local stage="$1"
  local status="${2:-running}"
  STAGE="$stage" STATUS="$status" RUN_DIR="$RUN_DIR" "${PYBIN}" - <<'PY'
import json, os, time
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
path = run / "progress.json"
cur = {}
if path.exists():
    try:
        cur = json.loads(path.read_text())
    except Exception:
        cur = {}
cur.update({
    "status": os.environ["STATUS"],
    "stage": os.environ["STAGE"],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "last_progress_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
tmp = run / "progress.tmp"
tmp.write_text(json.dumps(cur, indent=2) + "\n")
tmp.replace(path)
PY
}

# shellcheck disable=SC2086
set -- ${COUNTRIES}

echo "[$(date -Iseconds)] STAGE export countries=${COUNTRIES}" | tee -a "${RUN_DIR}/job.log"
write_progress "export" "running"
"${PYBIN}" -u "${ROOT}/scripts/export_usa_embedding_globe_sample.py" \
  --n-per-city "${N_PER_CITY}" \
  --n-per-city-weak "${N_PER_CITY_WEAK}" \
  --seed 42 \
  --countries "$@" \
  --workers "${WORKERS}" \
  --fold-workers "${FOLD_WORKERS}" \
  --run-dir "${EXPORT_RUN}" \
  >> "${RUN_DIR}/job.log" 2>&1

echo "[$(date -Iseconds)] STAGE boundary_walks" | tee -a "${RUN_DIR}/job.log"
write_progress "boundary_walks" "running"
"${PYBIN}" -u "${ROOT}/scripts/build_boundary_walks.py" \
  --run-dir "${WALK_RUN}" \
  --workers "${WALK_WORKERS}" \
  >> "${RUN_DIR}/job.log" 2>&1

RUN_DIR="$RUN_DIR" ROOT="$ROOT" "${PYBIN}" - <<'PY'
import json, os, time
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
root = Path(os.environ["ROOT"])
export_sum = {}
walk_sum = {}
ep = run / "export" / "summary.json"
wp = run / "boundary" / "summary.json"
if ep.exists():
    export_sum = json.loads(ep.read_text())
if wp.exists():
    walk_sum = json.loads(wp.read_text())
status = "completed"
if export_sum.get("chips_failed_cells", 0):
    status = "completed_with_failures"
if walk_sum.get("n_failed", 0):
    status = "completed_with_failures"
summary = {
    "status": status,
    "countries": os.environ.get("COUNTRIES", "USA IND JPN").split(),
    "export": export_sum,
    "boundary_walks": walk_sum,
    "points_json": str(root / "viz/usa_embedding_globe/data/points.json"),
    "boundary_walks_json": str(root / "viz/usa_embedding_globe/data/boundary_walks.json"),
    "viewer": str(root / "viz/usa_embedding_globe/index.html"),
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
(run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
prog = {
    "status": status,
    "stage": "done",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "last_progress_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "percent_complete": 100.0,
    "export_n_points": export_sum.get("n_points"),
    "walks_n_ok": walk_sum.get("n_ok"),
    "walks_n_failed": walk_sum.get("n_failed"),
}
tmp = run / "progress.tmp"
tmp.write_text(json.dumps(prog, indent=2) + "\n")
tmp.replace(run / "progress.json")
(run / "DONE").write_text("ok\n" if status.startswith("completed") else "failed\n")
print(json.dumps(summary, indent=2))
PY
