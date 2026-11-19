#!/usr/bin/env bash
# Wait for africa_ingest RUN_DIR, then LORO + GHSL-BUILT-S vs AE intl eval.
set -euo pipefail

INGEST_DIR="${1:?usage: run_africa_after_ingest.sh <africa_ingest_run_dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
PY="${REPO}/.venv/bin/python"
CHAIN_DIR="${ROOT}/runs/africa_chain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CHAIN_DIR}"/{checkpoints,logs,artifacts,failures}
echo "$$" > "${CHAIN_DIR}/job.pid"
{
  echo "ingest=${INGEST_DIR}"
  echo "chain_dir=${CHAIN_DIR}"
  echo "started=$(date -Iseconds)"
} > "${CHAIN_DIR}/launch.txt"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${CHAIN_DIR}/job.log"; }

prog() {
  "$PY" - <<PY
import json, time
from pathlib import Path
p = Path("${CHAIN_DIR}") / "progress.json"
d = json.loads(p.read_text()) if p.exists() else {}
d.update(status="running", stage="$1", percent_complete=float($2),
         updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         last_progress_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
  sleep 30
done

prog loro 40
log "submitting africa LORO"
# Run LORO inline (not nested nohup) so chain owns completion
set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a
LORO_DIR="${ROOT}/runs/africa_loro_chain_$(date +%Y%m%d_%H%M%S)"
EXTRACT_DIR="${LORO_DIR}/extract"
mkdir -p "${LORO_DIR}"/{checkpoints,logs,artifacts,failures,extract}
export RUN_DIR="${LORO_DIR}" EXTRACT_DIR REPO
export COUNTRIES="${COUNTRIES:-EGY,GHA,KEN,NGA,ZAF}"
export WORKERS="${WORKERS:-20}"
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export SKIP_CITY_TRANSFER="${SKIP_CITY_TRANSFER:-0}"
echo "loro_dir=${LORO_DIR}" >> "${CHAIN_DIR}/launch.txt"

"$PY" -u - <<'PY'
import json, os, subprocess, time
from pathlib import Path
run_dir = Path(os.environ["RUN_DIR"])
extract_dir = Path(os.environ["EXTRACT_DIR"])
repo = Path(os.environ["REPO"])
py = str(repo / ".venv" / "bin" / "python")
root = repo / "week4" / "international_geo_generalization"
countries = os.environ["COUNTRIES"]
workers = os.environ.get("WORKERS", "20")
log = run_dir / "job.log"

def progress(**kw):
    path = run_dir / "progress.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(kw)
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev["last_progress_at"] = prev["updated_at"]
    path.write_text(json.dumps(prev, indent=2) + "\n")

def run(cmd):
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n"); f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

progress(status="running", stage="export", started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
# export_balanced_extract --countries is nargs=+; split comma lists into tokens.
country_list = [c.strip() for c in countries.replace(",", " ").split() if c.strip()]
run([py, "-u", str(root / "scripts" / "export_balanced_extract.py"),
     "--out-dir", str(extract_dir), "--run-dir", str(run_dir), "--countries", *country_list])
progress(stage="analysis")
cmd = [py, "-u", str(root / "scripts" / "run_international_analysis.py"),
       "--run-dir", str(run_dir), "--extract-dir", str(extract_dir), "--workers", workers]
if os.environ.get("SKIP_CITY_TRANSFER") == "1":
    cmd.append("--skip-city-transfer")
run(cmd)
progress(status="completed", stage="done", percent_complete=100.0)
(run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
PY
echo "loro=${LORO_DIR}" >> "${CHAIN_DIR}/launch.txt"
log "LORO done → ${LORO_DIR}"

prog ghsl 75
GHSL_DIR="${REPO}/outputs/ghsl_built_s_vs_ae_intl/africa_$(date +%Y%m%d_%H%M%S)"
log "GHSL vs AE intl → ${GHSL_DIR}"
COUNTRIES="${COUNTRIES}" bash "${REPO}/scripts/submit_ghsl_built_s_vs_ae_intl.sh" "${GHSL_DIR}"
# wait for GHSL job
GHSL_PID="$(cat "${GHSL_DIR}/job.pid")"
log "waiting GHSL PID=${GHSL_PID}"
while kill -0 "${GHSL_PID}" 2>/dev/null; do sleep 30; done
if [[ -f "${GHSL_DIR}/DONE" ]] || [[ -f "${GHSL_DIR}/summary.json" ]]; then
  log "GHSL finished"
else
  log "GHSL process exited; check ${GHSL_DIR}"
fi
echo "ghsl=${GHSL_DIR}" >> "${CHAIN_DIR}/launch.txt"

prog done 100
echo ok > "${CHAIN_DIR}/DONE"
log "CHAIN COMPLETE ingest=${INGEST_DIR} loro=${LORO_DIR} ghsl=${GHSL_DIR}"
