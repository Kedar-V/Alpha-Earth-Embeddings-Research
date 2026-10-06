#!/usr/bin/env bash
# Install a user-space PostgreSQL 16 + PostGIS 3.4 via micromamba.
# Used when Docker daemon access is unavailable.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${RUN_DIR:-$(cat /tmp/postgis_env_run_dir.txt 2>/dev/null || true)}"
MM="${REPO}/.tools/micromamba"
PREFIX="${REPO}/.tools/postgis"
export MAMBA_ROOT_PREFIX="${REPO}/.tools/mamba"
export OMP_NUM_THREADS="${CPUS:-70}"
export MAMBA_EXTRACT_THREADS="${CPUS:-70}"

update_progress() {
  local stage="$1" completed="$2" status="$3"
  python3 - "$RUN_DIR" "$stage" "$completed" "$status" <<'PY'
import json, sys, time
from pathlib import Path
run, stage, completed, status = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
p = run / "progress.json"
data = json.loads(p.read_text()) if p.exists() else {}
data.update({
    "status": status,
    "stage": stage,
    "completed": completed,
    "total": 3,
    "percent_complete": round(100.0 * completed / 3, 2),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "last_progress_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "prefix": str(Path(sys.argv[0]).resolve()) if False else data.get("prefix"),
})
tmp = run / "progress.tmp"
tmp.write_text(json.dumps(data, indent=2))
tmp.replace(p)
PY
}

if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR unset" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Creating micromamba env at ${PREFIX}"
update_progress micromamba_create 0 running

"$MM" create -y -p "$PREFIX" -c conda-forge \
  "postgresql=16" "postgis=3.4" "psycopg=3" "python=3.11"

update_progress micromamba_create 1 running
echo "[$(date -Iseconds)] Env created"

# Sanity: binaries exist
"$PREFIX/bin/psql" --version
"$PREFIX/bin/postgres" --version
"$PREFIX/bin/python" -c "import psycopg; print('psycopg', psycopg.__version__)"

update_progress ready_for_init 2 running

python3 - "$RUN_DIR" "$PREFIX" <<'PY'
import json, time, sys
from pathlib import Path
run, prefix = Path(sys.argv[1]), sys.argv[2]
summary = {
    "status": "completed",
    "stage": "micromamba_env_ready",
    "prefix": prefix,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
(run / "summary.json").write_text(json.dumps(summary, indent=2))
(run / "DONE").write_text("ok\n")
p = json.loads((run / "progress.json").read_text())
p.update(summary)
p["completed"] = 3
p["percent_complete"] = 100.0
p["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
tmp = run / "progress.tmp"
tmp.write_text(json.dumps(p, indent=2))
tmp.replace(run / "progress.json")
print("DONE", run)
PY

echo 0 > "${RUN_DIR}/exit_code.txt"
echo "[$(date -Iseconds)] install_postgis_env complete"
