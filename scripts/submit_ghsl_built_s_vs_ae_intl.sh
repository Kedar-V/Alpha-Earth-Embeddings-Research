#!/usr/bin/env bash
# Submit GHS-BUILT-S vs AE intl compare with submit-job observability.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load DB env (password stays in env, not argv)
set -a
# shellcheck disable=SC1091
source "$ROOT/week4/international_geo_generalization/db/.env"
set +a
export PGDATA_DIR="${PGDATA_DIR:-$ROOT/pgdata}"

# Ensure local Postgres from workspace pgdata/
if ! "$ROOT/.tools/postgis/bin/pg_ctl" -D "$PGDATA_DIR" status >/dev/null 2>&1; then
  echo "[submit] starting postgres from $PGDATA_DIR"
  DB_RUNTIME=local "$ROOT/week4/international_geo_generalization/scripts/init_db.sh" >/tmp/init_db_intl.submit.log 2>&1 || {
    # init_db may hang on migrations after ready; pg_ctl status is enough
    "$ROOT/.tools/postgis/bin/pg_ctl" -D "$PGDATA_DIR" status
  }
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${1:-$ROOT/outputs/ghsl_built_s_vs_ae_intl/${TS}}"
mkdir -p "$RUN_DIR"/{checkpoints,artifacts,figures,failures}

COUNTRIES="${COUNTRIES:-IND,JPN}"
echo "[submit] run_dir=$RUN_DIR countries=$COUNTRIES cpus=$(nproc)"
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/eval_ghsl_built_s_vs_ae_intl.py" \
  --run-dir "$RUN_DIR" \
  --countries "$COUNTRIES" \
  >"$RUN_DIR/stdout.log" 2>&1 &
PID=$!
echo "$PID" >"$RUN_DIR/job.pid"
echo "cmd=eval_ghsl_built_s_vs_ae_intl.py --run-dir $RUN_DIR --countries $COUNTRIES" >"$RUN_DIR/launch.txt"
echo "pid=$PID" >>"$RUN_DIR/launch.txt"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_DIR/launch.txt"
echo "submit_job=1" >>"$RUN_DIR/launch.txt"
echo "[submit] PID=$PID"
echo "[submit] log=$RUN_DIR/job.log"
echo "[submit] progress=$RUN_DIR/progress.json"
sleep 3
if kill -0 "$PID" 2>/dev/null; then
  echo "[submit] process alive"
  tail -n 15 "$RUN_DIR/stdout.log" || true
else
  echo "[submit] process died — see $RUN_DIR/stdout.log" >&2
  exit 1
fi
