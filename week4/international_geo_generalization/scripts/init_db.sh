#!/usr/bin/env bash
# Start PostgreSQL/PostGIS, wait for health, apply migrations, verify schema.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
DB_DIR="${ROOT}/db"
MIG_DIR="${DB_DIR}/migrations"
ENV_FILE="${DB_DIR}/.env"
PREFIX="${REPO}/.tools/postgis"
RUNTIME_MODE="${DB_RUNTIME:-auto}"  # auto | docker | local

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" >&2; }

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${DB_DIR}/.env.example" ]]; then
    log "Creating ${ENV_FILE} from .env.example"
    cp "${DB_DIR}/.env.example" "${ENV_FILE}"
    # Generate a random password if still placeholder
    if grep -q 'change_me_to_a_strong_password' "${ENV_FILE}"; then
      PW="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
      sed -i "s/change_me_to_a_strong_password/${PW}/" "${ENV_FILE}"
      log "Generated random POSTGRES_PASSWORD in db/.env (not printed)"
    fi
  else
    echo "Missing ${DB_DIR}/.env.example" >&2
    exit 1
  fi
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${POSTGRES_HOST:=127.0.0.1}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_DB:=aef_international}"
: "${POSTGRES_USER:=aef}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in db/.env}"
: "${PGDATA_DIR:=/data/users/kv133/Alpha-Earth-Embeddings-Research/pgdata}"

export PGPASSWORD="${POSTGRES_PASSWORD}"
export PGHOST="${POSTGRES_HOST}"
export PGPORT="${POSTGRES_PORT}"
export PGDATABASE="${POSTGRES_DB}"
export PGUSER="${POSTGRES_USER}"

docker_available() {
  command -v docker >/dev/null 2>&1 && timeout 3 docker info >/dev/null 2>&1
}

PSQL_BIN=""

resolve_psql() {
  if [[ -x "${PREFIX}/bin/psql" ]]; then
    PSQL_BIN="${PREFIX}/bin/psql"
  elif command -v psql >/dev/null 2>&1; then
    PSQL_BIN="$(command -v psql)"
  else
    echo "psql not found" >&2
    exit 1
  fi
}

start_docker() {
  log "Starting PostGIS via Docker Compose"
  (cd "${DB_DIR}" && docker compose up -d)
  resolve_psql
  echo "docker" > "${DB_DIR}/.runtime"
}

start_local() {
  log "Docker unavailable — starting local micromamba PostgreSQL/PostGIS"
  if [[ ! -x "${PREFIX}/bin/pg_ctl" ]]; then
    echo "Missing ${PREFIX}. Run scripts/install_postgis_env.sh first." >&2
    exit 1
  fi
  resolve_psql
  mkdir -p "${PGDATA_DIR}"
  if [[ ! -f "${PGDATA_DIR}/PG_VERSION" ]]; then
    log "Initializing cluster at ${PGDATA_DIR}"
    "${PREFIX}/bin/initdb" -D "${PGDATA_DIR}" -U "${POSTGRES_USER}" \
      --auth-local=trust --auth-host=scram-sha-256 \
      --pwfile=<(printf '%s\n' "${POSTGRES_PASSWORD}") >/dev/null
    cat >> "${PGDATA_DIR}/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1'
port = ${POSTGRES_PORT}
unix_socket_directories = '${PGDATA_DIR}'
shared_buffers = 2GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_worker_processes = 70
max_parallel_workers = 70
max_parallel_workers_per_gather = 8
EOF
    cat > "${PGDATA_DIR}/pg_hba.conf" <<EOF
local   all             all                                     trust
host    all             all             127.0.0.1/32            scram-sha-256
host    all             all             ::1/128                 scram-sha-256
EOF
  fi

  if ! "${PREFIX}/bin/pg_ctl" -D "${PGDATA_DIR}" status >/dev/null 2>&1; then
    log "Starting postgres on port ${POSTGRES_PORT}"
    "${PREFIX}/bin/pg_ctl" -D "${PGDATA_DIR}" -l "${PGDATA_DIR}/postgres.log" start -w
  else
    log "Local postgres already running"
  fi

  # Ensure role/database exist (socket; silence stdout so it cannot pollute callers)
  "${PREFIX}/bin/psql" -h "${PGDATA_DIR}" -p "${POSTGRES_PORT}" -d postgres -v ON_ERROR_STOP=1 >/dev/null <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${POSTGRES_USER}') THEN
    CREATE ROLE ${POSTGRES_USER} LOGIN PASSWORD '${POSTGRES_PASSWORD}' SUPERUSER;
  END IF;
END
\$\$;
SQL
  if ! "${PREFIX}/bin/psql" -h "${PGDATA_DIR}" -p "${POSTGRES_PORT}" -d postgres -tAc \
      "SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB}'" | grep -q 1; then
    "${PREFIX}/bin/createdb" -h "${PGDATA_DIR}" -p "${POSTGRES_PORT}" -O "${POSTGRES_USER}" "${POSTGRES_DB}"
  fi
  echo "local" > "${DB_DIR}/.runtime"
}

case "${RUNTIME_MODE}" in
  docker) start_docker ;;
  local) start_local ;;
  auto)
    if docker_available; then start_docker; else start_local; fi
    ;;
  *)
    echo "Unknown DB_RUNTIME=${RUNTIME_MODE}" >&2
    exit 1
    ;;
esac

log "Waiting for database health at ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB} via ${PSQL_BIN}"
ready=0
for i in $(seq 1 60); do
  if "${PSQL_BIN}" -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
      -c "SELECT 1" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if [[ -S "${PGDATA_DIR}/.s.PGSQL.${POSTGRES_PORT}" ]]; then
    if "${PSQL_BIN}" -h "${PGDATA_DIR}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
        -c "SELECT 1" >/dev/null 2>&1; then
      ready=1
      break
    fi
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "Database did not become ready" >&2
  "${PSQL_BIN}" -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -c "SELECT 1" >&2 || true
  if [[ -f "${PGDATA_DIR}/postgres.log" ]]; then
    tail -50 "${PGDATA_DIR}/postgres.log" >&2 || true
  fi
  exit 1
fi

log "Applying migrations from ${MIG_DIR}"
for sql in $(ls -1 "${MIG_DIR}"/*.sql | sort); do
  log "  -> $(basename "${sql}")"
  "${PSQL_BIN}" -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -v ON_ERROR_STOP=1 -f "${sql}" >/dev/null
done

log "Verifying PostGIS + schema"
"${PSQL_BIN}" -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT extname, extversion FROM pg_extension WHERE extname LIKE 'postgis%' ORDER BY 1;
SELECT version FROM schema_migrations ORDER BY version;
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
    'countries','cities','city_candidates','buildings','alphaearth_cells',
    'cell_building_intersections','cell_labels','morphology_features',
    'experiment_splits','experiment_runs','experiment_metrics','data_sources'
  )
ORDER BY 1;
SELECT PostGIS_Full_Version();
SQL

# Refresh consolidated schema reference
"${PSQL_BIN}" -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  -c "\dt+" > "${DB_DIR}/schema.sql.runtime.txt" || true

log "Connection OK: host=${POSTGRES_HOST} port=${POSTGRES_PORT} db=${POSTGRES_DB} user=${POSTGRES_USER}"
log "Runtime: $(cat "${DB_DIR}/.runtime" 2>/dev/null || echo "${RUNTIME_MODE}")"
log "init_db complete"
unset PGPASSWORD
