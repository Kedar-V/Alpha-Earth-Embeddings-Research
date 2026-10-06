#!/usr/bin/env bash
# Usage: restore_db.sh /path/to/aef_international_YYYYMMDD_HHMMSS.dump
set -euo pipefail
DUMP="${1:?Provide path to .dump file}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
ENV_FILE="${ROOT}/db/.env"
PREFIX="${REPO}/.tools/postgis"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export PGPASSWORD="${POSTGRES_PASSWORD}"

PG_RESTORE="${PREFIX}/bin/pg_restore"
command -v "${PG_RESTORE}" >/dev/null 2>&1 || PG_RESTORE="pg_restore"

echo "Restoring ${DUMP} into ${POSTGRES_DB} (existing objects may be overwritten with --clean)"
"${PG_RESTORE}" \
  -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  --clean --if-exists --no-owner --no-acl \
  "${DUMP}"

echo "Restore complete"
unset PGPASSWORD
