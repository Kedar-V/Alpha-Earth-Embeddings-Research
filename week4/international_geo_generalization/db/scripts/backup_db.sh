#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
ENV_FILE="${ROOT}/db/.env"
PREFIX="${REPO}/.tools/postgis"
OUT_DIR="${1:-${ROOT}/outputs/backups}"
mkdir -p "${OUT_DIR}"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
export PGPASSWORD="${POSTGRES_PASSWORD}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR}/aef_international_${STAMP}.dump"
PG_DUMP="${PREFIX}/bin/pg_dump"
command -v "${PG_DUMP}" >/dev/null 2>&1 || PG_DUMP="pg_dump"

"${PG_DUMP}" \
  -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
  -Fc --no-owner --no-acl \
  -f "${OUT}"

echo "Wrote ${OUT}"
ls -lh "${OUT}"
unset PGPASSWORD
