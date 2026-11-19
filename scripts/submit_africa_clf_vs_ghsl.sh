#!/usr/bin/env bash
# Submit Africa AE vs GHS-BUILT-S classification F1 compare (submit-job).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-$ROOT/outputs/africa_clf_vs_ghsl/${TS}}"
mkdir -p "$RUN_DIR"/{checkpoints,artifacts,figures,failures,logs}

SCORED="${SCORED:-$ROOT/outputs/africa_loro_vs_ghsl/20261113_125810/artifacts/scored.parquet}"
INTL="${INTL:-$ROOT/outputs/ghsl_built_s_vs_ae_intl/africa_20261113_020114/artifacts/scored.parquet}"
NCPU="$(nproc)"

{
  echo "job: africa_clf_vs_ghsl"
  echo "run_dir: ${RUN_DIR}"
  echo "scored: ${SCORED}"
  echo "intl_scored: ${INTL}"
  echo "cpus: ${NCPU} (all cores)"
  echo "gpus: 0 (eval-only; reuses existing OOF + BUILT-S samples)"
  echo "started: $(date -Iseconds)"
  echo "protocol: RES/NRES F1 · AE XGB clf / soft-argmax heads vs GHS-BUILT-S soft-argmax"
} > "${RUN_DIR}/launch.txt"

echo "[submit] run_dir=$RUN_DIR cpus=$NCPU"
nohup "$ROOT/.venv/bin/python" -u "$ROOT/scripts/compare_africa_clf_vs_ghsl.py" \
  --run-dir "$RUN_DIR" \
  --scored "$SCORED" \
  --intl-scored "$INTL" \
  --workers "$NCPU" \
  >"${RUN_DIR}/stdout.log" 2>&1 &

echo $! >"${RUN_DIR}/job.pid"
echo "[submit] PID=$(cat "${RUN_DIR}/job.pid")"
echo "[submit] log=${RUN_DIR}/job.log"
echo "[submit] progress=${RUN_DIR}/progress.json"
echo "[submit] stdout=${RUN_DIR}/stdout.log"
