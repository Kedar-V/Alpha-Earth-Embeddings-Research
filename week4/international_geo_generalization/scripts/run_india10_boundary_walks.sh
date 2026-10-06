#!/usr/bin/env bash
# Fit India-10 PLS3 LORO folds → export points → boundary walks → REPORT.md
set -euo pipefail

ROOT="${ROOT:-/home/kv133/Alpha-Earth-Embeddings-Research/week4/international_geo_generalization}"
PY="${PY:-/home/kv133/Alpha-Earth-Embeddings-Research/.venv/bin/python}"
EXTRACT="${EXTRACT:-$ROOT/runs/india_loro_20261005_202600/extract}"
LORO_DIR="${LORO_DIR:-$ROOT/runs/india_loro_20261005_202600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/viz/usa_embedding_globe/runs/india10_walks_${STAMP}}"
CPUS="$(nproc)"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set -a
# shellcheck disable=SC1091
source "$ROOT/db/.env"
set +a

mkdir -p "$RUN_DIR"/{checkpoints,logs,artifacts,failures}
echo "$$" > "$RUN_DIR/job.pid"
{
  echo "stamp=$STAMP"
  echo "extract=$EXTRACT"
  echo "run_dir=$RUN_DIR"
} > "$RUN_DIR/launch.txt"
{
  echo "hostname=$(hostname)"
  echo "date=$(date -Iseconds)"
  echo "cpus=$CPUS"
  echo "python=$PY"
} > "$RUN_DIR/environment.txt"

prog() {
  local stage="$1"
  local pct="$2"
  "$PY" - <<PY
import json, time
from pathlib import Path
p = Path("$RUN_DIR") / "progress.json"
d = json.loads(p.read_text()) if p.exists() else {}
d.update({
    "status": "running",
    "stage": "$stage",
    "percent_complete": float($pct),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "last_progress_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
if "started_at" not in d:
    d["started_at"] = d["updated_at"]
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

log() { echo "[$(date -Iseconds)] $*" | tee -a "$RUN_DIR/job.log"; }

prog starting 0
log "STAGE fit_folds IND×10 extract=$EXTRACT"
prog fit_folds 5
"$PY" -u "$ROOT/scripts/intl_globe_folds.py" \
  --countries IND --force --workers 10 --n-pls 32 \
  --extract-dir "$EXTRACT" \
  --out-dir "$ROOT/viz/usa_embedding_globe/data/intl_folds" \
  >> "$RUN_DIR/job.log" 2>&1

log "STAGE export IND points"
prog export 35
POINTS_OUT="$RUN_DIR/artifacts/points_india10.json"
"$PY" -u "$ROOT/scripts/export_usa_embedding_globe_sample.py" \
  --countries IND --n-per-city 200 --n-per-city-weak 500 --seed 42 \
  --skip-imagery \
  --extract-dir "$EXTRACT" \
  --intl-fold-dir "$ROOT/viz/usa_embedding_globe/data/intl_folds" \
  --fold-workers 10 \
  --out-points "$POINTS_OUT" \
  --run-dir "$RUN_DIR/export" \
  >> "$RUN_DIR/job.log" 2>&1

log "STAGE boundary_walks"
prog boundary_walks 70
WALK_RUN="$RUN_DIR/boundary"
mkdir -p "$WALK_RUN"/{checkpoints,logs,artifacts,failures}
WALKS_OUT="$RUN_DIR/artifacts/boundary_walks_india10.json"
"$PY" -u "$ROOT/scripts/build_boundary_walks.py" \
  --run-dir "$WALK_RUN" --workers "$CPUS" \
  --points "$POINTS_OUT" \
  --out "$WALKS_OUT" \
  --intl-fold-dir "$ROOT/viz/usa_embedding_globe/data/intl_folds" \
  >> "$RUN_DIR/job.log" 2>&1

log "STAGE report+merge"
prog report 90
"$PY" -u "$ROOT/scripts/report_india10_boundary_walks.py" \
  --run-dir "$RUN_DIR" \
  --loro-dir "$LORO_DIR" \
  --data-dir "$ROOT/viz/usa_embedding_globe/data" \
  --merge-globe \
  >> "$RUN_DIR/job.log" 2>&1

log "DONE"
