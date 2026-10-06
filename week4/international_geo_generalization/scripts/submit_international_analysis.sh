#!/usr/bin/env bash
# Submit week4-style USA/India/Japan transfer analysis with observability.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/intl_transfer_${STAMP}"
EXTRACT_DIR="${RUN_DIR}/extract"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures,extract}

export RUN_DIR EXTRACT_DIR
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export WORKERS="${WORKERS:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Analysis is CPU logistic/PCA/PLS — leave GPUs free
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: international_week4_transfer"
  echo "countries: USA IND JPN"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS}"
  echo "gpus: none (CPU analysis)"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "extract_dir: ${EXTRACT_DIR}"
  echo "protocol: pure_exclusive_clear + per-city 1:1 seed42; LORO + country/city transfer"
} > "${RUN_DIR}/launch.txt"

python3 - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, subprocess
print("host:", platform.node())
print("cpus:", os.cpu_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    print(subprocess.check_output(["nvidia-smi","-L"], text=True))
except Exception as e:
    print("nvidia-smi:", e)
print("free:")
print(subprocess.check_output(["free","-h"], text=True))
PY

PYBIN="${REPO}/.venv/bin/python"
if [[ ! -x "${PYBIN}" ]]; then
  PYBIN="${REPO}/.tools/postgis/bin/python"
fi

EXTRA_ANAL=()
if [[ "${SKIP_CITY_TRANSFER:-0}" == "1" ]]; then
  EXTRA_ANAL+=(--skip-city-transfer)
fi
MAX_PER_CLASS_ARGS=()
if [[ -n "${MAX_PER_CLASS:-}" ]]; then
  MAX_PER_CLASS_ARGS+=(--max-per-class "${MAX_PER_CLASS}")
fi

# Wrapper: export then analyze; write progress throughout
cat > "${RUN_DIR}/run_pipeline.py" <<'PY'
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
extract_dir = Path(os.environ["EXTRACT_DIR"])
repo = Path(os.environ["REPO"])
py = os.environ["PYBIN"]
workers = os.environ.get("WORKERS", "20")
root = repo / "week4" / "international_geo_generalization"
log = run_dir / "job.log"

def progress(**kw):
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prev, indent=2) + "\n")
    os.replace(tmp, path)

def run(cmd: list[str]) -> None:
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

progress(status="running", stage="export", started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
         output_location=str(run_dir), resource_allocation={"cpus": os.environ.get("CPUS"), "workers": workers, "gpus": 0})

export_cmd = [
    py, "-u", str(root / "scripts" / "export_balanced_extract.py"),
    "--out-dir", str(extract_dir),
    "--run-dir", str(run_dir),
    "--countries", "USA", "IND", "JPN",
]
export_cmd += json.loads(os.environ.get("MAX_PER_CLASS_JSON", "[]"))
run(export_cmd)

progress(status="running", stage="analysis")
anal_cmd = [
    py, "-u", str(root / "scripts" / "run_international_analysis.py"),
    "--run-dir", str(run_dir),
    "--extract-dir", str(extract_dir),
    "--workers", workers,
]
anal_cmd += json.loads(os.environ.get("EXTRA_ANAL_JSON", "[]"))
run(anal_cmd)
print("pipeline complete", flush=True)
PY

export REPO PYBIN
export MAX_PER_CLASS_JSON="$(python3 -c 'import json,os; print(json.dumps(["--max-per-class", os.environ["MAX_PER_CLASS"]] if os.environ.get("MAX_PER_CLASS") else []))')"
export EXTRA_ANAL_JSON="$(python3 -c 'import json,os; print(json.dumps(["--skip-city-transfer"] if os.environ.get("SKIP_CITY_TRANSFER")=="1" else []))')"

nohup env RUN_DIR="${RUN_DIR}" EXTRACT_DIR="${EXTRACT_DIR}" REPO="${REPO}" PYBIN="${PYBIN}" \
  CPUS="${CPUS}" WORKERS="${WORKERS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  MAX_PER_CLASS_JSON="${MAX_PER_CLASS_JSON}" EXTRA_ANAL_JSON="${EXTRA_ANAL_JSON}" \
  "${PYBIN}" -u "${RUN_DIR}/run_pipeline.py" \
  >> "${RUN_DIR}/job.log" 2>&1 &
echo $! | tee "${RUN_DIR}/job.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
echo "Results will be at: ${RUN_DIR}/RESULTS.md"
