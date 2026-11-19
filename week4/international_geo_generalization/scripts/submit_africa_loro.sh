#!/usr/bin/env bash
# Africa-5 balanced extract + city LORO (week4-comparable) after ingest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/africa_loro_${STAMP}"
EXTRACT_DIR="${RUN_DIR}/extract"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures,extract}

export RUN_DIR EXTRACT_DIR REPO
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export WORKERS="${WORKERS:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export SKIP_CITY_TRANSFER="${SKIP_CITY_TRANSFER:-0}"
export COUNTRIES="${COUNTRIES:-EGY,GHA,KEN,NGA,ZAF}"

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: africa_loro_5cities"
  echo "countries: ${COUNTRIES}"
  echo "cpus: ${CPUS}"
  echo "workers: ${WORKERS}"
  echo "gpus: none"
  echo "skip_city_transfer: ${SKIP_CITY_TRANSFER}"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
  echo "protocol: pure_exclusive_clear + per-city 1:1 seed42; Africa LORO + within-Africa city transfer"
} > "${RUN_DIR}/launch.txt"

python3 - <<PY > "${RUN_DIR}/environment.txt"
import os, platform, subprocess
print("host:", platform.node())
print("cpus:", os.cpu_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
except Exception as e:
    print("nvidia-smi:", e)
print("free:")
print(subprocess.check_output(["free", "-h"], text=True))
PY

PYBIN="${REPO}/.venv/bin/python"
cat > "${RUN_DIR}/run_pipeline.py" <<'PY'
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
extract_dir = Path(os.environ["EXTRACT_DIR"])
repo = Path(os.environ["REPO"])
py = os.environ.get("PYBIN", str(repo / ".venv" / "bin" / "python"))
workers = os.environ.get("WORKERS", "20")
countries = os.environ.get("COUNTRIES", "EGY,GHA,KEN,NGA,ZAF")
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
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev["last_progress_at"] = prev["updated_at"]
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

progress(
    status="running",
    stage="export",
    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    output_location=str(run_dir),
    countries=countries,
    resource_allocation={"cpus": os.environ.get("CPUS"), "workers": workers, "gpus": 0},
)
(run_dir / "job.pid").write_text(str(os.getpid()) + "\n")

# export_balanced_extract --countries is nargs=+; split comma lists into tokens.
country_list = [c.strip() for c in countries.replace(",", " ").split() if c.strip()]
export_cmd = [
    py, "-u", str(root / "scripts" / "export_balanced_extract.py"),
    "--out-dir", str(extract_dir),
    "--run-dir", str(run_dir),
    "--countries", *country_list,
]
run(export_cmd)

progress(status="running", stage="analysis")
anal_cmd = [
    py, "-u", str(root / "scripts" / "run_international_analysis.py"),
    "--run-dir", str(run_dir),
    "--extract-dir", str(extract_dir),
    "--workers", workers,
]
if os.environ.get("SKIP_CITY_TRANSFER") == "1":
    anal_cmd.append("--skip-city-transfer")
run(anal_cmd)
progress(status="completed", stage="done", percent_complete=100.0)
(run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
print("pipeline complete", flush=True)
PY

nohup env RUN_DIR="${RUN_DIR}" EXTRACT_DIR="${EXTRACT_DIR}" REPO="${REPO}" \
  PYBIN="${PYBIN}" WORKERS="${WORKERS}" CPUS="${CPUS}" COUNTRIES="${COUNTRIES}" \
  SKIP_CITY_TRANSFER="${SKIP_CITY_TRANSFER}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYBIN}" -u "${RUN_DIR}/run_pipeline.py" \
  > "${RUN_DIR}/wrapper.log" 2>&1 &
echo $! | tee "${RUN_DIR}/wrapper.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
