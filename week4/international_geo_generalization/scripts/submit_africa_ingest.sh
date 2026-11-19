#!/usr/bin/env bash
# Ingest selected African cities → Postgres.
# Defaults: EGY/NGA/ZAF with 5 cities each; skips cities already complete in DB
# unless CITY_IDS is set explicitly.
#
# Env overrides:
#   COUNTRIES=EGY,NGA,ZAF
#   CITY_IDS=EGY_ALEXANDRIA,NGA_ABUJA,...   # optional subset
#   SKIP_EXISTING=1 (default) — skip cities with labels already in DB
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${ROOT}/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT}/runs/africa_ingest_${STAMP}"
mkdir -p "${RUN_DIR}"/{checkpoints,logs,artifacts,failures}

export RUN_DIR REPO ROOT
export CPUS="${CPUS:-$(( $(nproc) - 2 ))}"
export GPUS="${GPUS:-all}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export DISCOVERY_RADIUS_M="${DISCOVERY_RADIUS_M:-40000}"
export RING_KM="${RING_KM:-5}"
export AREA_HEADROOM="${AREA_HEADROOM:-1.10}"
export TARGET_AREA_M2="${TARGET_AREA_M2:-24604226.98057363}"
export AEF_SHARD_SIZE="${AEF_SHARD_SIZE:-1000}"
export PARALLEL_DISCOVERY="${PARALLEL_DISCOVERY:-8}"
export PARALLEL_AEF="${PARALLEL_AEF:-2}"
export WORKERS="${WORKERS:-$(( CPUS > 2 ? CPUS - 2 : CPUS ))}"
export COUNTRIES="${COUNTRIES:-EGY,NGA,ZAF}"
export CITY_IDS="${CITY_IDS:-}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export MIN_LABELS_EXISTING="${MIN_LABELS_EXISTING:-1000}"

if command -v nvidia-smi >/dev/null 2>&1; then
  export CUDA_VISIBLE_DEVICES="$(nvidia-smi -L | sed -n 's/GPU \([0-9]*\):.*/\1/p' | paste -sd, -)"
fi

set -a
# shellcheck disable=SC1091
source "${ROOT}/db/.env"
set +a

{
  echo "job: africa_ingest_to_postgres"
  echo "countries: ${COUNTRIES}"
  echo "city_ids: ${CITY_IDS:-<selected for countries>}"
  echo "skip_existing: ${SKIP_EXISTING} (min_labels=${MIN_LABELS_EXISTING})"
  echo "cpus: ${CPUS}"
  echo "gpus: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
  echo "workers: ${WORKERS}"
  echo "discovery_radius_m: ${DISCOVERY_RADIUS_M}"
  echo "ring_km: ${RING_KM}"
  echo "area_headroom: ${AREA_HEADROOM}"
  echo "target_area_m2: ${TARGET_AREA_M2}"
  echo "aef_shard_size: ${AEF_SHARD_SIZE}"
  echo "parallel_discovery: ${PARALLEL_DISCOVERY}"
  echo "parallel_aef: ${PARALLEL_AEF}"
  echo "started: $(date -Iseconds)"
  echo "run_dir: ${RUN_DIR}"
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
try:
    print(subprocess.check_output(["free","-h"], text=True))
except Exception as e:
    print(e)
PY

PYBIN="${REPO}/.venv/bin/python"
cat > "${RUN_DIR}/run_africa_ingest.py" <<'PY'
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
root = Path(os.environ["ROOT"])
repo = Path(os.environ["REPO"])
py = str(repo / ".venv" / "bin" / "python")
sys.path.insert(0, str(root))
from lib.db import connect, wait_until_ready  # noqa: E402

countries = [c.strip() for c in os.environ.get("COUNTRIES", "EGY,NGA,ZAF").split(",") if c.strip()]
city_ids_env = [c.strip() for c in os.environ.get("CITY_IDS", "").split(",") if c.strip()]
skip_existing = os.environ.get("SKIP_EXISTING", "1") not in ("0", "false", "False")
min_labels = int(os.environ.get("MIN_LABELS_EXISTING", "1000"))
log = run_dir / "job.log"
manifest_path = run_dir / "manifest.jsonl"

def progress(**kw):
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kw)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev["updated_at"] = now
    prev["last_progress_at"] = now
    if "_t0" in prev and "completed" in prev and "total" in prev and int(prev["total"]) > 0:
        elapsed = max(1e-6, time.time() - float(prev["_t0"]))
        done = int(prev["completed"])
        total = int(prev["total"])
        rate = done / elapsed
        prev["elapsed_seconds"] = round(elapsed, 1)
        prev["rate_per_second"] = round(rate, 4)
        prev["percent_complete"] = round(100.0 * done / total, 2)
        prev["eta_seconds"] = round(max(0, total - done) / rate, 1) if rate > 0 else None
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prev, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def logline(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with log.open("a") as f:
        f.write(line + "\n")

def append_manifest(row: dict) -> None:
    with manifest_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

wait_until_ready(timeout_s=60)

# Resolve selected cities per country
by_country: dict[str, list[str]] = {c: [] for c in countries}
with connect() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT city_id, country_id FROM cities
            WHERE country_id = ANY(%s) AND inclusion_status = 'selected'
            ORDER BY country_id, city_id
            """,
            (countries,),
        )
        rows = list(cur.fetchall())
        for r in rows:
            by_country.setdefault(r["country_id"], []).append(r["city_id"])

        if city_ids_env:
            want = set(city_ids_env)
            for cid in list(by_country):
                by_country[cid] = [x for x in by_country[cid] if x in want]

        already_done: set[str] = set()
        if skip_existing:
            all_ids = [x for v in by_country.values() for x in v]
            if all_ids:
                cur.execute(
                    """
                    SELECT a.city_id, COUNT(*) AS n_lab
                    FROM cell_labels cl
                    JOIN alphaearth_cells a ON a.cell_id = cl.cell_id
                    WHERE a.city_id = ANY(%s)
                    GROUP BY a.city_id
                    """,
                    (all_ids,),
                )
                for r in cur.fetchall():
                    if int(r["n_lab"]) >= min_labels:
                        already_done.add(r["city_id"])

pending_by_country = {
    cid: [x for x in ids if x not in already_done]
    for cid, ids in by_country.items()
}
pending_by_country = {cid: ids for cid, ids in pending_by_country.items() if ids}
all_pending = [x for v in pending_by_country.values() for x in v]
all_selected = [x for v in by_country.values() for x in v]

t0 = time.time()
(run_dir / "job.pid").write_text(str(os.getpid()) + "\n")
progress(
    status="running",
    stage="init",
    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    _t0=t0,
    completed=0,
    total=len(all_pending),
    countries=countries,
    cities_selected=all_selected,
    cities_pending=all_pending,
    cities_skipped_existing=sorted(already_done),
    failed=0,
    success=0,
    output_location=str(run_dir),
    resource_allocation={
        "cpus": os.environ.get("CPUS"),
        "workers": os.environ.get("WORKERS"),
        "gpus": os.environ.get("GPUS"),
        "discovery_radius_m": os.environ.get("DISCOVERY_RADIUS_M"),
        "parallel_discovery": os.environ.get("PARALLEL_DISCOVERY"),
        "parallel_aef": os.environ.get("PARALLEL_AEF"),
    },
)
logline(
    f"plan countries={countries} selected={len(all_selected)} "
    f"skip_existing={sorted(already_done)} pending={all_pending}"
)

if not all_pending:
    summary = {
        "status": "completed",
        "message": "nothing to ingest — all selected cities already present",
        "countries": countries,
        "cities_skipped_existing": sorted(already_done),
        "runtime_seconds": 0.0,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    progress(status="completed", stage="done", percent_complete=100.0, completed=0, total=0)
    logline("DONE — nothing pending")
    raise SystemExit(0)

failed = 0
success = 0
done = 0
for i, (country_id, city_list) in enumerate(pending_by_country.items()):
    progress(
        stage=f"ingest_{country_id}",
        current_country=country_id,
        current_cities=city_list,
        completed=done,
        failed=failed,
        success=success,
    )
    logline(f"INGEST start country={country_id} cities={city_list} ({i+1}/{len(pending_by_country)})")
    child = run_dir / "countries" / country_id
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (child / sub).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RUN_DIR"] = str(child)
    cmd = [
        py, "-u", str(root / "scripts" / "ingest_usa_to_postgres.py"),
        "--country-id", country_id,
        "--discovery-radius-m", os.environ["DISCOVERY_RADIUS_M"],
        "--ring-km", os.environ["RING_KM"],
        "--area-headroom", os.environ["AREA_HEADROOM"],
        "--target-area-m2", os.environ["TARGET_AREA_M2"],
        "--aef-shard-size", os.environ["AEF_SHARD_SIZE"],
        "--parallel-discovery", os.environ["PARALLEL_DISCOVERY"],
        "--parallel-aef", os.environ["PARALLEL_AEF"],
        "--workers", os.environ["WORKERS"],
    ]
    for cid in city_list:
        cmd.extend(["--city-id", cid])
        append_manifest({
            "item_id": cid,
            "country_id": country_id,
            "status": "started",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    t_country = time.time()
    try:
        with log.open("a") as f:
            f.write(f"\n$ RUN_DIR={child} {' '.join(cmd)}\n")
            f.flush()
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if p.returncode != 0:
            raise SystemExit(p.returncode)
        success += len(city_list)
        done += len(city_list)
        logline(f"INGEST ok country={country_id} n_cities={len(city_list)}")
        (run_dir / "checkpoints" / f"country_{country_id}.DONE").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
        )
        for cid in city_list:
            append_manifest({
                "item_id": cid,
                "country_id": country_id,
                "status": "done",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": round(time.time() - t_country, 1),
            })
    except SystemExit as e:
        failed += len(city_list)
        done += len(city_list)
        logline(f"INGEST FAIL country={country_id} rc={getattr(e, 'code', e)}")
        (run_dir / "failures" / f"country_{country_id}.txt").write_text(
            f"rc={getattr(e, 'code', e)}\nchild={child}\ncities={city_list}\n"
        )
        for cid in city_list:
            append_manifest({
                "item_id": cid,
                "country_id": country_id,
                "status": "failed",
                "error": f"rc={getattr(e, 'code', e)}",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    progress(completed=done, failed=failed, success=success, last_country=country_id)

summary = {
    "status": "completed" if failed == 0 else "completed_with_failures",
    "countries": countries,
    "cities_pending": all_pending,
    "cities_skipped_existing": sorted(already_done),
    "success": success,
    "failed": failed,
    "runtime_seconds": round(time.time() - t0, 1),
    "run_dir": str(run_dir),
    "manifest": str(manifest_path),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
progress(
    status=summary["status"],
    stage="done",
    percent_complete=100.0,
    completed=len(all_pending),
    total=len(all_pending),
    success=success,
    failed=failed,
)
if failed == 0:
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
else:
    (run_dir / "FAILED").write_text(json.dumps(summary) + "\n")
    raise SystemExit(2)
logline(f"DONE africa ingest · success={success} failed={failed} · {run_dir}")
PY

nohup env RUN_DIR="${RUN_DIR}" ROOT="${ROOT}" REPO="${REPO}" \
  CPUS="${CPUS}" GPUS="${GPUS}" WORKERS="${WORKERS}" COUNTRIES="${COUNTRIES}" \
  CITY_IDS="${CITY_IDS}" SKIP_EXISTING="${SKIP_EXISTING}" \
  MIN_LABELS_EXISTING="${MIN_LABELS_EXISTING}" \
  DISCOVERY_RADIUS_M="${DISCOVERY_RADIUS_M}" RING_KM="${RING_KM}" \
  AREA_HEADROOM="${AREA_HEADROOM}" TARGET_AREA_M2="${TARGET_AREA_M2}" \
  AEF_SHARD_SIZE="${AEF_SHARD_SIZE}" PARALLEL_DISCOVERY="${PARALLEL_DISCOVERY}" \
  PARALLEL_AEF="${PARALLEL_AEF}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "${PYBIN}" -u "${RUN_DIR}/run_africa_ingest.py" \
  > "${RUN_DIR}/wrapper.log" 2>&1 &
echo $! | tee "${RUN_DIR}/wrapper.pid"
echo "RUN_DIR=${RUN_DIR}"
echo "Log: ${RUN_DIR}/job.log"
echo "Progress: ${RUN_DIR}/progress.json"
echo "Wrapper: ${RUN_DIR}/wrapper.log"
