#!/usr/bin/env python3
"""Rerun the week-5 subspace suite on a DoU-balanced extract.

Stages, each resumable via checkpoints/<stage>.DONE:
  01_extract                  country-level urban = peri_urban = rural, 1:1 res/nres
  02_global_local_subspace    hypothesis S_country ≈ S_global ⊕ S_local
  03_three_country            uses THIS run's global_local subspaces, not the Oct 5 prior
  04_task_invariant           uses THIS run's global_local and three_country outputs

max_per_class is 0 so the experiment samplers keep the extract as-is.
A positive cap would re-draw per city per binary label and undo the dou balance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
EXPORT = ROOT / "week4/international_geo_generalization/scripts/export_dou_balanced_extract.py"
GLOBAL = ROOT / "week5/scripts/global_local_subspace/run_experiment.py"
THREE = ROOT / "week5/scripts/three_country_falsification/run_experiment.py"
TASK = ROOT / "week5/scripts/task_invariant_subspace/run_experiment.py"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def write_progress(run_dir: Path, **kwargs) -> None:
    path = run_dir / "progress.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = now()
    prev["last_progress_at"] = prev["updated_at"]
    atomic_json(path, prev)


def log(run_dir: Path, msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def mark_done(run_dir: Path, name: str, payload: dict) -> None:
    p = run_dir / "checkpoints" / f"{name}.DONE"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"completed_at": now(), **payload}, indent=2) + "\n")


def stage_done(run_dir: Path, name: str) -> bool:
    return (run_dir / "checkpoints" / f"{name}.DONE").exists()


def resource_snapshot() -> dict:
    snap = {"cpus": os.cpu_count()}
    try:
        mem = Path("/proc/meminfo").read_text().splitlines()
        kv = {}
        for line in mem:
            k, v = line.split(":", 1)
            kv[k] = int(v.strip().split()[0])
        snap["ram_used_gb"] = round((kv["MemTotal"] - kv["MemAvailable"]) / 1024 / 1024, 1)
        snap["ram_total_gb"] = round(kv["MemTotal"] / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        load = Path("/proc/loadavg").read_text().split()
        snap["load_1m"] = float(load[0])
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        gpus = []
        for line in out.strip().splitlines():
            idx, util, mem = [p.strip() for p in line.split(",")]
            gpus.append({"index": int(idx), "util_pct": int(util), "mem_mib": int(mem)})
        snap["gpus"] = gpus
    except Exception:
        snap["gpus"] = []
    return snap


def child_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def run_child(run_dir: Path, stage: str, cmd: list[str], log_path: Path, child_progress_path: Path, env: dict) -> None:
    log(run_dir, f"start {stage}: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("a") as fh:
        fh.write(f"\n[{now()}] {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    (run_dir / "child.pid").write_text(f"parent={os.getpid()}\nchild={proc.pid}\nstage={stage}\n")
    while proc.poll() is None:
        child = child_progress(child_progress_path)
        elapsed = time.time() - t0
        write_progress(
            run_dir,
            status="running",
            stage=stage,
            child_pid=proc.pid,
            child_stage=child.get("stage"),
            child_status=child.get("status"),
            child_completed=child.get("completed"),
            child_total=child.get("total"),
            child_percent=child.get("percent_complete"),
            elapsed_stage_seconds=round(elapsed, 1),
            resources=resource_snapshot(),
        )
        time.sleep(20)
    if proc.returncode != 0:
        write_progress(run_dir, status="failed", stage=stage, child_returncode=proc.returncode)
        (run_dir / "FAILED").write_text(f"{stage} exit {proc.returncode}\n")
        raise SystemExit(f"{stage} failed with exit {proc.returncode}")
    log(run_dir, f"finished {stage} in {time.time() - t0:.1f}s")


def gpu_devices() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=10)
    except Exception:
        return ""
    ids = []
    for line in out.splitlines():
        if line.startswith("GPU "):
            ids.append(line.split()[1].rstrip(":"))
    return ",".join(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-bootstrap", type=int, default=50)
    ap.add_argument("--n-shuffle", type=int, default=5)
    args = ap.parse_args()

    run_dir = args.run_dir
    extract = run_dir / "extract"
    gl_dir = run_dir / "global_local_subspace"
    tc_dir = run_dir / "three_country_falsification"
    ti_dir = run_dir / "task_invariant_subspace"
    for d in (run_dir, extract, gl_dir, tc_dir, ti_dir, run_dir / "checkpoints", run_dir / "failures"):
        d.mkdir(parents=True, exist_ok=True)

    py = str(PY if PY.exists() else "python3")
    base_env = os.environ.copy()
    base_env["OMP_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["PYTHONUNBUFFERED"] = "1"
    seeds = [str(s) for s in args.seeds]
    t0 = time.time()

    write_progress(
        run_dir,
        status="running",
        stage="init",
        started_at=now(),
        completed=0,
        failed=0,
        total=4,
        output_location=str(run_dir),
        resource_allocation={
            "workers": args.workers,
            "cpus": os.cpu_count(),
            "gpus": gpu_devices() or "none",
            "max_per_class": 0,
            "note": "0 disables the per-city cap so the dou balance is what the models see",
        },
    )
    log(run_dir, f"workers={args.workers} seeds={seeds} extract={extract}")

    if not stage_done(run_dir, "01_extract"):
        write_progress(run_dir, stage="01_extract", status="running", completed=0, total=4)
        run_child(
            run_dir,
            "01_extract",
            [py, "-u", str(EXPORT), "--out-dir", str(extract), "--run-dir", str(run_dir / "extract_progress")],
            run_dir / "logs" / "01_extract.log",
            run_dir / "extract_progress" / "progress.json",
            base_env,
        )
        if not (extract / "manifest.json").exists():
            raise SystemExit("extract finished without manifest.json")
        mark_done(run_dir, "01_extract", {"extract": str(extract)})
    else:
        log(run_dir, "skip 01_extract")

    if not stage_done(run_dir, "02_global_local"):
        write_progress(run_dir, stage="02_global_local", status="running", completed=1, total=4, percent_complete=25.0)
        run_child(
            run_dir,
            "02_global_local",
            [
                py, "-u", str(GLOBAL),
                "--run-dir", str(gl_dir),
                "--extract-dir", str(extract),
                "--workers", str(args.workers),
                "--max-per-class", "0",
                "--seeds", *seeds,
            ],
            gl_dir / "job.log",
            gl_dir / "progress.json",
            base_env,
        )
        mark_done(run_dir, "02_global_local", {"run": str(gl_dir)})
    else:
        log(run_dir, "skip 02_global_local")

    if not stage_done(run_dir, "03_three_country"):
        write_progress(run_dir, stage="03_three_country", status="running", completed=2, total=4, percent_complete=50.0)
        run_child(
            run_dir,
            "03_three_country",
            [
                py, "-u", str(THREE),
                "--run-dir", str(tc_dir),
                "--prior-run", str(gl_dir),
                "--extract-dir", str(extract),
                "--workers", str(args.workers),
                "--max-per-class", "0",
                "--n-bootstrap", str(args.n_bootstrap),
                "--seeds", *seeds,
            ],
            tc_dir / "job.log",
            tc_dir / "progress.json",
            base_env,
        )
        mark_done(run_dir, "03_three_country", {"run": str(tc_dir)})
    else:
        log(run_dir, "skip 03_three_country")

    if not stage_done(run_dir, "04_task_invariant"):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices()
        write_progress(
            run_dir,
            stage="04_task_invariant",
            status="running",
            completed=3,
            total=4,
            percent_complete=75.0,
            resources=resource_snapshot(),
        )
        run_child(
            run_dir,
            "04_task_invariant",
            [
                py, "-u", str(TASK),
                "--run-dir", str(ti_dir),
                "--prior-run", str(gl_dir),
                "--exp04-run", str(tc_dir),
                "--extract-dir", str(extract),
                "--workers", str(args.workers),
                "--max-per-class", "0",
                "--n-bootstrap", str(args.n_bootstrap),
                "--n-shuffle", str(args.n_shuffle),
                "--seeds", *seeds,
            ],
            ti_dir / "job.log",
            ti_dir / "progress.json",
            env,
        )
        mark_done(run_dir, "04_task_invariant", {"run": str(ti_dir)})
    else:
        log(run_dir, "skip 04_task_invariant")

    manifest = json.loads((extract / "manifest.json").read_text())
    summary = {
        "status": "completed",
        "protocol": manifest.get("protocol"),
        "n_cells": manifest.get("n_cells"),
        "targets_per_stratum": manifest.get("targets_per_stratum"),
        "balance": manifest.get("balance"),
        "max_per_class": 0,
        "seeds": args.seeds,
        "workers": args.workers,
        "extract": str(extract),
        "global_local_subspace": str(gl_dir),
        "three_country_falsification": str(tc_dir),
        "task_invariant_subspace": str(ti_dir),
        "runtime_seconds": round(time.time() - t0, 1),
        "finished_at": now(),
    }
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("ok\n")
    write_progress(run_dir, status="completed", stage="done", completed=4, total=4, percent_complete=100.0, eta_seconds=0)
    log(run_dir, "DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
