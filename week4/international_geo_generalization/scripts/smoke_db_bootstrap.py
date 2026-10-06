#!/usr/bin/env python3
"""Observable smoke bootstrap: init verification, seed 30 countries + Japan candidates,
spatial checks, exports. Does NOT download the full 150-city building/embedding corpus.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))


def atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> int:
    run_dir = Path(os.environ["RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    stages = [
        "init_db",
        "seed_countries",
        "seed_japan_candidates",
        "validate",
        "export_city_selection",
        "backup",
    ]
    progress = {
        "status": "running",
        "stage": stages[0],
        "started_at": started,
        "updated_at": started,
        "completed": 0,
        "failed": 0,
        "total": len(stages),
        "percent_complete": 0.0,
        "resource_allocation": {
            "cpus": int(os.environ.get("CPUS", "70")),
            "gpus": os.environ.get("GPUS", "all"),
        },
        "output": str(run_dir),
    }
    atomic_write(run_dir / "progress.json", progress)

    py = str(REPO / ".tools" / "postgis" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable

    env = os.environ.copy()
    # Load db/.env into process env without echoing secrets
    env_file = ROOT / "db" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env.setdefault(k.strip(), v.strip().strip("'").strip('"'))

    def run_stage(name: str, cmd: list[str]) -> None:
        progress["stage"] = name
        progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        progress["last_progress_at"] = progress["updated_at"]
        atomic_write(run_dir / "progress.json", progress)
        log_path = run_dir / "logs" / f"{name}.log"
        print(f"[stage] {name}: {' '.join(cmd)}", flush=True)
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            progress["failed"] += 1
            progress["status"] = "failed"
            progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            atomic_write(run_dir / "progress.json", progress)
            (run_dir / "failures" / f"{name}.txt").write_text(
                f"exit={proc.returncode}\nsee logs/{name}.log\n"
            )
            raise SystemExit(proc.returncode)
        progress["completed"] += 1
        progress["percent_complete"] = round(
            100.0 * progress["completed"] / progress["total"], 2
        )
        progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        progress["last_progress_at"] = progress["updated_at"]
        atomic_write(run_dir / "progress.json", progress)
        # checkpoint marker
        (run_dir / "checkpoints" / f"{name}.DONE").write_text("ok\n")

    run_stage(
        "init_db",
        ["bash", str(ROOT / "scripts" / "init_db.sh")],
    )
    run_stage("seed_countries", [py, str(ROOT / "scripts" / "seed_countries.py")])
    run_stage(
        "seed_japan_candidates",
        [py, str(ROOT / "scripts" / "seed_test_country_candidates.py")],
    )
    run_stage("validate", [py, str(ROOT / "scripts" / "validate_db.py")])
    run_stage(
        "export_city_selection",
        [py, str(ROOT / "scripts" / "export_city_selection.py")],
    )
    run_stage(
        "backup",
        [
            "bash",
            str(ROOT / "db" / "scripts" / "backup_db.sh"),
            str(run_dir / "artifacts"),
        ],
    )

    # Final DB snapshot stats (no secrets)
    from lib.db import connect

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM countries")
            n_co = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM city_candidates")
            n_cand = cur.fetchone()["n"]
            cur.execute(
                "SELECT COUNT(*) AS n FROM cities WHERE inclusion_status='selected'"
            )
            n_sel = cur.fetchone()["n"]
            cur.execute("SELECT version FROM schema_migrations ORDER BY 1")
            migrations = [r["version"] for r in cur.fetchall()]
            cur.execute(
                """
                SELECT city_name FROM cities
                WHERE country_id='JPN' AND inclusion_status='selected'
                ORDER BY city_name
                """
            )
            japan_cities = [r["city_name"] for r in cur.fetchall()]

    summary = {
        "status": "completed",
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "countries": n_co,
        "city_candidates": n_cand,
        "selected_cities": n_sel,
        "japan_selected": japan_cities,
        "migrations": migrations,
        "exports": {
            "csv": str(ROOT / "outputs" / "city_selection.csv"),
            "md": str(ROOT / "outputs" / "CITY_SELECTION.md"),
        },
        "note": "Full 150-city building/AlphaEarth ingest deferred until smoke checks pass.",
    }
    atomic_write(run_dir / "summary.json", summary)
    progress["status"] = "completed"
    progress["stage"] = "done"
    progress["updated_at"] = summary["finished_at"]
    atomic_write(run_dir / "progress.json", progress)
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
