#!/usr/bin/env python3
"""Health-check an ingest RUN_DIR every 15 minutes until DONE / dead / max checks.

Usage:
  RUN_DIR=... python monitor_15m.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

RUN = Path(os.environ["RUN_DIR"]).resolve()
OUT = RUN / "monitor_15m.log"
ALERT = RUN / "monitor_ALERT"
INTERVAL_S = int(os.environ.get("MONITOR_INTERVAL_S", "900"))
MAX_CHECKS = int(os.environ.get("MONITOR_MAX_CHECKS", "48"))  # 12h default
STUCK_LOG_S = int(os.environ.get("MONITOR_STUCK_LOG_S", "900"))


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pid_alive(pid: int) -> tuple[bool, str, str]:
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "etime=,%cpu="],
            text=True,
        ).strip()
        if not out:
            return False, "-", "-"
        parts = out.split()
        return True, parts[0], parts[1] if len(parts) > 1 else "-"
    except subprocess.CalledProcessError:
        return False, "-", "-"


def check(i: int) -> str:
    lines: list[str] = []
    pid_s = (RUN / "job.pid").read_text().strip() if (RUN / "job.pid").exists() else ""
    pid = int(pid_s) if pid_s.isdigit() else None
    alive, etime, cpu = (False, "-", "-")
    if pid is not None:
        alive, etime, cpu = pid_alive(pid)

    prog = {}
    if (RUN / "progress.json").exists():
        prog = json.loads((RUN / "progress.json").read_text())
    log = RUN / "job.log"
    log_age = time.time() - log.stat().st_mtime if log.exists() else None
    log_size = log.stat().st_size if log.exists() else 0
    cities = sorted(p.parent.name for p in (RUN / "checkpoints").glob("*/CITY.DONE"))
    fails = sorted(p.name for p in (RUN / "failures").glob("*.txt"))
    done = (RUN / "DONE").exists()

    # Flag vacuous "success" (buildings but zero cells)
    empty_cells = []
    for jp in (RUN / "checkpoints").glob("*/CITY.json"):
        try:
            d = json.loads(jp.read_text())
            if int(d.get("buildings") or 0) > 0 and int(d.get("cells") or 0) == 0:
                empty_cells.append(jp.parent.name)
        except Exception:
            pass

    lines.append(
        f"[{ts()}] check={i} alive={int(alive)} pid={pid} etime={etime} cpu%={cpu}"
    )
    lines.append(
        f"  status={prog.get('status')} completed={prog.get('completed')}/{prog.get('total')} "
        f"failed={prog.get('failed')} embedding={prog.get('embedding')} "
        f"log_age_s={None if log_age is None else round(log_age)} log_bytes={log_size}"
    )
    lines.append(f"  CITY={cities}")
    lines.append(f"  failures={fails}")
    if empty_cells:
        lines.append(f"  empty_cells={empty_cells}")

    alerts: list[str] = []
    if done and not empty_cells:
        lines.append("  OK: DONE present")
        state = "done"
    elif done and empty_cells:
        alerts.append("DONE_BUT_EMPTY_CELLS=" + ",".join(empty_cells))
        state = "failures"
    elif not alive:
        alerts.append("PROCESS_DEAD")
        state = "dead"
    elif log_age is not None and log_age > STUCK_LOG_S:
        alerts.append(f"LOG_STUCK>{STUCK_LOG_S}s")
        state = "stuck"
    else:
        state = "ok"

    if fails:
        alerts.append("FAILURES=" + ",".join(fails))
        if state == "ok":
            state = "failures"

    if alerts:
        lines.append("  ALERT: " + "; ".join(alerts))
        ALERT.write_text("\n".join(alerts) + f"\n{ts()}\n")
    else:
        if state == "ok":
            lines.append("  OK: healthy")
        if ALERT.exists() and state in {"ok", "done"}:
            ALERT.unlink()

    text = "\n".join(lines) + "\n"
    with OUT.open("a") as f:
        f.write(text)
    print(text, end="", flush=True)
    return state


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(f"===== monitor start {ts()} interval={INTERVAL_S}s =====\n")
    print(f"===== monitor start {ts()} interval={INTERVAL_S}s =====", flush=True)

    for i in range(1, MAX_CHECKS + 1):
        state = check(i)
        if state == "done":
            with OUT.open("a") as f:
                f.write(f"[{ts()}] monitor exit: job DONE\n")
            return 0
        if state == "dead":
            with OUT.open("a") as f:
                f.write(f"[{ts()}] monitor exit: process dead\n")
            return 1
        if i < MAX_CHECKS:
            time.sleep(INTERVAL_S)

    with OUT.open("a") as f:
        f.write(f"[{ts()}] monitor exit: max checks reached\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
