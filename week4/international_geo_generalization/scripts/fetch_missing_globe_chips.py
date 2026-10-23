#!/usr/bin/env python3
"""Fetch missing Esri chips for globe points.json, then rebuild boundary walks.

Used after India/Japan 10-city exports that ran with --skip-imagery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "viz" / "usa_embedding_globe"
DATA = OUT / "data"
POINTS = DATA / "points.json"
CHIPS = DATA / "chips"

sys.path.insert(0, str(ROOT / "scripts"))
from export_usa_embedding_globe_sample import (  # noqa: E402
    atomic_json,
    chip_job,
    update_progress,
)


def main() -> int:
    n_cpus = os.cpu_count() or 8
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=str, default=str(POINTS))
    ap.add_argument("--workers", type=int, default=max(32, 2 * n_cpus))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild-walks", action="store_true")
    ap.add_argument("--walk-workers", type=int, default=0)
    ap.add_argument("--run-dir", type=str, default=os.environ.get("RUN_DIR", ""))
    ap.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city_id filter for chip fetch",
    )
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = OUT / "runs" / f"chips_{stamp}"
    for sub in ("checkpoints", "logs", "artifacts", "failures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    points_path = Path(args.points)
    payload = json.loads(points_path.read_text())
    pts = payload["points"]
    if args.cities:
        want = set(args.cities)
        pts_f = [p for p in pts if p["city_id"] in want]
    else:
        pts_f = pts

    CHIPS.mkdir(parents=True, exist_ok=True)
    need = []
    for p in pts_f:
        cid = int(p["id"])
        patch = CHIPS / f"{cid}_patch.jpg"
        ctx = CHIPS / f"{cid}_context.jpg"
        ctx500 = CHIPS / f"{cid}_context500.jpg"
        if args.force or not (
            patch.exists()
            and patch.stat().st_size > 1000
            and ctx.exists()
            and ctx.stat().st_size > 1000
            and ctx500.exists()
            and ctx500.stat().st_size > 1000
        ):
            need.append(
                {
                    "cell_id": cid,
                    "west": p["west"],
                    "south": p["south"],
                    "east": p["east"],
                    "north": p["north"],
                }
            )

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    update_progress(
        run_dir,
        status="running",
        stage="imagery",
        started_at=started,
        completed=0,
        failed=0,
        total=len(need),
        percent_complete=0.0,
        n_points=len(pts_f),
        n_need=len(need),
        workers=args.workers,
    )
    print(
        f"fetching chips for {len(need)}/{len(pts_f)} cells workers={args.workers}",
        flush=True,
    )

    ok = fail = skipped = wrote = 0
    t0 = time.time()
    if need:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(chip_job, r, args.force): r["cell_id"] for r in need}
            for i, fut in enumerate(as_completed(futs), 1):
                cid = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    fail += 1
                    (run_dir / "failures" / f"{cid}.txt").write_text(str(e) + "\n")
                    print(f"  FAIL {cid}: {e}", flush=True)
                    continue
                if res.get("ok"):
                    ok += 1
                    skipped += int(res.get("skipped") or 0)
                    wrote += int(res.get("wrote") or 0)
                else:
                    fail += 1
                    (run_dir / "failures" / f"{cid}.txt").write_text(
                        str(res.get("error")) + "\n"
                    )
                if i % 50 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-6)
                    eta = (len(futs) - i) / max(rate, 1e-6)
                    update_progress(
                        run_dir,
                        stage="imagery",
                        completed=i,
                        success=ok,
                        failed=fail,
                        total=len(futs),
                        percent_complete=round(100.0 * i / len(futs), 2),
                        rate_per_second=round(rate, 3),
                        eta_seconds=round(eta, 1),
                        wrote_chip_files=wrote,
                        skipped_chip_files=skipped,
                    )
                    print(
                        f"  [{i}/{len(futs)}] ok={ok} fail={fail} "
                        f"wrote={wrote} skip={skipped} "
                        f"{rate:.2f}/s eta={eta/60:.1f}m",
                        flush=True,
                    )

    # Refresh patch paths on ALL points (not only filtered)
    update_progress(run_dir, stage="update_points")
    n_linked = 0
    for p in pts:
        cid = int(p["id"])
        patch = f"chips/{cid}_patch.jpg"
        ctx = f"chips/{cid}_context.jpg"
        ctx500 = f"chips/{cid}_context500.jpg"
        p["patch"] = patch if (CHIPS / f"{cid}_patch.jpg").exists() else None
        p["context"] = ctx if (CHIPS / f"{cid}_context.jpg").exists() else None
        p["context500"] = (
            ctx500 if (CHIPS / f"{cid}_context500.jpg").exists() else None
        )
        if p["patch"]:
            n_linked += 1
    payload["points"] = pts
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["imagery_note"] = "Esri World Imagery XYZ · patch=exact 10m · context=50m · context=500m"
    tmp = points_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload))
    tmp.replace(points_path)
    print(f"updated {points_path} linked_chips={n_linked}/{len(pts)}", flush=True)

    walk_summary = None
    if args.rebuild_walks:
        update_progress(run_dir, stage="boundary_walks")
        print("rebuilding boundary walks …", flush=True)
        from build_boundary_walks import main as walks_main

        # build_boundary_walks reads argv; call via subprocess for clean args
        import subprocess

        walk_run = run_dir / "boundary"
        walk_run.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "build_boundary_walks.py"),
            "--run-dir",
            str(walk_run),
            "--points",
            str(points_path),
            "--workers",
            str(args.walk_workers or min(32, n_cpus)),
        ]
        with (run_dir / "job.log").open("a") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            update_progress(run_dir, status="failed", stage="boundary_walks")
            raise SystemExit(rc)
        if (walk_run / "summary.json").exists():
            walk_summary = json.loads((walk_run / "summary.json").read_text())

    summary = {
        "status": "completed" if fail == 0 else "completed_with_failures",
        "n_need": len(need),
        "n_ok": ok,
        "n_failed": fail,
        "wrote": wrote,
        "skipped": skipped,
        "n_linked": n_linked,
        "n_points": len(pts),
        "runtime_seconds": round(time.time() - t0, 1),
        "points_json": str(points_path),
        "walks": walk_summary,
    }
    atomic_json(run_dir / "summary.json", summary)
    update_progress(
        run_dir,
        status=summary["status"],
        stage="done",
        percent_complete=100.0,
        completed=len(need),
        failed=fail,
        runtime_seconds=summary["runtime_seconds"],
    )
    (run_dir / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
