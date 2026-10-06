#!/usr/bin/env python3
"""Build REPORT.md + merge India-10 walks/points into the globe viewer."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--loro-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--merge-globe", action="store_true")
    args = ap.parse_args()

    run = args.run_dir
    loro = args.loro_dir
    data = args.data_dir
    points = json.loads((run / "artifacts" / "points_india10.json").read_text())
    walks = json.loads((run / "artifacts" / "boundary_walks_india10.json").read_text())
    loro_m = pd.read_csv(loro / "artifacts" / "loro_IND_metrics.csv")

    key = next(
        (c for c in ("held_out", "region", "city_id", "held_out_city") if c in loro_m.columns),
        loro_m.columns[0],
    )
    f1_col = next(
        (c for c in ("macro_f1", "f1", "macro_f1_lr64") if c in loro_m.columns),
        None,
    )

    rows = []
    for w in walks["cities"]:
        ep = w.get("endpoint_selection") or {}
        proba = w.get("endpoint_proba") or {}
        frames = w.get("frames") or []
        path_p = [f.get("proba_res") for f in frames if f.get("proba_res") is not None]
        mono = None
        if len(path_p) >= 2:
            diffs = np.diff(path_p)
            if (diffs <= 1e-9).all():
                mono = "decreasing"
            elif (diffs < 0).mean() >= 0.7:
                mono = "mostly_dec"
            else:
                mono = "non_mono"
        cid = w["city_id"]
        f1 = None
        if f1_col is not None:
            sub = loro_m[loro_m[key] == cid]
            if len(sub):
                f1 = float(sub.iloc[0][f1_col])
        p0, p1 = proba.get("start_model"), proba.get("end_model")
        rows.append(
            {
                "city_id": cid,
                "city_name": w.get("city_name"),
                "status": w.get("status"),
                "endpoint_mode": ep.get("endpoint_mode"),
                "proba_space": w.get("proba_space"),
                "n_hi_res": w.get("n_hi_res"),
                "n_hi_nres": w.get("n_hi_nres"),
                "p_start": p0,
                "p_end": p1,
                "delta_p": None if p0 is None else float(p0) - float(p1),
                "path_mono": mono,
                "loro_f1": f1,
                "reason": w.get("reason"),
            }
        )

    csv_path = run / "artifacts" / "india10_walks_table.csv"
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    pred = points.get("prediction") or {}
    ok = [r for r in rows if r["status"] == "ok"]
    modes: dict[str, int] = {}
    for r in ok:
        modes[r["endpoint_mode"] or "none"] = modes.get(r["endpoint_mode"] or "none", 0) + 1
    fallback = [
        r
        for r in ok
        if r["endpoint_mode"] and not str(r["endpoint_mode"]).startswith("correct_hi")
    ]
    hi_ok = [r for r in ok if r["endpoint_mode"] == "correct_hi_0.9_0.1"]
    mean_dp = (
        float(np.mean([r["delta_p"] for r in ok if r["delta_p"] is not None]))
        if ok
        else float("nan")
    )

    lines = [
        "# India 10-city PLS-3 boundary walks",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Protocol: leave-one-city-out folds on `{loro.name}` extract;",
        "walk probability = LR on PLS dims 1–3; NN retrieval = PLS-32;",
        "endpoints = high-conf correct res→nres with fallback ladder.",
        "",
        "## Summary",
        "",
        f"- Walks: **{walks.get('n_ok', 0)}/{walks.get('n_cities', 0)} ok** "
        f"(skipped={walks.get('n_skipped')}, failed={walks.get('n_failed')})",
        f"- Viz sample: **{points.get('n_points')}** points across "
        f"**{len(points.get('cities', []))}** cities (1:1 + weak top-up)",
        f"- Sample LORO macro-F1 (scored cells): **{pred.get('macro_f1')}**  "
        f"acc=**{pred.get('accuracy')}**",
        f"- Endpoint modes: `{modes}`",
        f"- Strict hi-conf endpoints (0.9/0.1): **{len(hi_ok)}/10** cities",
        f"- Needed fallback endpoints: **{len(fallback)}/10**",
        "",
        "## Per-city walks",
        "",
        "| City | status | mode | P(res) start→end | ΔP | hi pools | LORO F1 | path |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda x: x["city_id"]):
        ps = (
            f"{r['p_start']:.3f}→{r['p_end']:.3f}"
            if r["p_start"] is not None
            else "—"
        )
        dp = f"{r['delta_p']:.3f}" if r["delta_p"] is not None else "—"
        hi = f"{r['n_hi_res']}/{r['n_hi_nres']}"
        f1 = f"{r['loro_f1']:.3f}" if r["loro_f1"] is not None else "—"
        lines.append(
            f"| {r['city_name']} | {r['status']} | `{r['endpoint_mode']}` | "
            f"{ps} | {dp} | {hi} | {f1} | {r['path_mono'] or '—'} |"
        )

    lines += ["", "## Findings", ""]
    if len(ok) == 10:
        lines.append(
            "1. **All 10 cities produce a PLS-3 boundary walk** on the India 10-city LORO panel."
        )
    else:
        lines.append(
            f"1. **{len(ok)}/10 cities** produced walks; check skipped/failed rows above."
        )
    lines.append(
        f"2. Mean endpoint ΔP(res) (start−end) = **{mean_dp:.3f}** — "
        "larger = clearer res↔nres separation in PLS-3."
    )
    if hi_ok:
        lines.append(
            "3. Cities with strict hi-conf endpoints: "
            + ", ".join(r["city_name"] for r in hi_ok)
            + "."
        )
    else:
        lines.append(
            "3. **No city** had enough hi-conf correct cells (P>0.9 / P<0.1) — "
            "India LORO remains weak vs USA."
        )
    if fallback:
        lines.append(
            "4. Fallback modes (mid/any/gt): "
            + ", ".join(f"{r['city_name']} (`{r['endpoint_mode']}`)" for r in fallback)
            + "."
        )
    with_f1 = [r for r in ok if r["loro_f1"] is not None and r["delta_p"] is not None]
    if len(with_f1) >= 5:
        xs = np.array([r["loro_f1"] for r in with_f1])
        ys = np.array([r["delta_p"] for r in with_f1])
        corr = (
            float(np.corrcoef(xs, ys)[0, 1])
            if xs.std() > 0 and ys.std() > 0
            else float("nan")
        )
        lines.append(
            f"5. Corr(LORO macro-F1, walk ΔP) ≈ **{corr:.2f}** (n={len(with_f1)})."
        )
    worst = sorted(
        ok, key=lambda r: r["delta_p"] if r["delta_p"] is not None else 999
    )[:3]
    best = sorted(
        ok, key=lambda r: -(r["delta_p"] if r["delta_p"] is not None else -999)
    )[:3]
    lines.append(
        "6. Weakest separation (small ΔP): "
        + ", ".join(
            f"{r['city_name']} ({r['delta_p']:.3f})"
            for r in worst
            if r["delta_p"] is not None
        )
        + "."
    )
    lines.append(
        "7. Strongest separation (large ΔP): "
        + ", ".join(
            f"{r['city_name']} ({r['delta_p']:.3f})"
            for r in best
            if r["delta_p"] is not None
        )
        + "."
    )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Walks: `{run / 'artifacts' / 'boundary_walks_india10.json'}`",
        f"- Points: `{run / 'artifacts' / 'points_india10.json'}`",
        f"- Table: `{csv_path}`",
        f"- Folds: `{data / 'intl_folds'}` (fold_IND_*.joblib, PLS-3 LR)",
        f"- Extract: `{loro / 'extract'}`",
        "",
    ]
    report = "\n".join(lines) + "\n"
    (run / "REPORT.md").write_text(report)
    (loro / "BOUNDARY_WALKS.md").write_text(report)
    print(report)

    if args.merge_globe:
        globe_pts_path = data / "points.json"
        globe_w_path = data / "boundary_walks.json"
        if globe_pts_path.exists():
            gp = json.loads(globe_pts_path.read_text())
            keep = [
                p
                for p in gp["points"]
                if not str(p.get("city_id", "")).startswith("IND_")
            ]
            merged_pts = keep + points["points"]
            cities = sorted({p["city_id"] for p in merged_pts})
            gp["points"] = merged_pts
            gp["n_points"] = len(merged_pts)
            gp["cities"] = [
                {
                    "city_id": c,
                    "city_name": next(
                        p["city_name"] for p in merged_pts if p["city_id"] == c
                    ),
                    "country": next(
                        p["country"] for p in merged_pts if p["city_id"] == c
                    ),
                    "n": sum(1 for p in merged_pts if p["city_id"] == c),
                }
                for c in cities
            ]
            gp["countries"] = sorted({p["country"] for p in merged_pts})
            gp["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp = globe_pts_path.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(gp))
            tmp.replace(globe_pts_path)
            print("merged points →", globe_pts_path, "n=", len(merged_pts))

        if globe_w_path.exists():
            gw = json.loads(globe_w_path.read_text())
            keep_w = [
                c
                for c in gw.get("cities", [])
                if not str(c.get("city_id", "")).startswith("IND_")
            ]
            merged_w = keep_w + walks["cities"]
            gw["cities"] = sorted(merged_w, key=lambda c: c.get("city_id") or "")
            gw["n_cities"] = len(gw["cities"])
            gw["n_ok"] = sum(1 for c in gw["cities"] if c.get("status") == "ok")
            gw["n_skipped"] = sum(
                1 for c in gw["cities"] if c.get("status") == "skipped"
            )
            gw["n_failed"] = sum(1 for c in gw["cities"] if c.get("status") == "failed")
            gw["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp = globe_w_path.with_suffix(".tmp.json")
            tmp.write_text(json.dumps(gw))
            tmp.replace(globe_w_path)
            print(
                "merged walks →",
                globe_w_path,
                "n=",
                gw["n_cities"],
                "ok=",
                gw["n_ok"],
            )

    summary = {
        "status": "completed",
        "n_ok": walks.get("n_ok"),
        "n_cities": walks.get("n_cities"),
        "modes": modes,
        "mean_delta_p": mean_dp,
        "report": str(run / "REPORT.md"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run / "progress.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stage": "done",
                "percent_complete": 100.0,
                "updated_at": summary["finished_at"],
                "walks_n_ok": summary["n_ok"],
            },
            indent=2,
        )
        + "\n"
    )
    (run / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
