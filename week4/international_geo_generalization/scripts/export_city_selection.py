#!/usr/bin/env python3
"""Export city selection CSV/MD from PostgreSQL (exports only; DB is source of truth)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "city_selection.csv"
    md_path = out_dir / "CITY_SELECTION.md"

    wait_until_ready(timeout_s=30)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    co.continent,
                    co.iso3 AS country_id,
                    co.name AS country_name,
                    cc.city_name,
                    cc.admin_region,
                    cc.latitude,
                    cc.longitude,
                    cc.selected,
                    cc.selection_rank,
                    cc.rejection_reason,
                    cc.selected_city_id,
                    cc.quality_scores,
                    cc.morphology_features,
                    cc.geographic_features
                FROM city_candidates cc
                JOIN countries co ON co.country_id = cc.country_id
                ORDER BY co.continent, co.name, cc.selected DESC NULLS LAST, cc.selection_rank NULLS LAST, cc.city_name
                """
            )
            rows = list(cur.fetchall())

            cur.execute(
                """
                SELECT co.continent, co.name, COUNT(*) FILTER (WHERE cc.selected) AS n_selected,
                       COUNT(*) AS n_candidates
                FROM countries co
                LEFT JOIN city_candidates cc ON cc.country_id = co.country_id
                GROUP BY 1, 2
                HAVING COUNT(cc.candidate_id) > 0
                ORDER BY 1, 2
                """
            )
            summary = list(cur.fetchall())

    fieldnames = [
        "continent",
        "country_id",
        "country_name",
        "city_name",
        "admin_region",
        "latitude",
        "longitude",
        "selected",
        "selection_rank",
        "rejection_reason",
        "selected_city_id",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    lines = [
        "# City selection (export from PostgreSQL)",
        "",
        "Canonical source: `city_candidates` + `cities` + `countries`.",
        "This file is generated — do not edit as the primary record.",
        "",
        "## Coverage",
        "",
    ]
    for s in summary:
        lines.append(
            f"- **{s['name']}** ({s['continent']}): {s['n_selected']} selected / {s['n_candidates']} candidates"
        )
    lines += ["", "## Selected cities", ""]
    for r in rows:
        if r["selected"]:
            lines.append(
                f"- {r['country_name']} / {r['city_name']} "
                f"(rank {r['selection_rank']}, {r['latitude']}, {r['longitude']})"
            )
    lines += ["", "## Rejected candidates", ""]
    for r in rows:
        if not r["selected"]:
            lines.append(
                f"- {r['country_name']} / {r['city_name']}: {r['rejection_reason']}"
            )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
