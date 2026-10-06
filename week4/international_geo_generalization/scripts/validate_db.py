#!/usr/bin/env python3
"""Run database health / data-quality checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402


def main() -> int:
    wait_until_ready(timeout_s=30)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT PostGIS_Version() AS v")
            print("postgis:", cur.fetchone()["v"])
            cur.execute("SELECT * FROM run_data_quality_checks() ORDER BY check_name")
            checks = list(cur.fetchall())
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                WHERE schemaname = 'public'
                  AND indexname LIKE '%gix%'
                ORDER BY 1
                """
            )
            gix = [r["indexname"] for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) AS n FROM countries")
            n_countries = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM city_candidates")
            n_cand = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM cities WHERE inclusion_status = 'selected'")
            n_sel = cur.fetchone()["n"]

            # Spatial index usage smoke: bounding-box filter on Japan candidates
            cur.execute(
                """
                SELECT city_name
                FROM city_candidates
                WHERE centroid && ST_MakeEnvelope(129.0, 30.0, 146.0, 46.0, 4326)
                ORDER BY city_name
                """
            )
            bbox_hits = [r["city_name"] for r in cur.fetchall()]

    print(json.dumps({"countries": n_countries, "candidates": n_cand, "selected_cities": n_sel}, indent=2))
    print("gist_indexes:", ", ".join(gix) or "(none)")
    print("japan_bbox_hits:", ", ".join(bbox_hits))
    print("quality_checks:")
    failed = 0
    for c in checks:
        mark = "OK" if c["n_issues"] == 0 else "ISSUE"
        if c["n_issues"]:
            failed += 1
        print(f"  [{mark}] {c['check_name']}: {c['n_issues']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
