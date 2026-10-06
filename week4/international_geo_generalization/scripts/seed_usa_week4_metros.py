#!/usr/bin/env python3
"""Insert the 10 week4 US metros into cities (international run includes USA)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402


def main() -> None:
    metros = json.loads((ROOT / "data" / "usa_week4_metros.json").read_text())
    assert len(metros) == 10

    wait_until_ready(timeout_s=30)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM countries WHERE country_id = 'USA'")
            if cur.fetchone() is None:
                raise SystemExit("USA missing from countries — run seed_countries.py first")

            for m in metros:
                meta = {
                    "week4_region_id": m["week4_region_id"],
                    "legacy_downtown_bbox_wgs84": {
                        "xmin": m["legacy_downtown_bbox"][0],
                        "ymin": m["legacy_downtown_bbox"][1],
                        "xmax": m["legacy_downtown_bbox"][2],
                        "ymax": m["legacy_downtown_bbox"][3],
                        "note": "week4 AOI for audit reference only; international AOI is center+R",
                    },
                }
                cur.execute(
                    """
                    INSERT INTO cities (
                        city_id, country_id, city_name, admin_region,
                        latitude, longitude, centroid, aoi_center,
                        selection_role, selection_reason,
                        quality_status, inclusion_status, metadata
                    ) VALUES (
                        %(city_id)s, 'USA', %(city_name)s, %(admin_region)s,
                        %(latitude)s, %(longitude)s,
                        ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326),
                        ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326),
                        'week4_anchor',
                        'Week4 US metro retained for structure/analysis audit against downtown-bbox study.',
                        'provisional', 'selected',
                        %(metadata)s::jsonb
                    )
                    ON CONFLICT (city_id) DO UPDATE SET
                        city_name = EXCLUDED.city_name,
                        admin_region = EXCLUDED.admin_region,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        centroid = EXCLUDED.centroid,
                        aoi_center = COALESCE(cities.aoi_center, EXCLUDED.aoi_center),
                        selection_role = EXCLUDED.selection_role,
                        selection_reason = EXCLUDED.selection_reason,
                        inclusion_status = 'selected',
                        metadata = cities.metadata || EXCLUDED.metadata,
                        updated_at = now()
                    """,
                    {
                        "city_id": m["city_id"],
                        "city_name": m["city_name"],
                        "admin_region": m["admin_region"],
                        "latitude": m["latitude"],
                        "longitude": m["longitude"],
                        "metadata": json.dumps(meta),
                    },
                )

            cur.execute(
                """
                SELECT city_id, city_name, metadata->>'week4_region_id' AS week4_region_id
                FROM cities
                WHERE country_id = 'USA'
                ORDER BY city_id
                """
            )
            rows = list(cur.fetchall())

    print(f"usa_selected_cities={len(rows)}")
    for r in rows:
        print(f"  {r['city_id']}  {r['city_name']}  <- {r['week4_region_id']}")


if __name__ == "__main__":
    main()
