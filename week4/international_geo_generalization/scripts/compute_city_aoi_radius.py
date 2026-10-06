#!/usr/bin/env python3
"""Compute city AOI = center + radius R that covers all buildings.

Usage (after buildings are loaded for a city):
  .tools/postgis/bin/python scripts/compute_city_aoi_radius.py --city-id JPN_TOKYO

Provisional mode (no buildings yet):
  .tools/postgis/bin/python scripts/compute_city_aoi_radius.py --city-id JPN_TOKYO --provisional-radius-m 25000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-id", required=True)
    ap.add_argument("--buffer-m", type=float, default=0.0)
    ap.add_argument(
        "--provisional-radius-m",
        type=float,
        default=None,
        help="If set, skip building scan and write a provisional R",
    )
    args = ap.parse_args()

    wait_until_ready(timeout_s=30)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT city_id FROM cities WHERE city_id = %s",
                (args.city_id,),
            )
            if cur.fetchone() is None:
                raise SystemExit(f"unknown city_id={args.city_id}")

            cur.execute(
                """
                UPDATE cities
                SET aoi_center = COALESCE(aoi_center, centroid),
                    updated_at = now()
                WHERE city_id = %s
                """,
                (args.city_id,),
            )

            if args.provisional_radius_m is not None:
                radius_m = float(args.provisional_radius_m) + float(args.buffer_m)
                method = "provisional"
                n_buildings = None
            else:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS n,
                        MAX(
                          ST_Distance(
                            b.geom::geography,
                            c.aoi_center::geography
                          )
                        ) AS r_raw
                    FROM buildings b
                    JOIN cities c ON c.city_id = b.city_id
                    WHERE b.city_id = %s
                    """,
                    (args.city_id,),
                )
                stats = cur.fetchone()
                if not stats["n"] or stats["r_raw"] is None:
                    raise SystemExit(
                        f"No buildings for {args.city_id}; "
                        "load buildings first or pass --provisional-radius-m"
                    )
                radius_m = float(stats["r_raw"]) + float(args.buffer_m)
                method = "building_enclosing_radius"
                n_buildings = int(stats["n"])

            cur.execute(
                """
                UPDATE cities SET
                    aoi_radius_m = %(radius_m)s,
                    aoi_buffer_m = %(buffer_m)s,
                    aoi_method = %(method)s,
                    aoi_building_count = %(n_buildings)s,
                    aoi_disk = city_aoi_disk_from_radius(aoi_center, %(radius_m)s),
                    aoi_computed_at = now(),
                    updated_at = now()
                WHERE city_id = %(city_id)s
                RETURNING city_id, aoi_method, aoi_radius_m, aoi_building_count,
                          ST_Area(aoi_disk::geography)/1e6 AS area_km2
                """,
                {
                    "city_id": args.city_id,
                    "radius_m": radius_m,
                    "buffer_m": args.buffer_m,
                    "method": method,
                    "n_buildings": n_buildings,
                },
            )
            out = cur.fetchone()

    print(json.dumps(dict(out), indent=2, default=str))


if __name__ == "__main__":
    main()
