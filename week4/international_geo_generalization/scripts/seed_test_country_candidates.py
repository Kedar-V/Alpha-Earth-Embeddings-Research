#!/usr/bin/env python3
"""Seed city candidates + selected cities for ONE smoke-test country (Japan)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402

COUNTRY_ID = "JPN"


def main() -> None:
    path = ROOT / "data" / "japan_city_candidates.json"
    candidates = json.loads(path.read_text())
    wait_until_ready(timeout_s=30)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM countries WHERE country_id = %s", (COUNTRY_ID,))
            if cur.fetchone() is None:
                raise SystemExit(f"Country {COUNTRY_ID} missing — run seed_countries.py first")

            selected_ids: list[str] = []
            for c in candidates:
                city_id = f"{COUNTRY_ID}_{c['city_name'].upper().replace(' ', '_')}"
                if c["selected"]:
                    cur.execute(
                        """
                        INSERT INTO cities (
                            city_id, country_id, city_name, admin_region,
                            latitude, longitude, centroid,
                            selection_role, selection_reason,
                            morphology_metadata, environment_metadata,
                            quality_status, inclusion_status, metadata
                        ) VALUES (
                            %(city_id)s, %(country_id)s, %(city_name)s, %(admin_region)s,
                            %(latitude)s, %(longitude)s,
                            ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326),
                            'selected_primary', %(selection_reason)s,
                            %(morphology_features)s::jsonb, %(geographic_features)s::jsonb,
                            'provisional', 'selected',
                            %(quality_scores)s::jsonb
                        )
                        ON CONFLICT (city_id) DO UPDATE SET
                            selection_reason = EXCLUDED.selection_reason,
                            morphology_metadata = EXCLUDED.morphology_metadata,
                            environment_metadata = EXCLUDED.environment_metadata,
                            inclusion_status = 'selected',
                            updated_at = now()
                        """,
                        {
                            "city_id": city_id,
                            "country_id": COUNTRY_ID,
                            "city_name": c["city_name"],
                            "admin_region": c["admin_region"],
                            "latitude": c["latitude"],
                            "longitude": c["longitude"],
                            "selection_reason": c.get("selection_reason"),
                            "morphology_features": json.dumps(c["morphology_features"]),
                            "geographic_features": json.dumps(c["geographic_features"]),
                            "quality_scores": json.dumps(c["quality_scores"]),
                        },
                    )
                    selected_ids.append(city_id)

                cur.execute(
                    """
                    INSERT INTO city_candidates (
                        country_id, city_name, admin_region, latitude, longitude, centroid,
                        morphology_features, geographic_features, quality_scores,
                        metadata, selected, rejection_reason, selected_city_id, selection_rank
                    ) VALUES (
                        %(country_id)s, %(city_name)s, %(admin_region)s,
                        %(latitude)s, %(longitude)s,
                        ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326),
                        %(morphology_features)s::jsonb,
                        %(geographic_features)s::jsonb,
                        %(quality_scores)s::jsonb,
                        '{}'::jsonb,
                        %(selected)s, %(rejection_reason)s,
                        %(selected_city_id)s, %(selection_rank)s
                    )
                    ON CONFLICT (country_id, city_name, admin_region) DO UPDATE SET
                        selected = EXCLUDED.selected,
                        rejection_reason = EXCLUDED.rejection_reason,
                        selected_city_id = EXCLUDED.selected_city_id,
                        selection_rank = EXCLUDED.selection_rank,
                        morphology_features = EXCLUDED.morphology_features,
                        geographic_features = EXCLUDED.geographic_features,
                        quality_scores = EXCLUDED.quality_scores,
                        updated_at = now()
                    """,
                    {
                        "country_id": COUNTRY_ID,
                        "city_name": c["city_name"],
                        "admin_region": c["admin_region"],
                        "latitude": c["latitude"],
                        "longitude": c["longitude"],
                        "morphology_features": json.dumps(c["morphology_features"]),
                        "geographic_features": json.dumps(c["geographic_features"]),
                        "quality_scores": json.dumps(c["quality_scores"]),
                        "selected": c["selected"],
                        "rejection_reason": c.get("rejection_reason"),
                        "selected_city_id": city_id if c["selected"] else None,
                        "selection_rank": c.get("selection_rank"),
                    },
                )

            cur.execute(
                "SELECT COUNT(*) AS n FROM city_candidates WHERE country_id = %s",
                (COUNTRY_ID,),
            )
            n_cand = cur.fetchone()["n"]
            cur.execute(
                "SELECT COUNT(*) AS n FROM cities WHERE country_id = %s AND inclusion_status = 'selected'",
                (COUNTRY_ID,),
            )
            n_sel = cur.fetchone()["n"]

            # Spatial smoke query: candidates within 500 km of Tokyo
            cur.execute(
                """
                SELECT city_name,
                       ROUND(
                         (ST_Distance(
                            centroid::geography,
                            ST_SetSRID(ST_MakePoint(139.6917, 35.6895), 4326)::geography
                          ) / 1000.0)::numeric, 1
                       ) AS dist_km_tokyo
                FROM city_candidates
                WHERE country_id = %s
                ORDER BY dist_km_tokyo
                """,
                (COUNTRY_ID,),
            )
            dists = list(cur.fetchall())

    print(f"country={COUNTRY_ID} candidates={n_cand} selected_cities={n_sel}")
    print("selected:", ", ".join(selected_ids))
    print("distance_to_tokyo_km:")
    for row in dists:
        print(f"  {row['city_name']}: {row['dist_km_tokyo']}")


if __name__ == "__main__":
    main()
