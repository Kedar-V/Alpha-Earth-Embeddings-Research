#!/usr/bin/env python3
"""Insert the canonical 30-country list into PostgreSQL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.db import connect, wait_until_ready  # noqa: E402


def main() -> None:
    countries_path = ROOT / "data" / "countries_30.json"
    countries = json.loads(countries_path.read_text())
    assert len(countries) == 30, f"Expected 30 countries, got {len(countries)}"

    wait_until_ready(timeout_s=30)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO countries (country_id, name, iso2, iso3, continent, metadata)
                VALUES (%(country_id)s, %(name)s, %(iso2)s, %(iso3)s, %(continent)s, '{}'::jsonb)
                ON CONFLICT (country_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    iso2 = EXCLUDED.iso2,
                    iso3 = EXCLUDED.iso3,
                    continent = EXCLUDED.continent,
                    updated_at = now()
                """,
                countries,
            )
            cur.execute("SELECT COUNT(*) AS n FROM countries")
            n = cur.fetchone()["n"]
            cur.execute(
                "SELECT continent, COUNT(*) AS n FROM countries GROUP BY 1 ORDER BY 1"
            )
            by_cont = list(cur.fetchall())
    print(f"countries_upserted={len(countries)} countries_total={n}")
    for row in by_cont:
        print(f"  {row['continent']}: {row['n']}")


if __name__ == "__main__":
    main()
