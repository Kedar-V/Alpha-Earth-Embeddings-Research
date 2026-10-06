# PostgreSQL + PostGIS backend

Canonical datastore for the international AlphaEarth geographic-generalization study.

Files under `outputs/` (CSV/MD/Parquet) are **exports**. PostgreSQL is authoritative.

## Quick start

```bash
# 1) Credentials
cp week4/international_geo_generalization/db/.env.example \
   week4/international_geo_generalization/db/.env
# edit POSTGRES_PASSWORD

# 2) Initialize (Docker Compose if available, else local micromamba PostGIS)
./week4/international_geo_generalization/scripts/init_db.sh

# 3) Smoke seed (30 countries + Japan candidate pool)
.tools/postgis/bin/python week4/international_geo_generalization/scripts/seed_countries.py
.tools/postgis/bin/python week4/international_geo_generalization/scripts/seed_test_country_candidates.py
.tools/postgis/bin/python week4/international_geo_generalization/scripts/validate_db.py
.tools/postgis/bin/python week4/international_geo_generalization/scripts/export_city_selection.py
```

Or one observable job:

```bash
# see scripts/submit_smoke_bootstrap.sh
```

## Start / stop

### Docker (preferred when you have daemon access)

```bash
cd week4/international_geo_generalization/db
docker compose up -d          # start
docker compose ps             # status
docker compose down           # stop (volume kept)
docker compose down -v        # stop + DELETE data volume
```

### Local micromamba fallback

Used automatically by `init_db.sh` when Docker is unavailable (current machine: user not in `docker` group).

```bash
# start/stop via pg_ctl
.tools/postgis/bin/pg_ctl -D "$PGDATA_DIR" status
.tools/postgis/bin/pg_ctl -D "$PGDATA_DIR" stop
```

`PGDATA_DIR` defaults to `/data/users/kv133/Alpha-Earth-Embeddings-Research/pgdata`.

Install/rebuild the user-space toolchain:

```bash
./week4/international_geo_generalization/scripts/install_postgis_env.sh
```

## Environment variables

See [`.env.example`](.env.example). Required:

| Variable | Purpose |
|----------|---------|
| `POSTGRES_HOST` | hostname (`127.0.0.1`) |
| `POSTGRES_PORT` | port (`5432`) |
| `POSTGRES_DB` | database name |
| `POSTGRES_USER` | role |
| `POSTGRES_PASSWORD` | secret — never commit, never put on CLI for long jobs |

Experiment / training jobs launched via submit-training-job should inherit these from the environment (or `db/.env` loaded by the job wrapper). Do not embed passwords in argv, logs, or committed configs.

## Migrations

Versioned SQL in `migrations/`:

1. `001_extensions.sql` — PostGIS + `schema_migrations`
2. `002_core_tables.sql` — entities
3. `003_indexes.sql` — GiST + B-tree
4. `004_validation.sql` — quality views / `run_data_quality_checks()`

Applied in sorted order by `scripts/init_db.sh`. Migrations are idempotent (`IF NOT EXISTS` / `ON CONFLICT`).

Do not manually mutate the live schema — add `005_*.sql` instead.

## Schema overview

See [SCHEMA.md](SCHEMA.md). Logical lineage:

`continent → country → city → building/cell → label → experiment sample`

## Backup / restore

```bash
./week4/international_geo_generalization/db/scripts/backup_db.sh [outdir]
./week4/international_geo_generalization/db/scripts/restore_db.sh path/to/file.dump
```

Dumps are custom-format (`pg_dump -Fc`). **Do not commit dumps.**

## City selection exports

Canonical tables: `countries`, `cities`, `city_candidates`.

Generated for review only:

```text
outputs/city_selection.csv
outputs/CITY_SELECTION.md
```

```bash
.tools/postgis/bin/python scripts/export_city_selection.py
```

## Example SQL

```sql
-- Five selected cities for one country
SELECT city_id, city_name, latitude, longitude
FROM cities
WHERE country_id = 'JPN' AND inclusion_status = 'selected';

-- All cities on one continent
SELECT c.city_id, c.city_name, co.name AS country
FROM cities c
JOIN countries co ON co.country_id = c.country_id
WHERE co.continent = 'Asia' AND c.inclusion_status = 'selected';

-- Candidate pool (including rejects) for Japan
SELECT city_name, selected, rejection_reason, selection_rank
FROM city_candidates
WHERE country_id = 'JPN'
ORDER BY selected DESC, selection_rank NULLS LAST;

-- Leave-one-country-out style split listing
SELECT country_id, split, fold, split_strategy
FROM experiment_splits
WHERE experiment_id = 'loro_v1';

-- Transfer matrix reconstruction
SELECT source_geography, target_geography, value AS macro_f1
FROM experiment_metrics
WHERE run_id = '...' AND metric = 'macro_f1';

-- Quality checks
SELECT * FROM run_data_quality_checks();
```

## How experiment scripts access the DB

```python
from lib.db import connect, wait_until_ready

wait_until_ready()
with connect() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT city_id FROM cities WHERE inclusion_status = 'selected'")
        cities = [r["city_id"] for r in cur.fetchall()]
```

Heavy training may write NumPy/Parquet extracts, but each extract must register a row in `experiment_extracts` (SQL text, label version, city set, timestamp). Caches are disposable.

## AOI sampling (center + radius)

International cities do **not** use US downtown bboxes by default.

```text
AOI = disk(city center, R)
R   = max distance from center to discovered buildings (+ optional buffer)
```

No urban/peri/rural **AOI zone rings** — just the enclosing disk.  
Distance rings may still be used as **AlphaEarth sampling strata** (area-balance RES/NRES per ring); that is a download gate, not an AOI partition. See [`../README.md`](../README.md) (Dataset creation) and [`../docs/DATASET_CONTRACT.md`](../docs/DATASET_CONTRACT.md).

```bash
# After buildings exist for a city:
.tools/postgis/bin/python scripts/compute_city_aoi_radius.py --city-id JPN_TOKYO

# Provisional R before buildings (smoke only):
.tools/postgis/bin/python scripts/compute_city_aoi_radius.py --city-id JPN_TOKYO --provisional-radius-m 25000
```

## Scaling gate

Before full 150-city ingest:

1. DB up + migrations applied  
2. 30 countries inserted  
3. Candidates for **one** test country (Japan) inserted  
4. Spatial queries + GiST indexes verified  
5. City-selection CSV/MD exported from PostgreSQL  
6. AOI contract locked (center + enclosing R)  

Only then scale building / AlphaEarth ingestion.
