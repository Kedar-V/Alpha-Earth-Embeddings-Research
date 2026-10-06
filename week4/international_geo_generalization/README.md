# International geographic generalization

Cross-country AlphaEarth residential / non-residential transfer study.

**PostgreSQL + PostGIS is the canonical datastore.** See [`db/README.md`](db/README.md).

## Layout

```text
db/                 Docker Compose, migrations, SCHEMA.md
scripts/            init_db, seed, validate, export, smoke job
lib/db.py           connection helpers (env-based credentials)
data/               seed inputs (30 countries, Japan + India candidates)
outputs/            generated exports (not authoritative)
runs/               job logs / progress / checkpoints
```

## Bootstrap gate (required before full ingest)

1. Bring DB up (`./scripts/init_db.sh` from repo root, or package script)
2. Apply migrations
3. Insert 30 countries
4. Insert candidates for **one** test country (Japan)
5. Verify spatial indexes / quality checks
6. Export `city_selection.csv` + `CITY_SELECTION.md` from PostgreSQL

Only then download buildings / AlphaEarth for all 150 cities.

## Dataset creation

PostgreSQL holds lineage. AlphaEarth download is the expensive step, so we **discover broadly, sample for embeddings, expand later if needed**.

### Goals

- Find similar RES vs NRES patterns across cities (urban → fringe), not reproduce week4 downtown pixel IDs.
- End with a **balanced pixel training set** per city (pure RES/NRES, 1:1 counts).
- Cut AlphaEarth volume when one class dominates by footprint area; pull more samples later without redoing Overture.

### Pipeline

```text
1. Discover   Track-A Overture buildings in a disk around city center
              (default discovery up to 30 km; class allowlist = week4 Track-A)

2. Persist    Store full discovery footprints in Postgres (+ buildings.parquet)
              AOI R = enclosing radius of discovered buildings (no urban/peri/rural AOI rings)

3. Sample     Stratified area-balance for AlphaEarth (download gate):
              - Bin buildings by distance to center (e.g. 0–5, 5–10, …, 25–30 km)
              - Within each ring: keep the minority class by footprint area;
                thin the majority to ≈ minority_area × headroom (default 1.10)
              - Cap every city to the same footprint budget: week4 total Track-A
                area / 10 ≈ **24.60 km²** (`--target-area-m2`), preserving class
                mix and ring shares
              - Rings are sampling strata only — not stored AOI zones

4. Embed      Pull AlphaEarth 10 m / 2024 / 64-D only for sampled buildings’ cells

5. Label      Pixel fractions / purity from building∩cell geometry
              (use enough footprints so labels on sampled cells stay correct)

6. Balance    Later: pure-exclusive cells, then per-city 1:1 undersample by
              pixel count (seed recorded). Headroom exists because some cells
              are mixed/empty or pixels-per-m² is uneven (~5–15% slack is normal)
```

### Why this is not identical to week4 (and why that’s OK)

Week4 took **all** Track-A buildings in a downtown bbox, embedded everything, then 1:1 pure-pixel undersample.  
Here we **thin majority footprints before embed**, stratified by radius, then still 1:1 at pixel level.

- Training is still pixel-level; polygons are GT for labels only.
- Fixed 10 m cells do **not** make footprint-area balance equal pixel counts — hence headroom + final count balance.
- USA audit = structure/definitions/analysis match, not identical samples to downtown week4.

### Expand later

Keep the full discovery building set in the DB. To grow the extract: raise headroom, add rings, or raise per-ring caps, then run AlphaEarth only for newly selected building ids (resume-friendly). No need to re-pull Overture.

### Rough cost

Overture discovery is cheap (minutes/city). AlphaEarth dominates wall time and scales with **sampled** buildings. On RES-heavy metros, stratified area-balance can keep ~30% of buildings (large speedup); on already area-balanced metros, keep ≈100% (little change).

---

## Concurrency & batching (implemented)

Focus: **multiprocessing**, **multithreading**, and **batching** at every stage.

### Rules of thumb

| Pattern | Use when | Avoid when |
|---------|----------|------------|
| **Multiprocessing** (`ProcessPoolExecutor`) | CPU-heavy Python (AEF extract, polygon∩pixel, tile assign) | Tiny tasks where spawn/pickle &gt; work |
| **Multithreading** (`ThreadPoolExecutor`) | Concurrent HTTP/GCS opens (optional) | Heavy pure-Python CPU (GIL) |
| **Batching** | DB writes (`COPY` chunks), AEF building shards, label cell chunks | Huge batches that OOM (cap shard size) |

Flags: `--workers`, `--parallel-discovery` (default 8), `--parallel-aef` (default 2; AEF workers = `workers // N`), `--aef-shard-size`, `--ring-km`, `--area-headroom`, `--target-area-m2` (default week4/10 ≈ 24.60 km²), `--discovery-radius-m`.

### Stage matrix

```text
stage                    parallel unit              model              batch unit
───────────────────────  ─────────────────────────  ─────────────────  ────────────────────────
Discovery pipeline       up to --parallel-discovery threads            Overture+COPY+sample / city
AEF pipeline             up to --parallel-aef       **processes (spawn)**  one city process; workers // N inside
Tile assign / AEF shards tile × building shard      processes          --aef-shard-size
Label / intersections    cell chunks                processes          ~n_cells / (2×writers)
Cell/label → Postgres    cell / label chunk         processes + COPY   COPY ~10k cells
```

Discovery feeds AEF as soon as each city finishes sample — no waiting for a full wave. Cities already past `aef_sample` skip straight into the AEF queue.

**Deadlock avoidance:** AEF cities run in a **`spawn` ProcessPool** (not threads). Nested tile/shard/label process pools therefore never `fork` from a worker thread (that hung the equal-area run). Discovery stays on threads (I/O / COPY only).

### Implemented fixes

1. **Within-tile AEF sharding** + shard checkpoints  
2. **Stratified area-balance** sample gate  
3. **Buildings COPY** upsert  
4. **Parallel tile assign**  
5. **Cell/label parallel COPY**  
6. **Decoupled pipeline** — parallel discovery (I/O) + parallel AEF with split worker budget; discovery completion immediately queues AEF  
7. **COG window cache** — before sharding, materialize each tile’s building-union window to a local GeoTIFF (`artifacts/<city>/cog_windows/`); shard workers read disk (full AEF tiles are ~2GB; we only cache the needed window)  
8. **Prune temps** — after `cells.parquet` is written (and again when CITY completes), delete `cog_windows/` + `aef_shards/` to avoid multi-GB bloat
