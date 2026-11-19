# Google Maps + Overture overlay (US metros) — Design

**Date:** 2026-11-10  
**Status:** draft for review  
**Approved choices:** interactive Google Maps JS (option 1) · stratified sample (option 3) · RES/NRES + study-label color toggle (option 3)

## Goal

Interactive map with a **Google Maps basemap** and **Overture building footprints** as a toggleable overlay for the ten Track-A US metros, so we can visually check residential vs non-residential polygons on real imagery/roads.

## Non-goals (v1)

- Full 492k building load
- GHSL / AlphaEarth / CNN layers (existing Leaflet UI covers those)
- Google Static Maps PNG gallery
- Editing or uploading footprints
- Committing the API key into HTML or git

## Inputs

| Source | Role |
|---|---|
| `datasets/building_bbox_grid/buildings.parquet` | Overture footprints (`wkt`, `region`, `y_res`, `label`, `class`, `lon`, `lat`) |
| `.env` → `GOOGLE_MAPS_API_KEY` | Maps JavaScript API key (gitignored) |
| Ten `region` values already in the parquet | City switcher |

## Sampling

Per metro:

- Target **n = 3,000** buildings (or all if fewer).
- Stratify **1:1 RES / NRES** by `y_res` when both classes have enough rows; otherwise take all of the minority class and fill the remainder from the majority.
- `seed = 42`, deterministic.
- Keep columns: `id`, `region`, `y_res`, `label`, `class`, `lon`, `lat`, geometry from `wkt`.

## Outputs

```text
outputs/google_maps_overture/
├── index.html              # Maps JS UI (no API key embedded)
├── regions.json            # metro centers, bounds, sample counts
├── geojson/
│   ├── Atlanta_SE.geojson
│   ├── Boston_NE.geojson
│   └── … (10 metros)
├── manifest.json           # sample n, seed, color maps, build time
└── README.md               # how to serve
```

## UI behavior

- **Basemap:** Google Maps (roadmap default; optional satellite toggle if cheap).
- **City switcher:** dropdown of 10 metros; loads that metro’s GeoJSON; fits bounds.
- **Layer toggle:** Overture polygons on/off.
- **Color mode toggle:**
  - Default **RES / NRES:** RES `#1F6FA8`, NRES `#C2691A`.
  - **Study label:** fixed palette for `{single_family, multi_family, education, medical, retail, office, hotel, industrial, religious, civic}`.
- **Click:** info window with `label`, `class`, `y_res`, `id`.
- **Legend:** updates with color mode.

## Serving / API key

- `scripts/build_google_maps_overture.py` — sample + write GeoJSON + HTML shell.
- `scripts/serve_google_maps_overture.py` — HTTP server on `127.0.0.1:8770` that:
  - serves static files from `outputs/google_maps_overture/`
  - injects `GOOGLE_MAPS_API_KEY` from `.env` into a tiny `/config.js` (or HTML template) at request time so the key is never written into the built artifact committed to the repo
- Browser URL: `http://127.0.0.1:8770/`

## Performance notes

- 3k polygons/metro is within comfortable Maps Data layer / GeoJSON load size.
- Load **one metro at a time** (do not preload all 10).
- Optional later: simplify geometries; not required for v1 sample size.

## Success criteria

1. Opening the local URL shows Google basemap for a default metro (e.g. Boston).
2. Overture polygons render and can be toggled.
3. Color mode switches between RES/NRES and study labels with a matching legend.
4. All 10 metros selectable; each loads its own sample GeoJSON.
5. API key is read from `.env` only; `git status` does not show the key in tracked files.

## Implementation touchpoints

| File | Action |
|---|---|
| `scripts/build_google_maps_overture.py` | create |
| `scripts/serve_google_maps_overture.py` | create |
| `outputs/google_maps_overture/**` | generated artifacts |
| `.env` | already has `GOOGLE_MAPS_API_KEY` |

## Spec self-review

- No placeholders left.
- Scope matches approved options (JS map, stratified sample, dual color mode).
- Key handling explicit (serve-time inject, not baked into build).
- Ambiguity resolved: n=3000, seed=42, one metro at a time.
