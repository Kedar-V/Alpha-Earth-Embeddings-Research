# Phase 1 meeting notes

Briefing notebook: `notebooks/01_phase1_alphaearth_overture_briefing.ipynb`

## Agenda answers (short)

### AlphaEarth — global?

Yes. Public annual Satellite Embedding V1 covers terrestrial land + shallow coastal waters worldwide at 10 m for 2017–2024. Poles limited by orbits. Weights not released. Prefer EE annual layers for Phases 1–3; Custom Embeddings later for sub-annual change.

### Overture — usable building GT?

Schema: yes (`subtype` / `class` map cleanly onto residential vs non-residential and most study classes). Completeness: no — OSM-only labels; ML footprints unlabeled; fill rate collapses in Global South sample extracts. Recommended stack: high-precision OSM classes → Places joins → optional GHS-OBAT weak residential flags.
