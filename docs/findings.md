# Findings: Tuesday meeting story

**Project:** AlphaEarth embeddings for building-type classification and carbon-emissions estimation  
**Audience:** First project meeting — embeddings availability + Overture ground truth  
**Evidence base:** Notebooks 01–05 in this repo (Phase 1 briefing through locked US LORO baseline)

---

## The story in 60 seconds

We need two things to run this study: **global embeddings** and **building-use labels**.

1. **Embeddings are not the blocker.** Google’s public AlphaEarth Satellite Embedding V1 is already global (land + shallow coast), 10 m, annual layers 2017–2024, 64-dimensional, accessible via Earth Engine. We can sample the same representation in Boston, Berlin, or Lagos.

2. **Labels are the hinge.** Overture Maps *can* supply effective ground truth for **residential vs non-residential** when a use tag is present — the categories are clear. But fill rates swing from ~75% in Berlin to ~1% in Lagos, because use labels come from OSM, not from the ML footprint sources. So the plan is: lock a **clean Overture core**, then use **automated noisy GT** (Places joins, GHS-style flags, etc.) to go global — never treating unlabeled buildings as “non-residential.”

3. **We’ve already stress-tested the pipeline.** On a US 10-class, AlphaEarth-only baseline with leave-one-region-out, macro-F1 is ~0.22 — above chance, far below multi-modal SOTA. That supports starting with **binary residential** and high-precision classes for carbon archetypes, not claiming fine building-type SOTA from embeddings alone.

---

## Agenda item 1 — What embeddings are available, and across what geographies?

### Bottom line

**Yes — we have global access up front.** We do not need to wait on a custom global training run to start Phases 1–3.

### Options

| Option | Geography | Role |
|--------|-----------|------|
| **Annual Satellite Embeddings V1** (recommended now) | Global terrestrial land + shallow coastal water | Building-type work, geographic transfer |
| **Custom Satellite Embeddings** (Maps Platform preview) | User-defined AOI / time windows | Later: sub-annual change / stretch temporal goals |

**Access:** Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (also GCS). Model weights are **not** released; we consume precomputed vectors.

### How the embeddings are developed

- **Product:** AlphaEarth Foundations → public Satellite Embedding V1 layers  
- **Representation:** 64-D unit-length vector per cell (`A00`–`A63`)  
- **Resolution:** **10 m**  
- **Time:** calendar years **2017–2024** (eight annual mosaics)  
- **Sensors / inputs:** multimodal Earth observation — optical (Landsat / Sentinel-2 class) and **Sentinel-1 C-band SAR**, among other streams used in foundation training (not a single commercial VHR façade sensor)  
- **Training setup (paper-scale):** millions of year-length video sequences from globally stratified ~1.28 km sites; the model learns a continuous embedding field of the Earth’s surface  
- **Caveats:** poles limited by orbits; deep ocean not the target; 10 m is coarse for a small house (~1–4 pixels) but neighborhood context is baked in via large training chips  

### Why this matters for carbon work

Same embedding space worldwide means we can train in label-rich metros and **test transfer** to unseen cities — which is required if we want scalable building-sector emissions estimates without hand-labeling every city.

**References**

- [Earth Engine — Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)  
- [AlphaEarth Foundations (arXiv)](https://arxiv.org/abs/2507.22291)  

**Supporting notebook:** [`notebooks/01_phase1_alphaearth_overture_briefing.ipynb`](notebooks/01_phase1_alphaearth_overture_briefing.ipynb)

---

## Agenda item 2 — Overture Maps as ground truth (residential vs non-residential)

### Bottom line

**Schema: clear. Completeness: uneven.**  
Overture is a viable GT source for residential vs non-residential **where `subtype` is filled**. It is not, by itself, a complete global building-use layer.

### What we looked at

Site: [https://overturemaps.org/](https://overturemaps.org/)  
Buildings schema: [`subtype`](https://docs.overturemaps.org/schema/reference/buildings/types/building_subtype/) (broad use) and [`class`](https://docs.overturemaps.org/schema/reference/buildings/types/building_class/) (OSM-style detail), plus footprints from OSM and ML sources (Microsoft / Google / Esri).

### Clarity of residential vs non-residential

**Locked definition:** [`docs/residential_definition_v1.md`](residential_definition_v1.md) (`resi_def_v1`).

- **Building GT:** high-precision Overture `class` → study labels; `y_res = 1` iff SF/MF. Ambiguous tags (`residential`, `terrace`, unlabeled) excluded — never negatives.
- **10 m cells:** area-weighted residential fraction \(f_{\mathrm{res}}\) with τ=0.7 and an explicit **mixed** band (not majority class).
- **Weak labels** (subtype-only, Places, OBAT): separate tracks; not pooled into GT F1.

Historical note (pre-lock): when only `subtype` is present, `residential` vs other non-null subtypes is schema-clean for a *weak* track, but **`resi_def_v1` GT does not use subtype alone.**


### Completeness (live extracts from public Overture GeoParquet)

| City | Approx. % with subtype | Read |
|------|------------------------:|------|
| Berlin | ~75% | Strong GT density |
| Amsterdam | ~70% | Strong |
| Boston | ~22% | Usable labeled core |
| Houston | ~16% | Thin |
| Nairobi | ~2% | Almost no use GT |
| Lagos | ~1% | Almost none |

**Why:** use attributes are OSM-derived. ML footprints add geometry but **zero** subtype/class. Unlabeled buildings must **not** be treated as negatives.

### GT scenario matching the meeting expectation

He asked whether Overture labels are clear enough to extract as GT, and noted ideas for **automated noisy GT** if a clean core exists.

| Layer | What | Role |
|-------|------|------|
| **A. Clean binary** | Overture `subtype` present → residential vs other | Gold for first models / eval |
| **B. Clean multi-class** | High-precision `class` only | Carbon archetypes (SF/MF, school, hospital, …) |
| **C. Noisy automated GT** | Places→building joins, GHS-OBAT-style flags, land-use heuristics | Scale into OSM-poor regions |
| **D. Unlabeled** | Missing subtype/class | **Excluded** — never “non-residential by default” |

**One-sentence GT story:**  
Train and evaluate on Overture’s clear residential vs non-residential (and high-precision classes where available); use automated noisy methods to expand coverage; never poison training with unlabeled footprints.

**Supporting notebooks:** 01 (Overture feasibility) · 02 (binary residential trial) · 03–04 (multi-class GT construction)

---

## What we learned after the briefing (strengthens Tuesday)

We did not stop at literature. We built GT, joined AlphaEarth, and ran a locked evaluation.

### Pipeline already standing

- Multi-class US dataset: **~11.3k** buildings × **10** metros × **10** types + 2024 AlphaEarth centroid embeddings ([`notebooks/04_…`](notebooks/04_sota_comparable_building_type_dataset.ipynb))  
- Protocol B baseline: leave-one-region-out + macro-F1 ([`notebooks/05_…`](notebooks/05_us_loro_building_type_baseline.ipynb))

### Headline result (AlphaEarth-only, 10-class)

| Protocol | Best model | Macro-F1 | Accuracy |
|----------|------------|---------:|---------:|
| **LORO (primary)** | HistGradientBoosting | **0.22 ± 0.03** | ~0.28 |
| Pooled 80/20 (optimistic) | HistGradientBoosting | ~0.40 | ~0.46 |
| Chance / majority | — | ~0.10 | ~0.10 / ~0.18 |

**Regional variance:** HGB is more stable than logistic regression (std ~0.03 vs ~0.06). Best holdout ~Chicago; hardest ~Phoenix and NYC. Transfer is real but uneven.

**vs multi-modal SOTA (fair framing):** Liang et al. HeteroGNN reports ~86–96% accuracy and mean F1 ~0.70–0.73 using street view + VHR + morphology + graphs on Amsterdam / DC / Berlin. Our number is a **lower-modality floor**, not a bake-off loss — different inputs, cities, and splits.

**Takeaway for the room:** embeddings carry *some* building-type signal and support a global method; they are **not enough alone** (at centroid sampling) for fine-grained, geographically robust typing. First carbon-relevant win is still **residential vs non-residential + coarse archetypes**, with multi-modal or pooled context as upgrades.

---

## What to say Tuesday (talk track)

1. **“Global embeddings: yes.”** Public AlphaEarth V1, worldwide land coverage, 10 m, 2017–2024, 64-D, EE access. Custom embeddings later for fine time windows.  
2. **“Overture GT: clear categories, sparse fill.”** Residential vs non-residential is effective when labeled; Berlin-class cities are rich, many Global South extracts are not.  
3. **“So the GT design is a stack.”** Clean Overture core → noisy automated labels for scale → never use unlabeled as negatives. That matches the noisy-GT ideas on the table.  
4. **“We’ve already run the loop.”** US multi-class AlphaEarth baseline under locked LORO sits ~0.22 macro-F1 — enough to justify the study design, not enough to claim SOTA building typing from embeddings alone.  
5. **“Proposed next steps.”** (a) Lock binary residential eval globally where subtype exists; (b) prototype Places / weak-label expansion; (c) footprint or buffer pooling before chasing multi-modal SOTA parity.

---

## Suggested decision asks

- Confirm **Annual Satellite Embedding V1 via EE** as the Phase 1–3 embedding source.  
- Confirm **Overture subtype binary** as primary clean GT, with high-precision `class` for archetype experiments.  
- Align on a first **noisy GT** prototype (Places join vs GHS-OBAT vs both).  
- Keep **fine multi-class SOTA comparison** as a later Protocol A (matched paper cities/modalities), not a claim from the current US centroid baseline.

---

## Artifact map

| Notebook | Role in this story |
|----------|--------------------|
| `01_phase1_alphaearth_overture_briefing.ipynb` | Agenda answers + live Overture city stats |
| `02_trial_residential_classifier.ipynb` | Binary residential × AlphaEarth trial |
| `03_building_type_gt.ipynb` | Multi-class GT + alignment QA |
| `04_sota_comparable_building_type_dataset.ipynb` | 10-class US dataset export |
| `05_us_loro_building_type_baseline.ipynb` | Locked LORO + regional variance |

Outputs (local / gitignored when large): `outputs/sota_building_type/`, `outputs/us_loro_baseline/`.
