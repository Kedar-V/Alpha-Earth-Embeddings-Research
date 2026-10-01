# Alpha-Earth-Embeddings-Research

Private study: **AlphaEarth embeddings for building-type classification and carbon-emissions estimation.**

Repo: [Kedar-V/Alpha-Earth-Embeddings-Research](https://github.com/Kedar-V/Alpha-Earth-Embeddings-Research)

## Status

**Phase 1** — literature / dataset identification and Overture label feasibility.  
**Trial** — AlphaEarth + logistic regression for residential vs non-residential across US metros.  
**Phase 2 GT** — multi-class building-type labels + AlphaEarth centroid embeddings + spatial alignment QA.  
**Protocol B baseline** — locked LORO + macro-F1 on US AlphaEarth-only features (notebook 05).  
**Residential definition** — locked as [`resi_def_v1`](docs/residential_definition_v1.md) (building GT + 10 m cell fraction).

### Canonical building-type dataset

Reusable labeled set for later modeling (research-brief categories + AlphaEarth):

```bash
python scripts/build_building_type_dataset.py
# outputs/building_type_dataset/dataset.parquet
# outputs/building_type_dataset/GAPS.md          # what's missing
```

| Brief category | Study label |
|---|---|
| Single-family residential | `single_family` |
| Multi-family residential | `multi_family` |
| Schools | `education` |
| Hospitals | `medical` |
| Commercial or office | `retail` + `office` |
| Additional | hotel, industrial, religious, civic |

## Weekly briefings (read these first)

Short story notebooks with figures already under `outputs/` (no retrain needed):

| Week | Notebook |
|---|---|
| 1 | [`notebooks/week1/00_briefing.ipynb`](notebooks/week1/00_briefing.ipynb): embeddings, Overture GT, binary + 10-class US baselines |
| 2 | [`notebooks/week2/00_briefing.ipynb`](notebooks/week2/00_briefing.ipynb): DoU / hinterland, missingness bias, bbox CNN |
| 3 | [`notebooks/week3/00_briefing.ipynb`](notebooks/week3/00_briefing.ipynb): building vs pixel ladder, soft-fraction 3-class, 3×3 CNN |

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/01_phase1_alphaearth_overture_briefing.ipynb`](notebooks/01_phase1_alphaearth_overture_briefing.ipynb) | Tuesday meeting briefing: AlphaEarth coverage & methodology; live Overture buildings/Places queries for residential vs non-residential label clarity |
| [`notebooks/02_trial_residential_classifier.ipynb`](notebooks/02_trial_residential_classifier.ipynb) | Trial classifier: Overture labels + AlphaEarth 64-D embeddings → residential vs non-residential; leave-one-region-out eval + US performance map |
| [`notebooks/03_building_type_gt.ipynb`](notebooks/03_building_type_gt.ipynb) | Multi-class GT construction: Overture study-rollup labels ↔ AlphaEarth embeddings; parquet outputs + spatial alignment visuals |
| [`notebooks/04_sota_comparable_building_type_dataset.ipynb`](notebooks/04_sota_comparable_building_type_dataset.ipynb) | Multi-class building-type dataset (SOTA-comparable taxonomy) + AlphaEarth embeddings; multi-modal-ready schema for future street-view / VHR / graphs |
| [`notebooks/05_us_loro_building_type_baseline.ipynb`](notebooks/05_us_loro_building_type_baseline.ipynb) | Protocol B: locked LORO + macro-F1 AlphaEarth-only baseline on the US notebook-04 dataset |
| [`notebooks/06_overture_eda.ipynb`](notebooks/06_overture_eda.ipynb) | Overture EDA: buildings/Places feature catalog, fill rates, subtype/class distributions, Places vs tags, US region→state→city hierarchy |
| [`notebooks/07_building_type_dataset.ipynb`](notebooks/07_building_type_dataset.ipynb) | **Building-type dataset construction** (research brief categories) + AlphaEarth join; loads `outputs/building_type_dataset/` |
| [`notebooks/08_alphaearth_embedding_eda.ipynb`](notebooks/08_alphaearth_embedding_eda.ipynb) | Exploratory AlphaEarth embedding analysis: PCA, UMAP, clustering, distances, k-NN retrieval, similarity |

## Quick start

```bash
git clone https://github.com/Kedar-V/Alpha-Earth-Embeddings-Research.git
cd Alpha-Earth-Embeddings-Research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# On Debian without python3-venv: virtualenv -p python3 .venv && ln -sfn local/bin .venv/bin
earthengine authenticate --auth_mode=notebook   # required for notebook 02 (no gcloud needed)
export EE_PROJECT=your-ee-cloud-project-id
jupyter notebook notebooks/03_building_type_gt.ipynb
```

Map UI (after artifacts exist): `python scripts/serve_map_overlay.py` → [http://127.0.0.1:8765/](http://127.0.0.1:8765/) — see **Interactive map UI** below.

Overture cells need network access (public S3 GeoParquet). Notebooks 02–03 need a Google Earth Engine account and Cloud project.

## Interactive map UI (Overture × AlphaEarth × CNN)

Leaflet overlay for building polygons, enclosing bboxes, AlphaEarth 10 m grid cells, CNN embedding color (**PC1 → cividis**), address / lat-lon search, and **misclassified-only** inspection.

### Start the UI

```bash
# from repo root
source .venv/bin/activate   # or: .venv/bin/python …

# serve (GeoJSON already under outputs/week2_cnn_bbox/map_overlay/)
python scripts/serve_map_overlay.py
# → open http://127.0.0.1:8765/
```

Optional host/port: `python scripts/serve_map_overlay.py --host 127.0.0.1 --port 8765`

If `outputs/week2_cnn_bbox/map_overlay/index.html` is missing, build it first:

```bash
python scripts/build_map_overlay.py
python scripts/serve_map_overlay.py
```

### In the UI

1. **Region** → **Load region** (or search an address — auto-loads nearest metro and zooms in).
2. Toggle layers: polygons, **enclosing bbox**, CNN color, 10 m grid cells, residential outlines.
3. **Cell color** dropdown: AE PC1 (cividis), similarity to building mean, or building CNN PC1.
4. **Errors**: show misclassified only (residential or 10-class); color by FP / FN / wrong type.
5. Paste `lat, lon` → **Go to lat/lon**, or use the map geocoder.

**Requires** prior bbox-grid + CNN outputs (`datasets/building_bbox_grid/`, `outputs/week2_cnn_bbox/cnn_embeddings_oof.parquet`). Rebuild those with:

```bash
python scripts/build_building_bbox_grid_dataset.py --rebuild
python scripts/extract_cnn_bbox_embeddings.py --epochs 25 --workers 8
python scripts/export_cnn_oof_predictions.py   # for misclassified toggle
python scripts/build_map_overlay.py
```

Related notebook: [`notebooks/week2/02_cnn_bbox_grid_training.ipynb`](notebooks/week2/02_cnn_bbox_grid_training.ipynb).

## Key findings (Phase 1)

See **[`docs/findings.md`](docs/findings.md)** for the full Tuesday meeting story (embeddings + Overture GT + baseline results).

1. **AlphaEarth Satellite Embedding V1 is global** at 10 m for calendar years 2017–2024 (land + shallow coastal water). Weights are not released; consume precomputed layers via Earth Engine / GCS.
2. **Overture building-use labels are schema-clear but sparsely filled.** Subtype/class come from OSM only; Microsoft/Google ML footprints are unlabeled. Label rates in sample extracts: Berlin ~75%, Amsterdam ~70%, Boston ~22%, Houston ~16%, Nairobi ~2%, Lagos ~1%.
3. **Residential vs non-residential is clean when subtype is present**; SF vs MF is clean only for `house`/`detached` vs `apartments`. Prefer Places joins / GHS-OBAT as noisy GT outside well-mapped cities. Never treat unlabeled footprints as negatives.

## Layout

```
notebooks/          Research notebooks
scripts/            Notebook builders / helpers
docs/               Notes and meeting artifacts
outputs/            Generated CSVs / figures (gitignored when large)
requirements.txt
```

## References

- [Satellite Embedding V1 (Earth Engine)](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
- [AlphaEarth Foundations paper](https://arxiv.org/abs/2507.22291)
- [Overture Maps](https://overturemaps.org/) · [Buildings guide](https://docs.overturemaps.org/guides/buildings/)
