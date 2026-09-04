# Alpha-Earth-Embeddings-Research

Private study: **AlphaEarth embeddings for building-type classification and carbon-emissions estimation.**

Repo: [Kedar-V/Alpha-Earth-Embeddings-Research](https://github.com/Kedar-V/Alpha-Earth-Embeddings-Research)

## Status

**Phase 1** — literature / dataset identification and Overture label feasibility.  
**Trial** — AlphaEarth + logistic regression for residential vs non-residential across US metros.  
**Phase 2 GT** — multi-class building-type labels + AlphaEarth centroid embeddings + spatial alignment QA.  
**Protocol B baseline** — locked LORO + macro-F1 on US AlphaEarth-only features (notebook 05).

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/01_phase1_alphaearth_overture_briefing.ipynb`](notebooks/01_phase1_alphaearth_overture_briefing.ipynb) | Tuesday meeting briefing: AlphaEarth coverage & methodology; live Overture buildings/Places queries for residential vs non-residential label clarity |
| [`notebooks/02_trial_residential_classifier.ipynb`](notebooks/02_trial_residential_classifier.ipynb) | Trial classifier: Overture labels + AlphaEarth 64-D embeddings → residential vs non-residential; leave-one-region-out eval + US performance map |
| [`notebooks/03_building_type_gt.ipynb`](notebooks/03_building_type_gt.ipynb) | Multi-class GT construction: Overture study-rollup labels ↔ AlphaEarth embeddings; parquet outputs + spatial alignment visuals |
| [`notebooks/04_sota_comparable_building_type_dataset.ipynb`](notebooks/04_sota_comparable_building_type_dataset.ipynb) | Multi-class building-type dataset (SOTA-comparable taxonomy) + AlphaEarth embeddings; multi-modal-ready schema for future street-view / VHR / graphs |
| [`notebooks/05_us_loro_building_type_baseline.ipynb`](notebooks/05_us_loro_building_type_baseline.ipynb) | Protocol B: locked LORO + macro-F1 AlphaEarth-only baseline on the US notebook-04 dataset |

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

Overture cells need network access (public S3 GeoParquet). Notebooks 02–03 need a Google Earth Engine account and Cloud project.

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
