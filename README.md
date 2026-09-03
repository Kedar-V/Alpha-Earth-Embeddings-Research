# Alpha-Earth-Embeddings-Research

Private study: **AlphaEarth embeddings for building-type classification and carbon-emissions estimation.**

Repo: [Kedar-V/Alpha-Earth-Embeddings-Research](https://github.com/Kedar-V/Alpha-Earth-Embeddings-Research)

## Status

**Phase 1** — literature / dataset identification and Overture label feasibility.

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/01_phase1_alphaearth_overture_briefing.ipynb`](notebooks/01_phase1_alphaearth_overture_briefing.ipynb) | Tuesday meeting briefing: AlphaEarth coverage & methodology; live Overture buildings/Places queries for residential vs non-residential label clarity |

## Quick start

```bash
git clone https://github.com/Kedar-V/Alpha-Earth-Embeddings-Research.git
cd Alpha-Earth-Embeddings-Research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_phase1_alphaearth_overture_briefing.ipynb
```

Overture cells need network access (public S3 GeoParquet). AlphaEarth sampling needs a Google Earth Engine account (optional for the briefing narrative).

## Key findings (Phase 1)

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
