# Claude prompt — US metro generalisation performance

Attach these four images (same folder as this file), then paste the prompt below.

| # | File | What it shows |
|---|------|----------------|
| 1 | `01_generalisation_capacity_and_metro.png` | Left: model capacity vs LORO F1. Right: **metro-wise held-out F1** (linear 10 m vs ViT v4 100 m) |
| 2 | `02_classification_metro_loro.png` | Left: balanced pixel LR leave-one-metro-out F1 by metro |
| 3 | `03_embedding_umap_wrong_side_by_metro.png` | UMAP per metro + 2D LR cut; coloured = wrong side of the cut |
| 4 | `04_embedding_gap_vs_f1.png` | ViT v4 cluster-gap vs held-out F1 by metro |

---

## Prompt (copy from here)

```text
You are editing stakeholder-facing prose about AlphaEarth residential vs non-residential classification on 10 US metros. I am attaching four figures from our Weeks 1–5 report.

Task
1. Write a tight “Generalisation performance” section (≈250–400 words) plus a short “What this means” bullet block.
2. Metro-wise performance must be first-class, not a footnote. Include a compact table of all 10 metros.
3. Then run an avoid-ai-writing pass (detect + rewrite): cut AI-isms, keep numbers exact, no invented claims.

Hard constraints
- Ground every number in the attached figures or the tables below. Do not invent metros, scores, or mechanisms.
- Protocols differ slightly across panels (10 m cell linear vs 100 m tile ViT). Compare shape / ranking, not absolute level across families, unless the same protocol is stated.
- Tone: blunt technical stakeholder brief. No chatbot openers, no “in a rapidly evolving…”, no significance inflation.
- Prefer per-metro floor language over mean-only celebration.

Facts you may use (source CSVs behind the figures)

Leave-one-metro-out macro-F1 — linear head on balanced 10 m cells
(mean 0.779, std 0.090):

Metro        Linear 10m F1
SF_W         0.657
NYC_NE       0.673
Chicago_MW   0.711
Boston_NE    0.719
Seattle_NW   0.765
Houston_S    0.768
LA_W         0.857
Denver_W     0.867
Atlanta_SE   0.868
Phoenix_SW   0.906

Same held-out metros — 3D ViT v4 on 100 m tiles (mean 0.738):

Metro        ViT v4 F1
SF_W         0.584
NYC_NE       0.586
Chicago_MW   0.553
Boston_NE    0.693
Houston_S    0.709
LA_W         0.793
Seattle_NW   0.824
Denver_W     0.862
Phoenix_SW   0.878
Atlanta_SE   0.898

Aggregate model comparison (leave-one-metro-out macro-F1):
- 64-D LR: 0.779 (10 m cells) / 0.761 (100 m tiles)
- PLS-DA best k: 0.775 / 0.762
- PLS t1 only: 0.735 / 0.720
- 3D ViT v1–v4 (100 m): 0.732 / 0.632 / 0.735 / 0.738

Other report facts (use only if useful):
- Pooled splits overstate quality by ~0.03–0.06 (Week 1: 0.83 vs 0.77; Week 4 OOF vs LORO similarly).
- Metro range on linear LORO: Phoenix 0.91 → SF 0.66.
- UMAP wrong-side share of a 2D LR cut: Phoenix ~20% → NYC ~43%, mean ~31%.
- Hard metros under both families: SF, NYC, Chicago, Boston. Easy: Phoenix, Atlanta, Denver.
- Seattle and Atlanta are the main places ViT v4 beats the linear head; elsewhere linear wins or ties.

Required structure
A. One-paragraph verdict (capacity did not buy robustness; metros dominate).
B. Metro-wise table (all 10 rows; Linear 10m + ViT v4; sort hard→easy by linear F1).
C. 3–5 bullets under “What this means” (acceptance = per-metro floor; keep model small; dense mixed-use cores as a separate slice).
D. Optional one-line caption notes for figures 1 and 3 only.

Output
- Final rewrite only (no process narration).
- After the section, a short “Changes vs naive draft” list if you rewrote AI phrasing; else say “clean”.
```

---

## Paths (absolute)

- Pack: `/home/kv133/Alpha-Earth-Embeddings-Research/week5/reports/claude_gen_performance_pack/`
- Source gen figure: `/home/kv133/Alpha-Earth-Embeddings-Research/week5/reports/stakeholder_figures/03_generalisation.png`
- Linear LORO CSV: `/home/kv133/Alpha-Earth-Embeddings-Research/outputs/week4_pixel_balanced_train/loro_metrics.csv`
- ViT LORO CSV: `/home/kv133/Alpha-Earth-Embeddings-Research/outputs/week4_vit_100m_3d_v4/loro_metrics.csv`
