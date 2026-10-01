# Week 2 — Degree of Urbanization (GHSL) findings

**Kyle’s question:** Does residential vs non-residential behave the same in the city, the suburbs, and the countryside?

**Method (short):** Attach a GHSL GHS-SMOD **Degree of Urbanization (DoU)** label to each building (1 km neighborhood → urban / peri-urban / rural). Expand each notebook-02 downtown box by ~39 km so suburbs and countryside exist. Compare Overture tag fill rate and leave-one-region-out (LORO) F1 by DoU.

**Source notebook:** [`notebooks/09_week2_ghsl_degree_of_urbanization.ipynb`](../notebooks/09_week2_ghsl_degree_of_urbanization.ipynb)  
**Artifacts:** [`outputs/week2_ghsl_dou/`](../outputs/week2_ghsl_dou/)

---

## Plot 1 — Downtown sample was almost entirely urban

![Downtown boxes are almost entirely urban](../outputs/week2_ghsl_dou/core_is_urban.png)

**Finding.** Notebook 02’s tight metro cores are **98.7% urban** (n = 3,554 of 3,600). Peri-urban and rural are essentially absent. Last week’s LORO F1 is an **urban-core** number — it cannot answer Kyle’s urban → rural question until the sample includes hinterland.

---

## Plot 2 — Expanding the box brings in suburbs and countryside

![Where the hinterland sample sits](../outputs/week2_ghsl_dou/hinterland_maps_by_dou.png)

**Finding.** The small black rectangle is last week’s downtown; the larger area is this week’s hinterland. Gold (peri-urban) and green (rural) points appear outside the core — especially wide suburban rings in places like Atlanta. Cities differ: LA’s expanded box stays mostly urban; Denver/Seattle show clearer urban → rural transitions.

---

## Plot 3 — Overture building-use tags get rarer outside the city

![Tags get rarer as you leave the city](../outputs/week2_ghsl_dou/fill_rate_story.png)

**Finding.** Subtype fill rate falls **35% → 20% → 14%** (urban → peri-urban → rural). Rural ground truth is thinner. Among buildings that *are* tagged, residential still dominates (~84–92%). Unlabeled footprints stay excluded — never treated as non-residential.

| DoU | Footprints | Fill rate | Tagged n | % residential among tagged |
|---|---:|---:|---:|---:|
| Urban | 7,692 | 35.0% | 2,692 | 91% |
| Peri-urban | 2,098 | 20.3% | 425 | 92% |
| Rural | 1,010 | 14.0% | 141 | 84% |

---

## Plot 4 — Classifier holds with a mild urban → rural drop

![F1 by DoU and city](../outputs/week2_ghsl_dou/f1_by_dou_and_city.png)

**Finding.** Same model as notebook 02 (logistic regression on 64-D AlphaEarth, LORO), with hinterland labels added:

| DoU | n (OOF) | Macro F1 (train all DoU) | Macro F1 (train urban only) |
|---|---:|---:|---:|
| Urban | 5,092 | **0.78** | 0.78 |
| Peri-urban | 460 | **0.76** | 0.75 |
| Rural | 208 | **0.74** | 0.71 |

- Mild drop urban → rural (**0.78 → 0.76 → 0.74**).
- Training on **urban only** hurts rural most (0.74 → 0.71) — Kyle’s “works in cities, fails in rural?” test is real but not catastrophic on this sample.
- Rural n is small; treat rural F1 as **directional**. City holdout still varies a lot (LA ~0.85 … NYC ~0.63).

---

## One-slide takeaway

1. **Downtown-only LORO ≠ urban/peri/rural story** — need hinterland.
2. **Label availability is the bigger DoU gap** (fill 35→14%) than classifier collapse.
3. **Report LORO F1 overall and by DoU going forward**; keep hinterland (or another rural draw) in the sample.
4. Do not treat unlabeled rural footprints as negatives.

---

## How to present (suggested order)

| Slide | Plot file | One line |
|---|---|---|
| 1 | `core_is_urban.png` | Last week = urban-core F1 |
| 2 | `hinterland_maps_by_dou.png` | Expanded boxes fix the sample |
| 3 | `fill_rate_story.png` | Tags thin outside the city |
| 4 | `f1_by_dou_and_city.png` | Mild F1 drop; urban-only training hurts rural |

Optional backup: `takeaway_cards.png`, `taxonomy_rollup.png`, `boston_atlanta_classifier_maps.png`.
