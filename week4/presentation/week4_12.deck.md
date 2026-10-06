---
deck: Week 4 — residential signal in AlphaEarth
audience: Advisor who knows the LORO residential setup and wants the week-4 findings, claims, and limits.
register: presented
structure: dark open and close; light evidence slides between; section kickers carry the arc
motif: one accent claim per slide; orange marks hard metros only
---

## Slide 1
layout: title
Brief: Open with the through-line; title role; title leads; dark open.
Title: Residential signal in frozen embeddings
Subtitle: Compact · partly shared · geographically brittle
Notes: Ten downtowns, leave-one-metro-out, frozen 64-D AlphaEarth, then 100 m tiles and 3D ViTs. One claim: the useful residential direction is short; geography lives in the rest.

## Slide 2
layout: composed
Brief: Protocol as sibling facts; card-grid; cards lead.
Title: Standardized experimental protocol
Block: card-grid
10 metros | Downtown core boxes
1.75M cells | Balanced 1:1 RES vs NRES
64-D latent | Frozen AlphaEarth
LORO split | Hold-out; threshold 0.50
Notes: Pure cells under resi_def_v1. Balance seed 42. Same folds across linear and tile runs.

## Slide 3
layout: statement
Brief: Baseline number is the whole point; statement earns the hero line; statement leads.
Statement: Mean LORO F1 0.779 — metro spread ±0.089
Notes: Phoenix and Atlanta sit high. NYC and SF sit near 0.66. The spread is the geographic uncertainty that matters.
Visual: week4/geo_subspace_analysis/figures/baseline_f1_by_metro.png

## Slide 4
layout: composed
Brief: Dimensionality as three siblings; card-grid; PLS card leads.
Title: Building signal sits in a short supervised map
Block: card-grid
PLS k=4 | ≥95% of full 64-D F1
PCA k=15 | Same 95% retention
L1 → t₁ | F1 0.735 at C=10
Notes: Supervised PLS beats PCA at the same width. Strong L1 collapses the head onto t₁.
Visual: week4/geo_subspace_analysis/figures/paper/01_retention_vs_k.png

## Slide 5
layout: composed
Brief: Scale cost as exact values; table; 100 m column leads the cost.
Title: Mean-pooling to 100 m costs 0.01–0.02 F1
Block: table
Head | 10 m | 100 m
64-D LR | 0.779 | 0.761
PLS best-k | 0.775 | 0.762
t₁ only | 0.735 | 0.720
Notes: Same sparsity story at both scales. This is mean-pool 64-D, not the 10×10 grid model.
Visual: outputs/week4_pls_10m_vs_100m/figures/f1_family_10m_vs_100m.png

## Slide 6
layout: composed
Brief: ViT ladder as exact values; table; F1 column leads.
Title: Linear 100 m still wins on F1
Block: table
Run | LORO F1 | Mean gap
Mean-pool LR | 0.761 | —
ViT 128-D | 0.756 | —
v1 Supervised Contrastive (SupCon) | 0.732 | 1.76
v3 longer | 0.735 | 1.71
v4 gap loss | 0.738 | 2.13
Notes: Gap target was 3. Raising the held-out gap did not beat linear F1. Each fold has its own 3D axes.

## Slide 7
layout: section
Brief: Open the ViT deep-dive; section kicker; dark.
Title: ViT 3D tile experiment
Subtitle: Training pipeline · loss functions · what Supervised Contrastive (SupCon) did and did not do

## Slide 8
layout: figure
Brief: Pipeline diagram; figure leads.
Title: 100 m tile → 3D residential embedding
Notes: 10×10 AE cells + mask → PLS-DA k=9 → ViT width 128 class token → z∈R³ → loss on z. LORO; v4 stops on val sep/within targeting 3.
Visual: week4/presentation/vit_diagrams/vit_training_pipeline.png

## Slide 9
layout: figure
Brief: Loss equations side by side; figure leads.
Title: Two objectives tried on the same 3D vector
Notes: v1–v3 BCE + Supervised Contrastive / SupCon (detach in v2/v3). v4 BCE+pull+gap on the same z; drop SupCon.
Visual: week4/presentation/vit_diagrams/vit_loss_functions.png

## Slide 10
layout: figure
Brief: Contrastive arc v1→v4; figure leads.
Title: Supervised Contrastive (SupCon) did not deliver separable held-out clusters
Notes: Detached classifier → encoder drifts to chance. Gap loss helps where metros are easy; mean-pool LR still wins F1 (0.761 vs 0.738).
Visual: week4/presentation/vit_diagrams/vit_contrastive_findings.png

## Slide 11
layout: figure
Brief: Results ladder diagram; figure leads.
Title: Ladder vs linear 100 m mean-pool
Notes: v4 mean gap 2.13 at F1 0.738. Atlanta/Phoenix clear; NYC gap 0.63.
Visual: week4/presentation/vit_diagrams/vit_results_ladder.png

## Slide 12
layout: two-column
Brief: Easy vs hard metro; two-column contrast; left Atlanta leads.
Title: Atlanta clears a gap of 3.5 · NYC does not
Left:
- Atlanta F1 0.898
- Held-out gap 3.51
Right:
- NYC F1 0.586
- Held-out gap 0.63
Notes: Validation metros in the NYC fold still separate (best val gap 3.33). Failure is the held-out cloud. Full LORO + NYC-fold validation cluster gallery is appended after the close.
Visual: outputs/week4_vit_100m_3d_v4/figures/vit3d_Atlanta_SE.png | outputs/week4_vit_100m_3d_v4/figures/vit3d_NYC_NE.png

## Slide 13
layout: composed
Brief: Three geo questions as siblings; card-grid; cards lead.
Title: Geographic subspaces — three questions
Block: card-grid
Signal? | LORO 64-D mean F1 0.779
How short? | PLS ≥95% at k=4
Shared? | Partly; short PLS transfers
Notes: Units are metros and 45 pairs. Pair correlations are descriptive.
Visual: week4/geo_subspace_analysis/figures/subspace_k2.png

## Slide 14
layout: composed
Brief: Transfer numbers; table; PLS k=2 row leads.
Title: Short PLS transfers better than full 64-D
Block: table
Representation | Off-diagonal F1
64-D LR | 0.586
PLS k=2 | 0.704
PLS k=9 | 0.607
Notes: Extra components fit the source metro and travel worse. k=2 similarity tracks retention (ρ≈0.32); geography does not.
Visual: week4/geo_subspace_analysis/figures/paper/04_transfer_matrices.png

## Slide 15
layout: composed
Brief: Metro-ID probe; table; metro accuracy drop at k=2 leads.
Title: PLS k=2 keeps the label · drops metro identity
Block: table
Rep | Building F1 | Metro acc
raw 64 | 0.819 | 0.999
PLS k=2 | 0.775 | 0.298
PLS k=9 | 0.818 | 0.926
Notes: Pixel diagnostic. Geography re-enters after the first few components.
Visual: week4/geo_subspace_analysis/figures/paper/02_pareto_building_vs_metro.png

## Slide 16
layout: title-content
Brief: NYC geometry; title plus facts; Chicago vs Atlanta cosines lead.
Title: NYC aligns with Chicago — not easy targets
Body:
- NYC ↔ Chicago cosine 0.815
- NYC ↔ Atlanta 0.479
- v4 NYC failure is not a rotated axis (min cosine 0.995)
- Axis shared · cloud still overlaps
Notes: Cosine of means is not a subspace test; k=2 similarity tells the same story.
Visual: week4/geo_subspace_analysis/figures/class_direction_cosine.png

## Slide 17
layout: two-column
Brief: Supported vs not; two-column; left supported leads; dark close in notes.
Title: What holds · what does not
Left:
- ~0.78 LORO F1 in frozen AE
- Short PLS carries most of it
- Subspaces only partly aligned
- PLS k=2 ships better than 64-D
Right:
- Alignment causes transfer
- Global geographic generalization
- v4 fails because the axis is wrong
Notes: Keep the short residential direction. Geography lives in the rest of the 64-D probe. Ten US downtowns only.
