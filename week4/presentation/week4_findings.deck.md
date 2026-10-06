---
deck: Week 4 — residential signal in AlphaEarth
audience: Advisor who knows the LORO residential setup and wants the week-4 findings, claims, and limits.
register: presented
structure: dark open and close around light evidence slides; section dividers mark the three arcs
motif: one accent callout number per evidence beat, never a full deck of hero metrics
---

## Slide 1
layout: title
Brief: Open with the through-line; title role; title leads; research briefing register.
Title: Week 4 · residential signal in AlphaEarth
Subtitle: Short · partly shared · geographically brittle
Notes: Three arcs: linear compression, 3D ViTs, geographic subspaces. Claim: the useful residential direction is short; the rest of the 64-D probe is where geography re-enters and transfer breaks.

## Slide 2
layout: composed
Brief: Protocol facts as a compact set; card-grid because the items are siblings; cards lead; lab one-pager.
Title: Same protocol throughout
Block: card-grid
10 metros | downtown boxes
1.75M cells | pure RES/NRES, 1:1 per metro
64-D | frozen AlphaEarth
LORO | hold out one metro; threshold 0.5
Notes: Pure cells under resi_def_v1. Balance seed 42. Same LORO as the week-4 PLS runs.

## Slide 3
layout: section
Brief: Part I divider; section role; title leads.
Title: Part I · Linear compression
Notes: Notebooks 08–10. PLS, L1, 10 m vs 100 m.

## Slide 4
layout: statement
Brief: Baseline number is the whole point; statement earns the hero line; statement leads.
Statement: Mean LORO macro-F1 0.779 — metros still disagree by ±0.089
Notes: Phoenix 0.905, Atlanta 0.868 at the top. NYC 0.673, SF 0.658 at the bottom. The spread is the geographic uncertainty that matters.
Visual: week4/geo_subspace_analysis/figures/baseline_f1_by_metro.png

## Slide 5
layout: title-content
Brief: PLS parity with full LR; title plus three facts; title leads.
Title: PLS matches the full 64-D logistic
Body:
- Baseline LR: 0.779
- PLS-DA → LR (k=32): 0.775
- Δ OOF F1 ≈ −0.001 · AE bands are collinear; PLS scores are orthogonal
Notes: Notebook 08. Supervised compression, not just dim reduction. PCA at the same width lags.
Visual: outputs/week4_pls_loro/figures/best_f1_compare.png

## Slide 6
layout: composed
Brief: Width rules as exact values; table because the numbers must be read; table leads.
Title: Most of the F1 lives in a short supervised map
Block: table
Rule | PLS | PCA
≥95% of baseline | k = 4 | k = 15
within 0.02 F1 | k = 9 | k = 64
within 0.01 F1 | k = 16 | k = 64
Notes: Geo-subspace LORO sweep. PLS reaches the 64-D F1 at smaller width than PCA.
Visual: week4/geo_subspace_analysis/figures/paper/01_retention_vs_k.png

## Slide 7
layout: title-content
Brief: L1 sparsity path; title plus facts; title leads.
Title: L1 collapses the head onto t₁
Body:
- nnz 32 → ~16 (C=100) → ~6 (C=30) → ~1 (C=10)
- C=10 LORO F1 ≈ 0.735 on nearly t₁ alone
- logit ≈ 0.011 + 0.304 · t₁
Notes: Notebook 09. Sparse weights on orthogonal scores are easier to read than sparse weights on raw AE bands.

## Slide 8
layout: composed
Brief: Scale cost as exact values; table because three heads must be read side by side; 100 m column leads the cost.
Title: Mean-pooling to 100 m costs about 0.01–0.02 F1
Block: table
Head | 10 m | 100 m
64-D LR | 0.779 | 0.761
PLS best-k | 0.775 | 0.762
t₁ only | 0.735 | 0.720
Notes: Notebook 10. Same sparsity story at both scales. This is not the 10×10 grid model.
Visual: outputs/week4_pls_10m_vs_100m/figures/f1_family_10m_vs_100m.png

## Slide 9
layout: section
Brief: Part II divider; section role; title leads.
Title: Part II · 3D tile models
Notes: Notebooks 12–16. Forcing a 3D embedding raises the gap, not the LORO F1.

## Slide 10
layout: composed
Brief: Run ladder as exact values; table; F1 column leads.
Title: Linear 100 m still wins on F1
Block: table
Run | LORO F1 | Mean held-out gap
100 m mean-pool LR | 0.761 | —
ViT 128-D | 0.756 | —
v1 BCE + SupCon | 0.732 | 1.76
v3 longer train | 0.735 | 1.71
v4 within-region gap | 0.738 | 2.13
Notes: Gap target was 3. Each fold has its own 3D axes — do not pool them across metros.

## Slide 11
layout: two-column
Brief: Easy vs hard metro; two-column contrast; left Atlanta leads as the clear case.
Title: Atlanta clears a gap of 3.5 · New York does not
Left:
- Atlanta F1 0.898
- Held-out gap 3.51
Right:
- NYC F1 0.586
- Held-out gap 0.63
Notes: v4 encoder outputs. The cut sorts mass along a thin line; supports still overlap.
Visual: outputs/week4_vit_100m_3d_v4/figures/vit3d_Atlanta_SE.png | outputs/week4_vit_100m_3d_v4/figures/vit3d_NYC_NE.png

## Slide 12
layout: title-content
Brief: NYC fold validation vs holdout; title plus facts; the 3.33 vs 0.63 gap leads.
Title: Validation metros separate · held-out NYC collapses
Body:
- Best val gap in the NYC fold: 3.33
- Same checkpoint on NYC: 0.63
- Phoenix in that val split: gap 5.44
- Failure is the held-out cloud, not a broken training loop
Notes: Notebook 16. Other cities in the NYC fold still separate.
Visual: outputs/week4_vit_100m_3d_v4/figures/val_nycfold_Phoenix_SW.png

## Slide 13
layout: section
Brief: Part III divider; section role; title leads.
Title: Part III · Geographic subspaces
Notes: Are task subspaces shared, and does sharing track transfer?

## Slide 14
layout: composed
Brief: Three questions as siblings; card-grid; cards lead.
Title: What this suite answers
Block: card-grid
Signal in frozen AE? | LORO 64-D mean F1 0.779
How low-dimensional? | PLS ≥95% at k=4; PCA needs k=15
Shared + transfer? | Partly shared; short PLS transfers better
Notes: Units are metros and 45 pairs. Pair p-values are descriptive.

## Slide 15
layout: title-content
Brief: Subspace alignment numbers; title plus facts; 0.292 leads.
Title: Task subspaces are only partly aligned
Body:
- Mean off-diagonal PLS similarity at k=2: 0.292
- At k=9: 0.462
- Class-direction cosine mean: 0.688
- Neither identical nor orthogonal
Notes: Within-metro PLS bases, QR-orthonormalized.
Visual: week4/geo_subspace_analysis/figures/subspace_k2.png

## Slide 16
layout: composed
Brief: Transfer comparison; table of off-diagonal F1; PLS k=2 row leads.
Title: Short PLS transfers better than full 64-D LR
Block: table
Representation | Off-diagonal F1
64-D LR | 0.586
PLS k=2 | 0.704
PLS k=9 | 0.607
Notes: Extra components fit the source metro and travel worse. Retention tells the same story.
Visual: week4/geo_subspace_analysis/figures/paper/04_transfer_matrices.png

## Slide 17
layout: title-content
Brief: What predicts transfer; title plus facts; k=2 ρ leads.
Title: k=2 similarity tracks retention · geography does not
Body:
- PLS similarity k=2: Spearman ρ = 0.321 (perm p ≈ 0.033)
- Class-direction cosine: ρ = 0.115
- Great-circle distance: ρ = −0.100
- 45 pairs from 10 metros — descriptive, not causal
Notes: Partial correlation of k=2 similarity given MMD and geo still positive (r ≈ 0.29).
Visual: week4/geo_subspace_analysis/figures/predictor_spearman.png

## Slide 18
layout: composed
Brief: Metro-ID probe; table; metro accuracy column leads the drop at k=2.
Title: PLS k=2 keeps the building label · drops metro identity
Block: table
Rep | Building F1 | Metro accuracy
raw 64 | 0.819 | 0.999
PLS k=2 | 0.775 | 0.298
PLS k=9 | 0.818 | 0.926
Notes: Pixel diagnostic, not the geographic sample. Geography re-enters after the first few components.
Visual: week4/geo_subspace_analysis/figures/paper/02_pareto_building_vs_metro.png

## Slide 19
layout: title-content
Brief: NYC geometry; title plus facts; Chicago vs Atlanta cosines lead.
Title: NYC’s class direction sits with Chicago — not easy targets
Body:
- NYC ↔ Chicago cosine 0.815
- NYC ↔ Atlanta 0.479
- v4 NYC failure is not a rotated axis (min source cosine 0.995, probe F1 0.578)
- Axis shared · cloud still overlaps
Notes: Cosine of means is not a subspace test; k=2 similarity tells the same story.
Visual: week4/geo_subspace_analysis/figures/class_direction_cosine.png

## Slide 20
layout: two-column
Brief: Supported vs not; two-column; left supported leads.
Title: What holds · what does not
Left:
- ~0.78 LORO F1 in frozen AE
- Short PLS carries most of it
- Subspaces only moderately aligned
- PLS k=2 ships better than 64-D
Right:
- Alignment causes transfer
- Global geographic generalization
- v4 fails because the axis points wrong
Notes: Supported for these ten US downtowns only. Correlations on 45 dependent pairs are descriptive.

## Slide 21
layout: statement
Brief: Closing claim; statement earns big text; statement leads.
Statement: Keep the short residential direction — geography lives in the rest
Notes: Offer to walk figures or geo_subspace RESULTS.md if they want depth.
