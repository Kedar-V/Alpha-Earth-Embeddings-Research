# Geographic subspaces in frozen AlphaEarth

Within-US study. Ten downtown metros, pure residential vs non-residential cells, frozen 64-D AlphaEarth. The question is whether the task-discriminative subspace is shared, and whether that sharing tracks cross-metro transfer.

Geographic claims use the metro as the unit. There are 10 metros and 45 unordered pairs. Pair p-values are not ten independent experiments.

## Key findings

| Finding | Evidence | Statistical support | Caveat |
|---|---|---|---|
| Frozen 64-D AlphaEarth already separates the two classes under leave-one-metro-out | Mean macro-F1 0.779 ± 0.089 | Metro bootstrap 95% CI for the mean 0.729–0.832 | Ten US downtowns. The CI resamples metros. |
| A few supervised components keep most of that F1 | PLS: smallest k at ≥95% of baseline F1 = 4; within 0.01 F1 = 16; within 0.02 F1 = 9. PCA: smallest k at ≥95% of baseline F1 = 15; within 0.01 F1 = 64; within 0.02 F1 = 64. | Same 10 LORO folds as the baseline. No pixel bootstrap. | k was a fixed grid, not tuned on the held-out metro. |
| Task subspaces are only partly shared | Mean off-diagonal PLS projection similarity 0.292 at k=2 and 0.462 at k=9. Mean class-direction cosine 0.688. | Full 10×10 matrices saved. | Stage 2 uses one global feature scale so the bases share an inner product. |
| A short PLS map transfers better than the full 64-D probe | Off-diagonal macro-F1 0.704 for PLS k=2 vs 0.586 for 64-D LR | All 90 ordered pairs, diagonal held out of that mean. | Diagonal is an 80/20 split, so it is not the same protocol as the off-diagonal. |
| k=2 subspace similarity lines up with transfer; the 1-D class direction and geographic distance do not | Spearman ρ = 0.321 for k=2 similarity, 0.115 for class-direction cosine, -0.100 for great-circle distance | Permutation p = 0.033 for k=2. Pair-bootstrap CI is in the table below. | 45 pairs from 10 metros. The pairs share metros, so the p-value is descriptive. |
| PLS k=2 keeps the building label and drops metro identity; k=9 brings metro identity back | Pixel probe: raw 64-D metro accuracy 0.999, PLS k=2 0.298, PLS k=9 0.926. Building F1 stays 0.775 at k=2 and 0.818 at k=9 | One 80/20 pixel split, seed 42. Not a metro-level interval. | The split is by pixels inside metros, so building F1 here is not leave-one-metro-out. |
| NYC's class direction sits with Chicago and away from Atlanta, Phoenix, LA, and Seattle | NYC cosine with Chicago 0.815; with Atlanta 0.479 | Single full-sample cosine per metro, no sampling error attached. | Cosine of means is not a subspace test. k=2 similarity tells the same story. |
| v4's held-out NYC failure is not a rotated class axis | In the NYC fold, source-metro directions match NYC's direction (minimum cosine 0.995) while the linear probe F1 is 0.578 | One frozen checkpoint, all tiles. | v4 is a 100 m tile model. It is not the 10 m matrix. |
| Explicit direction alignment in a learned map | Best eval cell is class at d=16, mean held-out F1 0.738. Class-only at that width is 0.738. | 10 metros × 3 seeds. Lambda chosen on a source metro. | One validation metro per fold, not a full inner LORO. Linear map only. |

## 1. How much residential signal is in frozen AlphaEarth?

Leave-one-metro-out logistic regression on the 64-D vectors, scaler fit on the training metros, threshold 0.5.

Mean macro-F1 0.779 ± 0.089. Metro-bootstrap 95% interval for that mean: 0.729–0.832. This matches the published 10 m reference of 0.779.

| Metro | Macro-F1 | Residential F1 | Non-residential F1 | Accuracy | Balanced accuracy | AUROC | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phoenix | 0.905 | 0.901 | 0.910 | 0.906 | 0.906 | 0.967 | 148658 |
| Atlanta | 0.868 | 0.873 | 0.862 | 0.868 | 0.868 | 0.944 | 112772 |
| Denver | 0.864 | 0.867 | 0.862 | 0.865 | 0.865 | 0.941 | 260218 |
| Los Angeles | 0.857 | 0.868 | 0.847 | 0.858 | 0.858 | 0.947 | 456818 |
| Houston | 0.770 | 0.729 | 0.811 | 0.777 | 0.777 | 0.895 | 167934 |
| Seattle | 0.767 | 0.806 | 0.729 | 0.774 | 0.774 | 0.931 | 76990 |
| Boston | 0.719 | 0.754 | 0.684 | 0.723 | 0.723 | 0.833 | 186520 |
| Chicago | 0.710 | 0.665 | 0.755 | 0.717 | 0.717 | 0.837 | 146950 |
| New York | 0.673 | 0.687 | 0.660 | 0.674 | 0.674 | 0.734 | 159998 |
| San Francisco | 0.658 | 0.597 | 0.719 | 0.669 | 0.669 | 0.777 | 34956 |

Phoenix, Atlanta, Denver, and Los Angeles sit at the top. San Francisco and New York sit at the bottom. The spread across metros (0.089) is the geographic uncertainty that matters.

## 2. How low-dimensional is the task signal?

PCA and PLS were fit inside each LORO fold on training metros only, then a logistic regression was fit on the scores.

| k | Model | Mean macro-F1 | Retention of the 64-D mean |
|---:|---|---:|---:|
| 1 | pca | 0.600 ± 0.132 | 0.770 |
| 1 | pls | 0.735 ± 0.123 | 0.943 |
| 2 | pca | 0.729 ± 0.123 | 0.936 |
| 2 | pls | 0.738 ± 0.132 | 0.947 |
| 3 | pca | 0.722 ± 0.136 | 0.926 |
| 3 | pls | 0.739 ± 0.108 | 0.948 |
| 4 | pca | 0.730 ± 0.129 | 0.937 |
| 4 | pls | 0.753 ± 0.096 | 0.967 |
| 5 | pca | 0.725 ± 0.138 | 0.930 |
| 5 | pls | 0.749 ± 0.098 | 0.961 |
| 8 | pca | 0.730 ± 0.131 | 0.937 |
| 8 | pls | 0.757 ± 0.093 | 0.972 |
| 9 | pca | 0.732 ± 0.124 | 0.940 |
| 9 | pls | 0.761 ± 0.098 | 0.977 |
| 12 | pca | 0.734 ± 0.118 | 0.942 |
| 12 | pls | 0.765 ± 0.092 | 0.982 |
| 15 | pca | 0.748 ± 0.095 | 0.959 |
| 15 | pls | 0.767 ± 0.094 | 0.985 |
| 16 | pca | 0.739 ± 0.093 | 0.949 |
| 16 | pls | 0.771 ± 0.094 | 0.989 |
| 24 | pca | 0.708 ± 0.088 | 0.909 |
| 24 | pls | 0.775 ± 0.094 | 0.994 |
| 32 | pca | 0.759 ± 0.088 | 0.974 |
| 32 | pls | 0.775 ± 0.093 | 0.994 |
| 48 | pca | 0.751 ± 0.100 | 0.964 |
| 48 | pls | 0.780 ± 0.093 | 1.001 |
| 64 | pca | 0.779 ± 0.089 | 1.000 |
| 64 | pls | 0.779 ± 0.089 | 1.000 |

PLS: smallest k at ≥95% of baseline F1 = 4; within 0.01 F1 = 16; within 0.02 F1 = 9.

PCA: smallest k at ≥95% of baseline F1 = 15; within 0.01 F1 = 64; within 0.02 F1 = 64.

PLS reaches the 64-D F1 at a smaller width than PCA. By k=9, PLS mean F1 is 0.761 (retention 0.977). PCA at k=9 is 0.732. At k=64 the two maps are full-rank linear reparameterizations; their F1 meets the baseline (0.779 PLS, 0.779 PCA, baseline 0.779).

## 3. Are the task subspaces consistent across metros?

Within each metro, PLS was fit on that metro alone. Bases were orthonormalized with QR. Projection similarity is the mean squared cosine of the principal angles.

Mean off-diagonal similarity: k=1 0.457, k=2 0.292, k=3 0.339, k=5 0.370, k=9 0.462.

k=2 is the least aligned width. The subspaces are neither identical nor orthogonal. Class-mean directions in the raw 64-D space agree more than the k=2 subspaces: mean cosine 0.688, range 0.475–0.884.

New York's nearest class direction is Chicago (0.815). Its farthest are Atlanta (0.479) and Seattle (0.515). At k=2 the same pattern appears: nearest Boston (0.560), farthest Los Angeles (0.173).

## 4. Does alignment predict transfer?

Directional transfer: train on all of metro A, test on all of metro B. The diagonal trains on 80% of B and tests on the other 20%, seed 42, so the same pixels are not in both. Retention is T(A→B) / T(B,B). Symmetric retention averages the two directions.

| Representation | Mean diagonal F1 | Mean off-diagonal F1 | Mean off-diagonal retention | Worst off-diagonal F1 |
|---|---:|---:|---:|---:|
| lr64_k64 | 0.848 | 0.586 | 0.692 | 0.333 |
| pca_k2 | 0.764 | 0.653 | 0.852 | 0.342 |
| pca_k9 | 0.818 | 0.701 | 0.853 | 0.414 |
| pls_k2 | 0.819 | 0.704 | 0.856 | 0.407 |
| pls_k9 | 0.842 | 0.607 | 0.723 | 0.333 |

PLS at k=2 and PCA at k=9 keep more of the target's own F1 under transfer than the 64-D logistic regression. PLS at k=9 gives some of that back: the extra components fit the source metro and travel worse. The useful residential direction is short. The rest of the 64-D probe is geographically brittle.

Spearman and Pearson correlations are over the 45 unordered pairs. The permutation test shuffles one side of the pair. The interval is a pair bootstrap and ignores the dependence among pairs that share a metro.

| Predictor of symmetric 64-D retention | Spearman ρ | Spearman p | Permutation p | Pair-bootstrap 95% CI | Pearson r |
|---|---:|---:|---:|---|---:|
| class_direction_cosine | 0.115 | 0.4503 | 0.4519 | -0.184–0.402 | 0.169 |
| pls_similarity_k1 | 0.113 | 0.4618 | 0.4655 | -0.171–0.387 | 0.148 |
| pls_similarity_k2 | 0.321 | 0.0314 | 0.0334 | 0.046–0.570 | 0.356 |
| pls_similarity_k9 | 0.222 | 0.1420 | 0.1494 | -0.133–0.535 | 0.257 |
| aef_mmd | -0.266 | 0.0775 | 0.0812 | -0.563–0.061 | -0.319 |
| aef_mean_distance | -0.279 | 0.0634 | 0.0668 | -0.578–0.043 | -0.286 |
| aef_sliced_wasserstein | -0.195 | 0.1987 | 0.2040 | -0.485–0.130 | -0.206 |
| geo_distance_km | -0.100 | 0.5123 | 0.5073 | -0.377–0.203 | -0.137 |

Partial correlation of PLS similarity with retention, after linear adjustment for MMD and great-circle distance:

- pls_similarity_k2_partial_given_mmd_and_geo: partial Pearson r = 0.290
- pls_similarity_k9_partial_given_mmd_and_geo: partial Pearson r = 0.046

Pairs with the largest residual from a straight line through class-direction cosine and symmetric retention:

- LA_W|Phoenix_SW: class-direction cosine 0.850, symmetric retention 0.981
- Boston_NE|Phoenix_SW: class-direction cosine 0.688, symmetric retention 0.436
- NYC_NE|Phoenix_SW: class-direction cosine 0.542, symmetric retention 0.413
- SF_W|Seattle_NW: class-direction cosine 0.672, symmetric retention 0.443
- Atlanta_SE|Phoenix_SW: class-direction cosine 0.620, symmetric retention 0.908

Reading: compare the Spearman values in the table above. A larger |ρ| means that predictor lines up more tightly with transfer in these 45 pairs. With 10 metros, a small p-value is a description of this set, not a license to claim a universal mechanism.

## 5. Raw embedding shift and geographic distance

Metro centroids are median pixel latitude and longitude. Embedding distances use 2,000 pixels per metro: distance between standardized means, RBF MMD, and sliced Wasserstein (32 directions). None of these use UMAP.

Their correlations are the rows `geo_distance_km`, `aef_mean_distance`, `aef_mmd`, and `aef_sliced_wasserstein` in the table above. Geographic distance and raw distribution shift are the competitors to task-subspace similarity. The partial correlations ask whether PLS alignment still lines up with retention after those two are accounted for.

## 6. Why New York transfers poorly

64-D leave-one-metro-out macro-F1 is 0.673 for New York, against 0.905 Phoenix, 0.868 Atlanta, and 0.864 Denver.

When New York is the target, other metros' 64-D probes score 0.521 on average (its own diagonal is 0.742). When New York is the source, it scores 0.571 on the other metros. Atlanta as target averages 0.730 (diagonal 0.918); Phoenix as target averages 0.538 (diagonal 0.928); Denver as target averages 0.562 (diagonal 0.887).

New York's class-mean direction agrees with Chicago and disagrees with the metros that are easy targets. That is a geometric fact about the frozen embeddings. It is not, by itself, a cause.

The frozen v4 3D model, scored inside the New York fold, tells a different story. Source-metro class directions in that fold match New York's direction (minimum cosine 0.995, mean 0.998). The linear probe on those 3D coordinates still only reaches macro-F1 0.578, against a 10-fold mean of 0.735. The axis is shared. The held-out cloud is not separated along it. v4 was trained on 100 m tiles. These numbers are not on the same samples as the 10 m matrix.

## 7. Does PLS remove metro identity?

A pixel subsample (2,000 per class per metro) was split 80/20 by metro. PCA and PLS were fit on the training rows only. The same rows were used to predict building function and to predict metro identity.

| Representation | Building-function macro-F1 | Metro accuracy |
|---|---:|---:|
| raw64 | 0.819 | 0.999 |
| pca2 | 0.717 | 0.538 |
| pls2 | 0.775 | 0.298 |
| pca9 | 0.751 | 0.937 |
| pls9 | 0.818 | 0.926 |
| pca32 | 0.815 | 0.993 |
| pls32 | 0.818 | 0.993 |

This probe is a pixel diagnostic. It is not the geographic sample. Raw 64-D metro accuracy is 0.999: metro identity is almost readable from the frozen embedding. PLS k=2 cuts that to 0.298 while building-function F1 stays 0.775. PCA k=2 only gets metro accuracy down to 0.538 and keeps less of the building label (0.717). At k=9, PLS metro accuracy is back to 0.926. The components past the first few are where geography re-enters, which matches the transfer drop from PLS k=2 to PLS k=9.

## 8. Does a learned alignment loss help a held-out metro?

A linear map 64→d, d in {2, 4, 8, 16}, AlphaEarth frozen. Four losses: classification only, classification plus supervised contrastive, classification plus cross-region supervised contrastive, classification plus λ times class-direction alignment. λ in {0.1, 1, 10} was chosen per held-out metro and width by macro-F1 on the median-sized source metro. The stopping epoch uses that metro's classification loss. The held-out metro was not used for either choice. Reported numbers are three fresh seeds at the chosen λ.

| Method | d | Held-out macro-F1 | Within-source pixel-holdout F1 | Source-metro accuracy | Source direction cosine | Held-out direction cosine | Mean gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| align | 2 | 0.732 ± 0.112 | 0.817 | 0.282 | 0.999 | 0.997 | 2.225 |
| align | 4 | 0.733 ± 0.099 | 0.829 | 0.557 | 0.998 | 0.994 | 2.094 |
| align | 8 | 0.731 ± 0.096 | 0.829 | 0.807 | 0.998 | 0.990 | 1.934 |
| align | 16 | 0.732 ± 0.099 | 0.831 | 0.952 | 0.997 | 0.986 | 1.810 |
| class | 2 | 0.732 ± 0.101 | 0.824 | 0.235 | 0.996 | 0.996 | 2.245 |
| class | 4 | 0.733 ± 0.099 | 0.831 | 0.518 | 0.988 | 0.988 | 2.027 |
| class | 8 | 0.736 ± 0.094 | 0.831 | 0.794 | 0.976 | 0.976 | 1.805 |
| class | 16 | 0.738 ± 0.090 | 0.833 | 0.946 | 0.965 | 0.962 | 1.616 |
| supcon | 2 | 0.501 ± 0.178 | 0.547 | 0.303 | 0.929 | 0.921 | 1.497 |
| supcon | 4 | 0.640 ± 0.148 | 0.771 | 0.331 | 0.932 | 0.910 | 1.481 |
| supcon | 8 | 0.688 ± 0.112 | 0.793 | 0.549 | 0.898 | 0.876 | 1.146 |
| supcon | 16 | 0.722 ± 0.125 | 0.822 | 0.574 | 0.893 | 0.852 | 1.169 |
| xsupcon | 2 | 0.470 ± 0.156 | 0.495 | 0.269 | 0.885 | 0.835 | 1.382 |
| xsupcon | 4 | 0.653 ± 0.143 | 0.775 | 0.271 | 0.930 | 0.908 | 1.444 |
| xsupcon | 8 | 0.683 ± 0.103 | 0.789 | 0.385 | 0.896 | 0.870 | 1.086 |
| xsupcon | 16 | 0.726 ± 0.118 | 0.822 | 0.390 | 0.902 | 0.854 | 1.141 |

Held-out F1 is the mean over 10 metros and 3 seeds, with the standard deviation across those rows. Within-source F1 is a 20% pixel holdout inside metros the map was trained on. Source-metro accuracy is a linear metro classifier on those same training metros. The gap is centroid distance divided by mean within-class radius, averaged over metros, in the projected space.

The comparison that matters is whether higher source-direction cosine comes with higher F1 on the metro that was absent from training. The scatter is `figures/learned_alignment_vs_heldout.png`.

## 9. What is supported, and what is not

Supported, for these ten US metros:

- Frozen AlphaEarth carries a residential vs non-residential signal that survives leave-one-metro-out, at about 0.78 macro-F1.
- Most of that signal fits in a short supervised subspace. PLS needs fewer components than PCA.
- The short subspaces are only moderately aligned. k=2 similarity averages about 0.292.
- Training a 64-D logistic regression on one metro and shipping it to another loses more accuracy than shipping a 2-component PLS.
- New York's raw class direction is aligned with Chicago and misaligned with Atlanta, Los Angeles, Seattle, and Phoenix.

Not supported:

- A claim that subspace alignment causes transfer. The correlations are descriptive, on 45 dependent pairs.
- A claim of global geographic generalization. The sample is ten US downtowns.
- A claim that v4 fails on New York because the class axis points the wrong way. In the New York fold the axes agree and the cloud still overlaps.
- Any result from the smoke directories `baseline_loro/smoke1`, `subspace_analysis/smoke1`, `transfer_matrix/smoke1`, or `geo_projection/smoke1`.

## Paper figures

All of these are redrawn from the saved tables by `scripts/make_figures.py`.

1. `figures/f1_vs_k.png` — PLS vs PCA leave-one-metro-out F1 against width.
2. `figures/retention_vs_k.png` — the same F1 divided by the 64-D baseline.
3. `figures/subspace_k2.png` — k=2 similarity and the first principal angle.
4. `figures/class_direction_cosine.png` — cosine of raw class-mean directions.
5. `figures/transfer_f1_lr64.png` and `figures/transfer_f1_pls_k2.png` — the directional transfer matrices.
6. `figures/class_direction_vs_retention.png` — class-direction cosine against symmetric transfer retention.
7. `figures/predictor_spearman.png` — which predictor lines up with transfer.
8. `figures/learned_projection_f1.png` — held-out F1 of the four learned maps.

## Runs

| Stage | Directory | Resources |
|---|---|---|
| 1 LORO sweep | `outputs/baseline_loro/20261002T0015` | 10 processes, 4 BLAS threads each, no GPU |
| 2 subspaces | `outputs/subspace_analysis/20261002T0015` | 10 processes, 2 threads, no GPU |
| 3 transfer | `outputs/transfer_matrix/20261002T0015` | 10 processes, 2 threads, no GPU |
| 4 correlations | `outputs/statistical_tests/20261002T0031` | 1 process, 8 threads, no GPU |
| v4 directions | `outputs/v4_frozen_directions/20261002T0016` | 1× TITAN RTX, inference only |
| 5 projection | `outputs/geo_projection/20261002T0025` | 8× TITAN RTX |

The first v4 launch, run id `20261002T0015`, never started: the log redirect pointed at a directory that did not exist. That run was left empty of a process and the inference was submitted again as `20261002T0016`.

The first correlation launch, `outputs/statistical_tests/20261002T0021`, was stopped after nine minutes. Its RBF kernel built a 2000×2000×64 array for every metro pair. `FAILED` records that. `20261002T0031` is the same analysis with a Gram-form kernel. Retention matrices in the stopped run match the formula and were recomputed.

## Deviations

- Stages 2 and 3 do not use stage 1's fitted models, so they ran at the same time as stage 1.
- Stage 2 standardizes columns with all ten metros before PLS, unsupervised, so principal angles are in one inner product. Documented above.
- Stage 5 holds out one source metro, chosen by sample count, for λ and early stopping. It does not run a full inner leave-one-source-metro-out. The contrastive coefficient is 1. Each epoch is 80 minibatches, not a full pass. Three seeds are reported for every method and width.
- v1–v3 checkpoints were not re-encoded. The frozen-direction analysis is v4 only.
