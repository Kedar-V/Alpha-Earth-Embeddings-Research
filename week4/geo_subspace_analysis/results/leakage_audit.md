Overall: PASS
- PASS — Transfer diagonal uses a within-metro 20% holdout. 50 diagonal rows, protocols=['within_metro_20pct']
- PASS — Off-diagonal transfer trains on the source metro only. 450 off-diagonal rows
- PASS — No NaN macro-F1 in the transfer matrix. min=0.333 max=0.928
- PASS — LORO baseline has ten metros and finite F1. metros=Atlanta_SE, Boston_NE, Chicago_MW, Denver_W, Houston_S, LA_W, NYC_NE, Phoenix_SW, SF_W, Seattle_NW
- PASS — Each LORO fold's train and test counts add to the full table. sum of n_train+n_test over 10 folds = 17518140
- PASS — k=2 projection similarity is finite and diagonal is 1. off-diagonal mean=0.292
- PASS — Projection validation metro is never the held-out metro. 10 held-out metros, 480 eval rows
- PASS — Projection eval grid is complete. rows=480 expected=480

What the training code asserts, and this audit does not refit:
- Stage 1 raises if the scaler mean differs from the training-block mean, and if the held-out metro appears in the training index.
- Stage 3 raises if the diagonal train and test indices overlap, or if a target slice contains the source metro.
- Stage 5 raises if a training row belongs to the held-out metro or the validation metro.
- The classification threshold is 0.5 in every probe. It is not tuned.

Known limits, not silent leaks:
- Stage 2 divides each AlphaEarth column by a standard deviation computed on all ten metros, with no labels, so the ten PLS bases share one inner product. That scale uses the held-out metro's feature spread. Class-mean directions are computed in the raw 64-D vectors, and every LORO probe fits its own scaler on training metros only.
- Stage 5 picks lambda and the stopping epoch on one source metro, the median metro by pixel count. That metro is not the held-out test metro. SupCon's coefficient is fixed at 1 and was not searched.
- Pairwise p-values treat 45 pairs from 10 metros. They are descriptive. The geographic sample size is 10.
