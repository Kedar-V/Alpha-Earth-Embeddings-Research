# Distributions

No one chart shows the whole distribution. Pick the form from the sample size and the insight. State that choice when you build or critique the figure.

## Choose the chart

| Situation | Prefer | Why |
|---|---|---|
| Few observations | Strip or jitter, plus a median line | Every point is visible. A box hides the structure. |
| Many observations, and the audience only needs a summary | Boxplot | Compact median, IQR, and whiskers. Can hide a second mode or a gap. |
| The shape of the density | Violin, or a half-violin (ridgeline) | A half-violin reads more like a density. Mirrored violins are less intuitive and can still hide features. |
| Shape plus a summary | Half-violin with a box, or a raincloud (density plus dots) | The most complete picture. Rainclouds overwhelm audiences who do not read densities. |
| Very many observations | Distribution heatmap or faceted histogram | Strips turn into a solid bar. Bins such as `<10`, `10–<15`, … keep the groups comparable. |
| Natural bins (letter grades, age bands) | Faceted histograms | The bins match categories people already use. |

Mean ± standard-error bars alone understate shape. Pair them with the points or a fuller distribution when shape is the claim. Standard error, standard deviation, and a confidence interval are not interchangeable; name the quantity (see uncertainty-and-models rules if that file was opened).

## Histogram vs bar chart

| | Bar chart | Histogram |
|---|---|---|
| Variable | Categorical | Numeric |
| Order | Can be reordered | Fixed |
| Gaps | Yes | No |

## Bins and kernels change the story

Histogram interpretation depends on bin width. Choose the width that shows the insight you need, and do not pretend another width would say the same thing.

- Wider bins can support "more employees at higher pay than lower pay."
- Narrower bins can reveal two groups with nobody between $55k and $60k.

Density curves depend on kernel and bandwidth. A Gaussian kernel at bandwidth 0.5, 2, and 5, or a rectangular kernel at the same bandwidth, can turn one sample into a spike, a smooth bump, or a blob. A density can also paint probability past the range of the data. Say which kernel and bandwidth you used when the shape is doing the arguing. If you are unsure, show the histogram beside the curve.

## Comparing groups

- Overlapping histograms or densities are hard to read past two or three groups. Readers cannot tell overlap from stacking, and a grey fill with no label is an unlabeled series.
- Small transparency and color changes help for two groups. They do not rescue five groups.
- Facets (shared axis, one group per panel, plus an "all" panel if the total matters) make each shape readable. Comparing a specific bin across groups gets harder; use facets when shape-within-group matters more than a pointwise overlay.
- For a handful of groups and a modest n, dots plus a median beat a boxplot. The three groups that look identical as boxes are often obviously different as strips.
- With a large n, switch to a faceted histogram, a half-violin plus box, or a distribution heatmap. Do not overplot thousands of points and call it transparent.
