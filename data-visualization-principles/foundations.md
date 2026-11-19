# Foundations

Use on every figure, story, and slide. Working memory is limited. Design so fast visual processing does the decoding and executive attention stays free for the claim.

Helping the audience see the pattern is not manipulation if the values, comparisons, and context stay honest. A truncated bar, a cherry-picked window, or a title that contradicts the marks is manipulation. A sorted bar, a direct label, and an active title are not.

## One story

Do not "just show the data." Pick what is important.

- One primary claim per figure. Facet or split when a second claim matters.
- If every subgroup is on the chart at once, the comparison is invisible. Filter to the comparison, or highlight one series and mute the rest.
- A map is for spatial adjacency. When the story is magnitude, use a sorted bar or a table. Choropleths of raw totals usually need a rate (sales per customer, growth per capita), because large places dominate counts.

## Active title

The title states the finding, not the variable names. Include population, time, and the question when the reader needs them.

- Weak: `Accuracy of Color and Shape vs Control`
- Strong: `Accuracy is Improved by Color, not by Shape`
- Weak: `Sales by region`
- Strong: `North Division is closest to goal`

The title must be true of the marks. A ROC curve where random forest has the higher AUC cannot be titled as if logistic regression won.

Left-justify the title and subtitle. Hierarchy, largest to smallest: title, subtitle, annotations, axis labels, footnote and source.

## Do not make the audience do math

If the claim is a change, a difference, a rank, or progress toward a goal, plot that quantity.

- Prefer percent of goal, change, or an index over stacked yearly totals the reader must subtract.
- "Grew more" is a change or a percent change, not a stack of levels.
- Normalize when raw totals are trivial. Big states have more sales; show sales per customer.
- Label values on the marks when that removes axis hunting.
- Round. `4.30241` is not a finding. Drop jargon and raw variable names (`NN_t0`, `Nb_MinQBLo`).

## Order

- Sort categories by value unless the order is already meaningful.
- Do not reorder ordinal data: age bands, Likert scales, time, ranked stages, letter grades.
- Alphabetical bars hide rank. Sort them.

## Data-ink

Data-ink ratio = ink that encodes data / all ink in the figure.

Remove chart junk: 3D, heavy gridlines, redundant legends, extra borders, decorative icons, unused decimals, template chrome. Keep marks, labels, and reference lines that serve the claim.

## Decision shortcuts

| If the draft looks like… | Do this |
|---|---|
| Every subgroup at once | Filter, or highlight one comparison |
| Descriptive title (`Sales by region`) | Active title (`North Division is closest to goal`) |
| Legend off to the side | Direct labels, or a legend at the top |
| Stacked yearly totals for a "grew more" claim | Show change or percent change |
| Alphabetical bars | Sort by value, unless the axis is ordinal |
| Choropleth of totals | Normalize; consider a bar chart |
| Rainbow palette | Sequential, diverging, or one accent plus neutrals |
| A unique bright hue for every series | Highlight the focal series; mute the rest |
| Truncated bar axis | Start bars at zero |
| Dual y-axis "correlation" | Side-by-side panels or an index chart |
| Boxplot only, and few points | Strip plus median |
| Overlapping densities for many groups | Facet |
| Spaghetti hourly series | Aggregate, or use a cycle plot for seasonality |
| Vertical unreadable dates | Year bands and fewer labels |
| Codes and long decimals | Plain labels and rounded numbers |
