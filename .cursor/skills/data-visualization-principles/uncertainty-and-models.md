# Uncertainty and model outputs

A summary without the uncertainty reads as a fact. Show uncertainty in a form the audience can interpret, and name what the mark means.

## Do not collapse a distribution into one bar

A single split bar ("71.4% vs 28.6% chance of winning") reports a point probability. It hides how the simulations are spread and how often the other outcome happens.

Stronger encodings, when you have the draws:

- A dot histogram of simulations, with counts in words (`72 in 100`, `28 in 100`), a line at the decision threshold (a dead tie), and color for who wins. The reader sees both the probability and the margins.
- Small multiples of simulated outcomes (one map per draw, or a sample of draws). "Favored" and "certain" look different when most maps go one way and a visible minority go the other.
- For a spatial forecast, an explicit cone and an implicit ensemble answer different questions. The cone is a summary region. The ensemble of tracks shows the paths that produced it. Label which one you are showing. A cone alone is often read as "the storm grows," which is the wrong reading.

## Fan charts and intervals

A line plus a shaded band needs a sentence on the graphic:

- What the line is (published data, the mean forecast).
- What the band is, in frequencies if you can (`in 90 of 100 future three-month periods, the change falls in this band`).
- Whether the band is revision uncertainty on data already published, or forecast uncertainty about the future. Split those regions visually (a vertical line at "now" is enough).

Do not use an unlabeled ribbon and hope the legend carries it.

## Error bars

Error bars are ambiguous. The same dot can wear a standard deviation, a standard error, or a 99% confidence interval, and the picture changes from "the sample is spread out" to "the mean is precisely estimated."

Rules:

- Write the quantity and, for an interval, the confidence level on the figure. "Error bars" is not a label.
- When the sample is small enough to plot, show the points and the summary. A mean sitting in a cloud of dots is harder to misread than a floating interval.
- Do not mix SD, SE, and CI in one chart without saying which bar is which.
- Mean ± SE is a statement about the mean, not about the shape of the data. If the claim is about shape, use a distribution chart.

## The title has to match the model chart

A descriptive title (`Model comparison for identifying spam email`) forces the audience to decide who won. An active title must match the geometry.

Failure mode from the course: an ROC curve titled as if logistic regression were better at catching spam and avoiding false alarms, while the legend shows random forest AUC 0.98 and logistic regression AUC 0.91. The curve and the title disagreed. Either retitle to the model that actually dominates, or change the chart.

For a non-technical audience, spell the comparison in outcomes ("fewer real messages marked as spam at the same detection rate"), and keep AUC in the footnote. Include the chance diagonal only if you will say what it is.

## Q-Q plots

A Q-Q plot is a diagnostic for people who already read them. It is a weak way to show skew to everyone else.

- Audience that knows Q-Q: the plot is enough; extra polish is optional. Annotate the departure you care about (a long tail of low scores).
- Everyone else: show the histogram or density, annotate the tail, and use the Q-Q plot in the analysis notebook to confirm. Do not make the Q-Q the figure they have to trust.

## Odds ratios and other ratios

The null value of an odds ratio is 1, not 0. "No difference from the reference group" is a line at 1. Above 1: higher odds. Below 1: lower odds.

Do not use a bar chart.

- Bars encode length from zero. Zero is not the null, so the length is not the comparison.
- A bar cannot point below zero, so a protective ratio (0.2) becomes a stub. An 8-fold increase and a 5-fold decrease do not look like inverses of each other. The 0.2 bar looks like a tiny effect; it is not.
- Sorting and highlighting the largest bar (opioids, for example) still leaves the baseline wrong.

Use a forest plot:

- One row per group, sorted by the estimate or by a meaningful order, not alphabetically unless the list is a lookup.
- Point estimate and confidence interval.
- A reference line at 1, labeled in words: increased odds vs decreased odds relative to the reference group (patients with no diagnosis, male arrests in a white neighborhood, whatever the model used).
- The numeric value on the point.
- Highlight the row the title is about; mute the rest.
- Footnote the interval (95% CI) and the method if the audience might ask.
- Title the finding, not the chart type: patients with a recent substance-use diagnosis had higher odds of COVID-19, highest for opioid use.

A log axis can make multiplicative steps even, but it does not fix a bar that starts at 0. If you use a log scale, say so, and still mark 1.

## Teaching a chart the audience has not seen

Use this only when they have the time and the chart is the right tool. Order:

1. Name the chart and the decision it supports. ("To decide whether we have enough samples, we use a species accumulation curve.")
2. Define each axis in words.
3. Say what pattern to look for before you show the result. Steep: still finding new species. Flat: more samples add little.
4. Then show the data with those labels on the curves (control vs treatment, current n versus the projection).
5. End on the action. ("Double the treatment samples from 250 to at least 500.")

Do not open with the fitted curves and the recommendation in the same breath. The audience needs the grammar first, then the evidence, then the ask.

Same pattern for any specialist view you cannot translate away: calibration plots, accumulation curves, ordinations, survival curves. If you cannot afford the teaching sequence, do not use the specialist view in the main talk.
