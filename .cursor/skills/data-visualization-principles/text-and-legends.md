# Text, hierarchy, and legends

Text tells the reader what the marks mean and where to look. It is part of the graphic, not a caption bolted on afterward.

## Hierarchy

Guide the eye in this order, and make the type sizes match:

1. Active title, largest.
2. Subtitle, smaller. Context: who, when, what was measured.
3. Legend, if one is truly required, at the top of the figure.
4. Annotation next to the mark it explains, only where the reader needs the insight.
5. Axis titles and tick labels, large enough to read. Horizontal text. A horizontal y-axis title is preferred; a vertical one is acceptable.
6. Footnote: source, definition, sample, method.

Axis-title placement: above the axis, aligned with the ticks, or appended to the first tick. Do not rotate category labels to fix a crowded vertical bar chart. Flip to a horizontal bar chart.

Put the color word in the title when that replaces a legend: `The **melt rate** was lower in 2024`, with "melt rate" set in the series color.

## Labels

- Human-readable names. No raw field names, no abbreviations the audience must remember, no decorative decimals.
- Label the marks when the reader would otherwise hunt along the axis for the number the title depends on.
- Direct-label lines at the end of the series or at the focal point. A side legend makes the eye bounce.

## When a legend is allowed

Default: no legend.

| Situation | Legend? |
|---|---|
| A few categories you can name on the marks | No. Direct labels. |
| Qualitative color, few series | No, unless the marks are too small or too tangled to label. |
| Continuous sequential or diverging ramp (unclassed choropleth, heatmap) | Yes. A ramp cannot be labeled on every county. Put endpoints and the midpoint in words (`lower`, `national average`, `higher`). |
| Classed color, even with only a few classes (a binned map) | Yes, if you cannot write the class on each region. |
| Stacked segments (low / medium / high) | Yes, unless every segment is labeled in place. Order the legend the same way as the stack. |
| Grouped bars for several models | Prefer a one-line key at the top (`Model 1 \| Model 2 \| Model 3`) over a legend parked on the right. If comparison is still hard, small multiples with the model name in the panel title remove the key. |

Do not put the legend off to the side. Top of the figure, close to the title.

Do not label every bar in a repeating group if the first group (or a top key) already defines the colors. Repeating "Model 1, Model 2, Model 3" on every cluster is noise.

## Annotations

Annotate the pattern, not the obvious. "Lower validation loss with RoBERTa than with BERT" next to the panel that shows it, plus "higher is better" or "lower is better" when the metric's direction is not common knowledge.

Do not write "as you can see." If the reader can see it, the annotation should say what it means. If they cannot see it, the chart is wrong.

## One visualization as a slide

A stand-alone chart slide can carry, in this order:

- One-line highlight: the most important number or result.
- The question the chart answers.
- The chart.
- The finding, in a sentence.
- The recommendation: what to do next.
- A link or footnote to the long report, for the person who wants the method.

That is the inverted pyramid at the size of one figure. The highlight is the lede. The chart is the proof. The appendix is the general information.
