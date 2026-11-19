# Color

Perception is not physics. Equidistant wavelengths are not equidistant to the eye. Rainbow and spectrum colormaps have no intuitive order and create false stripes. Do not use them.

Color draws attention to the story. It is not decoration for every category.

## Pick the scale

| Data | Scale | Use |
|---|---|---|
| Unordered categories | Qualitative | Distinct hues. Highlight the focal category. Mute the rest with grey or lower saturation. |
| Numeric or ordered | Sequential | One hue, or a two-hue ramp. Default for most magnitudes. |
| Numeric with a real midpoint, and the story is about that midpoint | Diverging | Two hues meeting at the midpoint. |

Quantitative scales are ramps (one or two hues). Qualitative scales are different hues.

### Sequential vs diverging

Use diverging only when both are true:

- There is a meaningful midpoint (50%, a national average, surplus vs deficit, above vs below a requirement).
- The story is the discrepancy around that point, not "which values are high."

A sequential ramp emphasizes high values. The same data on a diverging scale, split at 50% or at the national average, changes the story to "above or below the reference." If there is no meaningful midpoint, sequential is easier to read.

A diverging scale can be right even on a line chart of counts when a threshold is the point. Cumulative challenges colored against a minimum of 50 support the title "most competitors met or exceeded the minimum." Without that threshold, a sequential ramp is the better default.

Ordinal data (low / medium / high, letter grades) gets a sequential or diverging ramp, not a rainbow of unrelated hues. Order does not have to be numeric.

## Do not waste color

- One accent for the series or category the sentence is about. Grey or desaturated color for context.
- If the other series still must be distinguishable, give them muted distinct hues and keep the focal series saturated. Do not give every country its own loud color unless comparing all of them is the claim.
- On a line chart, direct-label the highlighted series. Do not rely on a rainbow legend.

## Craft

- Avoid pure saturated hues. Slightly desaturated colors read better and sit together more cleanly.
- Keep the same mapping across related figures. A category that was blue in figure 1 stays blue in figure 2.
- Red-green is the most common color-vision deficiency. Many decision-makers are colorblind. Avoid true red paired with true green. Check with a colorblind simulator (Datawrapper's colorblind check, or equivalent).
- Label contrast: white text on dark fills, black text on light fills.
- Brand palettes are a constraint, not a reason to use a rainbow. Pick the brand ramps that still match sequential, diverging, or a single accent.

Tools: ColorBrewer, `scale_color_brewer()`, `matplotlib.colormaps`.

## Remember

- Different shades for numeric or ordinal data.
- Different hues for categories, then mute all but the focal one.
- Avoid rainbow scales, red-green pairs, and oversaturated colors.
