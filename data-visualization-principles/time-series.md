# Time series

Not every series of dates is a line chart. Match the geometry to whether the value moves continuously, jumps and holds, accumulates, or only the rank matters.

## Make the dates readable

- Do not stack `01-01-2021` vertically along the axis.
- Prefer a horizontal year band with quarter ticks (`2024` under `Q1 Q2 Q3 Q4`) when the series is dense.
- An active title still has to say what changed (`Click rate has increased on all pages this year`), not only `Clicks per min`.
- Drop point markers when the audience does not need them to read a value. Markers add load on a line that is already the encoding.

## Missing data

Show the gap on purpose. Never silently treat a missing point as zero, and never hide an interpolation.

| Treatment | What the reader believes |
|---|---|
| Connect across the hole, or drop the line to zero | There was a real zero or a real smooth path |
| Interpolate and draw it like data | The filled values were observed |
| Leave a gap | Those times were not observed |
| Draw the interpolated segment in a lighter or transparent stroke | The middle is estimated; the ends are data |

Use a gap by default. Use transparency only when you must show an estimate and you label it as one.

## Chart type and metric

- **Line** for a value that changes continuously between observations.
- **Step chart** when the value holds and then jumps: a price, a policy rate, a posted tariff. A line through April and June prices invents prices that were never charged. A step shows the price that was actually in effect.
- **Levels vs change vs share.** "Home-equity auto-pay increased more than other loan types" is hard to see on four raw lines with different baselines. A bar of the latest change, or a bar of percent change, matches the sentence. Keep the line only if the path is the claim.
- **Cumulative vs new counts.** A cumulative chart of exoplanets by discovery method shows the stock. It hides which method is contributing now. If the claim is about recent discoveries, plot the flow (new planets per period), or show both and title the one that answers the question.
- **Bump chart** when the claim is rank, not magnitude. Click-rate levels can hide a reorder at the bottom of the axis ("B is now first, C is lagging"). A bump chart shows that reorder and does not show how large the gap is. Do not use it when the size of the gap is the story. Direct-label the lines (B, A, G, …) instead of a legend.
- Direct-label series. Do not add a bright color per line unless every line is the story.

## Trend, seasonal, cyclic

| Pattern | Meaning |
|---|---|
| Trend | Long rise or fall |
| Seasonal | Rises and falls on a fixed calendar period: quarter, month, day of week, hour |
| Cyclic | Rises and falls that are not locked to a calendar period |

Look for seasonality in this order:

1. Plot the raw series first. Note spikes, flat stretches, and repeating peaks.
2. Overlay cycles on one panel: one line per year, or per week, so peaks either line up or they do not.
3. Use a **cycle plot** to see one season across years: a panel (or a grouped line) per month or per weekday, with time moving inside the panel. That shows whether Wednesdays rose while weekends stayed flat.

Autocorrelation and correlograms can confirm a period. Stakeholders usually cannot read them. Show the aligned peaks and valleys. Example: donations near zero on weekends, with regular peaks on Wednesday and Friday. A small-multiple or a labeled week strip beats a correlogram in a briefing.

## Smoothing

Rolling means and exponential smoothing show a trend under noise. Show the raw series and the smooth together when the audience needs both the wiggles and the trend. Document the window or the smoothing parameter in the footnote. Do not replace the data with a smooth and leave the method unnamed.

Cumulative curves look smooth for a different reason: they integrate. Do not describe that smoothness as a trend you discovered. If you smooth, do it on the series whose variation you are explaining (often the per-period counts, not the cumulative sum).
