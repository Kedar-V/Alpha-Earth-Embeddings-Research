---
name: data-visualization-principles
description: >-
  Apply IDS 707 visualization, storytelling, and presentation principles:
  cognitive load, active titles, color scales, axes, distributions, time
  series, legends, uncertainty, model outputs, inverted-pyramid and hourglass
  narrative, personas, ABT, and PARC slide design. Use only when the user
  explicitly invokes this skill to create, edit, review, or critique a chart,
  figure, data story, memo, or presentation.
disable-model-invocation: true
---

# Data Visualization Principles

Reduce cognitive load so the audience can evaluate the claim instead of decoding the chart. Helping them is not manipulation when the data stay accurate and contextualized. Give them a claim they can agree or disagree with.

This file routes the work. The rules live in the concept files. Read those files before you design, edit, or critique. Do not answer from this page alone.

## Read these files

Read [foundations.md](foundations.md) on every use.

Then open every other file the task touches. For a new figure, a full critique, or a slide deck, read all of them.

| The work involves | Read |
|---|---|
| Claim, title, sorting, normalization, data-ink | [foundations.md](foundations.md) |
| Hue, ramps, highlighting, colorblind safety | [color.md](color.md) |
| Zero baselines, axis limits, dual axes, index charts | [quantitative-scales.md](quantitative-scales.md) |
| Shape of a distribution, bins, box, violin, strip | [distributions.md](distributions.md) |
| Dates, trend, season, cycle, smoothing, step or bump charts | [time-series.md](time-series.md) |
| Titles, labels, legends, hierarchy, annotations | [text-and-legends.md](text-and-legends.md) |
| Who will see it, polish, code vs post-processing | [audience-and-process.md](audience-and-process.md) |
| Error bars, forecasts, odds ratios, ROC, Q-Q, teaching a new chart | [uncertainty-and-models.md](uncertainty-and-models.md) |
| What to say, what to cut, narrative order, personas | [storytelling.md](storytelling.md) |
| Slides, a talk, a poster, layout, attention | [presentations.md](presentations.md) |

Concept files do not point onward. If a rule in one file depends on another topic, this table already told you to open that file.

## Workflow

**Creating or editing a figure**

1. Name the audience and the single claim. If two claims compete, split the figure.
2. Read the files above that match the chart, the audience, and the deliverable (figure, memo, or slides).
3. Encode the claim directly. Do not leave a difference, rank, or progress-to-goal for the reader to calculate.
4. Put that claim in an active title with who, when, and what.
5. Delete anything the audience would have to look up, compute, or ignore.
6. State the claim in one sentence in your reply, then the choices that matter: chart type, color scale, axis limits, distribution form, polish level.

**Critiquing a figure, story, or deck**

Answer all four prompts. Name the broken rules and the file they come from.

```markdown
1. Who was the intended audience? What was the original purpose?
2. Which principles of effective visualization does the figure implement? Which does it not implement?
3. What could be clearer, more accurate, or more visually effective — given the audience and purpose?
4. For “Other” visualization types only — What type of visualization is it?
```

For a story or presentation, also say whether the order is a lede (problem, decision, proof) or a chronology of the analysis, and whether the audience could stop early and still know what to do.

## Checklist gate

Before you deliver, confirm:

- One claim, and the title states it.
- The audience is not doing arithmetic the chart could have done.
- Color, labels, and legends follow [color.md](color.md) and [text-and-legends.md](text-and-legends.md).
- Bars start at zero; other axes use a justified range from [quantitative-scales.md](quantitative-scales.md).
- Uncertainty, if shown, names the quantity (see [uncertainty-and-models.md](uncertainty-and-models.md)).
- A talk or memo leads with the decision, not the lab notebook (see [storytelling.md](storytelling.md) and [presentations.md](presentations.md)).

Apply every rule in the files you opened unless the user forbids it.

Course source for these files: IDS 707 slide decks on color and quantitative scales, distributions, time series, uncertainty and model outputs, data storytelling, and effective presentations.
