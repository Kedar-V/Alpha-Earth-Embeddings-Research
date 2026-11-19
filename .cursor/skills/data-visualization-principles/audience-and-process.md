# Audience, polish, and process

Match the explanation to the person in the room. Over-polishing for analysts wastes time. Under-explaining for executives fails.

## Who is in the room

Place the audience on two related axes:

- Access to the raw data, ability to analyze it, and whether they already do.
- How much plain-language explanation, insight, recommended action, and business impact they need.

| Audience | Data access and analysis | What to add |
|---|---|---|
| Data scientists and analysts on the team | High. Exploratory figures can stay dense. Legends and technical encodings are acceptable. | The pattern, the method caveat, the next test. Low polish is fine. |
| Managers (product, marketing, program) | Medium | A clear takeaway and light annotation. Less raw diagnostic detail. |
| VPs | Lower time, still some domain context | The decision and the evidence that justifies it. |
| C-suite, public, program staff with low data literacy | Low access, low time for method | High polish: active title, plain language, a recommended action, the business or operational impact. |

Also score the specific person, not the job title alone:

- Time available
- Domain knowledge
- Technical knowledge
- Access to the data

A program manager can have low technical knowledge, low data access, and high domain knowledge. A data scientist in another field can have high technical skill, medium data access, low domain knowledge, and no time. Those two people do not get the same figure.

There is a cost to sitting in the wrong corner. Heavy polish for an expert who did not need it burns your time. A raw diagnostic for an audience with no time and no technical background does not get read.

If they lack expertise but have time, you may teach the chart (how to do that is in the uncertainty and model-output file). If they lack both, translate the result into a sentence and a simple comparison instead of teaching a new grammar.

## What rises as literacy falls

Insights, recommended actions, and impact. Method, alternative models, and diagnostic plots move to the appendix or get cut.

An NMDS scatter titled with axis names is appropriate for collaborators who already know ordination. The same plot for a broader audience needs an active title ("Outdoor and indoor bacterial communities overlap, with some separation"), a sentence that closeness means similar composition, a note that small points are communities with fewer species, and the next step. The ordination can stay in a backup slide.

## Reproducibility vs one-off polish

| | One-off presentation | Figure that updates |
|---|---|---|
| Construction | Most of the geometry in code; polish can be finished by hand | All of it in code |
| Titles and annotations | Can be written by hand | Must be conditional on the data |
| Post-processing | Allowed for polish | Not allowed |

Do as much as possible in code. PowerPoint, Illustrator, or Figma are for polish, never for fixing the data.

If you post-process:

1. Keep the plotting code.
2. Keep the original exported figure.
3. Show the edited figure.
4. Set axis scales and ranges in code before export.
5. Do not drag marks, resize bars, or nudge points.
6. Leave gridlines and reference guides in place until you have checked the edit against the original. Then remove the ones that are not data.

Prefer vector export with text as text, not outlines, so labels can still be edited. In matplotlib, set `svg.fonttype` to `none` before `savefig` of an SVG so type stays text.

PDF is a reasonable exchange format when the next step is layout, not re-analysis.
