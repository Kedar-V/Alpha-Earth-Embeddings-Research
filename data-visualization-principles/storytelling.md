# Data storytelling

A statistic states a number. A story gives that number a person, a stake, and a decision.

"23% of first births happen by 38 weeks" is a statistic. "There is a 23% chance Nick leads class next week, because first pregnancies tend to run long and the due date is still ahead" is a story. Stories recruit imagery, emotion, and social reasoning; a bare statistic is easier to forget. Use the story structure without inventing facts the data do not support.

Data storytelling is a narrative built from data and the methods that produced it. It conveys what the result means. Visuals often carry it. It can be simple or complex.

It is not:

- Simplifying the analysis until the result is no longer true
- A plot the data do not support
- Atmosphere, adjectives, or decoration with no claim

A visualization is not automatically a story. It can be one piece of a story, and a single chart can be the whole story if the title, the evidence, and the action are all there.

## Do not narrate the work in the order you did it

The lab path is a common and weak structure: problem, exploration, collection, cleaning, a first model, more collection, class imbalance, a metric table, tuning, an appendix of caveats.

The audience is busy. They want the bottom line. Start with the problem and the solution, then justify the approach with the evidence they need. The chronology of your effort is not the argument.

## Inverted pyramid

Use this for a memo or a report the reader may abandon.

1. **Lede (nut graph).** The most important information. If they stop here, they know what they must know. Cover who, what, where, when, why, so what, and the decision in front of them. Problem, recommendation, and the choice they have to make.
2. **Important details.** Proof the recommendation works, and the justification. A skeptical reader stays for this.
3. **General information.** Useful, not required. Loose ends, method catalogs, extra cuts.

Example of a lede, compressed: anomalies in PayFlow's web logs often become billing outages; a detector flags about 9 in 10 of them before the outage; the open decision is how many false alarms operations will tolerate if the threshold is tightened to catch the rest. Everything after that explains the tradeoff and helps them pick a rate.

The rest of the document supports and qualifies the lede. It does not introduce a new decision in paragraph twelve.

Scale: the same three layers work for one chart and for a 30-minute talk. On one chart, the title is the lede and the marks are the proof.

## Hourglass

Same opening and middle as the inverted pyramid. The ending is different: after the proof, widen back to the big picture. Tie the evidence to the recommendation and to the problem you opened with.

- Inverted pyramid: better for some written reports, where trailing background is optional.
- Hourglass: better for presentations, and for written pieces that need a closing the audience will actually hear.

A talk that ends on a metric table feels unfinished. A talk that ends by restating the decision, now backed by the evidence, gives them the action.

## What to pack

Treat content like a suitcase. You cannot bring the closet. Pack for this destination: this audience, this decision, the time they have.

Ask, before you keep a figure:

- Do they need to know other models were tried?
- Do they need to know why this one was chosen?
- Will this change the decision?

Three ways to handle a model-comparison table, in the order you should consider them:

1. **Ditch the figure.** If they do not care that alternatives existed, one sentence is enough: we can sort constituent mail into positive, negative, and mixed, well enough to act. Put the table in the appendix for the person who will ask.
2. **Visualize and annotate.** Replace the grid of six metrics with small bar panels. Mark which direction is good (`Higher = better`, `Lower = better`). Highlight the metric that decided it, and say so in the title ("We chose RoBERTa because validation loss was lower"). Mute the also-ran models.
3. **Slow reveal.** Use this when they must understand the metric and they have the time. One idea per beat: what accuracy counts, that higher is better, that accuracy can hide a failure on the negative class, then the metric you actually trusted. Do not dump the glossary and the results on one slide.

Never write "as you can see." Say what the comparison means.

Translate rates into counts when the audience does not live in confusion matrices. "About 1 in 34 messages we mark negative is actually neutral" is a claim. A cell in a normalized confusion matrix is a puzzle. Show the misses that matter in the audience's words (a sarcastic complaint classified as support), and say what you are doing about them.

## Design the story for a person

Before you choose charts, write down:

- How much time they have
- Technical expertise
- Familiarity with the domain
- Their biggest concerns
- Their main interests

Personas make that specific. A useful persona is fictional but realistic, specific rather than "stakeholders," and built for a purpose: to predict what this audience will ask.

Two personas for the same bike-lane sentiment project:

- A city council member with no NLP background cares whether residents support the lanes, whether the city can trust the sort, and how a polarized neighborhood will look. They do not want training loss.
- The head of an NLP and ethics group cares whether the model is fair, where it fails on nuanced or sarcastic text, and how it performs by demographic group.

Build the main deck for the decision-maker. Add backup slides that anticipate the specialist's questions: fairness cuts, error examples, performance by group. Do not make the council sit through the backup, and do not make the ethics lead accept a slide that only says "mostly positive."

For each audience, answer three questions before outlining:

1. What are they trying to do?
2. Which metrics or comparisons change their mind?
3. What tone and level of detail can they use?

A data-science team, a group of clinicians, and an executive team get three different decks from one clustering analysis. Motivations, metrics, and tone all change. Label illustrative data as illustrative.

## Personas inside the story

Use a character to make a table tangible when the audience has to follow a procedure (missing data, a threshold, a recommendation rule).

- One athlete with a complete recruiting profile, and one with height and test scores missing, shows the hole. Filling height with the median for that position shows the fix. Dropping income and SAT from the feature list shows the choice. The audience tracks Jane; they do not track a column of NaNs.
- The same move explains a method. Content-based recommendations: two students with similar attributes, so the college one attended is a candidate for the other. Collaborative: two people who both liked one film, so the second film one of them liked is a candidate for the other.
- A persona can be a case instead of a person. "The September server crash" is a single episode on the error-log series. Model A catches that day and raises one false alarm. Model B catches it and raises many. The threshold becomes visible.
- A persona does not have to be human. Two named sourdough starters, with age, flour, water, and origin, can carry a microbiology result. Name them, ground them in real samples, and show the microbes that distinguish them.

Constraints:

- Ground the persona in real data. Do not invent a case that flatters the model.
- Avoid stereotypes. A range of personas beats one caricature.
- Be transparent that the character is an example, especially when the underlying rows are real people.
- Do not let the character replace the aggregate. Jane illustrates the rule; the claim still needs the full sample.
