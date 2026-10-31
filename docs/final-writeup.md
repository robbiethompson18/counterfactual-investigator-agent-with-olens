# Final write-up: can metamodel reads tell an investigator what Qwen will do?

**Status (2026-10-31):** ACTIVE. Draft for the sprint interview; headers only, Robbie is writing it
by hand.

Note: for sections that are fully done I will write DONE in the header. For sections I want a Kylie
review I will write (KTR). FOr Claude I will write (CTR).

## Executive Summary (KTR)

We originally set out to test whether metamodels are useful in counterfactual investigations. We
replicated results from CHIVE ([Karvonen et al., 2026](https://arxiv.org/abs/2608.16747)), and
tested whether 🔮-lenses help where NLAs and J-lenses did not.

Contrary to our expectations, 🔮-lens were unhelpful on counterfactuals in the CHIVE dataset. They
performed at or below a transcript-only baseline. We replaced the whole dataset with handbuilt
environments to make the task easier and on-policy, but this did not just results.

We suspect that counterfactuals are a harder task than the literature appreciates: on both our
environments and on CHIVE, adding a transcript does not help a counterfactual investigator when
compared to simply giving the investigator both system prompts. TODO waiting on CHIVE results here

We successively simplified the counterfactual prediction tasks we asked metamodels to do until we
found ones that work. They are: predicting Qwen's grade for an essay; predicting Qwen's answer to a
multiple choice question; predicting Qwen's idiosyncratic preferences; predicting Qwen's answer to a
factual query which is past its knowledge cutoff.

## Claims, with credences (CTR, KTR)

Current SOTA metamodels (J++ lens, 🔮-lens, NLA) do not provide uplift > 1% on 'wild' counterfactual
investigations (eg CHIVE, WeirdChat) or on 'agentic' counterfactuals (eg Palisade one, Aditya ones,
@c link) under any harness configuation: 95%.

SOTA metamodels do provide uplift on the simple prediction tasks we detail below (1,2,3,4, @c put
actual titles): 95%. They are worse than linear probes (where such probes are practicable): 95%.

SOTA metamodels would provide uplift on 'wild' counterfactuals if the investigator and subject model
was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

SOTA metamodels would provide uplift on 'agentic' counterfactuals if the investigator and subject
model was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

I think the last two questions are terribly interesting! If you are reading this and you work at a
lab: this should be a straighforward experiment to run!

## Part 0: counterfactual investigation is null

Bar chart on chives showing nulls Bar chart: score on all of our envs, showing nulls Bar chart:
abaltions by model showing null

We ablated everything we could think of, and in none of it did the O-lens reliably beat
transcript-only or its own placebo. We swapped the predictor across seven models, from Haiku 5.5 to
GPT-6 Astra, plus thinking-off and deliberately weak predictors. We varied the predictor's reasoning
effort from low to high, and its tool-call budget from 5 to 40 calls. We ran the target with and
without chain of thought, both pre-registered. We changed how reads reach the predictor: as an
agentic tool, or pasted straight into its prompt. We changed the lens itself: the old lens vs the
space-fixed one, greedy vs temperature-1 decoding, and reads at stop tokens vs at every token. We
tried other metamodels, J-lens, NLAs and Thought Anchors, each against its own placebo. We tried
other models' lenses: Camila's DeepSeek V4 Flash lens on alignment faking and Celeste's GLM 5.3 lens
on over-eagerness. We iterated the predictor's harness through six versions. We rescored everything
with log skill, RMSE, and binary flip/no-flip at thresholds from 5 to 30pp. And on CHIVE itself we
tried Flash vs Sonnet 5.5 as the predictor, and free-form vs yes/no answers. The one thing we never
ablated is which layers the lens reads. Every chart is in the appendix,
[Counterfactual investigation graphs](#counterfactual-investigation-graphs).

TODO link to appendix and read the Claudespeak paragraph above.

TODO @Kylie plz add your appendices with your data here

### Chive Shows No Not Much Uplift From The Transcript

<img src="img/chive-inline.png" alt="AUROC on CHIVE's yes/no claims: Haiku edit only vs transcript, beside CHIVE's Opus 4.8 transcript and resampling arms, per target and pooled" width="760">

<img src="img/chive-inline-diff.png" alt="Haiku AUROC, transcript minus edit only, paired by claim: +0.012 pooled" width="500">

TODO fix this to use qwen 3.6 27b if we can, then use our jlens / olens / NLA.

### Blog sentence deletion

On 1,316 selected sentence deletions from 520 bloggers, we predict whether deleting a sentence
substantially lowers Qwen's confidence in its answer about the author's age or sex. Text-only Haiku
scores 0.797 AUROC, Haiku with O-lens reads 0.803, and a standalone linear probe 0.784. The O-lens
gain over text-only is not statistically significant; the standalone probe is also below text-only
on this metric. The probe is supervised on this deletion task with bloggers held out, and its
configuration was chosen during exploration of this dataset.

<img src="img/author-blog-drop.png" alt="Blog sentence-deletion AUROC: text-only Haiku 0.797, Haiku with O-lens 0.803, standalone L2 probe 0.784, with 95% blogger-bootstrap confidence intervals" width="500">

The chart shows the probe alone. Details and the separate ensemble analysis are in
[Author evidence](author-mcq-probe.md). Regenerate with `uv run python -m graphs.author_blog_drop`.

## Part 1: Simple Investigations That Work

All of our environments that work, with the exception of essay grading, are ideas stolen from
WorkspaceBench. The additions we make to their dataset:

- We do not grade with pass@thousands, running a regex on metamodel output at many layers and
  tokens. Instead, we pass metamodel output to an investigator model and ask it to pick the correct
  answer. This penalizes noise and hallucinations appropriately.
- We scale up dataset size to between 1,000 and 3,000 cases per category. This allows us to pick up
  small but statistically significant effects of metamodels, to train probes on unseen data, and
  most importantly, to _not filter our dataset in potentially biasing ways_. To be concrete:
  WorkspaceBench filters their `knowledge_cutoff` data to answers which disagree with Sonnet and
  Opus. So if they ask Qwen "Who is the CEO of Intel?" and both Qwen and Opus reply "Pat Gelsinger"
  then they drop this datapoint. This makes any transcript-only baseline confounded: we have already
  filtered to examples where our prompt only judge will be wrong. We drop this filter and others
  like it.

### Dataset Descriptions:

In each diagram, dark green is what we evaluate, with Qwen-right and Qwen-wrong cases in the same
color because both are valid. Light green is usable but held out, either for probes or simply
untouched. Gray is not usable, typically because Qwen is inconsistent.

@c add two docent link examples per dataset here plz.

#### Multiple Choice Questions

We ask Qwen3.6-27B (thinking off) multiple-choice questions from MMLU-Pro and SuperGPQA, 8 samples
each, and the investigator must name the letter Qwen picks from the prompt alone. We keep every
topic, but drop SuperGPQA's calculation questions, because Qwen writes out its working on them
instead of giving a letter. The only other filter is consistency, the same letter on at least 7 of 8
samples: that leaves 2,523 cases from a random 4,000 MMLU-Pro questions (Qwen right on 82%) and a
random 3,000 of the 6,328 consistent questions among 10,000 sampled from SuperGPQA (Qwen right on
63%). We never filter on Qwen being wrong, so most labels are also the correct answer, and we report
the Qwen-wrong slice separately.

<img src="img/data-filter-sankey-mmlu.png" alt="MMLU-Pro: 12,032 questions to 2,523 eval cases, consistency the only filter" width="500">

<img src="img/data-filter-sankey-supergpqa.png" alt="SuperGPQA: 26,529 questions to 3,000 eval cases, calculation questions dropped, then consistency the only filter" width="500">

To fit a probe here we use nested two-fold cross-fitting over all consistent questions, grouped by
question: each question's prediction comes from a probe trained on the other half (at most 1,000
questions). This is valid, but training on a held-out pool would have been cleaner. We used the same
protocol for Essay Grading.

#### Knowledge Cutoff Q&A

We ask Qwen present-tense factual questions, many of whose answers changed around its knowledge
cutoff, such as "Who is the CEO of Intel?", written by Sonnet subagents with no date in the
question. The label is Qwen's own modal answer, not the truth, and the only filters are the same
7-of-8 consistency gate and dropping numeric answers. We don't filter on Qwen being wrong: of the
999 items, 339 get a stale answer and 660 a current one. This is WorkspaceBench's `knowledge_cutoff`
category, scaled from 109 items to 999, and not filtered on Qwen disagreeing with Opus and Sonnet.

<img src="img/data-filter-sankey-cutoff.png" alt="Knowledge cutoff: 5,467 questions to 999 eval cases, stale and current answers both kept" width="500">

#### Idiosyncratic Preferences

We ask Qwen free-response questions about its own preferences, such as "Which text editor would you
choose for editing a config file over SSH? Answer with just the name.", from a bank of 5,187 that
Sonnet subagents wrote. This is WorkspaceBench's `basic_readout` implicit family scaled from 32
items to 1,500. There is no truth, the label is Qwen's modal answer, and the only filter is
consistency: we keep every question where at least 7 of 8 samples agree, never filtering on whether
a text reader can guess it, so transcript-only already gets 80%
([favorites-qa-inline.md](favorites-qa-inline.md)).

<img src="img/data-filter-sankey-favorites.png" alt="Idiosyncratic preferences: 5,187 questions to 1,500 eval cases" width="500">

#### Essay Grading

We take 204 student intro paragraphs from
[PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) (Crossley et al., 2024), 34 at each
human score from 1 to 6, and open each with one of 225 sentences about who wrote it or what the user
wants ("My rival for a scholarship wrote this essay."), giving 1,800 cases. Qwen grades each prompt
out of 10 about 8 times, and the investigator predicts whether its mean score goes up when the
opener becomes the plain "A student in my class wrote this essay."; it does on 54% of cases. There
is no filter at all, not even consistency: the openers were written before any was run, and every
case is kept.

<img src="img/data-filter-sankey-essay.png" alt="Essay grading: 1,800 cases, no filter, all evaluated" width="500">

## Results and Ablations

### Headline

Metamodels work on these four tasks! They are all about equally good. @c bar graph with just four:
transcript, 3 metamodels.

All three convincingly beat their placebo versions, where we draw metamodel data from entirely
different tasks; @c bar graph with six, each metamodel and their placebo.

No metamodel performs as well as a probe: @c show three metamodels and probe, split bar chart by the
two datasets where probe workes, MCQ (combined), essays.

The difference is more noticeable when you measure by error rate (lower better): @c should be pretty
intuitive what I want here, same as above but 1-y.

### Headline by dataset

@c 4 different bar charts, each with 7 bars.

TODO: ablate grader model on the cutoff dataset.

### By metamodel

### By token position

<img src="img/read-positions-diagram.png" alt="One MMLU-Pro prompt as Qwen sees it, question reads in gold, tail reads in maroon, the last token boxed" width="450">

We expected that reading the very last tokens before the reply would improve performance, especially
for J++. That holds for the O-lens and the NLA, but barely: tail reads beat question reads by about
1pp (p = 0.010 and 0.001), and all of that comes from the preferences set. J++ goes the other way:
question reads beat tail reads by 2.3pp (p < 0.001), and J++ at the tail is no better than the
prompt-only baseline.

<img src="img/menus-inline-position.png" alt="Question reads minus tail reads: O-lens -1.0, NLA -1.1, J++ +2.3" width="400">

<!-- @c We expected that reading the very last tokens before the reply would improve performance over
here, especially for J++. Surprisingly, outputs close to the question itself perform the best. This
is? significant? @c you tell me.
@robbie significant, but only J++ goes your way, so I rewrote the paragraph. Paired question minus
tail, pooled over the four sets (each weighted equally), Haiku 5.5, one draw: O-lens -1.0 [-1.7,
-0.2] p 0.010; NLA -1.1 [-1.8, -0.4] p 0.001; J++ +2.3 [+1.4, +3.2] p < 0.001. Per set, the O-lens
and NLA tail wins are all favorites (-4.1, -4.4); cutoff, MMLU-Pro and SuperGPQA are within 1pp.
Accuracy: prompt only 79.9; O-lens question 81.9, tail 82.9; NLA 81.3, 82.4; J++ 81.1, 78.8.
@c one bar chart, aggregate across all four datasets, show 3 bars, one per metamodel.
@robbie done: I read "3 bars" as each metamodel's question-minus-tail difference.
`uv run python -m graphs.menus_inline` (PR #516).
@c help methink of an image here to explain all three token positions? You can make one and try it
out, should be an actual prompt and highlight the split of our tokens for early vs late.
@robbie done, the diagram above: `uv run python -m graphs.read_positions_diagram`. -->

### Why do Placebos Help?

## How we kept LLM agents from producing BS

We read so so so many transcripts.

We understand the important things deeply: scoring functions, sampling methods, any drafts.

## What we'd do differently

We spent way too long trying to get hard environments to work. We should have done some 'binary
search' in which we picked a tremendously easy task on which metamodels do show uplift, then went
halfway to our hard task and tried to get something of medium difficulty to work, and so on. If we
had done this, we would've noticed much sooner that even on workspace bench there are only ~4 tasks
which show genuine uplift from metamodels.

## What we learned

Robbie is much more bearish on metamodels than he was when we started out. He is not convinced that
they do more than noisy text inversion. If and when models lose their CoT, it's not obvious how you
would train them to do more than text inversion on the CoT-less replies of the model, meaning that
your metamodels degrade exactly when CoT degrades and might never be useful.

## Pre-sprint work

Robbie spent a lot of time trying to understand how 🔮-lenses work and should've spent more time
deeply understanding CHIVE and WorkspaceBench. He did not run code or experiments related to this
project idea.

## Cost

STILL TODO LEAVE UNTIL 4pm.

@kylie you should add your stuff here too.

| item                                          | spend  | source                                                                                        |
| --------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| RunPod GPUs, 2026-10-19 to 2026-10-31         | $2,051 | `runpodctl billing pods`; $1,997 of it H200s                                                  |
| RunPod GPUs before the work log, Oct 2 to 18  | $1,192 | same; not counted as sprint spend. This is an overestimate... some of this is NDRL. TODO fix. |
| OpenRouter, personal account (from Oct 27)    | $331   | `GET /api/v1/credits`, `total_usage`                                                          |
| OpenRouter, old usebits account (to Oct 27)   | TODO   | key retired; it was burning ~$2,100/month on Oct 26                                           |
| Anthropic API (BYOK Haiku, Sonnet, Opus 5.5)  | TODO   | Chase                                                                                         |
| Google AI Studio (Gemini 3.8 Flash, BYOK)     | TODO   | Chase; part of it came out of a $300 free-trial credit                                        |
| Docent, Hugging Face, Cloudflare/GitHub Pages | $0     | free tiers                                                                                    |

Things that will not have receipts:

- $2.5k of OpenRouter credits that Robbie had lying around from his startup
- Robbie's two claude max plans, which provided coding agent tokens + $600 in free API credits ($200
  a month, we hit a boundary on one).
- Robbie's Codex max plan.

## Where everything is

- **Code:** this repo,
  [robbiethompson18/counterfactual-investigator-agent-with-olens](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens).
- **Data and every run:** the HF dataset
  [Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens):
  O-lens banks, batch reads, run outputs.
- **Transcripts:** public [Docent](https://docent.transluce.org) collections, one per env plus
  cfbench ([docent.md](docent.md)).
- **Runs, judge calls and lens reads:**
  [expview](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/).
- **Token-level O-lens reads:** [olens-viewer.pages.dev](https://olens-viewer.pages.dev).
- **CHIVE incidents with every draw:** [chive-cases.pages.dev](https://chive-cases.pages.dev).
- **Every result and dead end:** [INDEX.md](INDEX.md) plus the active docs listed in the repo's
  `CLAUDE.md`.

# Appendix:

## 🔮-lens Now Speak English (DONE)

We investigated why so many o-lens bullets lead with a Mandarin character and then become English.
The canonical o-lens Agam gave us stripped leading whitespace before feeding AV output back to AR.
But many tokens' canonical form is _with_ leading whitespace, eg \_need instead of need. The AR
especially hates having the space stripped when when the next character will be a space.

Agam fixed this in RL, and the o-lens speaks English again.

<img src="img/olens-mandarin.png" alt="O-lens Mandarin before vs after keeping the bullet's leading space" width="500">

## Our issues with the CHIVE dataset (DONE)

- 40% of the data is off-policy in that it's one target model continuation at the end of a WildChat
  ChatGPT transcript. This makes some 'weird behavior' not that surprising, e.g.
  [000054](https://chive-cases.pages.dev/?case=prompt_000054), where the model correctly points out
  its own prior mistake (some of the time).
- We hardcode hypotheses to validate or not based on a raw move in % space (rather than say a
  likelihood ratio) so 20%->71% is a validated finding but 30% -> 0% is null and 10% --> 0% is
  'validated doesn't matter, moves < 15%.' It's not obvious what a better solution is. At least for
  positive examples Fisher's Exact Test seems better. If you p=0.01 (w/ Benjamini-Hochberg
  Correction) you get 1.5x experiments marked positive and usable in the dataset.
- On 8.6k experiments (12%, affecting 32% of investigations) we have a base behavior rate between
  10% and 50%, and a counterfactual which moves that rate to 0%, but because of our cutoffs this is
  marked inconclusive and not shown to the predictor. In 18% of investigations we have no positive
  experiments (and thus only negative claims to show the predictor) but we do have a 10%+ → 0 claim
  that we throw away. These 10%+ → 0 experiments actually aren't statistically significant; feeling
  confident that a counterfactual is 'inert' is a hard problem that likely requires more than 30
  samples.
  - The mirror case where CF bumps rate to 100% is much less common

## WorkspaceBench Shows Less Than We Thought (DONE)

On 28 of 33 categories on WorkspaceBench, Sonnet 5.5 performing text inversion matches or exceeds
the accuracy of every metamodel. On the remaining five categories, stock Qwen matches or exceeds the
performance of any metamodel.

The five exceptions are: two filters on Qwen being factually inaccurate, two requests for Qwen’s
favorite of something, and a steering task which cannot be run on other models.

On four other tasks, the benchmark shows metamodels beating a prompt-only baseline by comparing
Sonnet’s pass@1 to metamodels’ pass@hundreds-to-thousands. Metamodels get one (sometimes four)
grade(s) per token per layer and get credit if any of these pass.

Our full review is the "WorkspaceBench Review" tab of the
[sprint doc](https://docs.google.com/document/d/1Ghq1WhCyjcUV1XhG1x04mMW26o7saJV-S-yPxLhysI4/edit).

## Counterfactual investigation graphs (ALMOST DONE, Robbie reviewed, but I do want to add CHIVE here if possible)

Every chart we made for the counterfactual task. 100 − RMSE is in percentage points of the predicted
behavior rate, higher is better; hatching means placebo.

### Null Results on Counterfactual Predictions

We built ~30 environments, borrowing ideas from previous work (eg Aditya Singh and Gerson Kroiz's
[agent-interp-envs](https://github.com/gkroiz/agent-interp-envs) and Palisade's
[shutdown resistance setup](https://arxiv.org/abs/2509.14260) (Schlatter et al., 2025)). We did not
see metamodels help in evaluating counterfactuals in these environments. We did not even see
transcripts helping, which led us to believe that these counterfactuals were too difficult to
predict.

Nothing helper an investigator make better counterfacutla predictions:

<img src="img/cfbench-report/skill_pooled.png" alt="All targets pooled" width="404">

This was true across target models:

<img src="img/cfbench-report/skill_by_target.png" alt="By target model, with paired gains over transcript-only" width="500">

One task, essay grading, looked like it worked, but when we tried a different investigator model it
did not replicate:

<img src="img/cfbench-report/skill_hidden_goal.png" alt="hidden_goal.grade" width="404">

### Harness Sanity Chceks:

Models searched for reasonable things when reading transcripts and lens outputs:

<img src="img/cfbench-report/olens_search_topics.png" alt="What the O-lens agent searches for" width="489">

They read mostly transcripts instead of lookign at lenses:

<img src="img/cfbench-report/tool_calls.png" alt="Which tools the predictor calls" width="500">

### Harness Ablations

We tested whether our results were sensitive to investigator model, effort, and tool budget. These
sweeps use four hand-picked cases, so most differences between lines are noise. Stronger predictors
gain more from transcripts (GPT-6 Astra +20pp [+3, +36] over no transcript; Haiku about 0), but no
predictor reliably gains from the 🔮-lens: averaged over all seven, O-lens minus transcript-only is
+0.4pp [−0.9, +1.7] on these cases.

<img src="img/cfbench-report/models.png" alt="Predictor model by condition" width="400">

<img src="img/cfbench-report/flash_effort.png" alt="Gemini 3.8 Flash by reasoning effort" width="400">

<img src="img/cfbench-report/flash_calls.png" alt="Gemini 3.8 Flash by tool-call budget" width="400">

No model seems to get more or less uplift from having access to 🔮-lens outputs:

<img src="img/cfbench-report/ts_vs_olens.png" alt="Transcript-only vs O-lens by predictor" width="321">
