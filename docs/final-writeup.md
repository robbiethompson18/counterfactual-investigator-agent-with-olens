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

## Claims, with credences

## How the question changed

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

## Part 1: Simple Investigations That Work

All of our environments that work, with the exception of essay grading, are ideas stolen from
WorkspaceBench. The additions we make to their dataset:

- We do not grade with pass@thousands, running a regex on metamodel output at many layers and
  tokens. Instead, we pass metamodel output to an investigator model and ask it to pick the correct
  answer. This penalizes noise and hallucinations appropriately.
- We scale up dataset size to several thousand per category. This allows us to pick up small but
  statistically significant effects of metamodels, to train probes on unseen data, and most
  importantly, to _not filter our dataset in potentially biasing ways_. To be concrete:
  WorkspaceBench filters their idiosyncratic preference data (`directed_modulation_obscure`) to
  answers which disagree with their judge, Opus. So if they ask Qwen "What is your favorite flower?"
  and both Qwen and Opus reply "Lilies" then they drop this datapoint. This makes any
  transcript-only baseline confounded: we have already filtered to examples where our prompt only
  judge will be wrong. We drop this filter.

<!-- @c they call it what, basic recall? help me here
@robbie it's `directed_modulation_obscure` ("Think about your favorite Middle Eastern spice blend
while you write. Now write exactly this sentence: ..."). Their gate removed every item Opus or
Sonnet could answer from the text, so "disagree with Opus" is slightly narrow: it's Opus and
Sonnet. `basic_readout`'s 32 own-favourite items are not filtered that way (consistency only), and
that is where the lens wins only by reading the next token (docs/wsb-flash-prompt-only.md). -->

### Dataset Descriptions:

#### Multiple Choice Questions

We ask Qwen3.6-27B (thinking off) multiple-choice questions from MMLU-Pro and SuperGPQA, 8 samples
each, and the investigator must name the letter Qwen picks from the prompt alone. The only filter is
consistency, the same letter on at least 7 of 8 samples: that leaves 2,523 cases from a random 4,000
MMLU-Pro questions (Qwen right on 82%) and a random 3,000 of SuperGPQA's 6,328 consistent questions
(Qwen right on 63%). We never filter on Qwen being wrong, so most labels are also the correct
answer, and we report the Qwen-wrong slice separately.

#### Knowledge Cutoff Q&A

We ask Qwen present-tense factual questions whose answer changed around its knowledge cutoff, such
as "Who is the CEO of Intel?", written by Sonnet subagents with no date in the question. The label
is Qwen's own modal answer, not the truth, and the only filter is the same 7-of-8 consistency gate.
We don't filter on Qwen being wrong: of the 999 items, 339 get a stale answer and 660 a current one.

#### Idiosyncratic Preferences

We ask Qwen free-response questions about its own preferences, such as "Which text editor would you
choose for editing a config file over SSH? Answer with just the name.", from a bank of 5,187 that
Sonnet subagents wrote across six domains. This is WorkspaceBench's `basic_readout` implicit family
scaled from 32 items to 1,500. There is no truth, the label is Qwen's modal answer, and the only
filter is consistency: we keep every question where at least 7 of 8 samples agree, never filtering
on whether a text reader can guess it, so transcript-only already gets 80%
([favorites-qa-inline.md](favorites-qa-inline.md)).

<!-- @c three sentences on this dataset. again mention how we don't filter on qewn disagreeing, only on
being consistent.
@robbie done. -->

#### Essay Grading

We take 204 student intro paragraphs from PERSUADE 2.0, 34 at each human score from 1 to 6, and open
each with one of 225 sentences about who wrote it or what the user wants ("My rival for a
scholarship wrote this essay."), giving 1,800 cases. Qwen grades each prompt out of 10 about 8
times, and the investigator predicts whether its mean score goes up when the opener becomes the
plain "A student in my class wrote this essay."; it does on 54% of cases. There is no filter at all,
not even consistency: the openers were written before any was run, and every case is kept.

<!-- @c three sentences on this dataset. No fitlers here right besides consistency?
@robbie no filters, not even consistency: the truth is a mean over ~8 samples, so a noisy essay
just gets a small shift. The only selection is the balanced 34 intros per human score.
(cfbench/evals/essay_grading_cf_inline/eval.py, slides/talk/talk.tex) -->

### Data Filter

@c I like this sankey diagram! We should just align on colors!

Dark green: evaluated in dataset. You can split to qwen right and wrong, but keep these same color
so that it's obvious they are both valid and both users.

light green: usable but held out (either for probes or simply untouched)

gray: not usable (typically because qwen inconsistent).

Also @c put every sankey diagram below its 3 sentence description.

<img src="img/data-filter-sankey-supergpqa.png" alt="SuperGPQA: 10,000 questions to 3,000 eval cases, consistency the only filter" width="500">

## Results and Ablations

### Headline

### Headline by dataset

### By metamodel

### By token position

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

We built ~30 environments, borrowing ideas from previous work (eg Aditya and Palisade, @c links here
plz). We did not see metamodels help in evaluating counterfactuals in these environments. We did not
even see transcripts helping, which led us to believe that these counterfactuals were too difficult
to predict.

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
