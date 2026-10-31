# Final write-up: can metamodel reads tell an investigator what Qwen will do?

**Status (2026-10-31):** ACTIVE. Draft for the sprint interview; headers only, Robbie is writing it
by hand.

## Executive Summary

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
answer, and we report the Qwen-wrong slice separately ([mcq-unselected.md](mcq-unselected.md),
[supergpqa-inline.md](supergpqa-inline.md)).

<!-- @c three sentences on this dataset. Mention how we don't filter on qwen wrong.
@robbie done. -->

#### Knowledge Cutoff Q&A

We ask Qwen present-tense factual questions whose answer changed around its knowledge cutoff, such
as "Who is the CEO of Intel?", written by Sonnet subagents with no date in the question. The label
is Qwen's own modal answer, not the truth, and the only filter is the same 7-of-8 consistency gate.
We don't filter on Qwen being wrong: of the 999 items, 339 get a stale answer and 660 a current one
([cutoff-qa-inline.md](cutoff-qa-inline.md), [envs/cutoff_qa.md](envs/cutoff_qa.md)).

<!-- @c three sentences on this dataset. We don't filter on qwen wrong here either
@robbie done. Correct: no wrong filter, stale and current are both in. -->

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

TODO new graphs here. We should show some Sankey diagrams maybe?

<img src="img/data-filter-sankey-supergpqa.png" alt="SuperGPQA: 10,000 questions to 3,000 eval cases, consistency the only filter" width="500">

<!-- @c make a sankey for one dataset to test it out
@robbie done for SuperGPQA: `uv run python -m graphs.data_filter_sankey` (code PR #502, waiting
on your review; the PNG is in docs/img locally). Counts are constants from
docs/supergpqa-inline.md. If you like it I'll do the other datasets. -->

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

## 🔮-lens Now Speak English

We investigated why so many o-lens bullets lead with a Mandarin character and then become English.
The canonical o-lens Agam gave us stripped leading whitespace before feeding AV output back to AR.
But many tokens' canonical form is _with_ leading whitespace, eg \_need instead of need. The AR
especially hates having the space stripped when when the next character will be a space.

Agam fixed this in RL, and the o-lens speaks English again.

<img src="img/olens-mandarin.png" alt="O-lens Mandarin before vs after keeping the bullet's leading space" width="500">

## Our issues with the CHIVE dataset

- 40% of the data is off-policy in that it's one {Qwen,Gemma,etc} continuation at the end of a
  WildChat ChatGPT transcript. This makes some 'weird behavior' not that surprising, eg
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
  - the mirror case where CF bumps rate to 100% is much less common

## WorkspaceBench Shows Less Than We Thought

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

## Counterfactual investigation graphs

Every chart we made for the counterfactual task. 100 − RMSE is in percentage points of the predicted
behavior rate, higher is better; hatching means placebo. The PNGs are gitignored, so regenerate them
from the repo root, then rebuild `final-writeup.html`, which embeds them, with
`sh scripts/build_writeup.sh`:

```bash
uv run python -m graphs.cfbench_report        # the ten cfbench charts
uv run python -m graphs.honest_binary
uv run python -m graphs.honest_binary_olens --out docs/img/honest-grader/binary_rmse_olens.png
uv run python -m graphs.essay_curious_answer data/runs/cfbench/2026-10-28_essay-curious
uv run python -m graphs.essay_curious_score 'data/runs/cfbench/2026-10-28_essay-ten-duke3*'
uv run python -m graphs.hint_mcq_answer
uv run python -m graphs.hint_mcq_total
uv run --with transformers python -m graphs.olens_mandarin
```

### cfbench, pooled and by target

<img src="img/cfbench-report/skill_pooled.png" alt="All targets pooled" width="404">

All targets pooled. Env diff only, transcript-only, O-lens and its placebo land within about a point
of each other, all well above "edit does nothing".

<img src="img/cfbench-report/skill_by_target.png" alt="By target model, with paired gains over transcript-only" width="500">

By target model, with each condition's paired gain over transcript-only on the same cases.

<img src="img/cfbench-report/skill_hidden_goal.png" alt="hidden_goal.grade" width="404">

hidden_goal.grade, the one env with a pre-registered O-lens win (Qwen without CoT, 38 cases). It did
not replicate: on the same 38 cases, GPT-6.1 Sol under a later harness is null (RMSE:
transcript-only 31.7pp, O-lens 32.0pp, placebo 32.0pp). We never reran Flash on all 38 under that
harness. The task that failed when we sized it up is a different one, the
[hinted MCQ answer task](#hinted-mcq-answer-task-legacy-before-hints-were-dropped).

<!-- @c When we sized up sample size here this one failed right? read
https://docs.google.com/document/d/1Ghq1WhCyjcUV1XhG1x04mMW26o7saJV-S-yPxLhysI4/edit?tab=t.ikhkgkb68db4#heading=h.256u14widhfp
and include the graph here showign the fail too?
@robbie no: the linked report is about hint_mcq, not hidden_goal. Chart is in the hint section. -->

<img src="img/cfbench-report/skill_other_envs.png" alt="Every other Qwen-without-CoT case" width="404">

Every Qwen-without-CoT case outside hidden_goal.grade.

<img src="img/cfbench-report/olens_search_topics.png" alt="What the O-lens agent searches for" width="489">

What the O-lens agent searches the lens for.

<img src="img/cfbench-report/tool_calls.png" alt="Which tools the predictor calls" width="500">

Which tools the predictor calls, at a 20-call budget.

### cfbench, predictor ablations

<img src="img/cfbench-report/models.png" alt="Predictor model by condition" width="400">

Seven predictor models by condition, on the model chart's four cases.

<!-- @c remove the celing line from this graph plz ty
@robbie done.
@c possible to add error bars to this grpah somehow? don't chagne anything yet but this is mostly
noise right?
@robbie the lens gaps are noise; the GPT transcript gains are not. Paired case bootstrap on the 4
cases: O-lens minus transcript-only is -2.7 to +3.0pp, and only Haiku's +3.0 excludes 0 (but its
placebo matches it). Transcript-only minus no transcript is +16.5 [+5.9, +26.0] for Sol and
+20.7 [+9.2, +28.5] for Astra. No error bars added, per your note. -->

<img src="img/cfbench-report/flash_effort.png" alt="Gemini 3.8 Flash by reasoning effort" width="400">

Gemini 3.8 Flash by reasoning effort. Shading is a 95% bootstrap over the four cases, so it mostly
shows the spread between cases, which every line shares.

<!-- @c likewise error shading here would be nice.
@robbie done. -->

<img src="img/cfbench-report/flash_calls.png" alt="Gemini 3.8 Flash by tool-call budget" width="400">

Gemini 3.8 Flash by tool-call budget, shaded the same way.

<!-- @c likewise would like to see error bars.
@robbie done. -->

<img src="img/cfbench-report/ts_vs_olens.png" alt="Transcript-only vs O-lens by predictor" width="321">

Transcript-only vs O-lens, one dot per predictor and thinking level.

### cfbench, binary regrade

Graded as flip/no-flip instead of RMSE, O-lens, its placebo and transcript-only stay within a point
or two of each other at every threshold from 5 to 30pp. At 10pp, balanced accuracy for
transcript-only, O-lens and placebo is 73, 72 and 73 on GLM 5.3, and 78, 79 and 77 on Qwen without
CoT. Method and tables are in [cfbench-binary-regrade.md](cfbench-binary-regrade.md).

<!-- @c these ones aren't rendering? Do we even want them?
@robbie they came from a scratch script that isn't in the repo, so they can't be regenerated. Cut;
the numbers above say the same thing.
@c all of these graphs fucking massive can we resize every graph in the appenix to be max maybe 300x500 pixels?
@robbie done: every appendix image fits 500 wide by 300 tall. -->

### Honest grader

<img src="img/honest-grader/binary_rmse.png" alt="Honest grader, all 242 cases" width="500">

All 242 cases, without the lens.

<img src="img/honest-grader/binary_rmse_olens.png" alt="Honest grader, cases with lens reads" width="500">

The cases the lens runs cover, with O-lens and its placebo.

### Essay grading (essay_curious)

<img src="img/essay-curious/accuracy.png" alt="Letter-grade answer task by condition" width="500">

Share of draws naming the letter grade Qwen gives with no scenario.

<img src="img/essay-curious/per_case.png" alt="Letter-grade answer task per case" width="318">

The same, per case.

<img src="img/essay-curious/score_rmse.png" alt="Out-of-10 score RMSE by condition" width="500">

RMSE of the predicted no-scenario score out of 10.

<img src="img/essay-curious/score_per_case.png" alt="Every out-of-10 prediction per case" width="487">

Every out-of-10 prediction per case against the truth.

### Hinted MCQ answer task (legacy, before hints were dropped)

<img src="img/hint-mcq-answer/accuracy.png" alt="Hint answer task by hint type" width="500">

<img src="img/hint-mcq-answer/per_item_gold_hint.png" alt="Correct draws per item, gold-letter hint" width="500">

<img src="img/hint-mcq-answer/per_item_wrong_hint.png" alt="Correct draws per item, wrong-letter hint" width="500">

Share of draws naming Qwen's no-hint letter, overall and per item.

<img src="img/hint-mcq-total/by_hint.png" alt="Running total by hint type" width="500">

<img src="img/hint-mcq-total/combined.png" alt="Running total, all items" width="405">

The running total over the original items and the 2k scale-up. We thought this task was tremendously
easy for the 🔮-lens: Qwen consistently picks one wrong answer, a hint hides it, and the
investigator says what Qwen picks without the hint. It looked positive on 20 items. On all 241 items
it is null: O-lens 27.2% [22.0, 32.6], placebo 27.1% [21.9, 32.6], transcript-only 24.7% [19.8,
29.9].

## Appendix: List of Bugs We Caught:

Bugs in our own pipeline:

1. **The O-lens wrote Chinese.** The lens we were given stripped each bullet's leading space before
   the AR scored it during RL, so about half its bullets opened with a Chinese word. Agam retrained
   it with the space kept, and we redid every read made with the old lens
   ([olens-space-migration.md](olens-space-migration.md)).
2. **Reads of the wrong render.** Batch reads of thinking-off conversations rendered the thinking-on
   generation header, a prompt Qwen never saw (#477).
3. **Reads attached to the wrong tokens.** The reader found each message field by plain text search,
   so a short field like the answer "e" could match chat-template text and mislabel every token
   after it (#487).
4. **Labels nobody could match.** cutoff_qa labels were normalized twice, so answers with
   apostrophes ("N'Djamena") could never be matched by any investigator (9 of 999 items).
5. **A flipped sign.** cutoff_qa reported "positive" as Qwen correct instead of Qwen wrong, which
   made downstream tools skip every episode (#486).
6. **Stale predictions reused.** cfbench's `--reuse` matched rows by case, not by the prompt sent,
   so a changed reads file could pass off another eval's predictions as new ones (#483).
7. **The grader in the predictor's system prompt.** A merge put the grader function where the new
   system-prompt argument went; caught by the type checker (24c98af5).
8. **Two Qwens pooled.** OpenRouter's fp8 Qwen and our bf16 vLLM Qwen give different rates (14/20 vs
   36/40 on one env), and we had pooled them as one target (1237cedb).
9. **Truncated answers scored as complete.** Replies cut off by a token limit were labelled as
   finished episodes until the runner raised on them (#473).
10. **The gate and the env sampled differently.** cutoff_qa's env ran at different sampling settings
    from the stability gate that made its labels (55f53874).
11. **A placebo that was the answer.** The first essay placebo was another opener's reads on the
    same essay, which was sometimes the counterfactual prompt itself.
12. **Reads of the counterfactual.** One cfbench condition read the lens on the counterfactual
    prompt, which is outside the zero-counterfactual setting; we dropped it.
13. **Greedy reads were shared.** Before 2026-10-27 lens reads were greedy, so every rollout of one
    prompt got the identical read, not independent draws.

Bugs and biases in the setups we inherited:

14. **CHIVE's thresholds discard clean kills.** The absolute 50pp rule marks 8,623 experiments where
    a 10–50% behavior drops to 0% as inconclusive ([chive-critique.md](chive-critique.md)).
15. **A filter that leaked the label.** Our first MCQ eval kept only questions where DeepSeek V4
    Flash disagreed with Qwen, which imported DeepSeek's letter bias: the hinted letter alone
    predicted "Qwen knew" at AUROC 0.70, vs 0.53 unfiltered
    ([mcq-unselected.md](mcq-unselected.md)).
16. **WorkspaceBench's text baseline was truncated.** Its prompt-only floor is one 256-token
    generation, and 40–73% of them stop mid-sentence
    ([wsb-flash-prompt-only.md](wsb-flash-prompt-only.md)).
17. **WorkspaceBench compares pass@1 to pass@hundreds.** The lens passes if any readout over its
    read grid passes; the text reader gets one answer.

<!-- @c add 3-20 here and I will trim.
@robbie done, 17. Sources: the fix commits on main and the docs linked. -->
