# Using Metamodels for Counterfactual Investigations

**Status (2026-10-31):** ACTIVE. Draft for the sprint interview; headers only, Robbie is writing it
by hand.

Note: for sections that are fully done I will write DONE in the header. For sections I want a Kylie
review I will write (KTR). FOr Claude I will write (CTR).

TODO BROAD:

- put essay grading and blog posts into some null tasks.

## Executive Summary (KTR)

We originally set out to test whether metamodels are useful in counterfactual investigations. We
replicated results from CHIVE ([Karvonen et al., 2026](https://arxiv.org/abs/2608.16747)), and
tested whether 🔮-lenses help where NLAs and J-lenses did not.

Contrary to our expectations, 🔮-lens were unhelpful on counterfactuals in the CHIVE dataset. They
performed at or below a transcript-only baseline. We replaced the whole dataset with handbuilt
environments to make the task easier and on-policy, but this did not just results.

We suspect that counterfactuals are a harder task than the literature appreciates: on both our
environments and on CHIVE, adding a transcript helps a counterfactual investigator surprisingly
little compared to simply giving the investigator both system prompts. TODO waiting on CHIVE results
here

We retreated from counterfactuals to simpler tasks until we found ones that work, and none of them
is a counterfactual. They are: predicting Qwen's answer to a multiple choice question; predicting
Qwen's idiosyncratic preferences; predicting Qwen's answer to a factual query which is past its
knowledge cutoff.

## Claims, with credences (CTR, KTR)

Current SOTA metamodels (J++ lens, 🔮-lens, NLA) do not provide uplift > 1% on 'wild' counterfactual
investigations (eg CHIVE, WeirdChat) or on 'agentic' counterfactuals (eg
[Palisade one](https://arxiv.org/abs/2509.14260),
[Aditya ones](https://github.com/gkroiz/agent-interp-envs)) under any harness configuation: 95%.

SOTA metamodels do provide uplift on the simple prediction tasks we detail below (Multiple Choice
Questions, Knowledge Cutoff Q&A, Idiosyncratic Preferences): 95%. They are worse than linear probes
(where such probes are practicable): 95%.

SOTA metamodels would provide uplift on 'wild' counterfactuals if the investigator and subject model
was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

SOTA metamodels would provide uplift on 'agentic' counterfactuals if the investigator and subject
model was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

I think the last two questions are terribly interesting! If you are reading this and you work at a
lab: this should be a straighforward experiment to run!

## Part 0: 🔮-lens Does Not Provide Uplift on CHIVE

Bar chart on chives showing nulls Bar chart: score on all of our envs, showing nulls Bar chart:
abaltions by model showing null

We ablated everything we could think of, and in none of it did the O-lens reliably beat
transcript-only or its own placebo:

- We swapped the predictor across seven models, from Haiku 5.5 to GPT-6 Astra, plus thinking-off and
  deliberately weak predictors.
- We varied the predictor's reasoning effort from low to high, and its tool-call budget from 5 to 40
  calls.
- We ran the target with and without chain of thought.
- We changed how reads reach the predictor: as an agentic tool, or pasted straight into its prompt.
- We changed the lens itself: the old lens vs the space-fixed one, greedy vs temperature-1 decoding,
  and reads at stop tokens vs at every token.
- We tried other tools: J-lens, NLAs and [Thought Anchors](https://arxiv.org/abs/2506.19143) (Bogdan
  et al., 2025), each against its own placebo.
- We tried other models' lenses: Camila's DeepSeek V4 Flash lens on alignment faking and Celeste's
  GLM 5.3 lens on over-eagerness.
- We iterated the predictor's harness through six versions.
- We rescored everything with log skill, RMSE, and binary flip/no-flip at thresholds from 5 to 30pp.
- And on CHIVE itself we tried Flash vs Sonnet 5.5 as the predictor, and free-form vs yes/no
  answers.
- The one thing we never ablated (which we regret) is which layers the lens reads.

More details and charts is in the appendix,
[Counterfactual investigation graphs](#counterfactual-investigation-graphs).

### Chive Shows Not Much Uplift From The Transcript

<img src="img/chive-inline.png" alt="AUROC on CHIVE's yes/no claims: Haiku edit only vs transcript, beside CHIVE's Opus 4.8 transcript and resampling arms, per target and pooled" width="760">

<img src="img/chive-inline-diff.png" alt="Haiku AUROC, transcript minus edit only, paired by claim: +0.012 pooled" width="500">

TODO fix this to use qwen 3.6 27b if we can, then use our jlens / olens / NLA.

## Part 1: Our New Simple Evals

All of our environments that work are ideas stolen from WorkspaceBench. The additions we make to
their dataset:

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

### Dataset Descriptions: (BACK TODO)

In each diagram, dark green is what we evaluate, with Qwen-right and Qwen-wrong cases in the same
color because both are valid. Light green is usable but held out, either for probes or simply
untouched. Gray is not usable, typically because Qwen is inconsistent.

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
questions). We don't use the light green pools for this. This is valid, but training on a held-out
pool would have been cleaner. We used the same protocol for Essay Grading.

Example Qwen transcripts on Docent: MMLU-Pro,
[Qwen right](https://docent.transluce.org/dashboard/bd9d3ef4-13bb-4e85-9883-f7718b8801ee/agent_run/0ea03746-b9c1-4404-9fb6-e7e6f1524795)
and
[Qwen wrong](https://docent.transluce.org/dashboard/bd9d3ef4-13bb-4e85-9883-f7718b8801ee/agent_run/a7135119-dd21-4857-9918-2dc57013f14e);
SuperGPQA,
[Qwen right](https://docent.transluce.org/dashboard/e75c1169-f6dd-4f36-9137-35ede45f8997/agent_run/5370988b-5caa-4bc4-b7ca-2421075ef795)
and
[Qwen wrong](https://docent.transluce.org/dashboard/e75c1169-f6dd-4f36-9137-35ede45f8997/agent_run/16cb4b7d-d5bf-446a-bb4d-87e139ece6d3).

#### Knowledge Cutoff Q&A

We ask Qwen present-tense factual questions, many of whose answers changed around its knowledge
cutoff, such as "Who is the CEO of Intel?", written by Sonnet subagents with no date in the
question. The label is Qwen's own modal answer, not the truth, and the only filters are the same
7-of-8 consistency gate and dropping numeric answers. We don't filter on Qwen being wrong: of the
999 items, 339 get a stale answer and 660 a current one. This is WorkspaceBench's `knowledge_cutoff`
category, scaled from 109 items to 999, and not filtered on Qwen disagreeing with Opus and Sonnet.

<img src="img/data-filter-sankey-cutoff.png" alt="Knowledge cutoff: 5,467 questions to 999 eval cases, stale and current answers both kept" width="500">

Example Qwen transcripts on Docent:
[stale answer, the Foo Fighters' drummer](https://docent.transluce.org/dashboard/022092a9-27c8-4443-9fbc-c951c10fdc78/agent_run/389a242f-4ded-4f4b-a3c8-6c5a48b64b69)
and
[current answer, Revolut's headquarters](https://docent.transluce.org/dashboard/022092a9-27c8-4443-9fbc-c951c10fdc78/agent_run/fc6c6634-8ec2-4014-9b32-1727bc7d4b2a).

#### Idiosyncratic Preferences

We ask Qwen free-response questions about its own preferences, such as "Which text editor would you
choose for editing a config file over SSH? Answer with just the name.", from a bank of 5,187 that
Sonnet subagents wrote. This is WorkspaceBench's `basic_readout` implicit family scaled from 32
items to 1,500. There is no truth, the label is Qwen's modal answer, and the only filter is
consistency: we keep every question where at least 7 of 8 samples agree, never filtering on whether
a text reader can guess it, so transcript-only already gets 80%.

<img src="img/data-filter-sankey-favorites.png" alt="Idiosyncratic preferences: 5,187 questions to 1,500 eval cases" width="500">

Example Qwen transcripts on Docent:
[name an amphibian](https://docent.transluce.org/dashboard/fa7a372c-dbb1-47c1-a685-489e46579bcf/agent_run/45abd398-c6f3-436a-9f89-99bcb469b1db)
and
[text editor over SSH](https://docent.transluce.org/dashboard/fa7a372c-dbb1-47c1-a685-489e46579bcf/agent_run/aabae6b8-472e-474d-9076-7ebef10dd4f2).

#### Essay Grading

We take 204 student intro paragraphs from
[PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0) (Crossley et al., 2024), 34 at each
human score from 1 to 6, and open each with one of 225 sentences about who wrote it or what the user
wants ("My rival for a scholarship wrote this essay."), giving 1,800 cases. Qwen grades each prompt
out of 10 about 8 times, and the investigator predicts whether its mean score goes up when the
opener becomes the plain "A student in my class wrote this essay."; it does on 54% of cases. There
is no filter at all, not even consistency: the openers were written before any was run on these
essays, and every case is kept.

<img src="img/data-filter-sankey-essay.png" alt="Essay grading: 1,800 cases, no filter, all evaluated" width="500">

Example Qwen transcripts on Docent:
[rival opener](https://docent.transluce.org/dashboard/d0ca26f8-3fcb-46ae-934a-43ee4e9dc3f6/agent_run/88f65572-a720-48ac-b03a-af7615811af2)
and
[plain opener on the same intro](https://docent.transluce.org/dashboard/d0ca26f8-3fcb-46ae-934a-43ee4e9dc3f6/agent_run/0de617fd-dcd2-4717-bdc9-72676025408b).

#### Blog Authorship

We give Qwen a real blog post from the
[Blog Authorship Corpus](https://huggingface.co/datasets/tasksource/blog_authorship_corpus) (Schler
et al., 2006) and ask a first-person question about its author, such as "What is my sex, based on
the blog post I wrote below?". The investigator sees Qwen's answer and one sentence of the post, and
predicts whether deleting that sentence makes Qwen much less sure of its answer, meaning its
probability falls by more than max(0.15, 5 × the post's median drop). Half of the 1,316 cases are
such sentences, each paired with an ordinary sentence from the same post. Unlike our other datasets
this one does select on Qwen being wrong: of the questions where Qwen is confident and a typical
deletion barely moves it, we keep every one it gets wrong and sample the ones it gets right, so Qwen
is wrong on 722 of the 1,316 cases.

<img src="img/data-filter-sankey-blog.png" alt="Blog authorship: 8,016 questions to 533 eval questions, every confident Qwen-wrong question kept and Qwen-right questions sampled" width="500">

Example investigator transcripts on Docent, on the same post (Qwen's answer here is letter
probabilities, not a sampled reply, so its prompt is shown inside the investigator's):
[a significant sentence](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/4e1bd243-d610-4c33-b3a8-8e64c7b16e4b),
[an ordinary one](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/a6927a81-8e69-4409-bad1-e00ee7b0327c).

## Results

### Headline

Metamodels work on these three tasks! The 🔮-lens and NLA are about equally good, and J++ is weaker.
The next few charts show pooled correctness over all three tasks. Our investigator is always Haiku
5.5 on medium effort.

<img src="img/headline-metamodels.png" alt="Pooled over the three tasks: transcript only 78.0%, O-lens 84.5%, J++ 80.0%, NLA 83.6%" width="500">

All three beat their placebo versions (where we swap in metamodel data from a different question) by
statistically significant amounts, though J++'s gap is mostly its placebo misleading the
investigator (−9.7pp vs transcript-only, against its own +2.0pp):

<img src="img/headline-placebos.png" alt="Pooled over the three tasks: O-lens 84.5% vs placebo 74.1%, J++ 80.0% vs 68.3%, NLA 83.6% vs 79.7%" width="500">

No metamodel performs as well as a probe on multiple choice, the one of these tasks where we have
one:

<img src="img/headline-probes.png" alt="Multiple choice: O-lens 80.3%, J++ 77.4%, NLA 77.0%, probe 98.7%" width="500">

The difference is more noticeable when you measure by error rate (lower better):

<img src="img/headline-probes-error.png" alt="Error rate on multiple choice: O-lens 19.7%, J++ 22.6%, NLA 23.0%, probe 1.3%" width="500">

### Datasets That Work Convincingly

#### On Multiple Choice Questions

<img src="img/headline-mcq.png" alt="Multiple choice, transcript only, three metamodels and their placebos" width="500">

#### Knowledge Cutoff Q&A

<img src="img/headline-cutoff.png" alt="Knowledge cutoff, transcript only, three metamodels and their placebos" width="500">

This dataset mostly works because Haiku doesn't have the same knowledge cutoff as Qwen. If you tell
Haiku about Qwen's knowledge cutoff, or use a judge model with a similar knowlege cutoff, most or
all of the effect disappears.

<img src="img/cutoff-qa-investigators.png" alt="Knowledge cutoff by investigator: Haiku 5.5 (transcript only 47.1% on stale items, told the model and its cutoff 61.5%, O-lens 72.6%, placebo 55.6%) beside Gemini 2.5 Flash-Lite (answers the question itself 63.1%, transcript only 70.4%, O-lens 54.0%, placebo 8.3%), with not-stale and all-items panels" width="760">

Telling Haiku that the model is Qwen3.6-27B with a January 2025 cutoff closes about half of the gap
on stale items (transcript only goes from 47% to 62%, against 73% with the O-lens). Gemini 2.5
Flash-Lite, whose cutoff is about Qwen's, names Qwen's stale answer on 63% of stale items with no
investigator prompt at all and on 70% as a transcript-only investigator; given O-lens reads it falls
to 54%, and given placebo reads to 8%, so it cannot use the reads at all. Regenerate with
`uv run python -m graphs.cutoff_qa_investigators`.

#### Idionsynratic Preferences

<img src="img/headline-preferences.png" alt="Idiosyncratic preferences, transcript only, three metamodels and their placebos" width="500">

### Datasets That Do Not Work Convincingly

#### Blog Authorship

On 1,316 selected sentence deletions from 520 bloggers, we predict whether deleting a sentence
substantially lowers Qwen's confidence in its answer about the author's age or sex. Text-only Haiku
scores 0.797 AUROC, Haiku with O-lens reads 0.803, and a standalone linear probe 0.784. The O-lens
gain over text-only is not statistically significant; the standalone probe is also below text-only
on this metric. The probe is supervised on this deletion task with bloggers held out, and its
configuration was chosen during exploration of this dataset.

<img src="img/author-blog-drop.png" alt="Blog sentence-deletion AUROC: text-only Haiku 0.797, Haiku with O-lens 0.803, standalone L2 probe 0.784, with 95% blogger-bootstrap confidence intervals" width="500">

The chart shows the probe alone. Details and the separate ensemble analysis are in
[Author evidence](author-mcq-probe.md). Regenerate with `uv run python -m graphs.author_blog_drop`.

#### Essay Grading

## Ablations

### Filtering On Correctness

#### No Uplift When Qwen is Right

<img src="img/headline-qwen-right.png" alt="Cases where Qwen is right: transcript only vs O-lens, J++ and NLA on MMLU-Pro (94.4, 95.6, 95.1, 95.0), SuperGPQA (82.5, 88.8, 83.9, 84.6) and knowledge cutoff (94.6, 92.1, 91.8, 92.9)" width="760">

<img src="img/headline-qwen-wrong.png" alt="Cases where Qwen is wrong: transcript only vs O-lens, J++ and NLA on MMLU-Pro (29.5, 33.3, 38.9, 31.4), SuperGPQA (46.2, 57.0, 49.6, 49.6) and knowledge cutoff (47.1, 72.6, 67.1, 73.6)" width="760">

Split by whether Qwen's answer is correct (on MMLU-Pro and SuperGPQA its modal letter is the gold
one; on knowledge cutoff its answer is still true rather than stale; preferences have no truth, so
they are left out), nearly all of the uplift is on the cases where Qwen is wrong: there the lenses
add 4 to 26pp over transcript only, while on Qwen-right cases they add 0 to 6pp over an 82 to 95%
baseline and cost 2 to 3pp on knowledge cutoff. Regenerate with
`uv run python -m graphs.headline_ablations`.

#### Even Bad Results Look Good When You Filter On Qwen Being Wrong!

This is why we think it's so important to _not filter data based on subject model correctness_. If
you do this, then way of pushing your investigator off of 'Qwen will say the right answer' looks
like a significant effect.

<img src="img/headline-essay-correctness.png" alt="Essay grading AUROC on the 76 essays Qwen grades wrong (transcript only 0.743, O-lens 0.764, placebo 0.752, J++ 0.747 / 0.740, NLA 0.749 / 0.744) and the 128 it grades right (0.758, 0.781, 0.760, 0.766 / 0.755, 0.766 / 0.764)" width="760">

Qwen is "wrong" on an essay when it grades a top-half essay (PERSUADE holistic score 4 to 6) below
its own median grade, or a bottom-half essay above it, which is 76 of the 204 essays; the split
changes little here, the O-lens leading transcript only by about 0.02 AUROC on both halves.

<img src="img/headline-blog-correctness.png" alt="Blog sentence deletion AUROC on the 722 cases where Qwen is wrong about the author (transcript only 0.749, O-lens 0.766, placebo 0.746) and the 594 where it is right (0.849, 0.841, 0.831)" width="760">

On the blog deletions Qwen is wrong about the author's age or sex on 722 of the 1,316 cases: there
the O-lens leads transcript only by 0.017 AUROC and its placebo by 0.020, while on the Qwen-right
cases it trails transcript only by 0.008. Both charts regenerate with
`uv run python -m graphs.headline_ablations`.

### Which Metamodel is Best?

<img src="img/headline-lens-diffs.png" alt="Each lens minus transcript only, paired by case: multiple choice O-lens +5.1, J++ +2.2, NLA +1.8; knowledge cutoff +7.0, +4.9, +7.8; idiosyncratic preferences +7.3, -0.9, +7.0; pooled +6.5, +2.0, +5.5" width="620">

Paired by case, the O-lens leads transcript only on every task (+6.5pp pooled), NLA matches it on
knowledge cutoff and preferences but adds only +1.8pp on multiple choice, and J++ is the weakest
overall (+2.0pp pooled), beating NLA on multiple choice and adding nothing on preferences.
Regenerate with `uv run python -m graphs.headline_ablations`.

### By token position

This section pools the three tasks that work, with multiple choice split into MMLU-Pro and
SuperGPQA, so four sets, each weighted equally.

When testing J++ lens, we read every token in the prompt and the boilerplate at the start of the
assistant turn. When testing NLA and 🔮-lens, we read punctuation and newlines in the the prompt,
and assistant turn boilerplate. See image below:

<img src="img/read-positions-diagram.png" alt="One MMLU-Pro prompt as Qwen sees it, question reads in gold, tail reads in maroon, the last token boxed" width="450">

We ablated this, testing whether the early or late reads were more important. We expected that
reading the very last tokens before the reply would improve performance, especially for J++. In
fact, the opposite is true: late tokens are more helpful for NLA and 🔮-lens, and less important for
J++. This is makes us 'like' J++ lens more, and NLA and 🔮-lens less. J++ is reading Qwen's thoughts
earlier, suggesting it is doing something workspace-like. NLA and 🔮-lens read thoughts later and
can be accused of being next-token predictors.

<img src="img/menus-inline-position.png" alt="Question reads minus tail reads: O-lens -1.0, NLA -1.1, J++ +2.3" width="400">

### Why do Placebos Help?

#### Which tasks

Yes: paired by case, a placebo only helps on knowledge cutoff (J++ placebo +3.2pp, NLA placebo
+4.8pp, O-lens placebo −0.6pp over transcript only), where any stale read tells Haiku the model
thinks it is an earlier year; on multiple choice and preferences every placebo is neutral or hurts,
the J++ placebo badly (−5.5pp and −27.0pp).

<img src="img/headline-placebo-diffs.png" alt="Each lens's placebo minus transcript only, paired by case: multiple choice O-lens -1.2, J++ -5.5, NLA -0.5; knowledge cutoff -0.6, +3.2, +4.8; idiosyncratic preferences -9.8, -27.0, +0.8; pooled -3.9, -9.7, +1.7" width="620">

Regenerate with `uv run python -m graphs.headline_ablations`.

#### Total Tokens Per

<img src="img/headline-tokens.png" alt="Prompt tokens each lens adds to Haiku's prompt over transcript only, pooled and per set" width="500">

We see the most uplift from x, which uses less tokens than y. Thus we don't think that extra tokens alone drive placebo gains.

Reads also make Haiku think longer, so we checked whether more thinking explains the gains. It
doesn't look like it. Pooled, Haiku spends 154 thinking tokens per prediction on transcript-only,
224 with the 🔮-lens, 214 with NLA and 288 with J++. But the placebos make it think as long or
longer (297, 226 and 377) and score worse, and J++ thinks the most of the three lenses and scores
the lowest.

<img src="img/headline-thinking.png" alt="Thinking tokens pooled over the three tasks: transcript only 154, O-lens 224 vs placebo 297, J++ 288 vs 377, NLA 214 vs 226" width="500">

Within every condition and dataset, the predictions where Haiku thinks longer are the ones it gets
wrong more often (correlation −0.07 to −0.44), so long thinking looks more like a sign of confusion
than a source of accuracy. The one place this could still matter is preferences, where
transcript-only barely thinks at all (15 tokens, and none on 83% of draws), so reads may be
prompting thought it would otherwise skip. We didn't run transcript-only with a bigger thinking
budget, which is what would test this directly.

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

## Future Work (@CTR)

We will list a couple hypotheses for why metamodels do not provide uplift on counterfactuals, along
with the experiments one could run to test them:

- Qwen is too small and dumb of a model for there to be interesting counterfactuals to investigate.
  Rerun CHIVE using smarter subject models.
- Metamodels might be hamstrung primarily by their 'hallucinations,' split-brained ideation
  ([recommending drunk driving while warning against it](https://olens-viewer.pages.dev/#pattern=219&transcript=64bb7e454c2534db8fccd094a80416d8f7353753d3a9a9931a203fa6359b469c3c%2Ft053)),
  and non-workspace-content. Make architectural changes to prevent this.
- There exists some 'better' dataset than CHIVE or our environments which might show metamodel
  uplift. Use a search process similar to
  [WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat) or more recent live user data
  than what is on [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M).
- Our environments would have worked with a large enough sample size, cleaner graders, and/or a
  smarter subject model. We suspect there are many small issues in our environments' graders, which
  is why we do not ship them as part of our repo. We doubt that this hypothesis is true; if we
  didn't we would have spent more time cleaning the graders!

Additional

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
see metamodels help in evaluating counterfactuals in these environments. Transcripts helped only a
little, which led us to believe that these counterfactuals were too difficult to predict.

Nothing helper an investigator make better counterfacutla predictions:

<img src="img/cfbench-report/skill_pooled.png" alt="All targets pooled" width="404">

This was true across target models:

<img src="img/cfbench-report/skill_by_target.png" alt="By target model, with paired gains over transcript-only" width="500">

One task, an earlier essay grading environment (hidden_goal.grade), looked like it worked, but when
we tried a different investigator model it did not replicate:

<img src="img/cfbench-report/skill_hidden_goal.png" alt="hidden_goal.grade" width="404">

### Harness Sanity Chceks:

Models searched for reasonable things when reading transcripts and lens outputs:

<img src="img/cfbench-report/olens_search_topics.png" alt="What the O-lens agent searches for" width="489">

They read mostly transcripts instead of lookign at lenses:

<img src="img/cfbench-report/tool_calls.png" alt="Which tools the predictor calls" width="500">

### Harness Ablations

We tested whether our results were sensitive to investigator model, effort, and tool budget. These
sweeps use four hand-picked cases, so most differences between lines are noise. No predictor
reliably gains from the 🔮-lens: averaged over all seven, O-lens minus transcript-only is +0.4pp
[−0.9, +1.7] on these cases.

<img src="img/cfbench-report/models.png" alt="Predictor model by condition" width="400">

<img src="img/cfbench-report/flash_effort.png" alt="Gemini 3.8 Flash by reasoning effort" width="400">

<img src="img/cfbench-report/flash_calls.png" alt="Gemini 3.8 Flash by tool-call budget" width="400">

No model seems to get more or less uplift from having access to 🔮-lens outputs:

<img src="img/cfbench-report/ts_vs_olens.png" alt="Transcript-only vs O-lens by predictor" width="321">

# TODO WITH MORE TIME:

- Claude thinks we have some slightly significant astra result? Possibly we want to try this? doubt
  it. On some set of four CFs.
- Ship our environments in any usable way? We should at least write up that we think there are lots
  of grader issues.
