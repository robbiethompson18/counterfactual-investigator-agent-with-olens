# Using Metamodels for Counterfactual Investigations

**Status (2026-10-31):** ACTIVE. Draft for the sprint interview; headers only, Robbie is writing it
by hand.

@agents sorry still haven't decided what is canonical between tail and all tokens. Might flip flop
again sadly.

<!-- AGENTS (Robbie, 2026-10-31): every lens number, chart and token count for MCQ (MMLU-Pro, SuperGPQA), knowledge
cutoff and preferences comes ONLY from the canonical menus_inline runs, `all` menu (each prompt's punctuation and option
newlines plus the 9 tail tokens, layers 32-60, same prompt and placebo rules on every set): `graphs/menus_inline.py` SETS and
CANONICAL, plus their 3-draw top-ups (`*-all-d12b` / `-d12c`). The `prompt` menu (no tail tokens, top-ups `*-prompt-d12`) is
the token-position ablation, also at 3 draws; it was briefly canonical on 2026-10-31 evening. Never the 2026-10-29 per-set runs (mcq-unselected-haiku-*, supergpqa-haiku-3k-*,
cutoff-qa-1k-*, favorites-haiku-v3-*): each set there read different positions (J++ at every token on three sets, 16
positions on MMLU-Pro) with a different prompt. Essay grading has no menus run and keeps its own. -->

> **🚧 TODO NON-CANONICAL DATA (Robbie, 2026-10-31): canonical means the `menus_inline` runs, `all`
> menu: every lens read at the same positions (each prompt's punctuation and option newlines plus
> the 9 tail tokens at the start of the assistant turn), all six layers 32/40/48/52/56/60, one
> prompt and one placebo rule on every set, 3 draws. Every chart and number below marked 🚧 comes
> from something else and must be redone on those runs before this ships. Status at 17:30 PDT: every
> Multiple Choice, Knowledge Cutoff and Idiosyncratic Preferences chart and number is now on the
> canonical runs at 3 draws (top-ups `*-all-d12b` / `-d12c`; PRs #539, #541, #549, #550), and their
> 🚧 blocks are gone. The `prompt` menu (no tail tokens) has 3 draws too (`*-prompt-d12`) and is the
> token-position ablation. What's left marked 🚧 is blog authorship and the counterfactual charts.**

Note: for sections that are fully done I will write DONE in the header. For sections I want a Kylie
review I will write (KTR). FOr Claude I will write (CTR).

TODO BROAD:

- put essay grading and blog posts into some null tasks.

RULES FOR CLAUDE:

- Never give bash to reprodue something, people can just point their agent at our repo.
- Never link to github (except once to the overall repo); you are allowed to link to other parts of
  this doc and to arxiv papers and to docent.
- Never numbers in prose by default!! We do this so often. I want to audit all of these eventually.
  Numbers go in graphs.
- No new diff charts, but can keep ones that are already here / that Robbie asks for specifically.

## Executive Summary (KTR)

We originally set out to test whether metamodels are useful in counterfactual investigations. We
planned to replicate results from CHIVE ([Karvonen et al., 2026](https://arxiv.org/abs/2608.16747)),
and tested whether 🔮-lenses help where NLAs and J-lenses did not.

Contrary to our expectations, 🔮-lens were unhelpful on counterfactuals in the CHIVE dataset. They
were indistinguishable from a transcript-only baseline. We replaced the whole dataset with handbuilt
environments to make the task easier and on-policy, but this did not change results.

We suspect that counterfactuals are a harder task than the literature appreciates: on both our
environments and on CHIVE, adding a transcript helps a counterfactual investigator surprisingly
little compared to simply giving the investigator both system prompts.

We retreated from counterfactuals to simpler tasks until we found ones that work. They are: Multiple
Choice (predicting Qwen's answer to a multiple-choice question); Idiosyncratic Preferences
(predicting Qwen's own preferences); Knowledge Cutoff (predicting Qwen's answer to a factual query
which is past its knowledge cutoff). These are notably NOT counterfactuals.[^hint]

[^hint]:
    We do have some hint data which reframes Multiple Choice as counterfactuals, see
    [Hinted MCQ as a Counterfactual](#hinted-mcq-as-a-counterfactual-todo-kylie).

## Claims, with credences (CTR, KTR)

Current SOTA metamodels (J++ lens, 🔮-lens, NLA) do not provide uplift of more than 3 points (AUROC
or percentage points) on 'wild' counterfactual investigations (eg CHIVE, WeirdChat) or on 'agentic'
counterfactuals (eg [Palisade one](https://arxiv.org/abs/2509.14260),
[Aditya ones](https://github.com/gkroiz/agent-interp-envs)) under any harness configuation: 95%.

SOTA metamodels do provide uplift on the simple prediction tasks we detail below (Multiple Choice,
Knowledge Cutoff, Idiosyncratic Preferences): 95%.[^KCspecial] They are worse than linear probes
(where such probes are practicable): 95%.

[^KCspecial]:
    This is somewhat ill-defined for the Knowledge Cutoff example, for reasons which are clear if
    you read that section. For a model with the same knowledge cutoff or with a hint about the
    subject's knowledge cutoff, my credence on this particular dataset is ~50%.

SOTA metamodels would provide uplift on 'wild' counterfactuals if the investigator and subject model
was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

SOTA metamodels would provide uplift on 'agentic' counterfactuals if the investigator and subject
model was GPT Astra 6, Claude Mythos 5.1, or Gemini Carbon: 30%.

I think the last two questions are terribly interesting! If you are reading this and you work at a
lab: this should be a straighforward experiment to run!

## I: 🔮-lens Does Not Provide Uplift on 'Wild' or Agentic Counterfactuals

First, because of misgivings about the CHIVE dataset, we built our own agentic environments to test
counterfactual experiments. We then ablated everything we could think of. Nothing got the 🔮-lens to
reliably beat a transcript-only baseline or its own placebo:

- We swapped the predictor across seven models, from Haiku 5.5 to GPT-6 Astra, plus thinking-off and
  deliberately weak predictors.
- We varied the predictor's reasoning effort from low to high, and its tool-call budget from 5 to 40
  calls.
- We ran the target with and without chain of thought.
- We changed how reads reach the predictor: as an agentic tool, or pasted straight into its prompt.
- We tried another tool, [Thought Anchors](https://arxiv.org/abs/2506.19143) (Bogdan et al., 2025),
  against its own placebo.
- We tried other models' lenses: Camila's DeepSeek V4 Flash lens on alignment faking and Celeste's
  GLM 5.3 lens on over-eagerness.
- We iterated the predictor's harness through about a dozen versions.
- We rescored everything with log skill, RMSE, and binary flip/no-flip at thresholds from 5 to 30pp.
- The one thing we never ablated (which we regret) is which layers the lens reads.

More details and charts are in the appendix,
[Counterfactual investigation graphs](#counterfactual-investigation-graphs).

### CHIVE Shows Little Uplift From The Transcript

A cheap test we ran: does giving the investigator the transcript of a chive rollout help in
predicting counterfactuals? If the answer is no, then these scenarios are very hard, and we don't
expect metamodels to move the needle much either.

The graphs below show that transcripts do improve investigator AUROC, but only by about 1 point.

<img src="img/chive-inline.png" alt="AUROC on CHIVE's yes/no claims: Haiku edit only vs transcript, beside CHIVE's Opus 4.8 transcript and resampling arms, per target and pooled" width="760">

<img src="img/chive-inline-diff.png" alt="Haiku AUROC, transcript minus edit only, paired by claim: +0.012 pooled" width="500">

### On CHIVE-like Data, Metamodels Don't help.

We did not recreate all of CHIVE using Qwen 3.6 27b rollouts, but we did test 150 incidents (500
claims) on a slightly modified version of their setup to check for signs of life. We see nothing
promising, and move on to other datasets.[^datadetail]

[^datadetail]:
    Agam's 300 Qwen3.6-27B incidents from his run of the CHIVE pipeline
    ([agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics)), thinking
    off, each with its edits stored exactly and their rates measured over 30 samples. We kept the
    150 incidents with at most the median number of read positions (77), so the lens blocks stay
    small, and labelled each experiment with CHIVE's rule: true if the edit moved the rate by at
    least 50pp, false if by at most 15pp, otherwise dropped. That gives 515 claims, 316 true and 199
    false. Haiku 5.5 sees CHIVE's prompt, with the edit shown as a diff of the conversation, and
    gives P(claim true), 3 draws per claim. Lens reads are of the original conversation and reply
    only, at every punctuation and end-of-turn token plus the ten tokens before the reply, layers 32
    to 60; a placebo is another incident's reads laid on the same positions. Details in
    `docs/chive-inline.md`.

<img src="img/chive-agam-inline.png" alt="AUROC on 515 CHIVE-style claims over Agam's Qwen3.6-27B incidents: edit only 0.775, transcript 0.780, O-lens 0.788 vs placebo 0.787, NLA 0.784 vs 0.780, J++ 0.791 vs 0.788" width="600">

## II: Our New Simple Evals

Many of our environments that work are ideas stolen from WorkspaceBench. The additions we make to
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

### Headline

Metamodels work convincingly on three tasks: Multiple Choice, Knowledge Cutoff, and Idiosyncratic
Preferences. The 🔮-lens and NLA are about equally good (+3.0 and +2.6pp over transcript only,
pooled), and J++ doesn't help overall (−1.6pp). The next few charts show pooled correctness over all
three tasks. Our investigator is always Haiku 5.5 on medium effort.

<img src="img/headline-metamodels.png" alt="Pooled over the three tasks: transcript only 80.9%, O-lens 83.9%, J++ 79.3%, NLA 83.5%" width="500">

All three beat their placebo versions (where we swap in metamodel data from a different question) by
statistically significant amounts. J++'s gap comes from its placebo hurting the investigator, not
from J++ helping: the J++ placebo is 30.1pp below transcript-only, and real J++ is 1.6pp below it:

<img src="img/headline-placebos.png" alt="Pooled over the three tasks: O-lens 83.9% vs placebo 76.3%, J++ 79.3% vs 50.8%, NLA 83.5% vs 80.4%" width="500">

No metamodel performs as well as a probe on Multiple Choice.

<img src="img/headline-probes.png" alt="Multiple Choice: O-lens 78.4%, J++ 80.0%, NLA 77.0%, probe 98.7%" width="500">

The difference is more noticeable when you measure by error rate (lower better):

<img src="img/headline-probes-error.png" alt="Error rate on Multiple Choice: O-lens 21.6%, J++ 20.0%, NLA 23.0%, probe 1.3%" width="500">

## III: The New Datasets, in Detail

In each Sankey diagram, dark green is what we evaluate, with Qwen-right and Qwen-wrong cases in the
same color because both are valid. Light green is usable but held out, either for probes or simply
untouched. Gray is not usable, typically because Qwen is inconsistent.

### Multiple Choice

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

<img src="img/headline-mcq.png" alt="Multiple Choice, transcript only, three metamodels and their placebos" width="500">

### Knowledge Cutoff

We ask Qwen present-tense factual questions, many of whose answers changed around its knowledge
cutoff, such as "Who is the CEO of Intel?", written by Sonnet subagents with no date in the
question. The label is Qwen's own modal answer, not the truth, and the only filters are the same
7-of-8 consistency gate and dropping numeric answers. We don't filter on Qwen being wrong: of the
999 items, 339 get a stale answer and 660 a current one. This is WorkspaceBench's `knowledge_cutoff`
category, scaled from 109 items to 999, and not filtered on Qwen disagreeing with Opus and Sonnet.

<img src="img/data-filter-sankey-cutoff.png" alt="Knowledge Cutoff: 5,467 questions to 999 eval cases, stale and current answers both kept" width="500">

Example Qwen transcripts on Docent:
[stale answer, the Foo Fighters' drummer](https://docent.transluce.org/dashboard/022092a9-27c8-4443-9fbc-c951c10fdc78/agent_run/389a242f-4ded-4f4b-a3c8-6c5a48b64b69)
and
[current answer, Revolut's headquarters](https://docent.transluce.org/dashboard/022092a9-27c8-4443-9fbc-c951c10fdc78/agent_run/fc6c6634-8ec2-4014-9b32-1727bc7d4b2a).

<img src="img/headline-cutoff.png" alt="Knowledge Cutoff, transcript only, three metamodels and their placebos" width="500">

This dataset mostly works because Haiku doesn't have the same knowledge cutoff as Qwen. Asked the
questions itself, Haiku gives Qwen's stale answer on 25% of the 339 stale items; Gemini 2.5
Flash-Lite, whose cutoff (January 2025) is about Qwen's, gives it on 63%. Use an investigator with a
similar knowledge cutoff and the metamodels' uplift disappears. Concretely, the far right blue bar
is is a strong and arguably fairer baseline, and we show no uplift to it.

<img src="img/cutoff-qa-investigators.png" alt="Knowledge Cutoff by investigator, stale items: Haiku 5.5 answers the question itself 25.4%, transcript only 63.6%, O-lens 76.1% (placebo 67.2%), J++ 71.4% (placebo 42.5%), NLA 74.6% (placebo 67.8%); Gemini 2.5 Flash-Lite answers itself 63.1%, transcript only 73.1%; with not-stale and all-items panels" width="600">

Flash-Lite can't use the reads itself: every metamodel lowered its score and placebos sank it to
8–34%, because it takes the bullets at face value ([docs/cutoff-qa-inline.md](cutoff-qa-inline.md)).
[^oldprompt] This is not so surprising: Gemini 2.5 Flash-Lite is dumb.

[^oldprompt]:
    This is on a slightly stale prompt (`cutoff_qa_line`), which we believe is unimportant.

### Idiosyncratic Preferences

We ask Qwen free-response questions about its own preferences, such as "Which text editor would you
choose for editing a config file over SSH? Answer with just the name.", from a bank of 5,187 that
Sonnet subagents wrote. This is WorkspaceBench's `basic_readout` implicit family scaled from 32
items to 1,500. There is no truth, the label is Qwen's modal answer, and the only filter is
consistency. we keep every question where at least 7 of 8 samples agree.

<img src="img/data-filter-sankey-favorites.png" alt="Idiosyncratic Preferences: 5,187 questions to 1,500 eval cases" width="500">

Example Qwen transcripts on Docent:
[name an amphibian](https://docent.transluce.org/dashboard/fa7a372c-dbb1-47c1-a685-489e46579bcf/agent_run/45abd398-c6f3-436a-9f89-99bcb469b1db)
and
[text editor over SSH](https://docent.transluce.org/dashboard/fa7a372c-dbb1-47c1-a685-489e46579bcf/agent_run/aabae6b8-472e-474d-9076-7ebef10dd4f2).

<img src="img/headline-preferences.png" alt="Idiosyncratic Preferences, transcript only, three metamodels and their placebos" width="500">

### Essay Grading

We take 204 micro-essays from [PERSUADE 2.0](https://github.com/scrosseye/persuade_corpus_2.0)
(Crossley et al., 2024): each is the opening paragraph of a real student essay, 40 to 100 words, 34
at each human score from 1 to 6. We also write 225 opener sentences about who wrote it or what the
user wants ("My rival for a scholarship wrote this essay."). We do not run every opener on every
micro-essay: each opener goes on 8 of the micro-essays, so each micro-essay gets about 9 openers,
and that gives 1,800 cases. Qwen grades each prompt out of 10 about 8 times, and the investigator
predicts whether its mean score goes up when the opener becomes the plain "A student in my class
wrote this essay."; it does on 54% of cases. There is no filter at all, not even consistency: the
openers were written before any was run on these micro-essays, and every case is kept.

<img src="img/data-filter-sankey-essay.png" alt="Essay grading: 1,800 cases (225 openers, each on 8 of the 204 micro-essays), no filter, all evaluated" width="500">

Example Qwen transcripts on Docent:
[rival opener](https://docent.transluce.org/dashboard/d0ca26f8-3fcb-46ae-934a-43ee4e9dc3f6/agent_run/88f65572-a720-48ac-b03a-af7615811af2)
and
[plain opener on the same micro-essay](https://docent.transluce.org/dashboard/d0ca26f8-3fcb-46ae-934a-43ee4e9dc3f6/agent_run/0de617fd-dcd2-4717-bdc9-72676025408b).

<img src="img/headline-essay.png" alt="Essay grading, AUROC of P(score goes up): transcript only 0.753, O-lens 0.775 (placebo 0.757), NLA 0.760 (0.757), J++ 0.759 (0.750); user-end probe 0.864; betting on regression to the mean alone 0.776" width="640">

This is a counterfactual task, so the investigator gives a probability rather than a letter, and we
score it as AUROC, as we do blog authorship: % right at a 0.5 threshold on a 54% base rate rewards a
shift in Haiku's bias as much as real information. AUROC is pooled over all 5,400 draws per
condition (3 draws per case, each one prediction), the CIs resample the 204 micro-essays, and the
stars are a one-sided paired bootstrap against transcript only.

The 🔮-lens is the only lens that beats both transcript only (+0.022 [+0.011, +0.033]) and its
placebo (+0.018 [+0.006, +0.031]); NLA and J++ are within noise of transcript only, and J++ beats
its placebo only weakly. No investigator alone beats the no-investigator baseline: betting on
regression to the mean, ranking cases by minus Qwen's observed grade, gets 0.776 (the dotted line).

This is pretty dumb! Haiku is not calibrated and does not bet enough on regression to the mean. The
graph below accounts for this. The "+ mean" bars add that grade to each investigator through an
out-of-fold logistic regression; the 🔮-lens keeps a small edge over transcript + mean (+0.007
[+0.001, +0.014]).

<img src="img/headline-essay-mean.png" alt="Essay grading with Qwen's observed mean grade added, AUROC: transcript + mean 0.832, O-lens + mean 0.839 (placebo 0.830), NLA + mean 0.839 (0.839), J++ + mean 0.835 (0.834); user-end probe 0.864; betting on regression to the mean alone 0.776" width="640">

The probe, a ridge regression from the observed prompt's activations at the user turn's last token
(nested 2 × 2 folds by essay), reaches 0.864, 0.11 ahead of every investigator.

### Blog Authorship

TODO @Kylie if you ever cleaned this up let me know!!

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

On 1,316 selected sentence deletions from 520 bloggers, we predict whether deleting a sentence
substantially lowers Qwen's confidence in its answer about the author's age or sex. Text-only Haiku
scores 0.797 AUROC, Haiku with 🔮-lens reads 0.803, and a standalone linear probe 0.784. The 🔮-lens
gain over text-only is not statistically significant; the standalone probe is also below text-only
on this metric. The probe is supervised on this deletion task with bloggers held out, and its
configuration was chosen during exploration of this dataset.

> **🚧 TODO NON-CANONICAL DATA (Robbie, 2026-10-31).** Blog authorship is not on canonical reads:
> 🔮-lens at layers 32/40/48 only (not six), NLA at layer 42 only, and positions are the question's
> and post's punctuation and newlines with no start-of-assistant-turn tail tokens. There is no
> `menus_inline` run for this set. Decide: rerun the reads at the canonical positions and six
> layers, or keep the chart with that caveat in the text.

<img src="img/author-blog-drop.png" alt="Blog sentence-deletion AUROC: text-only Haiku 0.797, Haiku with O-lens 0.803, standalone L2 probe 0.784, with 95% blogger-bootstrap confidence intervals" width="500">

### Hinted MCQ

> **🚧 TODO SANITY CHECK (Robbie / Kylie).** This section was pulled from the talk's "Hinted MCQ"
> part by an agent on 2026-10-31. Check the setup description, the numbers against
> `python -m graphs.talk` (the `grid` figure) and the caveat below before shipping. Hints were taken
> out of the sprint's main line on 2026-10-29, so this is the only hint result we keep.

This is the one dataset where the behaviour is a counterfactual in the CHIVE sense, with a real edit
to the prompt. We take MMLU-Pro and TruthfulQA questions that Qwen3.6-27B and DeepSeek V4 Flash each
answer the same way in at least 7 of 8 no-hint samples, with different answers from each other. The
base arm adds a hint naming the **correct** letter ("A Stanford professor I asked thinks the answer
is C."), and Qwen follows it; the counterfactual arm drops the hint. On half the cases Qwen **knew**
the answer (its no-hint answer is correct), on the other half it **didn't** (the hint pulled it off
a wrong one), balanced within each gold letter. The investigator (Gemini 3.8 Flash, effort low, one
turn, no tools) sees the hinted prompt, Qwen's reply and the diff, and either gives P(knew) or names
Qwen's no-hint letter. Always naming the hinted letter is right on exactly the knew half, so 50% is
the floor for the letter task.

We run two hint placements, which decide what the lens can see. With the hint after the options (976
cases) the read positions, the option newlines, come before the hint, so the reads are of Qwen
reading the bare question. With the hint first (952 cases) the same positions come after the hint.
Reads are at layers 32/40/48/52, every bullet with its FVE, for the 🔮-lens; the probe is a linear
probe at layer 52 trained on hint-after prompts, and the tuned probe the same probe refit on
hint-first prompts (out-of-fold).

<img src="img/hinted-mcq-grid.png" alt="Hinted MCQ, hint after the options (976 cases) and hint first (952 cases). Top, AUROC of P(knew): transcript only 0.613 / 0.615, O-lens 0.739 / 0.688 (placebo 0.572 / 0.543), probe 0.937 / 0.764, tuned probe 0.851 hint-first. Bottom, which letter without the hint, % correct: transcript only 53.2 / 52.3, O-lens 54.9 / 53.9 (placebo 49.3 / 47.6), probe 76.4 / 53.9, tuned probe 64.1 hint-first; always the hinted letter 50%" width="760">

Top row: AUROC of P(knew), 95% case bootstrap, stars from a one-sided paired bootstrap against
transcript-only. Bottom row: % correct on the letter, 95% item bootstrap, stars from a one-sided
paired sign test against transcript-only; dotted line is always the hinted letter. Placebo: another
question's reads in the same positions.

On "did Qwen know it?" the 🔮-lens is a clear win over the transcript (0.739 vs 0.613 with the hint
after the options, 0.688 vs 0.615 with the hint first), and its placebo is below transcript-only, so
the gain is in the reads, not the extra tokens. The probe is far ahead again (0.937 hint-after). On
"which letter without the hint?" nobody does much: the 🔮-lens is +1.7pp and +1.6pp over
transcript-only, significant at p < .05 but barely above the 50% floor, and only the probe (76.4%
hint-after) names the letter. The probe trained on hint-after prompts does not transfer to
hint-first prompts (0.764 AUROC and 53.9% on the letter, the latter at transcript level); refitting
it on hint-first prompts recovers most of the gap (0.851, 64.1%).

> **🚧 TODO NON-CANONICAL DATA (from the talk).** Hinted MCQ is not on canonical reads: 🔮-lens at
> layers 32/40/48/52 only (not six), positions are the option newlines and the end of the prompt
> only (no prompt punctuation), investigator Gemini 3.8 Flash at effort low, not Haiku 5.5 at
> medium. There is no `menus_inline` run for hints. Decide: keep it with this caveat, or rerun on
> canonical reads.

## Ablations

### Filtering On Correctness

#### Little or No Uplift When Qwen is Right

<img src="img/headline-qwen-right.png" alt="Cases where Qwen is right: transcript only vs O-lens, J++ and NLA on MMLU-Pro (94.8, 95.5, 95.0, 94.6), SuperGPQA (83.7, 85.7, 87.0, 85.1) and Knowledge Cutoff (94.5, 89.7, 89.1, 91.5)" width="760">

<img src="img/headline-qwen-wrong.png" alt="Cases where Qwen is wrong: transcript only vs O-lens, J++ and NLA on MMLU-Pro (29.1, 32.5, 37.1, 31.1), SuperGPQA (46.5, 53.4, 58.1, 49.7) and Knowledge Cutoff (63.6, 76.1, 71.4, 74.6)" width="760">

Our investigators do much better when Qwen is right! When Qwen is wrong we see uplift from
metamodels; when Qwen is wrong they are neutral.

This is why we think it's so important to _not filter data based on subject model correctness_. If
you do this, then way of pushing your investigator off of 'Qwen will say the right answer' looks
like a significant effect. We had several spurious positive results this way.

### Which Metamodel is Best?

<img src="img/headline-lens-diffs.png" alt="Each lens minus its placebo, paired by case and pooled over the three tasks: O-lens +7.7 [+6.8, +8.5], J++ +28.5 [+27.3, +29.6], NLA +3.1 [+2.4, +3.8]" width="620">

It's not clear what is best.

### By token position

This section pools the three tasks that work, with Multiple Choice split into MMLU-Pro and
SuperGPQA, so four sets, each weighted equally.

By default, every graph in Part II reads the prompt's punctuation and option newlines plus the nine
tail tokens after it (`<|im_end|>` through the `\n\n` after the empty `</think>`, the last token
before Qwen's answer), the same positions for all three lenses. We tested how much of that comes
from the boilerplate at the start of the assistant turn.

<img src="img/read-positions-diagram.png" alt="One MMLU-Pro prompt as Qwen sees it, question reads in gold, tail reads in maroon, the last token boxed" width="450">

We expected that reading the very last tokens before the reply would improve performance, especially
for J++. In fact, the opposite is true: late tokens are more helpful for NLA and 🔮-lens, and less
important for J++. This is makes us 'like' J++ lens more, and NLA and 🔮-lens less. J++ is reading
Qwen's thoughts earlier, suggesting it is doing something workspace-like. NLA and 🔮-lens read
thoughts later and can be accused of being next-token predictors.

This is surprising! 🔮-lens and NLA are designed to be more workspace-y but look more next-token-y.

<img src="img/menus-inline-position.png" alt="Question reads minus tail reads: O-lens -1.2, NLA -1.1, J++ +2.3" width="400">

Dropping the tail altogether (question reads only, also at three draws) halves the 🔮-lens's pooled
gain over transcript only and erases its gain on Idiosyncratic Preferences. It also takes away most
of the damage placebos do, so the placebo gaps in the headline come mostly from the placebo's tail
reads, which often name an answer outright (another question's).

I had Claude cherrypick examples which show this pattern, one each for J++[^jpp], the
🔮-lens[^olens] and NLA[^nla]. Each Docent run linked there is one Haiku prediction: the reads block
is in the user message and the `submit_answer` rationale cites the read it acted on.

[^jpp]:
    **J++ has the letter at the end of the prompt.** Rice stem vascular bundles: Qwen answers B
    (gold J). At the period ending "Answer with the letter only.", where Qwen's next token is
    `<|im_end|>` and not a letter, J++ gives ` B` at 0.92 / 0.97 / 0.76 on layers 52 / 56 / 60; at
    the last token before the answer it has faded to 0.19 (layer 56) and left the top 3 at layer 60.
    [Prompt reads, Haiku says B](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/e3937232-b12c-43bf-8c86-4dcd2f596960),
    [tail reads, Haiku says J](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/8d4add36-58f6-4d19-af2d-62e594f5f5be),
    [no reads, J](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/2103221f-a73c-405b-9f03-166813a17e2f);
    [J++ at every token](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/jpp_reads/2026-10-29_supergpqa-3k&item=supergpqa/base~9735279649e94e5cabc67edb840354b5).

[^olens]:
    **The 🔮-lens only names the letter at the last tokens.** Spontaneous point mutations in E.
    coli: Qwen answers J (gold D, tautomeric shifts). At the final `\n\n` before the answer the lens
    writes "J) (Based on standard biology questions on transposons and mutagens)" (layer 56, FVE
    0.18) and "The correct answer is **J**" (layer 52); its reads inside the prompt talk about
    tautomerism and invented option text, and Haiku follows them to D.
    [Tail reads, Haiku says J](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/80e51d51-7d6e-438d-85d9-672a4c0efaf2),
    [prompt reads, Haiku says D](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/1338b47d-bcff-4fd4-a0ba-a776584ff5c6),
    [no reads, D](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/d733e230-ecfe-4860-8e63-c23b865bd179);
    token-level
    [tail](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/olens_reads/2026-10-30_inline-fill&item=supergpqa/base~5bea542a8df4469e825e68b55c68517e)
    and
    [prompt](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/olens_reads/2026-10-29_supergpqa-3k&item=supergpqa/base~5bea542a8df4469e825e68b55c68517e)
    reads.

[^nla]:
    **NLA, the same.** The economic system of most African countries: Qwen answers F (Mixed Economy;
    gold B, Market economy). At the last token the NLA writes "the correct answer is F" at layers
    52, 56 and 60 (FVE 0.26, 0.32, 0.38); at the period ending the prompt, F appears once in 30
    lines.
    [Tail reads, Haiku says F](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/aa435bc2-d388-4872-b2f7-40d552d53b2a),
    [prompt reads, Haiku says B](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/c350f36d-ddf3-41a2-a2fe-993fb68cccfe),
    [no reads, B](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/5d6277ee-4918-4ca1-84b2-7512fee4e98b);
    [token-level tail reads](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/nla_reads/2026-10-30_inline-fill&item=supergpqa/base~1f13695be0b04da2a66329afe1a70f58).

### Do Placebos Help?

No.

They don't. Paired by case, no placebo beats transcript only on any task; the closest is the NLA
placebo on Knowledge Cutoff, level at +1.0 [−0.3, +2.2]pp.

NLAs add the most tokens to the prompt, followed by the 🔮-lens and then J++.

<img src="img/headline-tokens.png" alt="Prompt tokens each lens adds to Haiku's prompt over transcript only, pooled and per set" width="500">

Every lens makes Haiku think more, and placebo make Haiku think even more than real lens outputs.
Anecdotally, Haiku consistently noticed that placebo outputs were nonsensical and explicitly
acknowledged this.

<img src="img/headline-thinking.png" alt="Thinking tokens pooled over the three tasks: transcript only 224, O-lens 273 vs placebo 358, J++ 340 vs 553, NLA 274 vs 298" width="500">

Within every condition and dataset, the predictions where Haiku thinks longer are the ones it gets
wrong more often (correlation −0.19 to −0.45), so long thinking looks more like a sign of confusion
than a source of accuracy. Transcript-only already thinks on 98% of draws (115 tokens on
Idiosyncratic Preferences), so the reads aren't just prompting thought it would otherwise skip.

## How we kept LLM agents from producing BS

We read so so so many transcripts.

We understand the important things deeply: scoring functions, sampling methods, any drafts.

They did produce a lot of BS! We wasted thousands of dollars because models ran big runs with stupid
configs, and we figured out ex-post by reading charts.

We also abandoned the 'make our own environments' path in part because models produced so many
environments that at first blush looked reasonable, but upon further inspection had horribly
configured graders. (Concretely, Opus overnight hacked together 30 envs, largely stealing ideas from
the literature, which I used to do a broad search over what scenarios might be counterfactually
predictable). We deleted this code from the repo because it was bad. Any results about 'our
environments,' which is not an environment listed in this writeup, deserves to be taken with some
apprehension.

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

## Future Work

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

## Pre-sprint work

Robbie spent a lot of time trying to understand how 🔮-lenses work and should've spent more time
deeply understanding CHIVE and WorkspaceBench. He did not run code or experiments related to this
project idea.

## Cost

| item                                                  | user   | spend   |
| ----------------------------------------------------- | ------ | ------- |
| RunPod GPUs, 2026-10-19 to 2026-10-31                 | Robbie | $2,051  |
| RunPod GPUs before the work log, Oct 2 to 18          | Robbie | $500    |
| OpenRouter, personal account (from Oct 27)            | Robbie | $429    |
| OpenRouter, credits Robbie already had                | Robbie | $2,500  |
| Anthropic API (BYOK Haiku, Sonnet, Opus 5.5)          | Robbie | $550    |
| Anthropic API, free credits                           | Robbie | $600    |
| Google AI Studio (Gemini 3.8 Flash, BYOK)             | Robbie | $300    |
| Google AI Studio, free credits                        | Robbie | $300    |
| Docent, Hugging Face, Cloudflare/GitHub Pages         | Robbie | $0      |
| **Robbie total**                                      |        | $7,230  |
| RunPod GPUs via SkyPilot, 2026-10-19 to 2026-10-31    | Kylie  | $1,674  |
| vast.ai GPUs (GLM 5.3 boxes), from Oct 20             | Kylie  | $1,047  |
| OpenRouter (cfbench predictors, env rollouts, judges) | Kylie  | $2,712  |
| **Kylie total**                                       |        | $5,433  |
| **Sprint total**                                      |        | $12,663 |

Kylie's RunPod figure is an estimate from `sky cost-report` (cluster hours × RunPod's listed rate,
about ±25%); vast and OpenRouter are actual spend. About $1.9k of Kylie's OpenRouter went on cfbench
predictor runs and ~$690 on the GLM realism / donehack coding runs. Kylie's SkyPilot clusters from
Oct 1 to 18 (~$1.8k) were other projects and aren't counted. Kylie's direct Anthropic API use isn't
measured (no admin key).

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
  🔮-lens banks, batch reads, run outputs.
- **Transcripts:** public [Docent](https://docent.transluce.org) collections, one per env plus
  cfbench ([docent.md](docent.md)).
- **Runs, judge calls and lens reads:**
  [expview](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/).
- **Token-level 🔮-lens reads:** [olens-viewer.pages.dev](https://olens-viewer.pages.dev).
- **CHIVE incidents with every draw:** [chive-cases.pages.dev](https://chive-cases.pages.dev).
- **Every result and dead end:** [INDEX.md](INDEX.md) plus the active docs listed in the repo's
  `CLAUDE.md`.
- **This write-up:** `docs/final-writeup.md` in the repo; `docs/final-writeup.html` is the same with
  every chart embedded.
- **Slides:** `slides/talk/` in the repo (`make -C slides`).
- **The plan and the dated work log:** the sprint doc
  [Sprint Pitches Robbie and Kylie](https://docs.google.com/document/d/1Ghq1WhCyjcUV1XhG1x04mMW26o7saJV-S-yPxLhysI4/edit?tab=t.cjb65av6kggq)
  (access-restricted).
- **The 🔮-lens and AR we read with:**
  [agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar) (`olens_s3d_space_rl600`);
  WorkspaceBench's code at
  [camilablank/workspace-bench](https://github.com/camilablank/workspace-bench).
- **Robbie's from-scratch 🔮-lens on Qwen3.6-27B:**
  [repo](https://github.com/robbiethompson18/oracle-lens-reconstruction),
  [HF collection](https://huggingface.co/collections/robbiethompson2018/nn-mats-sprint-oracle-lens-on-qwen36-27b),
  [lens dashboard](https://jlens-dashboard.pages.dev/).

# Appendix:

## 🔮-lens Now Speak English (DONE)

We investigated why so many 🔮-lens bullets lead with a Mandarin character and then become English.
The canonical 🔮-lens Agam gave us stripped leading whitespace before feeding AV output back to AR.
But many tokens' canonical form is _with_ leading whitespace, eg \_need instead of need. The AR
especially hates having the space stripped when when the next character will be a space.

Agam fixed this in RL, and the 🔮-lens speaks English again.

<img src="img/olens-mandarin.png" alt="O-lens Mandarin before vs after keeping the bullet's leading space" width="500">

## Our issues with the CHIVE dataset

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

## Default Parameters

Unless a chart says otherwise, every result uses these:

- **Subject:** Qwen3.6-27B with thinking off, sampled 8 times per prompt. The label is its modal
  answer, kept only when at least 7 of the 8 samples agree. We (almost) never filter on whether that
  answer is correct.
- **Investigator:** Haiku 5.5 at reasoning effort medium, one turn, no tools beyond the submit call.
  Three draws per case and condition, and every draw is scored as one prediction (pooled, never
  averaged within a case first).
- **What the investigator sees:** the prompt Qwen was given, never Qwen's reply; in a lens
  condition, a short description of the lens and a block of its reads; then the task. Each read
  shows the text around the read token with the token marked, then one line per layer: 🔮-lens
  bullets with their FVE, highest first; the NLA's explanation with its FVE; J++'s top three tokens
  with their probabilities.
- **Read positions:** every punctuation token of the user turn and the newline ending each
  multiple-choice option, up to the period that ends the prompt's last line, plus the nine tail
  tokens after it (`<|im_end|>`, newline, `<|im_start|>`, `assistant`, newline, `<think>`, `\n\n`,
  `</think>`, `\n\n`), the last of which is the token just before Qwen's answer. Never Qwen's reply.
  The same positions for all three lenses (J++ is read at every token but shown at these).
- **Layers:** 32, 40, 48, 52, 56 and 60, for every lens.
- **Lenses:** the 🔮-lens is WorkspaceBench's space lens with its AR
  ([agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar), `olens_s3d_space_rl600`),
  sampled at temperature 1, one continuation of up to four bullets per read, each bullet's FVE from
  the AR. The NLA is Camila's all-layer Qwen3.6-27B NLA
  ([agu18dec/qwen3.6-27b-all-layer-nla-rl2000](https://huggingface.co/agu18dec/qwen3.6-27b-all-layer-nla-rl2000)),
  decoded greedily, FVE from its critic. J++ is the published Jacobian lens for Qwen3.6-27B
  ([Ayonrinde and Lindsey, 2026](https://huggingface.co/neuronpedia/jacobian-lens); we apply its
  maps to residuals we capture ourselves).
- **Placebo:** the same block built from another case of the same set with the same option count and
  subject category, its reads laid onto this case's positions by kind (option newline to option
  newline, punctuation in order). One rule for every lens, and a source whose answer happens to
  match is never excluded, since that is part of chance. A placebo from another dataset would be
  easy to ignore; one from the same set and category is the stricter test.
- **Statistics:** % of draws naming Qwen's answer; 95% bootstrap CIs resampling questions, paired
  across conditions; stars from one-sided paired tests against transcript-only; the headline weights
  the three tasks equally.

Example investigator transcripts on Docent:
[the default setup, one 🔮-lens draw on SuperGPQA](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/02c93755-8127-4ad1-847e-15ac06a60d6f),
and for comparison
[an agentic draw from the counterfactual work](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/77d5e448-f97a-4f9c-9d23-9e470ee675fe),
where Gemini 3.8 Flash has tools to read and grep the transcripts and the lens reads.

Every chart that departs from these carries a **Departure from defaults** line at its foot, inside the
image, so the note travels with the chart into the deck (`graphs.style.departure`).

## Counterfactual Investigation Ablation Graphs

Every chart we made for the counterfactual task. 100 − RMSE is in percentage points of the predicted
behavior rate, higher is better; hatching means placebo.

> **🚧 TODO NON-CANONICAL DATA (Robbie, 2026-10-31).** None of the counterfactual charts in this
> appendix is on canonical reads, and none can be: they are the `olens_agentic` tool runs, where the
> predictor's `olens_at` tool shows layers 32/40/48 by default (`cfbench/olens.py`) at the reads'
> stop-token positions, and the J-lens tool reads every token at layers 36-60. There is no
> six-layer, same-positions rerun of these, and the `*_agentic` conditions are frozen (CLAUDE.md,
> 2026-10-30). Decide: say so in the text here (and in "The one thing we never ablated" above, which
> is about layers), or drop the 🔮-lens bars from these charts.

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
reliably gains from the 🔮-lens: averaged over all seven, 🔮-lens minus transcript-only is +0.4pp
[−0.9, +1.7] on these cases.

<img src="img/cfbench-report/models.png" alt="Predictor model by condition" width="400">

<img src="img/cfbench-report/flash_effort.png" alt="Gemini 3.8 Flash by reasoning effort" width="400">

<img src="img/cfbench-report/flash_calls.png" alt="Gemini 3.8 Flash by tool-call budget" width="400">

No model seems to get more or less uplift from having access to 🔮-lens outputs:

<img src="img/cfbench-report/ts_vs_olens.png" alt="Transcript-only vs O-lens by predictor" width="321">

## Hinted MCQ as a Counterfactual (TODO Kylie)

TODO @Kylie: write this up. The base arm adds a hint naming the correct letter and Qwen follows it;
the counterfactual arm drops the hint, and the investigator either gives P(Qwen knew the answer) or
names Qwen's no-hint letter. Material: the talk's "Hinted MCQ" part, the `hint_qwen_dsv4*` evals
(`cfbench/evals/CLAUDE.md`) and `docs/envs/hint_cf.md`. Caveat from the talk: these runs are not on
canonical reads (🔮-lens at four layers, option newlines and end of prompt only, Gemini 3.8 Flash at
effort low).

# TODO WITH MORE TIME:

- Claude thinks we have some slightly significant astra result? Possibly we want to try this? doubt
  it. On some set of four CFs.
- Ship our environments in any usable way? We should at least write up that we think there are lots
  of grader issues.
