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
compared to simply giving the investigator both system prompts. @c plz dig on this, I know we did
env only on our envs, did we ever do this on CHIVE? if not we should.

@robbie: we never ran it on CHIVE, and the CHIVE paper has no such arm either: every CHIVE predictor
sees the transcript plus the claim. On CHIVE the equivalent arm is the original prompt plus the edit
description, with no Qwen response. I didn't run it, because it needs a new harness condition, not a
docs change. On our own envs the evidence is mixed, so the sentence overclaims as written:

- cfbench, GLM 5.3, 280 cases: no transcript 30pp RMSE vs transcript-only 31pp
  ([cfbench-scoring.md](cfbench-scoring.md)). The binary regrade agrees: no transcript wins by 5
  points of balanced accuracy ([cfbench-binary-regrade.md](cfbench-binary-regrade.md)).
- cfbench, Qwen without CoT, 144 cases: 28pp vs 24pp. The transcript helps by 4pp, CI [−6.6, −1.8].
- Alignment faking: the prompt alone is the best condition, 7.4pp vs transcript 13.1pp
  ([alignment-faking-cfbench.md](alignment-faking-cfbench.md)).
- Honest grader, 242 cases: 75 vs 78 (100 − RMSE). The transcript helps by 3.5pp, CI [+1.6, +5.6]
  ([honest-grader-envs.md](honest-grader-envs.md)).
- Over-eagerness: 18.3pp vs 18.6pp, a tie ([overeager-main-cfbench.md](overeager-main-cfbench.md)).
- Model chart, 4 cases: the transcript helps every strong predictor, e.g. GPT-6 Astra 39 → 60 (100 −
  RMSE), and does nothing for Haiku ([cfbench-model-chart.md](cfbench-model-chart.md)).

A defensible version: "on several of our environments, adding the transcript helps a counterfactual
investigator little or not at all compared with giving it both prompts". Drop "and on CHIVE" unless
we run it.

We successively simplified the counterfactual prediction tasks we asked metamodels to do until we
found ones that work. They are: predicting Qwen's grade for an essay; predicting Qwen's answer to a
multiple choice question; predicting Qwen's idiosyncratic preferences; predicting Qwen's answer to a
factual query which is past its knowledge cutoff.

## Claims, with credences

## How the question changed

## Part 0: counterfactual investigation is null

Bar chart on chives showing nulls Bar chart: score on all of our envs, showing nulls Bar chart:
abaltions by model showing null @c what other ablatoins did I d ohere?

@robbie: every ablation on the counterfactual task. In none does the O-lens reliably beat
transcript-only or its placebo:

- **Predictor model:** seven predictors from Haiku 5.5 to GPT-6 Astra, plus thinking-off and weak
  predictors ([cfbench-model-chart.md](cfbench-model-chart.md),
  [cfbench-dumb-predictors.md](cfbench-dumb-predictors.md)). The lens beats placebo only for Gemini
  4 Argon and Flash, both through hidden_goal.
- **Predictor effort and tool-call budget:** effort low, medium and high; 5 to 40 tool calls (same
  doc).
- **Target thinking on vs off:** pre-registered; the O-lens is below transcript-only in both
  ([nocot-cfbench.md](nocot-cfbench.md), [cot-cfbench.md](cot-cfbench.md)).
- **How reads reach the predictor:** agentic tool vs reads pasted into the prompt (closed PR #116,
  [olens-cfbench-spike.md](olens-cfbench-spike.md)), and reads on the counterfactual prompt (dropped
  as outside the zero-counterfactual setting).
- **Lens settings:** old vs space lens, greedy vs temperature 1, stop-token vs every-token reads,
  reads from before the hint only ([olens-space-migration.md](olens-space-migration.md),
  [envs/hint_cf.md](envs/hint_cf.md)). We never ablated layers on this task.
- **Other tools, each with its placebo:** J-lens, NLA, Thought Anchors
  ([jlens-cfbench.md](jlens-cfbench.md), [nla-cfbench-hint-mcq.md](nla-cfbench-hint-mcq.md),
  [thought-anchors.md](thought-anchors.md)).
- **Other models' lenses:** Camila's DeepSeek V4 Flash lens on alignment faking, Celeste's GLM 5.3
  lens on over-eagerness.
- **Harness:** versions 1 to 6, and one shown run instead of 30
  ([cfbench-harness-smoke.md](cfbench-harness-smoke.md)).
- **Scoring:** log skill vs RMSE vs binary at 5 to 30pp thresholds
  ([cfbench-scoring.md](cfbench-scoring.md),
  [cfbench-binary-regrade.md](cfbench-binary-regrade.md)).
- **CHIVE itself:** Flash vs Sonnet 5.5, free-form vs yes/no
  ([smoke25-results.md](smoke25-results.md)), and the off-policy split of Agam's run
  ([chive-critique.md](chive-critique.md)).

Existing charts: `python -m graphs.cfbench_report` (pooled, by target model) and
`python -m graphs.cfbench_modelchart` (by predictor, effort, call budget). Nothing in `graphs/`
charts CHIVE yet.

### The WorkspaceBench review

## Part 1: predicting Qwen's answer from the prompt

## Part 2: essay grading counterfactual

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

Approximately none. I spent a lot of time trying to understand how 🔮-lenses work!

## Cost

@c can you fill this in for me, I will download something from Chase later.

@robbie: filled in what the provider APIs report on 2026-10-31. The TODO rows need the Chase
statement.

| item                                          | spend  | source                                                 |
| --------------------------------------------- | ------ | ------------------------------------------------------ |
| RunPod GPUs, 2026-10-19 to 2026-10-31         | $2,051 | `runpodctl billing pods`; $1,997 of it H200s           |
| RunPod GPUs before the work log, Oct 2 to 18  | $1,192 | same; not counted as sprint spend                      |
| OpenRouter, personal account (from Oct 27)    | $331   | `GET /api/v1/credits`, `total_usage`                   |
| OpenRouter, old usebits account (to Oct 27)   | TODO   | key retired; it was burning ~$2,100/month on Oct 26    |
| Anthropic API (BYOK Haiku, Sonnet, Opus 5.5)  | TODO   | Chase                                                  |
| Google AI Studio (Gemini 3.8 Flash, BYOK)     | TODO   | Chase; part of it came out of a $300 free-trial credit |
| Docent, Hugging Face, Cloudflare/GitHub Pages | $0     | free tiers                                             |

## Where everything is

@c what is this section supposed to bve?

@robbie: it's the map for a reader who wants to check our work: where the code, data, transcripts
and viewers live. Draft:

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

@c paste in from progress report doc.

@robbie done, pasted verbatim from the Oct 25 progress report's appendix ("Mandarin in O-lens
Bullets: 50% → 0.5%"). The repo's version with numbers is [olens-chinese.md](olens-chinese.md).

We investigated why so many o-lens bullets lead with a Mandarin character and then become English.
The canonical o-lens Agam gave us stripped leading whitespace before feeding AV output back to AR.
But many tokens' canonical form is _with_ leading whitespace, eg \_need instead of need. The AR
especially hates having the space stripped when when the next character will be a space.

Agam fixed this in RL, and the o-lens speaks English again.

## Our issues with the CHIVE dataset

@c plz paste in verbatim all of my messages from my Slack with Adam yesterday verbatim and I will
trim.

@robbie done: your three messages in the 2026-10-30 group DM with Adam and Kylie, verbatim, with
Slack markup turned into Markdown. This repo is public and `CLAUDE.md` bans private Slack content,
so where your third message quotes Adam I replaced his words with a bracketed paraphrase. The
numbers are reproducible in [chive-critique.md](chive-critique.md).

**10:38 PDT**

Hey Adam, should've sent this much sooner, but Kylie and I tested if the o-lens works on CHIVE data
for our sprint project, and found that not only does this not work, but it's not even close. We had
to dumb the task down to "predict what Qwen will say about this fact past its knowledge cutoff" to
start getting any results. Nothing more agentic or less contrived worked despite much tinkering.

1. Is this surprising to you?
2. What would convince you that our null results show 'metamodels are not up for this task' and not
   'harness issue'?
3. We spent a long time dissecting CHIVE to figure out why nothing worked and have opinions about
   the dataset, happy to share more thoughts if that is interesting. (eg due to cutoff quirks some
   scenarios have almost identical counterfactuals marked as a `supported cause` and a
   `refuted hypothesis`)

**11:48 PDT**

Ok well none of our complaints truly affect a majority of datapoints but here they are nonetheless:

- 40% of the data is off-policy in that it's one {Qwen,Gemma,etc} continuation at the end of a
  WildChat ChatGPT transcript. This makes some 'weird behavior' not that surprising, eg
  [000054](https://chive-cases.pages.dev/?case=prompt_000054), where the model correctly points out
  its own prior mistake (some of the time)
- We hardcode hypotheses to validate or not based on a raw move in % space (rather than say a
  likelihood ratio) so 20%->71% is a validated finding but 30% -> 0% is null and 10% --> 0% is
  'validated doesn't matter, moves < 15%.' It's not obvious what a better solution is (Fisher's
  exact test?), but this setup causes issues:
- On 8.6k experiments (12%, affecting 32% of investigations) we have a base behavior rate between
  10% and 50%, and a counterfactual which moves that rate to 0%, but because of our cutoffs this is
  marked inconclusive and not shown to the predictor. In 18% of investigations we have no positive
  experiments (and thus only negative claims to show the predictor) but we do have a 10%+ → 0 claim
  that we throw away.
  - the mirror case where CF bumps rate to 100% is much less common

**17:29 PDT**

> [Adam asked what we thought of CHIVE's data quality overall, and what we would do to improve or
> filter it.]

I think the weakest is deciding what counts as 'weird,' which IIUC is just Opus asked to rate
WildChat. (This skews very heavily toward 'mistakes' which I think you cover in the paper).
Weirdchat does something potentially more interesting with their search approach, but their
datapoints fall into a couple similar scenarios and have the sad property of being
'unrealistic/adversarial.' This is a hard question, I suspect there is more to be done here and that
someone with access to lots of real user data will be able to do the best research.

> [Adam: off-policy data is acceptable, if not ideal.]

Hot take! I agree simulating the user turn is also bad. Multi-turn agentic settings with no user
turn but the model is required to interact with a sandbox are on my (interesting, possible to create
at scale) Pareto frontier, which we did in the first week of our project before changing course
because metamodel uplift was 0.

> [Adam guessed the qualitative takeaways would not change if we filtered out the problem cases.]

agree

> [Adam: the likelihood cutoff should only cost usable data points, not add problematic ones.]

Not sure this is true, the 14% --> 0% 'inert' case is potentially problematic. Having thought about
it only briefly, it seems very hard to feel statistically confident that your rate of weird behavior
_hasn't_ changed. Claude wants to use Cohens's _h,_ which I agree trades simplicity for 'Platonic
Statistical Goodness.'

> [Adam: the more complex metrics he tried did not much raise the yield of experiments that are
> unlikely to be sampling noise.]

If you use Fisher's exact test you are selecting based on this exactly! If you p=0.05 (w/ BH
correction) you get **1.7x experiments marked positive**, +23% investigations with a positive result
(1.5x and 22% if you use p=0.01).

## WorkspaceBench Shows Less Than We Thought

@c plz just link that doc

@robbie done. Our review is the "WorkspaceBench Review" tab of the
[sprint doc](https://docs.google.com/document/d/1Ghq1WhCyjcUV1XhG1x04mMW26o7saJV-S-yPxLhysI4/edit)
(access-restricted). Its public counterpart in the repo is
[wsb-flash-prompt-only.md](wsb-flash-prompt-only.md).
