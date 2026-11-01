# Author evidence: does the O-lens show which sentence Qwen reads the author from?

**Status (2026-11-01):** NULL, and negative for the O-lens. On a natural-rate sample of real blog
posts (1,618 cases, 655 bloggers, flat 0.15 rule, labels averaged over both option orders, every
condition scored in one Haiku run) an investigator does worse with agu18dec's O-lens reads than with
the prompt alone: AUROC 0.811 vs 0.835, O-lens − transcript −0.024 [−0.041, −0.007], and no better
than the O-lens placebo (+0.004). A sentence-end linear probe reads 0.808, and probe + transcript
0.885 (+0.050, p < 0.001). ceselder's NLA is pending in the clean run (null, −0.018, in the earlier
pooled scoring). Numbers before 2026-11-01 pooled several Haiku runs with positives and negatives
mostly in different runs (an independent audit caught it), so they are superseded. An earlier pilot
on SynthPAI (synthetic comments) gave the same null and is retired.

Env-viewer page with the charts, paired differences and example cases:
`#eval=author_blog_drop_inline` on the
[env viewer](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/)
(after this branch merges).

## The question

Qwen3.6-27B (thinking off) reads a real blog post and answers a first-person multiple-choice
question about its author, asked before the post: "What is my sex / How old am I most likely to be,
based on the blog post I wrote below? [post] What do you think? A. … B. … Answer with the letter
only." Some sentences carry Qwen's evidence: deleting one makes it much less sure of its answer. Can
an investigator tell which, from the prompt alone, or better with O-lens reads of Qwen's activations
on the prompt?

- **Data:** the Blog Authorship Corpus (Schler et al. 2006,
  [`tasksource/blog_authorship_corpus`](https://huggingface.co/datasets/tasksource/blog_authorship_corpus)),
  one post of 200-450 words per blogger, balanced over gender × the corpus's three age groups (13-17
  / 23-27 / 33-48, asked as Under 20 / 20-29 / 30 or older). Bank on HF `banks/author_mcq_blog/`.
- **Qwen's answer:** letter log-probs in both option orders, renormalised over the letters and
  averaged. Accuracy vs the blogger's label: sex 0.83 (chance 0.50), age 0.75 (0.33).
- **Pairs kept:** confident (chance-rescaled top probability c ≥ 0.5, both orders agree) and stable
  (the median single-sentence deletion moves the pmf by TV < 0.05). Versions `qwen_wrong` (answer ≠
  label; every such pair) and `qwen_right` (300 per attribute per 2,004 bloggers).
- **Truth:** delete each sentence in turn; a case is positive when Qwen's probability on its answer
  drops by more than max(0.15, 5 × the pair's median drop). Each positive comes with one random
  non-significant sentence of the same post (base rate 0.50).
- **Investigator:** Claude Haiku 5.5, reasoning effort low, one draw, single turn
  (`cfbench/evals/author_blog_drop_inline`, the hint evals' layout): Qwen's exact prompt and letter,
  Qwen's probability, the sentence to delete; claim = "the deletion is significant".
  - `transcript_inline`: that alone.
  - `olens_inline`: plus agu18dec's Qwen3.6-27B O-lens, sampled at temperature 1 (top-p 0.95, top-k
    64, one continuation of up to 4 bullets per site), at every punctuation, symbol and newline
    token of the question and post (`bank.is_read_token`), layers 32/40/48, every bullet with its
    FVE.
  - `olens_placebo_inline`: another post's reads (same attribute, at least as many positions) in
    their place.

## Result

Batches 1 + 2: 1,316 cases from 520 bloggers (qwen_wrong 722, qwen_right 594); run
`runs/cfbench/2026-10-30_author-blog-drop-haiku-b2` (batch 1's draws reused; their prompts are
byte-identical). CIs and one-sided p (H1: O-lens better) from a 10k bootstrap over bloggers
(`scripts/author_probe/pvalues_drop.py`).

| version (cases)  | AUROC: transcript | O-lens | placebo |
| ---------------- | ----------------- | ------ | ------- |
| qwen_wrong (722) | 0.75              | 0.77   | 0.75    |
| qwen_right (594) | 0.85              | 0.84   | 0.83    |

| pooled              | AUROC                           | raw Brier gain                     | recalibrated Brier gain |
| ------------------- | ------------------------------- | ---------------------------------- | ----------------------- |
| O-lens − transcript | +0.006 [−0.015, +0.026], p 0.30 | +0.004 [−0.002, +0.010], p 0.09    | +0.002, p 0.36          |
| O-lens − placebo    | +0.016 [−0.005, +0.036], p 0.07 | +0.016 [+0.010, +0.022], p < 0.001 | +0.006, p 0.11          |

- No gain over the prompt alone, on ranking (AUROC) or on Brier once each condition is
  Platt-recalibrated (5-fold CV grouped by blogger).
- The O-lens beats the placebo on raw Brier only. Haiku's probabilities sit far below the 0.50 base
  rate (mean p 0.17-0.18), a little higher with reads in the prompt, which alone lowers raw Brier.
  The first batch alone gave O-lens − transcript raw Brier +0.010 (p = 0.008); that was this
  calibration effect, and it halved with twice the bloggers.
- Per attribute: sex is easy for every condition (AUROC 0.83-0.89), age harder (0.71-0.79); the
  O-lens leads by ~0.02 on qwen_wrong in both, inside the noise.
- Example: blog3344990:sex, a 45-year-old woman's post opening "Introducing my wife Hi honey, I am
  home......": Qwen answers Male at 0.97; deleting that sentence drops it to 0.16.

Cost: about $110 of RunPod H200s (SynthPAI and blog, deletions and reads) and $40 of API (DeepSeek
V4 Flash answers, Haiku investigator and judge) over 2026-10-29/31.

## Scale-up and natural sample (2026-10-31)

Kylie asked for about 1,000 posts, then for a sample with nothing suspicious in it ("filter on there
is evidence and Qwen is confident"). Truth changed to a flat rule: positive when deleting the
sentence lowers Qwen's probability on its answer by more than 0.15 (the old max(0.15, 5 × median)
rule flips 1 of the 1,316 old cases and adds 21 positives).

**How the earlier case sets were sampled** (the audit behind the natural sample):

- Bloggers were stratified: equal numbers per sex × age band (334 per cell in batches 1-2, 668 in
  batch 3 until the 30+ cells ran out at 474 F / 337 M). The eligible corpus is about 43% under 20,
  42% 20s and 15% 30+, so 30+ was oversampled about 2× and no unused 30+ blogger is left.
- Deletions went to every confident Qwen-wrong pair but only a random 300 (batches 1-2) or 521
  (batch 3) confident Qwen-right pairs per attribute, so Qwen-right is under-represented (about
  12-26% sampling rate).
- The stability filter (median deletion TV < 0.05) drops about 10% of deleted pairs, and about 25%
  of the pairs that have a > 0.15 drop. Kylie kept it ("stability filter is good").
- Nothing else selects cases: no length, position or model-output filter beyond confidence, order
  agreement, stability and at least one positive.

**Natural sample** (`scripts/author_probe/natural_plan.py`, `blog_u.py`; run
`runs/author_mcq_blog/2026-10-31_uniform-sample`, cell table in its `plan.json`). Cells are sex ×
age band × question × Qwen right/wrong. A cell's natural count of confident pairs is its
eligible-corpus bloggers times its confident-pair rate among the answered bloggers (within a cell
every answered post is a uniform draw, since the batches stratified only by sex × age band). Each
cell's target is proportional to that; a random subsample of the cell's already-deleted pairs fills
it, and the shortfall (461 Qwen-right pairs, all under 30) is drawn uniformly from the cell's
confident, answered, never-deleted pairs and deleted in both option orders. Then stable pairs with a
positive, 1:1 negatives. Labels average both option orders: 552 batch-3 pairs had been deleted in
one order only, and their second order was run on 2026-11-01, which moved 35 cases in and 43 out (no
kept label flipped). Result: 1,618 cases (809 positive) from 671 post-question pairs and 655
bloggers, 79% Qwen-right, no weights. Haiku 5.5 effort low, 1 draw, every case of each condition in
one run, harness 19 (the prompt states the flat rule):
`runs/cfbench/2026-11-01_author-blog-drop-haiku-natural2`. Scores:
`runs/author_mcq_blog/2026-11-01_natural-score` (the probe covers the 1,588 cases with a probe
capture; the other 30, from 11 pairs that entered with the second order, await it).

| natural sample, pooled (1,618 cases) | AUROC [95% CI]       |
| ------------------------------------ | -------------------- |
| transcript                           | 0.835 [0.815, 0.853] |
| O-lens                               | 0.811 [0.791, 0.831] |
| O-lens placebo                       | 0.806 [0.786, 0.826] |
| linear probe (L48 sentence end)      | 0.808 [0.786, 0.827] |
| probe + transcript                   | 0.885 [0.869, 0.900] |

| A − B                           | Δ AUROC [95% CI]        | one-sided p |
| ------------------------------- | ----------------------- | ----------- |
| O-lens − transcript             | −0.024 [−0.041, −0.007] | 0.998       |
| O-lens − O-lens placebo         | +0.004 [−0.015, +0.023] | 0.33        |
| probe + transcript − transcript | +0.050 [+0.040, +0.061] | < 0.001     |

NLA and NLA placebo: pending in the clean run.

**Sampling audit** (independent agent, 2026-10-31): the case set reproduces exactly from the
committed code; cell shares match the corpus (TV 0.008 from exact); earlier deleted pairs are
uniform within cells; the 461 new pairs behave like the old (stable-with-positive 0.215 vs
0.219-0.229); negatives are seeded and uniform. It found the run pooling and the single-order labels
fixed above, and that the draw is blogger-uniform (one post per blogger), not post-uniform.

**Batches 1-3** (secondary; `runs/author_mcq_blog/2026-10-31_scaleup-score`): the stratified,
Qwen-right-capped set grown with batch 3 (`scripts/author_probe/blog_c.py`), 2,456 cases from 960
bloggers, scored unweighted and with each Qwen-right case weighted by 1 / its batch and attribute's
sampling rate (ESS 1,744). Its predictions pool runs (harness 13/14/18, positives and negatives
mostly scored in different runs), so these are indicative only: O-lens − transcript +0.008 [−0.006,
+0.024] unweighted, +0.002 weighted; NLA − transcript +0.001 / −0.004; probe + transcript −
transcript +0.061 / +0.053 (both p < 0.001).

Probe and probe + transcript: the standard L2 probe (λ = 1) on the layer-48 residual at the deleted
sentence's last token, and a logistic on [logit of Haiku's transcript p, probe score], both fully
nested out of fold (5 folds grouped by blogger), refit within each case set
(`scripts/author_probe/scaleup.py`). Compute: about 1 h of a 6×H200 pod (deletions, O-lens and NLA
reads, captures) plus about $8 of SkyPilot H200s; Haiku about $25.

## NLA in place of the O-lens (2026-10-31)

[ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl) (layer 42 only; AV
= `av_base` + RL adapter `iter_000800`, 78% held-out FVE) at the same 26,706 tokens of the same
prompts, greedy (the O-lens sampled at temperature 1), each explanation scored by its critic (median
FVE 0.43 on this run's own ruler). Its AV is a merged full model, so it can't run on the one-engine
reader: `scripts/author_probe/nla_l42.py` does capture, decode and critic in plain transformers on
one GPU (`scripts/sky/nla_l42.yaml`, ~2.8 reads/s on an H200). Conditions `nla_inline` and
`nla_placebo_inline` (the O-lens placebo's post), Haiku 5.5 at effort low, 1 draw; run
`runs/cfbench/2026-10-31_author-blog-drop-haiku-nla`, reads
`runs/nla_reads/2026-10-31_author-blog-nla-l42` (also on the per-question expview pages beside the
O-lens).

| pooled, A − B    | AUROC                           | raw Brier gain | recalibrated Brier gain |
| ---------------- | ------------------------------- | -------------- | ----------------------- |
| NLA − transcript | −0.003 [−0.023, +0.017], p 0.60 | −0.008         | −0.003, p 0.70          |
| NLA − placebo    | +0.009, p 0.21                  | +0.006, p 0.02 | +0.002, p 0.36          |
| NLA − O-lens     | −0.008, p 0.78                  | −0.012         | −0.004                  |

Within-post paired accuracy: transcript 0.758, O-lens 0.771, NLA 0.772, NLA placebo 0.766 (CIs
±0.035).

## Earlier pilot on SynthPAI (synthetic), retired

[SynthPAI](https://huggingface.co/datasets/RobinSta/SynthPAI): synthetic Reddit comments by ~300
GPT-4 personas with attribute labels and per-comment human guesses. Passages of 120-300 words, five
attributes (sex, age band, education, relationship, income). Retired (Kylie, 2026-10-31); its evals
`author_mcq_inline` and `author_mcq_drop_inline` are in `RETIRED_EVALS`, runs left on HF under
`runs/author_mcq_probe/`.

- **Probes** (label-trained logistic, author-grouped 5-fold, all 65 layers): at the passage-only
  `<|im_end|>` AUROC 0.69 / 0.79 / 0.81 / 0.68 / 0.71, near a TF-IDF baseline except for sex and
  income, peaking early (L5-L28). At the question's "?" (question read, no options) 0.77 / 0.82 /
  0.88 / 0.77 / 0.77, matching Qwen's own answer pmf (0.81 / 0.80 / 0.88 / 0.74 / 0.79); an age
  ridge there reaches Spearman 0.82 vs Qwen's 0.77. The final prompt token carries Qwen's letter (a
  gold-letter probe decodes as well as the concept), so "?" is the site to probe. Asking the
  question first and probing after the passage ("What do you think?") is weaker (0.72-0.82).
- **Deletions** (leave one comment out): most confident pairs are stable; evidence comments are the
  ones the SynthPAI reviewer also guessed from (24-65% vs 7-10% of all comments). The "?" probe's
  shifts track Qwen's under deletion (r 0.13-0.36 vs a 0.07-0.09 shuffle null); the passage-only
  probe's do not.
- **Drop eval** (904 cases with two passages per author, same design as above): O-lens − transcript
  AUROC −0.000 (p 0.51), recalibrated Brier −0.001 (p 0.57); O-lens − placebo raw Brier +0.010 (p
  0.005), AUROC +0.006 (p 0.28). The same null.
- **Answer eval** (`author_mcq_inline`, 196 Qwen-wrong cases, predict Qwen's letter): transcript
  0.82, O-lens 0.79.

## Exploratory, post hoc: lexicons in the reads and sentence-end probes (2026-10-31)

Kylie's follow-ups after the null. Post hoc and multiply tested; leads, not results. Code
`scripts/author_probe/pronoun_probe.py`, `period_probe.py` and `period_check.py`; runs
`runs/author_mcq_blog/2026-10-31_pronoun-probe` and `…_period-probe` (activations included). Each
case's sentence is taken from its item id (8 cases delete a sentence whose text appears twice in its
post).

- **Answer words in the O-lens reads** along each post (FVE-weighted pronouns matching Qwen's answer
  for sex, a short word list per bucket for age; level, step, before/after shifts, slope; 16
  features): AUROC 0.55-0.60 alone (sex 0.68 on qwen_wrong for the 3-sentence signed shift, the best
  of 16 on 160 cases), placebo reads and the sentence text ~0.5. Added to Haiku's transcript p
  (5-fold CV): sex −0.004 (all lens features; level alone +0.001), age +0.021, but the same counts
  on the sentence text add +0.017 for age, so the age gain is lexical.
- **Sentence-end activations** (Qwen3.6-27B residual at the token holding the sentence's last
  character, question-first prompt, layer 48). An answer probe trained on 3,000 batch-3 pairs (no
  shared bloggers) and read along each post: AUROC 0.53-0.64 over 8 variants (the best, the step in
  P(answer), adds +0.017 over Haiku). The standard probe (`probes/fit.py`: training-only
  standardisation, unpenalised intercept, no PCA, L2 with λ = 1) fit to the task, 5-fold CV grouped
  by blogger, the combiner fully nested (`period_check.py`):

| score                                 | AUROC                | paired accuracy | + over Haiku, nested    |
| ------------------------------------- | -------------------- | --------------- | ----------------------- |
| Haiku transcript p                    | 0.797                | 0.758           |                         |
| L2 probe on h at the sentence end     | 0.784 [0.760, 0.808] | 0.782           | +0.058 [+0.043, +0.073] |
| L2 probe on h(end k) − h(end k−1)     | 0.737                | 0.712           | +0.039                  |
| PCA 128 + logistic on h (comparison)  | 0.780                | 0.766           |                         |
| TF-IDF of the sentence                | 0.645                | 0.652           | +0.011                  |
| sentence length + position in post    | 0.704                | 0.718           | +0.035                  |
| L2 probe over Haiku + length/position |                      |                 | +0.032 [+0.023, +0.042] |

Over 5 fold splits the nested increments are +0.057 ± 0.003 (over Haiku) and +0.030 ± 0.003 (over
Haiku + length/position); the mean rank of Haiku's p and the probe's out-of-fold score, with nothing
fit, adds +0.055. Per attribute (pooled fit): sex +0.045, age +0.067 over Haiku, the age gain mostly
on qwen_wrong (+0.087); age on qwen_right +0.033 (p 0.05).

Caveats: post hoc; nested given the layer, but L48 (and L2 rather than PCA) were chosen on this same
eval set; the probe is fit to this task with ~650 positives; the difference vector does worse than
h, so it is not obviously reading a belief change at the sentence; an L2 probe at layer 16 already
reaches 0.747 (the audit), so much of the signal is likely lexical, and TF-IDF is a weak text
control. A control with the question removed (the post alone in the user turn) is running.

## Where things live

- Code: [`scripts/author_probe/`](../scripts/author_probe/CLAUDE.md) (`blog.py` for the blog bank,
  answers and deletions; `bank.py`, `fit.py`, `ablate.py` for SynthPAI; `merge_reads.py`,
  `score_drop.py`, `pvalues_drop.py`), `cfbench/evals/author_blog_drop_inline/` (eval and its
  env-viewer page in `score.py`), forward passes with
  [`scripts/hs_at_positions.py`](../scripts/hs_at_positions.py) via
  `scripts/sky/hs_positions_any.yaml`, reads with `scripts/sky/olens_reads.yaml`.
- Blog runs on HF: `runs/author_mcq_blog/2026-10-30_qwen36-qfirst{,-b}` (answers; per-question
  expview pages with reads and deletion tables under `pairs`), `…_ablate-loo{,-b}` (deletions),
  `runs/olens_reads/2026-10-30_author-blog{,-b}`,
  `runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2` (every rendered prompt in its
  expview), `runs/cfbench/2026-10-30_author-blog-drop-haiku{,-b2}`. Batch 3 (`…-qfirst-c`,
  `…_ablate-loo-c`) was stopped: answers kept, deletions partial, marked abandoned.
- Expview: per-question pages
  [`#dataset=runs/author_mcq_blog/2026-10-30_qwen36-qfirst/pairs`](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/author_mcq_blog/2026-10-30_qwen36-qfirst/pairs),
  predictions
  [`#dataset=runs/cfbench/2026-10-30_author-blog-drop-haiku-b2`](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/cfbench/2026-10-30_author-blog-drop-haiku-b2).
