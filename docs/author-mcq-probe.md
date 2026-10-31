# Author evidence: does the O-lens show which sentence Qwen reads the author from?

**Status (2026-10-31):** NULL. On real blog posts the O-lens gives an investigator no gain over the
prompt alone in telling which sentence Qwen3.6-27B infers the author's sex or age from (AUROC p =
0.30, recalibrated Brier p = 0.36, 1,316 cases from 520 bloggers). An earlier pilot on SynthPAI
(synthetic comments) gave the same null and is retired. A third batch of bloggers was stopped
(Kylie: sample size won't change it).

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
`scripts/author_probe/pronoun_probe.py` and `period_probe.py`; runs
`runs/author_mcq_blog/2026-10-31_pronoun-probe` and `…_period-probe` (activations included).

- **Answer words in the O-lens reads** along each post (FVE-weighted pronouns matching Qwen's answer
  for sex, a short word list per bucket for age; level, step, before/after shifts, slope): AUROC
  0.55-0.60 alone (sex 0.68 on qwen_wrong for the 3-sentence signed shift, one of 16 features),
  placebo reads and the sentence text ~0.5. Added to Haiku's transcript p: sex +0.006 (n.s.), age
  +0.020 — but the same counts on the sentence text add +0.017, so the age gain is lexical.
- **Sentence-end activations** (Qwen3.6-27B residual at the token holding the sentence's last
  character, question-first prompt). An answer probe trained on 3,000 batch-3 pairs (no shared
  bloggers) and read along each post: AUROC 0.57-0.64, +0.007 to +0.017 over Haiku. A probe fit to
  the task (StandardScaler, PCA 128, logistic C 0.01, layer chosen by inner CV: L48; 5-fold CV
  grouped by blogger):

| score                              | AUROC                | paired accuracy | + over Haiku transcript p |
| ---------------------------------- | -------------------- | --------------- | ------------------------- |
| Haiku transcript p                 | 0.797                | 0.758           |                           |
| h at the sentence end              | 0.780 [0.755, 0.803] | 0.770           | +0.053 [+0.039, +0.068]   |
| h(end k) − h(end k−1)              | 0.724                | 0.712           | +0.033                    |
| TF-IDF of the sentence             | 0.645                | 0.652           | +0.011                    |
| sentence length + position in post | 0.704                | 0.719           | +0.035                    |
| h over Haiku + length/position     |                      |                 | +0.028 [+0.017, +0.038]   |

Caveats: post hoc; the probe is fit to this task with ~650 positives (CV grouped by blogger, layer
chosen inside each fold, but still the easiest thing here to overfit); the difference vector does
worse than h, so it is not obviously reading a belief change at the sentence; TF-IDF is a weak text
control. Next control: the same probe on the activation of the sentence alone (no question, no post
around it), which separates "what this sentence says" from "what Qwen made of it in context".

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
