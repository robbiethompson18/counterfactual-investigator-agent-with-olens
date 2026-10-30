# Author MCQ probe: what Qwen thinks about a passage's author

**Status (2026-10-30):** ACTIVE. SynthPAI pilot, Qwen3.6-27B: label probes per site and layer,
Qwen's own answers, DSv4 Flash answers, leave-one-comment-out ablations (the "?" probe moves with
Qwen, the passage-only probe does not). Question-first rerun, O-lens reads and a single-turn eval
(no O-lens gain) done.

Kylie's question: Qwen3.6-27B (thinking off) answers multiple-choice questions about the person who
wrote a passage. Can a linear probe read Qwen's view of the author from activations, ideally before
the question is asked? The probe is meant as the ground truth for later sentence ablations and
O-lens reads. Its label probe is "good by construction" only up to what the numbers below show: it
is trained on the dataset label, not on Qwen's answer.

Short answer: the probe of record is the label probe at the passage-only `<|im_end|>` (it never sees
the question): AUROC 0.69 / 0.79 / 0.81 / 0.68 / 0.71 (sex / age / education / relationship /
income). That is near a TF-IDF bag of words for age, education and relationship (0.76 / 0.81 / 0.68)
and above it for sex and income (0.56 / 0.62). Once the question has been read (the "?" token,
before any option), a label probe matches Qwen's own answer pmf, and beats it for age.

Page with every passage, the exact answer prompts, Qwen's letter pmf in both option orders and DSv4
Flash's replies (private artifact):
[claude.ai/artifact/GLNKN8xy5aXw3TijiTKyi2](https://claude.ai/artifact/GLNKN8xy5aXw3TijiTKyi2); the
same items on the public expview:
[expview#dataset=runs/author_mcq_probe/2026-10-29_synthpai-qwen36-v1](https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/author_mcq_probe/2026-10-29_synthpai-qwen36-v1).

## Setup

- **Data.** [SynthPAI](https://huggingface.co/datasets/RobinSta/SynthPAI) (CC-BY-NC-SA-4.0): 7,823
  synthetic Reddit comments by 300 GPT-4 personas with ground-truth profiles. Each comment also
  carries a human reviewer's estimate of each attribute, with hardness and certainty ratings (1–5);
  these are kept per question (`human`: how many of the passage's comments the reviewer guessed the
  attribute from, max certainty, min hardness).
- **Passages.** Authors are short (median ~225 words in all), so a passage is a random subset of one
  author's comments, 120–300 words, joined by blank lines. Up to 8 passages per author, so passages
  of one author overlap. That gives 1,372 passages from 291 authors; the effective sample size is
  closer to 291. Every split holds out by author.
- **Questions** (first person, Kylie 2026-10-29), each with lettered options:
  - sex: "Based on what I wrote above, what is my sex?" Male / Female
  - age: "... how old am I most likely to be?" 18-29 / 30-44 / 45-59 / 60 or older
  - education: "... what is my highest level of education (completed or in progress)?" High school
    or less / Some college (studying for a first degree) / Bachelor's / Graduate
  - relationship: "... what is my relationship status?" Single / In a relationship or engaged /
    Married / Divorced / Widowed
  - income: "... what is my income level, relative to where I live?" Low / Middle / High / Very high
- **Answer prompt.** One user turn: the raw passage, `\n\n`, the question, the options (`A. ...`),
  `\n\nAnswer with the letter only.`, then the thinking-off assistant prefix. Qwen's letter
  log-probs at the next token are renormalised over the option letters. Two option orders (canonical
  and reversed) are averaged in option space.
- **Sites.** All 65 `hidden_states` (0 = embeddings), bf16 HF transformers.
  - `text_end`, `im_end`: the passage's last token and the `<|im_end|>` closing the turn, in a
    passage-only prompt (the user turn is just the passage). The answer prompt's tokens are
    identical up to `text_end`.
  - `qmark`: the token holding the question's "?" in the answer prompt. It comes before the options,
    so it is the same in both orders.
  - `last`: the answer prompt's final token (where Qwen's letter is read). It has seen the options.
- **Probes.** Multinomial logistic regression per (site, layer, attribute), standardised features, C
  = 0.01 (0.001 and 0.1 within ±0.015), 5-fold GroupKFold by author. **Primary target: the dataset
  label**; secondary: Qwen's own answer. AUROC = macro one-vs-rest over classes with ≥ 10 positives.
  For age and education, also a single-axis ridge regression (exact age in years; education bin rank
  0–3), alpha chosen by inner 3-fold GroupKFold. The best layer is chosen on the same held-out
  predictions, so the "best" numbers are mildly optimistic; the every-4th-layer column is a check.

## Qwen's answers

| attribute    | Qwen acc vs label | majority acc | Qwen pmf AUROC vs label | orders agree | acc, human guessed / not |
| ------------ | ----------------- | ------------ | ----------------------- | ------------ | ------------------------ |
| sex          | 0.722             | 0.505        | 0.811                   | 0.875        | 0.864 / 0.630            |
| age          | 0.527             | 0.348        | 0.803                   | 0.823        | 0.640 / 0.436            |
| education    | 0.698             | 0.525        | 0.879                   | 0.826        | 0.782 / 0.595            |
| relationship | 0.350             | 0.240        | 0.739                   | 0.851        | 0.609 / 0.156            |
| income       | 0.631             | 0.504        | 0.786                   | 0.746        | 0.709 / 0.550            |

- Qwen's answers are skewed: "Single" on 1,061/1,372 relationship questions, 45+ on only 164 age
  questions (the labels have 547). The median top-option probability is 0.65–0.77, and 0.91 for
  relationship.
- Qwen is much more accurate where the SynthPAI human had guessed the attribute from one of the
  passage's comments (the last column), so the human annotations work as a hardness label.
- DeepSeek V4 Flash (OpenRouter, effort medium, order 0 and 1, 227/13,720 replies unparsed):
  accuracy 0.71 / 0.56 / 0.57 / 0.35 / 0.60, agreeing with Qwen 75% / 73% / 66% / 85% / 72% (same
  attribute order).

## Label probe by site

Best layer per site, AUROC vs label. `last` pools both option orders (see below).

| attribute    | text_end  | im_end    | qmark     | last (pooled) | Qwen's own pmf | TF-IDF baseline |
| ------------ | --------- | --------- | --------- | ------------- | -------------- | --------------- |
| sex          | 0.600 L22 | 0.686 L28 | 0.772 L55 | 0.802 L46     | 0.811          | 0.563           |
| age          | 0.726 L8  | 0.788 L5  | 0.815 L30 | 0.837 L57     | 0.803          | 0.760           |
| education    | 0.749 L20 | 0.814 L8  | 0.880 L59 | 0.871 L37     | 0.879          | 0.808           |
| relationship | 0.645 L21 | 0.675 L18 | 0.766 L34 | 0.810 L45     | 0.739          | 0.678           |
| income       | 0.677 L8  | 0.711 L10 | 0.774 L57 | 0.781 L61     | 0.786          | 0.618           |

- **Before the question** (`im_end` is better than `text_end` everywhere), the probe peaks early
  (L5–L28) and is barely above TF-IDF for age, education and relationship. It reads more than the
  words for sex and income, but stays well under Qwen's own pmf. Agreement of the label probe's
  argmax with Qwen's answer is 0.26–0.62 there.
- **At the "?"** the probe peaks late (L30–L59) and matches Qwen's pmf (within ±0.04, above it for
  age and relationship). Its argmax agrees with Qwen's answer 0.70 / 0.52 / 0.73 / 0.33 / 0.69, and
  its AUROC against Qwen's answer is 0.79 / 0.81 / 0.92 / 0.78 / 0.85: it tracks Qwen's view and the
  label about equally. Relationship agreement is low because Qwen says "Single" almost always.
- Every-4th-layer best is within 0.01 of the all-layer best at `qmark` and `im_end`.
- The secondary probe (trained on Qwen's answer) was fit at the passage sites only: best `im_end`,
  AUROC vs Qwen's answer 0.70 / 0.81 / 0.83 / 0.65 / 0.71.

### Ridge (age in years, education rank)

| attribute | site, layer    | Spearman | MAE       | AUROC at bin cuts     | Qwen E[bin] Spearman | Qwen cut AUROCs       |
| --------- | -------------- | -------- | --------- | --------------------- | -------------------- | --------------------- |
| age       | im_end L5      | 0.755    | 7.0 years | 0.907 / 0.860 / 0.829 | 0.768                | 0.899 / 0.869 / 0.902 |
| age       | qmark L52      | 0.818    | 6.1 years | 0.924 / 0.895 / 0.898 |                      |                       |
| age       | last L34, pool | 0.837    | 5.9 years | 0.931 / 0.913 / 0.919 |                      |                       |
| education | im_end L8      | 0.656    | 0.53      | 0.822 / 0.844 / 0.868 | 0.694                | 0.877 / 0.914 / 0.874 |
| education | qmark L31      | 0.720    | 0.49      | 0.908 / 0.917 / 0.890 |                      |                       |
| education | last L40, pool | 0.714    | 0.47      | 0.922 / 0.932 / 0.880 |                      |                       |

Age cuts: ≥ 30 / 45 / 60; education: ≥ some college / bachelor's / graduate. The question-side age
ridge beats Qwen's own expected bin (0.82–0.84 vs 0.77): the residual stream knows more about age
than Qwen's 4-way answer shows.

### The `last` site: letter controls (Kylie's worry: a letter or " Male"-logit direction)

Both orders pooled with labels in option space, both orders of a passage in one fold. Cross-order:
fit on one order's training authors, test on the other order's held-out authors. Letter control:
target = the letter of the gold option, pooled over orders.

| attribute    | pooled, vs label | order 0 → 1 | order 1 → 0 | letter control, vs gold letter |
| ------------ | ---------------- | ----------- | ----------- | ------------------------------ |
| sex          | 0.802 L46        | 0.727 L14   | 0.728 L14   | 0.792 L52                      |
| age          | 0.837 L57        | 0.819 L40   | 0.820 L12   | 0.826 L57                      |
| education    | 0.871 L37        | 0.844 L36   | 0.856 L43   | 0.884 L60                      |
| relationship | 0.810 L45        | 0.775 L43   | 0.771 L34   | 0.782 L48                      |
| income       | 0.781 L61        | 0.760 L7    | 0.761 L7    | 0.818 L60                      |

- The gold letter is as decodable as the concept at `last`: by then the site holds Qwen's chosen
  letter, which is right about as often as the probe. Cross-order transfer loses 0.02–0.08 (most for
  sex), so part of the `last` signal is order-specific.
- `qmark` has none of this problem (no options seen, identical in both orders) and is within 0.03 of
  pooled `last` except relationship.

**Probe of record (Kylie, 2026-10-30): `im_end` of the passage-only prompt**, label-trained: L28
sex, L5 age (ridge L5), L8 education (ridge L8), L18 relationship, L10 income. It has seen no
question, so nothing in its signal can come from the question or the options (no "random bullshit"
from the prompt). It is weaker: near the TF-IDF baseline for age, education and relationship.
`qmark` (L55 / L30 / L59 / L34 / L57) is the stronger alternative when reading Qwen's view after it
has been asked.

## Leave-one-comment-out ablations (2026-10-30)

Run `runs/author_mcq_probe/2026-10-30_ablate-loo` (`scripts/author_probe/ablate.py`). Candidates:
confident pairs, c = (p_top − 1/K)/(1 − 1/K) ≥ 0.5 with both option orders agreeing, relationship
dropped; one passage per author: 283 passages, 796 (passage, attribute) pairs, 3,249 deletions. For
each comment i, Qwen's order-averaged pmf without it (d_i = TV to the full pmf) and both frozen
probes of record (fit on all full passages, at their best layers) on the deleted passage. Stable =
median d_i < 0.05; evidence comment = d_i > max(0.15, 5 × median).

| attribute | pairs | stable | stable (argmax survives ≥ 90%) | stable with an evidence comment |
| --------- | ----- | ------ | ------------------------------ | ------------------------------- |
| sex       | 142   | 138    | 135                            | 63                              |
| age       | 254   | 186    | 224                            | 79                              |
| education | 202   | 179    | 188                            | 69                              |
| income    | 198   | 174    | 186                            | 43                              |

On the stable set. Shift r = Pearson r over all deletions between Qwen's change in p(original
answer) and the scorer's change in the same class; null = the scorer's shifts permuted within each
pair (95th percentile of 1,000 draws in brackets). Probe AUROCs are out-of-fold on the full
passages, restricted to the stable set.

| attribute | Qwen acc | `im_end` AUROC vs label / Qwen | "?" AUROC vs label / Qwen | r(Qwen, `im_end`) | r(Qwen, "?") | r(Qwen, TF-IDF) | evidence comments human-guessed (vs all comments) |
| --------- | -------- | ------------------------------ | ------------------------- | ----------------- | ------------ | --------------- | ------------------------------------------------- |
| sex       | 0.91     | 0.77 / 0.78                    | 0.92 / 0.96               | 0.04 [0.05]       | 0.36 [0.07]  | 0.05 [0.06]     | 0.62 (0.07)                                       |
| age       | 0.57     | 0.81 / 0.77                    | 0.82 / 0.85               | 0.06 [0.07]       | 0.13 [0.09]  | 0.03 [0.13]     | 0.24 (0.08)                                       |
| education | 0.78     | 0.85 / 0.85                    | 0.91 / 0.94               | 0.02 [0.05]       | 0.25 [0.07]  | 0.05 [0.03]     | 0.49 (0.10)                                       |
| income    | 0.66     | 0.71 / 0.72                    | 0.81 / 0.91               | 0.01 [0.05]       | 0.24 [0.09]  | 0.16 [0.11]     | 0.43 (0.07)                                       |

- Most confident pairs are stable (87–97%); a third to a half of those have one or more evidence
  comments.
- **The passage-only `<|im_end|>` probe does not move with Qwen when a comment is deleted** (r at
  the null for every attribute), although it predicts the label and Qwen's answer across passages.
  It reads features of the text that correlate with the attribute, not the evidence Qwen uses: the
  "random bullshit" risk, measured. **The "?" probe moves with Qwen** (r 0.36 / 0.13 / 0.25 / 0.24
  vs null ≤ 0.09; age is marginal), and on the stable set it predicts Qwen's answer at AUROC 0.85–
  0.96. The TF-IDF scorer does not track Qwen's shifts either (income slightly).
- Evidence comments are the ones a human also reads the attribute from: the SynthPAI reviewer
  guessed the attribute from 62% / 24% / 49% / 43% of evidence comments vs 7–10% of all comments.
- Qwen's age errors have a visible mechanism: several wrong 30-44 answers hinge on one comment about
  turning 30 ("Hit 30 farming...", "Turning 30? Cue responsibility's crescendo.") from authors aged
  45+.
- Key passages (ten per attribute, the clearest single evidence comment) are listed in the run's
  `ablate.md`; every stable pair with per-comment shifts is on the page
  [claude.ai/artifact/8BHQhsDWXZ8mFmxhPhmzjk](https://claude.ai/artifact/8BHQhsDWXZ8mFmxhPhmzjk)
  (and the public expview, `#dataset=runs/author_mcq_probe/2026-10-30_ablate-loo`).
- Cost ~$8 of H200 (~1.5 h incl. setup).

### Why did Qwen move? (judge, 2026-10-30)

Every evidence comment of a stable pair, plus each stable pair's top comment with TV > 0.1 that is
not evidence ("tier 2"), judged by Haiku 5.5 (effort medium, 1 replicate). The judge sees only the
comment, the question and the option whose probability fell when the comment was deleted, and
answers explicit / indirect / none ("no apparent reason"). All calls are in the run's
`expview/judge` and on
[claude.ai/artifact/52TF6yU4z8sJoXck5yrHBJ](https://claude.ai/artifact/52TF6yU4z8sJoXck5yrHBJ); the
list is in `judge.md`.

| attribute | tier     | n   | explicit | indirect | none | none, not human-guessed |
| --------- | -------- | --- | -------- | -------- | ---- | ----------------------- |
| sex       | evidence | 64  | 7        | 52       | 5    | 2                       |
| sex       | tier 2   | 25  | 0        | 17       | 8    | 7                       |
| age       | evidence | 88  | 8        | 70       | 10   | 9                       |
| age       | tier 2   | 43  | 0        | 29       | 14   | 14                      |
| education | evidence | 74  | 14       | 57       | 3    | 2                       |
| education | tier 2   | 51  | 1        | 38       | 12   | 12                      |
| income    | evidence | 49  | 10       | 35       | 4    | 4                       |
| income    | tier 2   | 50  | 3        | 34       | 13   | 11                      |

- Most evidence comments are "indirect" (life stage, stereotyped hobbies, wording); only 7–20% are
  explicit. "No apparent reason" is 5–11% of evidence comments and ~25% of tier 2.
- The judge is wrong on some "none" calls: "pretty rare for guys over here" (ikebana, Male), "psych
  wasn't always seen as 'masculine'" (Male) and "old 'woman' blueprints" (Female) are first-person
  cues a reader would take. Read the "none" list, don't count it.
- Patterns among the genuinely non-obvious ones: (1) numbers pull age to 30-44 ("Raskolnikov was
  23", "Caesar ... 25 ... his 40s", alongside the "turning 30" comments above); (2) complaining
  about gender stereotypes pulls sex toward Female, also for male authors ("gender stereotypes suck
  even in 'progressive' spots": Female 0.86 → 0.69, label Male); (3) Reddit-style "Question: ...
  Question description: ..." posts with no attribute content move income and sex by 0.1–0.3.

Stable pairs where Qwen is wrong: sex 13/138, age 80/186, education 39/179, income 59/174. The
ablation page's items carry `attribute`, `qwen_correct`, `has_evidence` and `judge_bucket` (the
pair's least-explained judged comment) as filterable fields.

## Question-first format (2026-10-30)

Run `runs/author_mcq_probe/2026-10-30_synthpai-qwen36-qfirst` (`bank.py qfprompts`). User turn = the
question without options ("What is my sex, based on the comments I wrote below?") + the comments +
"What do you think?"; Qwen's answer prompt appends the options and "Answer with the letter only."
(two orders). Sites: `qend`, the "?" of "What do you think?" (question and comments seen, no
options), and `qtop`, the question's own "?" (before the comments: a null). Label probes, all
layers, C = 0.01.

| attribute    | Qwen acc (old format) | Qwen pmf AUROC | `qend` AUROC | `qtop` | old-format "?" | passage-only `<\|im_end\|>` |
| ------------ | --------------------- | -------------- | ------------ | ------ | -------------- | --------------------------- |
| sex          | 0.75 (0.72)           | 0.82           | 0.72 (L17)   | 0.41   | 0.77           | 0.69                        |
| age          | 0.54 (0.53)           | 0.81           | 0.80 (L5)    | 0.44   | 0.82           | 0.79                        |
| education    | 0.67 (0.70)           | 0.88           | 0.82 (L14)   | 0.45   | 0.88           | 0.81                        |
| relationship | 0.35 (0.35)           | 0.75           | 0.71 (L24)   | 0.43   | 0.77           | 0.68                        |
| income       | 0.61 (0.63)           | 0.78           | 0.75 (L8)    | 0.46   | 0.77           | 0.71                        |

`qend` is only a little above the passage-only site and, like it, peaks early; age ridge Spearman
0.80 (MAE 6.5 years). Asking first does not move Qwen's late answer-forming representation to the
end of the passage: the old format's "?" right after the question stays the best site.

## O-lens reads and the single-turn eval (2026-10-30)

**Reads:** `runs/olens_reads/2026-10-30_author-qfirst` (expview:
`#dataset=runs/olens_reads/2026-10-30_author-qfirst`). The 677 stable pairs, question-first prompt
without options, agu18dec space lens, one-engine reader, T 1, up to 4 bullets, layers 32/40/48, at
every punctuation or newline token from the question to the final "?" (22,644 positions, 67,932
reads; two H200 readers, merged). The lens often names an age range or other attribute guesses at
the comments' punctuation ("20s or early 30s", ...).

**Eval:** `cfbench/evals/author_mcq_inline`. Cases: the stable pairs where Qwen's question-first
answer differs from the SynthPAI label (196: sex 14, age 79, education 40, income 63). The
investigator sees the question-first answer prompt and names Qwen's letter; `olens_inline` adds the
pair's reads. Investigator **Claude Haiku 5.5 at reasoning effort low** (Kylie; not the repo's
default Gemini 3.8 Flash), 3 draws, run `runs/cfbench/2026-10-30_author-mcq-haiku`, $1.78.

| attribute | `transcript_inline` | `olens_inline` | Qwen's most common answer on these cases |
| --------- | ------------------- | -------------- | ---------------------------------------- |
| sex       | 0.95                | 0.86           | 0.71                                     |
| age       | 0.82                | 0.76           | 0.78                                     |
| education | 0.77                | 0.72           | 0.60                                     |
| income    | 0.81                | 0.86           | 0.83                                     |
| all       | 0.82                | 0.79           | -                                        |

Paired O-lens − transcript: −0.024 (95% bootstrap CI −0.062 to +0.012, 585 draws): no gain. These
cases are not hard from the prompt: Qwen's wrong answers are the ones a reader of the comments also
gives (and age/income answers are mostly the modal 30-44 / Middle), so the transcript-only
investigator is already near the majority-answer rate or above it.

## Comment-deletion eval, question first (2026-10-30)

**Deletions:** `runs/author_mcq_probe/2026-10-30_ablate-loo-qfirst`
(`ablate.py prep/analyze --qfirst`, one H100, ~$4): 287 passages (one per author, preferring the
original-format run's passages), 738 confident pairs, 18,634 prompts. Stable / with an evidence
comment: sex 139/147, 56; age 156/226, 58; education 153/184, 66; income 138/181, 33. Evidence
comments are human-guessed 65 / 24 / 59 / 49% vs 7-10% of all comments.

**Eval:** `cfbench/evals/author_mcq_drop_inline` (hint_qwen_dsv4/inline.py's single-turn layout,
three conditions). Claim: deleting this comment lowers Qwen's probability on its answer by more than
max(0.15, 5 × the pair's median drop). Stable pairs, all positives + as many negatives per pair
(base rate 0.50), versions `qwen_wrong` (Qwen ≠ label: 124 cases, 62 positive, 54 pairs) and
`qwen_right` (346 cases, 173 positive, 161 pairs). Positive drops: median 0.23 / 0.26; 16 / 49 of
them flip the argmax. Investigator Haiku 5.5 at effort low (not the repo's default Gemini 3.8
Flash). Inputs run `runs/cfbench_evals/2026-10-30_author-mcq-drop-inline-inputs`, whose `expview/`
has every rendered prompt (case × condition, not sent) with the pair's deletion table and a link to
the pair's item.

**Results** (`runs/cfbench/2026-10-30_author-mcq-drop-haiku`, 1 draw, Haiku 5.5 effort low, $3.56;
base rate 0.50):

| version (n)      | condition  | AUROC | RMSE  | acc at p ≥ 0.5 |
| ---------------- | ---------- | ----- | ----- | -------------- |
| qwen_wrong (124) | transcript | 0.89  | 0.494 | 0.58           |
|                  | O-lens     | 0.91  | 0.491 | 0.63           |
|                  | placebo    | 0.91  | 0.502 | 0.60           |
| qwen_right (346) | transcript | 0.88  | 0.486 | 0.61           |
|                  | O-lens     | 0.90  | 0.492 | 0.59           |
|                  | placebo    | 0.88  | 0.506 | 0.58           |

Paired (bootstrap over cases): O-lens − transcript Brier gain +0.003 [−0.013, +0.020] (wrong),
−0.006 [−0.018, +0.006] (right); AUROC +0.02 in both, CIs over 0. O-lens − placebo Brier +0.011
[−0.007, +0.031] (wrong), +0.014 [+0.003, +0.027] (right). The investigator ranks evidence comments
well from the text alone (AUROC ~0.88) but its probabilities are poorly calibrated (RMSE ~0.49 at a
0.50 base rate); the O-lens adds nothing significant over the transcript and a little over the
placebo on qwen_right. Env viewer: `#eval=author_mcq_drop_inline` (after merge).

**Reads:** `runs/olens_reads/2026-10-30_author-qfirst` was topped up with the broader read-position
selector (`bank.is_read_token`: any punctuation, symbol or newline token, e.g. quotes, dashes, `&`):
+7,109 positions on the 677 pairs and 31 new pairs, 93,117 reads in all. Per-pair items with the
reads inline and the deletion tables:
`#dataset=runs/author_mcq_probe/2026-10-30_synthpai-qwen36-qfirst/pairs`.

## SynthPAI ×2 and the Blog Authorship Corpus (2026-10-30)

**SynthPAI, a second passage per author** (`runs/author_mcq_probe/2026-10-30_ablate-loo-qfirst-b`,
`ablate.py prep --qfirst --exclude`): 232 more passages, 706 confident pairs; stable / with
evidence: sex 128 / 53, age 135 / 59, education 146 / 56, income 137 / 32. Drop eval with both
passages (`AUTHOR_X2=1`, inputs `runs/cfbench_evals/2026-10-30_author-mcq-drop-inline-inputs-x2`):
904 cases (qwen_wrong 242 from 77 authors, qwen_right 662), base rate 0.50; score with CIs clustered
by author.

**Blog Authorship Corpus** (`scripts/author_probe/blog.py`; Schler et al. 2006,
`tasksource/blog_authorship_corpus`): 2,004 one-post passages (200-450 words) balanced over gender ×
the corpus's three age groups (13-17 / 23-27 / 33-48, asked as Under 20 / 20-29 / 30 or older), bank
on HF `banks/author_mcq_blog/`; question-first prompts ("... based on the blog post I wrote
below?"), the sentence as the deletion unit (median 19 per post).

| attribute | Qwen acc (chance) | confident | ablated | stable | stable with evidence | stable, Qwen wrong / right |
| --------- | ----------------- | --------- | ------- | ------ | -------------------- | -------------------------- |
| sex       | 0.83 (0.50)       | 1,279     | 413     | 395    | 107                  | 111 / 284                  |
| age       | 0.76 (0.33)       | 1,667     | 615     | 536    | 165                  | 256 / 280                  |

Ablated = every confident Qwen-wrong pair plus 300 Qwen-right pairs per attribute
(`runs/author_mcq_blog/2026-10-30_ablate-loo`, 42,696 prompts on two H200s). Drop eval
`cfbench/evals/author_blog_drop_inline` (same rule, prompt, conditions and investigator): qwen_wrong
366 cases (183 positive, 143 pairs), qwen_right 294 (147, 129). O-lens reads for its 272 pairs
(13,876 positions × layers 32/40/48) go to `runs/olens_reads/2026-10-30_author-blog`.

### Drop-eval results, SynthPAI ×2 and blog (Haiku 5.5 effort low, 1 draw)

`scripts/author_probe/score_drop.py`; CIs bootstrap over authors. AUROC / RMSE / accuracy at p ≥ 0.5
(base rate 0.50):

| eval (run, $)                                              | version (cases)  | transcript          | O-lens              | placebo             |
| ---------------------------------------------------------- | ---------------- | ------------------- | ------------------- | ------------------- |
| SynthPAI ×2 (`2026-10-30_author-mcq-drop-haiku-x2`, $6.95) | qwen_wrong (242) | 0.87 / 0.490 / 0.63 | 0.89 / 0.494 / 0.62 | 0.88 / 0.510 / 0.57 |
|                                                            | qwen_right (662) | 0.88 / 0.487 / 0.61 | 0.87 / 0.498 / 0.61 | 0.87 / 0.506 / 0.60 |
| blog (`2026-10-30_author-blog-drop-haiku`, $6.61)          | qwen_wrong (366) | 0.73 / 0.575 / 0.52 | 0.75 / 0.567 / 0.52 | 0.73 / 0.579 / 0.52 |
|                                                            | qwen_right (294) | 0.82 / 0.553 / 0.54 | 0.83 / 0.543 / 0.55 | 0.83 / 0.556 / 0.54 |

Paired Brier gain (positive = O-lens better) and AUROC difference:

| eval, version      | O-lens − transcript: Brier / AUROC                | O-lens − placebo: Brier / AUROC                   |
| ------------------ | ------------------------------------------------- | ------------------------------------------------- |
| SynthPAI ×2, wrong | −0.004 [−0.016, +0.007] / +0.020 [−0.021, +0.060] | +0.016 [+0.003, +0.029] / +0.010 [−0.036, +0.054] |
| SynthPAI ×2, right | −0.010 [−0.018, −0.001] / −0.007 [−0.031, +0.017] | +0.008 [−0.001, +0.017] / +0.004 [−0.017, +0.027] |
| blog, wrong        | +0.010 [−0.000, +0.020] / +0.015 [−0.029, +0.058] | +0.014 [+0.002, +0.026] / +0.013 [−0.035, +0.057] |
| blog, right        | +0.011 [−0.002, +0.025] / +0.006 [−0.035, +0.049] | +0.014 [+0.001, +0.027] / −0.001 [−0.041, +0.035] |

On SynthPAI the O-lens beats the placebo but not the transcript (slightly worse on qwen_right). On
the blog corpus, where the transcript is harder (AUROC 0.73 on qwen_wrong), the O-lens leans ahead
of the transcript (Brier +0.010, +0.011, CIs touching 0) and beats the placebo in both versions. The
investigator's probabilities stay too low and poorly calibrated throughout (RMSE 0.49-0.58 at a 0.50
base rate); its ranking (AUROC) carries the signal.

### Base-rate-free tests and the doubled blog set (2026-10-30)

`scripts/author_probe/pvalues_drop.py`: pooled qwen_wrong + qwen_right, 95% CIs and one-sided p from
a 10k bootstrap over authors; raw Brier also by an author-level sign-flip test; AUROC difference;
Brier after Platt recalibration of each condition (5-fold CV by author).

| eval (cases, authors)                      | O-lens − transcript: raw Brier / AUROC / recal. Brier | O-lens − placebo: raw Brier / AUROC / recal. Brier     |
| ------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------ |
| blog batch 1 (660, 264)                    | +0.010 (p 0.008) / +0.012 (p 0.22) / +0.004 (p 0.26)  | +0.014 (p 0.001) / +0.006 (p 0.36) / +0.001 (p 0.42)   |
| blog batches 1+2 (1,316, 520; `-haiku-b2`) | +0.004 (p 0.09) / +0.006 (p 0.30) / +0.002 (p 0.36)   | +0.016 (p < 0.001) / +0.016 (p 0.07) / +0.006 (p 0.11) |
| SynthPAI ×2 (904, 209)                     | −0.009 (p 0.99) / −0.000 (p 0.51) / −0.001 (p 0.57)   | +0.010 (p 0.005) / +0.006 (p 0.28) / +0.004 (p 0.28)   |

The raw-Brier wins are calibration: with reads in the prompt Haiku's probabilities sit a little
higher, closer to the 0.50 base rate. Ranking (AUROC) and recalibrated Brier show no O-lens gain
over the transcript; with twice the bloggers the batch-1 edge shrank to noise. Over the placebo the
O-lens keeps a marginal AUROC edge (p 0.07): the reads carry some case-specific signal, but not
beyond what the transcript already gives the investigator. Batch 2 reused batch 1's draws (its
prompts are byte-identical; the placebo is drawn within a batch).

## Where things live

- Code: [`scripts/author_probe/`](../scripts/author_probe/CLAUDE.md) (`bank.py`, `fit.py`,
  `dsv4.py`, `view.py`), forward passes with
  [`scripts/hs_at_positions.py`](../scripts/hs_at_positions.py) on one RunPod H200 via
  [`scripts/sky/hs_positions.yaml`](../scripts/sky/hs_positions.yaml).
- Bank on HF:
  [`banks/author_mcq/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/author_mcq)
  (`passages.jsonl`, `questions.jsonl`).
- Run on HF:
  [`runs/author_mcq_probe/2026-10-29_synthpai-qwen36-v1`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/author_mcq_probe/2026-10-29_synthpai-qwen36-v1):
  `qwen.md`, `qwen_answers.jsonl` (pmf per question and order), `probes.md/json` (all sites, per
  layer, C check), `qprobes.md/json` (`last` controls), `bow.md`, `dsv4_answers.jsonl`, `expview/`.
  `2026-10-29_synthpai-qwen36` is the abandoned v0 (framing line, third-person questions).
- Activations are not on HF: `data/probe_cache/2026-10-29_synthpai-qwen36-v1/` on bramble
  (`text_end.npy`, `im_end.npy`, `qsites/part-K/{qmark,last}.npy` for attribute K, [1372, 65, 5120]
  fp16). Order-1 `qsites_o1` is complete only for sex, age and education (the pod auto-downed
  mid-copy); regenerating takes ~15 min on one H200.
- Cost: ~$12 of RunPod H200 (~2.2 h) + $10 of DSv4 Flash on OpenRouter ($2.14 of it on v0).

```bash
uv run python -m scripts.author_probe.bank build --upload
uv run --with transformers --with jinja2 python -m scripts.author_probe.bank prompts RUN    # + qprompts RUN [--order 1]
uv run python -m scripts.author_probe.fit answers RUN
uv run --with torch --with scikit-learn python -m scripts.author_probe.fit probes RUN --device cuda --C 0.01 0.001 0.1
uv run --with torch --with scikit-learn python -m scripts.author_probe.fit qprobes RUN --device cuda
```
