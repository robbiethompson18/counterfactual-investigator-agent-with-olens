# essay_grading_cf_inline: Robbie's defaults, a corrected placebo, per-draw AUROC

**Status (2026-10-31):** REFERENCE. Supersedes the 2026-10-30 Haiku runs (PR #436) and Kylie's L42
NLA (PR #455, unmerged). Code: `cfbench/evals/essay_grading_cf_inline/`.

Question: the investigator sees one opener's prompt and all of Qwen's replies and scores, and gives
P(Qwen's mean grade rises when the opener becomes the plain base one). 1,800 cases (225 openers x 8
of 204 PERSUADE intros), Haiku 5.5 at effort medium through the Message Batches API, 3 draws, 5,400
predictions per condition, none invalid. Truth rate: 54% up.

## What changed from the 2026-10-30 runs

- **Placebo.** It was another opener's reads on the same essay, which could be `base`, the very
  prompt being predicted (12.7% of O-lens placebo draws). It is now another opener on a different
  essay of the same PERSUADE topic, never `base`. Audited on all 16,200 placebo draws: none uses
  `base`, the same essay or the same opener, and all share the topic.
- **Readers.** All-layer NLA at six layers with pooled FVE, J++ read at every token at six layers
  and shown top 3. Both at the O-lens's positions (20 per prompt on average). The O-lens reads are
  unchanged.
- **Effort.** Medium, not low.
- **Scoring.** Every draw is one prediction: pool draws, never average them first
  ([cfbench-scoring.md](cfbench-scoring.md)). Averaging gave a placebo an extra ensemble gain, since
  its draws see different reads.

## Results

Pooled AUROC, 95% CI resampling essays. "+ mean": out-of-fold logistic regression adding Qwen's
observed grade. Bet at the mean alone (no investigator) is 0.776.

| condition      | AUROC                | + mean |
| -------------- | -------------------- | ------ |
| transcript     | 0.753 [0.731, 0.774] | 0.832  |
| O-lens         | 0.775 [0.754, 0.795] | 0.839  |
| O-lens placebo | 0.757 [0.736, 0.777] | 0.830  |
| NLA            | 0.760 [0.738, 0.780] | 0.839  |
| NLA placebo    | 0.757 [0.737, 0.777] | 0.839  |
| J++            | 0.759 [0.737, 0.780] | 0.835  |
| J++ placebo    | 0.750 [0.727, 0.771] | 0.834  |

| paired difference   | AUROC                   | one-sided p | + mean                  |
| ------------------- | ----------------------- | ----------- | ----------------------- |
| O-lens − placebo    | +0.018 [+0.006, +0.031] | 0.002       | +0.009 [+0.001, +0.017] |
| O-lens − transcript | +0.022 [+0.011, +0.033] | < 0.001     | +0.007 [+0.001, +0.014] |
| NLA − placebo       | +0.003 [−0.006, +0.011] | 0.25        | −0.000 [−0.005, +0.005] |
| NLA − transcript    | +0.007 [−0.003, +0.017] | 0.08        | +0.007 [+0.002, +0.012] |
| J++ − placebo       | +0.009 [+0.001, +0.018] | 0.018       | +0.001 [−0.004, +0.006] |
| J++ − transcript    | +0.006 [−0.003, +0.016] | 0.10        | +0.003 [−0.002, +0.009] |

- The O-lens beats its placebo and transcript-only, and the placebo is level with transcript-only
  (+0.004 [−0.007, +0.016]).
- J++ beats its placebo weakly and not transcript-only. The NLA beats neither alone.
- No investigator alone beats bet at the mean. The standard probes (0.864 at user end, 0.888 at the
  final token) beat all of them.
- Haiku rarely notices a swapped placebo: 0.4% of O-lens placebo draws flag a mismatch, and some
  credit the swapped reads to this prompt's opener.

The single-draw and pooled comparisons agree; the mean of 3 draws gave O-lens − placebo +0.007
[−0.005, +0.020] on the same predictions. That analysis of metrics was post hoc.

## Runs

- Predictions: `cfbench/2026-10-31_essay-cf-b-{ts,olens2,nla4,jlens}`. Rows from `ts`, `olens2` and
  `jlens` say harness 15 (branch numbering); `nla4` says 18. Prompts are byte-identical across the
  merge (checked on one case's transcript, O-lens, O-lens placebo and J++ placebo prompts).
- Reads: `olens_reads/2026-10-30_essay-grading-inline`,
  `nla_reads/2026-10-31_essay-grading-inline-all-layer`,
  `jpp_reads/2026-10-31_essay-grading-inline-every-token`.
- Charts: `make -C slides` (`graphs/talk.py`), `python -m graphs.standard_probes`.
