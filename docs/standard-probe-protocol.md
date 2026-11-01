# Standard probe protocol: nested two-fold and two read positions

**Status (2026-10-31):** ACTIVE. Executed protocol; results below. This supersedes older probe
recipes for standardized comparisons, without invalidating or deleting their historical results.
Chart modules `graphs.mcq_unselected` and `graphs.supergpqa_inline` deleted 2026-10-31 (PR #579,
last at commit 62a4e31c); the last two commands under "Regenerate locally" no longer run.

## Scope and labels

Five training families: essay grading, hinted MCQ, SuperGPQA, MMLU-Pro, and SuperGPQA + MMLU-Pro.
The three unhinted MCQ families are each evaluated on both benchmarks (six cells). Every family
reports both read positions separately. The execution budget is 90 minutes from kickoff; reuse
existing responses and activations where compatible, parallelize extraction, and record unfinished
work honestly rather than silently changing the protocol.

Keep the existing MCQ consistency rule: at least seven of eight saved samples give the same bare
letter, with unparseable replies counting against consistency, as in the current evals. Do not
select on Qwen being wrong or on DeepSeek disagreeing with Qwen. Earlier chat explanations
incorrectly called this eight-of-eight. MCQ labels are Qwen's modal answer, not benchmark truth.
Correct/wrong are reporting subgroups only. Hinted MCQ predicts the no-hint modal answer from a
hinted prompt; do not mix hint placement variants or silently reuse a
wrong-only/disagreement-selected cohort as the standard population.

Essay grading uses only biased-prompt activations to predict the same essay's plain-opener mean
grade. Subtract the observed biased-opener mean to obtain the predicted change. Truth is
`plain mean > biased mean`; ties belong to not-up. The older second recipe used a counterfactual
forward pass; it is not a second standard essay probe.

## Known data inventory before rebuilding

| Family     | Previously counted data                                            | Required inventory at execution                                               |
| ---------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| Essays     | 204 essays; 1,800 biased-opener cases and 204 plain cases          | Essays, opener rows, response counts, eligible targets                        |
| SuperGPQA  | 6,328 consistent questions: old 3,000 eval + 3,328 probe pool      | 3,939 right / 2,389 wrong; all 6,328 used                                     |
| MMLU-Pro   | 7,943 consistent questions in the 12,032-question census           | 6,234 right / 1,709 wrong; old 3,883-question split is a subset               |
| Hinted MCQ | 7,943 eligible MMLU questions; rebuild canonical `hint+d1` prompts | Reuse no-hint labels; do not use the filtered old 976-case cohort             |
| Combined   | Union of eligible SuperGPQA and MMLU-Pro                           | 191 normalized stems overlap; group by whitespace-normalized, casefolded stem |

All eligible MCQs have exactly eight saved samples. Canonical hinted prompts put the hint in the
existing `hint+d1` position; hint-first is not an additional ablation. These are inventory facts,
not promised final completion counts. Existing SuperGPQA sampling excluded calculation questions;
this run removes correctness/disagreement filters, not that upstream bank construction restriction.
Record source paths, source revisions, population exclusions, unique questions/essays, training
rows, validation rows, evaluation rows and fold assignments in artifacts. Repeated Qwen samples and
essay opener variants are not independent questions/essays.

## Activation and position ablation

Use the same pinned Qwen3.6-27B checkpoint, bf16 forward passes and thinking-off prompt rendering.
No generated answer tokens enter features. Save exact token IDs, decoded boundary tokens and
position indices. Two sites, each with its own complete reported evaluation:

1. `user_end`: last token of the complete final user content, including options/instructions,
   immediately before the message closing token. This is not the stem before options.
2. `response_start`: actual final token of the thinking-off generation prefix, normally the newline
   token after `</think>`. Do not substitute the closing tag itself.

Use HF `hidden_states[L]` at L = 32, 40, 48, 52, 56, 60 (output of decoder block L - 1). Fit one
layer at a time; no concatenation. Inner validation selects a layer separately for each position.
Never select a position using outer evaluation results and report only its winner.

Single-position MCQs use a multiclass linear letter classifier with impossible options masked out of
the softmax. Both sites use exactly this architecture. This differs from the legacy shared
option-scoring probe on option-newline activations; label that distinction in comparisons. Essay
grades use linear ridge regression. Both use training-only per-dimension standardization,
unpenalized intercepts, no PCA/whitening and no optional calibration variants.

## Splits, training caps and selection

Seed 0; outer two folds and inner two folds, grouped by question or essay. All variants and exact
cross-benchmark question duplicates stay in one group. Save assignments once and share them across
positions, layers and training families. Neither scaling, fitting, nor selection sees outer-test
labels. The two outer folds may select different layers.

For each outer fold, cap the training subset at 1,000 questions per single-benchmark family. The
combined family uses 500 from each benchmark (1,000 total); if one source is short, lower each
source to the same available count. Inner validation splits the capped training subset in half,
selects the layer, and refits on that entire subset before predicting the untouched outer half. Keep
all eligible evaluation questions, including questions not selected for capped training in the other
fold. Essay training uses all available training-essay variants up to 1,000 rows, with essay groups
preserved. Never split variants of one essay between training and validation/test.

For each held-out MMLU half, evaluate the MMLU-only, SuperGPQA-only and combined fits trained using
only the corresponding opposite halves. Repeat symmetrically for SuperGPQA. Layer selection uses
only the source training pool, never target-benchmark test labels. No extra full-source transfer
fits are included in this standardized comparison.

Fix regularization rather than searching it: MCQ L2 lambda = 1, following the prior main probe;
essay mean-loss ridge lambda = 1 as an explicit default. The historical essay implementation did not
save selected alphas, so this is not a claim that 1 was its historical winner. Objectives are mean
cross-entropy + lambda/2 * squared weight norm, and mean squared error + lambda * squared weight
norm, respectively. Thus sklearn-style sum-loss ridge alpha = training_rows * lambda. The MCQ
architecture changed, so the old lambda is a reproducible default, not a newly validated optimum. No
regularization search in this run; at most two candidates would require a documented protocol
revision before outer results are inspected.

Inner selection: highest mean validation accuracy for MCQ and mean validation AUROC of predicted
grade change for essays; ties choose the smaller layer index. Report MCQ accuracy and NLL, overall
and Qwen-right/wrong; essay AUROC, grade/shift error and the minus-observed-grade baseline. Report
per-fold scores, chosen layers, effective sample counts, and pooled out-of-fold predictions. Any
uncertainty calculation resamples question/essay groups, not repeated draws or opener rows.

## Execution, provenance and legacy status

Use tracked runs through `scripts/runs.py`, sync partial results, and finish/sync even abandoned
runs. Record checkpoint revision, software/code SHA, sites/layers, seeds, folds, standardization,
objective convention, penalties, selected layers, runtime, cost, and any missing data. Publish the
input manifests and predictions on the public HF dataset, not in git. Preserve artifacts needed to
reproduce fitting; do not call a partial or selected-cohort result standard-complete.

Existing option-token, balanced/wrong-only, globally selected-layer, author-attribute and other
probe recipes are **legacy/non-standard for this comparison**. Their historical results remain
available; this protocol makes no claim they were all buggy or scientifically invalid. Compare new
probe and lens scores on explicitly identified common evaluation questions.

## Results (2026-10-31)

All five families completed, retaining every eligible evaluation question. Six GPUs captured 22,214
MCQ prompts at both positions and all six layers; all were deleted by 09:00 Pacific. Estimated GPU
cost: **$5.06**, excluding storage/billing rounding. All 260 inner/final fits converged. MCQ
training used exactly 1,000 questions per outer fold (combined: 500 from each benchmark); essays
used 901/899 training rows from 102 essays each.

Accuracy, **user end → response start**:

| Training family | MMLU-Pro (7,943) | SuperGPQA (6,328) |
| --------------- | ---------------- | ----------------- |
| MMLU-Pro        | 86.34% → 98.55%  | 82.24% → 98.63%   |
| SuperGPQA       | 86.06% → 97.77%  | 82.27% → 98.69%   |
| Combined        | 86.34% → 98.20%  | 82.81% → 98.58%   |

Hinted MCQ (7,943): **77.97% → 88.04%**; always choosing the hinted gold letter gives 78.48%. On
Qwen-wrong questions, the hinted probe gives 26.92% → 45.64%. Final-position unhinted fits selected
L60 in both outer folds; the hinted fits selected L56 at both positions.

Essay AUROC (1,800 cases / 204 essays): **0.8637 → 0.8881**; the minus-observed-grade baseline is
0.7758. Paired essay-group bootstrap difference: +0.0244, 95% CI [0.0083, 0.0410] (2,000 draws, seed
0, conditional on these fitted models). The historical essay activation cache did not record its
checkpoint revision; new MCQ captures pin Qwen revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

The final position wins every cell. Cross-benchmark transfer is close: using SuperGPQA training
instead of MMLU training costs 0.78pp on MMLU, while the reverse costs 0.06pp on SuperGPQA. The
combined source does not improve these final-position scores. These are fixed-seed standardized
reruns, not an untouched confirmation experiment.

[Full report, subgroup scores, predictions, fitted weights, input/fold manifests and provenance](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/standard_probes/2026-10-31_nested2-v1).
`report.md` holds all sixteen cells; `paired_comparisons.json` holds group-bootstrap comparisons.

## Code and figures

The maintained implementation is the [`probes` package](../probes/CLAUDE.md): run
`python -m probes.prepare`, `python -m probes.fit`, and `python -m probes.report`. The superseded
MCQ
[fit](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/638c3c56552dbc4368639245af5480a36702bf08/environments/hint_mcq/probe.py),
[eval](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/638c3c56552dbc4368639245af5480a36702bf08/environments/hint_mcq/probe_eval.py),
[unselected](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/638c3c56552dbc4368639245af5480a36702bf08/environments/hint_mcq/probe_unselected.py),
and
[essay](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/638c3c56552dbc4368639245af5480a36702bf08/cfbench/evals/essay_grading_cf_inline/probe.py)
entry points are preserved in git history, not as competing current recipes. Their saved HF
artifacts remain unchanged.

Regenerate locally, without model calls:

```bash
uv run python -m graphs.standard_probes
uv run python -m graphs.mcq_unselected
uv run python -m graphs.supergpqa_inline
```

- `docs/img/standard-probes-mcq.png`: all six source-to-target comparisons, both positions, full
  populations and Qwen-right/wrong subgroups.
- `docs/img/standard-probes-hinted.png`: canonical gold-letter hint after options, full 7,943 cases.
- `docs/img/standard-probes-essays.png`: both essay positions, historical transcript/O-lens
  investigators on the same 1,800 cases, and the mean-reversion baseline. Historical Haiku used low
  effort. The old placebo is excluded because some reads came from the same essay's counterfactual.
- `docs/img/mcq-unselected{,-diff}.png` and `docs/img/supergpqa-inline{,-diff}.png`: historical lens
  cohorts (2,523 and 3,000 questions), with new within-benchmark probes joined by question ID. These
  are paired comparison subsets, not the full-population scores above.

All confidence intervals resample question stems or essays, conditional on the fitted models. They
exclude training/selection variability. The figures use `graphs.style` (merged style PR #476).
Historical hint-selected figures remain legacy: their prompts, labels and cohorts differ, so their
probe bars cannot be replaced by the full-population canonical hint result.
