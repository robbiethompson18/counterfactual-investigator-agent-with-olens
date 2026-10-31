# Standard probe protocol: nested two-fold and two read positions

**Status (2026-10-31):** ACTIVE. Agreed execution protocol; results pending. This supersedes older
probe recipes for standardized comparisons, without invalidating or deleting their historical
results.

## Scope and labels

Five training families: essay grading, hinted MCQ, SuperGPQA, MMLU-Pro, and SuperGPQA + MMLU-Pro.
The three unhinted MCQ families are each evaluated on both benchmarks (six cells). Every family
reports both read positions separately. The execution budget is 90 minutes from kickoff; reuse
existing responses and activations where compatible, parallelize extraction, and record unfinished
work honestly rather than silently changing the protocol.

Keep the existing MCQ consistency rule: at least seven of eight valid sampled answers agree, with
strict bare-letter parsing as used by the current evals. Do not select on Qwen being wrong or on
DeepSeek disagreeing with Qwen. Earlier chat explanations incorrectly called this eight-of-eight.
MCQ labels are Qwen's modal answer, not benchmark truth. Correct/wrong are reporting subgroups only.
Hinted MCQ predicts the no-hint modal answer from a hinted prompt; do not mix hint placement
variants or silently reuse a wrong-only/disagreement-selected cohort as the standard population.

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
