# MCQ answer probe: Qwen's own answer from the option tokens

**Status (2026-10-28):** ACTIVE. MMLU-Pro only; TruthfulQA out-of-sample test pending.

Kylie's question: on multiple-choice questions Qwen3.6-27B answers consistently, does a linear probe
on the activations at each option's tokens say which option Qwen will pick, and does it track Qwen's
answer or the truth? It tracks Qwen's answer, sharply, before Qwen has written anything.

## Setup

- **Questions.** hint_mcq no-hint samples (MMLU-Pro, thinking off, 8 samples per item) from
  `2026-10-27_scale2k`, `2026-10-27_scale2k-topup` and `2026-10-26_or_nothink2`: 2,200 items, 1,683
  with a modal answer in ≥ 7/8 samples (1,324 right, 359 wrong). Balanced 50:50 by subsampling the
  right ones: 718 questions, 6,623 options.
- **Forward pass.** Qwen/Qwen3.6-27B bf16, HF transformers, the no-hint prompt
  (`env.build("nohint", item)`) with the thinking-off chat template; residual stream at every 2nd
  layer (0, 2, …, 64). Qwen's letter logits at the answer position agree with its sampled modal
  answer on 99.9% of questions.
- **Sites, per option.** `end`, the option text's last token (a "." for 41.5% of options; every
  option ends in "." in 284/718 questions), and `nl`, the newline token after it ("\n", or "\n\n"
  for the last option; never merged with the text). The label "." ("C.") was dropped: it comes
  before the option's text, so it can only reflect earlier options.
- **Labels.** `is_answer` (= Qwen's modal answer) and `is_gold`. On wrong questions they disagree,
  which is the test.
- **Probes.** Logistic regression per (site, layer), standardised features, C = 0.01, 5-fold
  GroupKFold by question. Top-1 = the option with the highest held-out score in its question. Chance
  top-1 is ~0.1 (≤ 10 options).

## Per-layer results

Full table: `table.md` in the run; plot `probe_by_layer.png` (regenerate with `fit`).

| probe                    | AUROC is_answer | top-1 = answer (right q) | top-1 = answer (wrong q) | top-1 = gold (wrong q) |
| ------------------------ | --------------- | ------------------------ | ------------------------ | ---------------------- |
| letter only              | 0.562           | 0.13                     | 0.19                     | 0.13                   |
| `end`, L50 (best `end`)  | 0.956           | 0.866                    | 0.721                    | 0.120                  |
| `nl`, L52 (best overall) | 0.974           | 0.936                    | 0.802                    | 0.064                  |

- **Depth.** Near chance through L12, ramp L14–28, jump L30–34, plateau L46–52, slow decline after.
  `nl` is a few points above `end` from L30 on.
- **is_gold probes don't find the truth.** Trained on the gold label, on wrong questions they pick
  the gold 18–26% at any layer and Qwen's answer 45–50% at late layers.
- **Period split (`end`, L50).** All-period questions 0.887 / 0.776 (right / wrong top-1), the rest
  0.853 / 0.685: ending on a "." helps a little.

## A pmf over options

`pmf` fits a conditional logit at the best (site, layer), `nl` L52: p(i | q) = softmax over the
question's options of w·h_i (+ an unpenalised per-letter bias b_letter in the `+letter` variants),
full-batch L-BFGS, L2 chosen by an inner GroupKFold on the first outer fold's training questions
(1.0 for every variant, interior of 0.01–100). Same outer folds. Targets: Qwen's modal answer
(one-hot) or Qwen's own letter pmf (its logits renormalised over the question's letters). Columns:
top-1 = answer / gold, NLL of Qwen's answer, KL(Qwen pmf ‖ probe pmf), mean top-option probability,
ECE of the top-option probability against "is Qwen's answer" (10 bins).

| method                            | split | top-1 ans | top-1 gold | NLL   | KL    | top p | ECE   |
| --------------------------------- | ----- | --------- | ---------- | ----- | ----- | ----- | ----- |
| uniform                           | all   | 0.120     | 0.145      | 2.196 | 1.765 | 0.116 | 0.007 |
| letter only (conditional logit)   | all   | 0.162     | 0.128      | 2.191 | 1.761 | 0.162 | 0.003 |
| binary is_answer probe, softmaxed | all   | 0.868     | 0.499      | 0.406 | 0.475 | 0.900 | 0.046 |
| listwise, target answer           | all   | 0.897     | 0.514      | 0.303 | 0.253 | 0.864 | 0.040 |
| listwise, target answer           | right | 0.964     | 0.964      | 0.138 | 0.146 | 0.926 | 0.044 |
| listwise, target answer           | wrong | 0.830     | 0.064      | 0.469 | 0.361 | 0.803 | 0.057 |
| listwise +letter, target answer   | all   | 0.897     | 0.515      | 0.306 | 0.256 | 0.865 | 0.044 |
| listwise, target Qwen pmf         | all   | 0.904     | 0.508      | 0.372 | 0.185 | 0.779 | 0.126 |
| listwise, target Qwen pmf         | wrong | 0.844     | 0.053      | 0.544 | 0.241 | 0.697 | 0.148 |
| listwise +letter, target Qwen pmf | all   | 0.911     | 0.506      | 0.373 | 0.186 | 0.780 | 0.132 |
| Qwen's own letter pmf (ceiling)   | all   | 0.999     | 0.501      | 0.160 | 0.000 | 0.869 | 0.130 |

- Listwise training beats renormalising the binary probe everywhere (NLL 0.30 vs 0.41, KL halved).
- The soft target fits Qwen's pmf best (KL 0.185) and helps top-1 on wrong questions a little, but
  is flatter and as miscalibrated against the modal answer as Qwen's own pmf (ECE ≈ 0.13).
- The per-letter bias adds nothing (±0.01), matching the letter-only model being at chance.
- The best-calibrated stand-in for Qwen's answer is the one-hot listwise probe: ECE 0.04, NLL 0.30
  vs Qwen's 0.16, top-1 ~9 points below Qwen's own.

## Where things live

- Code: [`environments/hint_mcq/probe.py`](../environments/hint_mcq/probe.py) (`collect`, `fit`,
  `pmf`).
- Run (selection, options with tokens, Qwen letter logprobs, results, tables):
  [runs/mcq_answer_probe/2026-10-28_scale2k](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/mcq_answer_probe/2026-10-28_scale2k)
  on HF.
- Activations (4.2 GB, `end.npy` and `nl.npy`, [options, 33 layers, 5120] fp16) are not on HF:
  `data/probe_cache/<run>/` on the machine that ran `collect` (Kylie's bramble for this run).
  `collect` regenerates them in ~15 min on one H100.

```bash
uv run --with torch --with transformers --with accelerate python -m environments.hint_mcq.probe collect \
    --runs 2026-10-27_scale2k 2026-10-27_scale2k-topup 2026-10-26_or_nothink2 --slug scale2k
uv run --with scikit-learn python -m environments.hint_mcq.probe fit data/runs/mcq_answer_probe/<run> --finish
uv run --with scikit-learn python -m environments.hint_mcq.probe pmf data/runs/mcq_answer_probe/<run>
```

HF transformers falls back to the slow reference kernel for Qwen3.6's linear attention without
`flash-linear-attention`; it is still ~50 questions/min on an H100.

## Not yet done

- TruthfulQA (hinted_truthfulqa questions) is the out-of-sample test, held back until the probe
  design is frozen and Kylie says go.
