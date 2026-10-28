# MCQ answer probe: Qwen's own answer from the option tokens

**Status (2026-10-28):** ACTIVE. MMLU-Pro in-sample, TruthfulQA out-of-sample done; O-lens reads at
the same tokens on both.

Kylie's question: on multiple-choice questions Qwen3.6-27B answers consistently, does a linear probe
on the activations at each option's tokens say which option Qwen will pick, and does it track Qwen's
answer or the truth? It tracks Qwen's answer, sharply, before Qwen has written anything, and the
frozen probe transfers to TruthfulQA (top-1 = Qwen's answer 0.885 out-of-sample, the truth 0.14 on
questions Qwen gets wrong). The O-lens at the same tokens points the same way but is a much weaker
readout.

Page with every question, the probe's per-option scores and pmf, Qwen's letter log-probs and the
lens reads (MMLU-Pro / TruthfulQA switch, private artifact):
[claude.ai/artifact/LdBx9nsbRZK2AusLDiGK6o](https://claude.ai/artifact/LdBx9nsbRZK2AusLDiGK6o).

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
  top-1 is ~0.1 (≤ 10 options). Layer L here is HF `hidden_states[L]`, the output of decoder block L
  − 1.

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
question's options of w·z_i (z = per-dimension standardised h; + an unpenalised per-letter bias
b_letter in the `+letter` variants), loss = mean over questions of −log p(target) + (λ/2)‖w‖²,
full-batch L-BFGS, λ chosen by an inner GroupKFold on the first outer fold's training questions (1.0
for every variant, interior of 0.01–100). Same outer folds. Targets: Qwen's modal answer (one-hot)
or Qwen's own letter pmf (its logits renormalised over the question's letters). Columns: top-1 =
answer / gold, NLL of Qwen's answer, KL(Qwen pmf ‖ probe pmf), mean top-option probability, ECE of
the top-option probability against "is Qwen's answer" (10 bins).

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
- **The chosen probe** (frozen for TruthfulQA): the one-hot listwise probe, diagonal
  standardisation, `nl`, `hidden_states[52]`, λ = 1. ECE 0.04, NLL 0.30 vs Qwen's 0.16, top-1 ~9
  points below Qwen's own.

### Whitening (diag kept)

`whiten` reruns the one-hot listwise probe at `nl` L52 with nested CV (outer 5, inner 4 by
question), choosing the whitening hyperparameter and λ per outer fold (`whiten.md` in the run). Full
whitening needs shrinkage: ~5.3k training options vs d = 5,120.

| whitening                                          | chosen (param, λ)  | top-1 all / right / wrong | NLL   | KL    |
| -------------------------------------------------- | ------------------ | ------------------------- | ----- | ----- |
| diag (per-dimension std)                           | λ 1 (one fold 0.1) | 0.894 / 0.958 / 0.830     | 0.303 | 0.294 |
| within-question centred + diag                     | λ 1                | 0.894 / 0.958 / 0.830     | 0.298 | 0.271 |
| PCA-whitened, top k                                | k 64, λ 0.01       | 0.882 / 0.955 / 0.808     | 0.338 | 0.302 |
| shrinkage (1−α)Σ + α·mean(eig)·I (Ledoit-Wolf / α) | α 0.3, λ 0.01      | 0.847 / 0.922 / 0.772     | 0.472 | 0.402 |
| within-question centred + shrinkage                | α 0.3, λ 0.01      | 0.851 / 0.925 / 0.777     | 0.491 | 0.421 |

Full whitening hurts, and both whitening families pick the least-whitening end of their grids
(isotropic L2 in whitened space makes low-variance noise directions cheap). Within-question centring
only nudges NLL. Diag is kept; the layer check (L44–56) puts L50 and L52 within noise.

## O-lens reads at the same tokens

agu18dec's space lens (`olens_s3d_space_rl600`) with the one-engine reader at each option's `nl`
token, blocks 20, 24, …, 60 (the blocks the lens and its whiteners exist for; the probe's block 51
has none, block 52 is nearest), sampling of record (T 1.0, up to 4 bullets), per-bullet and joint
whitened FVE: 72,853 reads on MMLU-Pro. Sanity check: the reader's capture at block 51 matches the
saved HF `hidden_states[52]` at those positions, cosine 0.986–0.9999 (median ~0.9995) over 30
options, so positions and layer indexing agree.

`lens-stats` (`lens_stats.md` in each run) is a crude regex reading: ENDORSE = "correct
answer/term", "is correct", "the answer is", "best fits", "most accurate", …; REJECT = "incorrect",
"is wrong", "not correct", "distractor". Share of options whose read matches, MMLU-Pro:

| block | endorse: own (R) | own (W) | gold (W) | other | reject: own (R) | own (W) | gold (W) | other     |
| ----- | ---------------- | ------- | -------- | ----- | --------------- | ------- | -------- | --------- |
| 24    | 0.38             | 0.37    | 0.36     | 0.33  | 0.11            | 0.11    | 0.09     | 0.14–0.17 |
| 32    | 0.48             | 0.37    | 0.25     | ~0.27 | 0.10            | 0.13    | 0.20     | ~0.27     |
| 40    | 0.55             | 0.40    | 0.23     | 0.19  | 0.06            | 0.08    | 0.19     | 0.25–0.28 |
| 52    | 0.51             | 0.43    | 0.21     | ~0.20 | 0.09            | 0.14    | 0.39     | ~0.49     |
| 56    | 0.58             | 0.53    | 0.31     | ~0.25 | 0.14            | 0.19    | 0.39     | ~0.53     |

- The lens separates Qwen's option from block 32 on; before that every option reads alike. On wrong
  questions the gold reads like a rejected option (slightly less rejected than other wrong ones).
- Qwen's option is endorsed less often when Qwen is wrong, but that is mostly confidence: binned by
  the probe's held-out p(Qwen's answer), block-52 endorse is 0.32 / 0.35 (right / wrong) in the
  bottom quintile and 0.66 / 0.64 in the top.
- Lens-only pick (most endorse − reject matches at block 52, ties → no pick): picks on 58% of
  questions; = Qwen's answer 0.31 (right q) / 0.23 (wrong q), = gold 0.07 on wrong q. The probe:
  0.96 / 0.83. The regex over bullets is a weak readout of what the probe reads linearly.

## TruthfulQA out-of-sample

Run with Kylie's go-ahead after the design above was frozen; nothing was tuned on it.

- **Questions.** hinted_truthfulqa `control` (no hint) runs `2026-10-27_control-n8-qwen36-nothink`
  and `…-q400-816-…` (TruthfulQA MC1, Qwen bf16 vLLM, 8 samples): 740 of 817 with a modal answer in
  ≥ 7/8 samples, 640 right / 100 wrong, kept unbalanced; 3,706 options, up to 13 per question
  (chance top-1 0.228).
- **Prompt.** `build("control", item)`: its system prompt and "(A) …" options, thinking-off
  template. The last option is followed by `<|im_end|>`, not a newline, and Qwen's template trims
  message content, so a "\n\n" appended to the message is dropped. `tqa-prep` inserts the "\n\n"
  token into the token stream after the last option (a single token, before `<|im_end|>`), so every
  option has a newline site as in hint_mcq. The unedited prompt is scored too (`_orig`). Qwen's
  replies open with "(X)" and "(X" is one token, so its letter pmf is over "(A", "(B", … at the
  reply's start; its argmax matches the sampled modal answer on 99.7% (edited) / 99.9% (original) of
  questions.
- **Probe.** The chosen listwise probe and the binary is_answer probe, each fit once on all 718
  MMLU-Pro questions (`oos`), applied unchanged.

| prompt / site                        | top-1 ans: all / right / wrong | top-1 gold (wrong q) | NLL                   | KL    | ECE   | binary AUROC |
| ------------------------------------ | ------------------------------ | -------------------- | --------------------- | ----- | ----- | ------------ | ----- | ----- |
| uniform                              | 0.247                          | 0.240                | 1.547                 | 1.395 | 0.022 |              |
| edited, `nl` (main)                  | 0.885 / 0.920 / 0.660          | 0.140                | 0.306                 | 0.225 | 0.030 | 0.953        |
| original, `nl` (last option at `<    | im_end                         | >`)                  | 0.893 / 0.930 / 0.660 | 0.110 | 0.307 | 0.221        | 0.040 | 0.955 |
| last option at its text's last token | 0.908 / 0.938 / 0.720          | 0.100                | 0.254                 | 0.182 | 0.028 | 0.953        |
| binary probe softmaxed, edited `nl`  | 0.858 / 0.894 / 0.630          | 0.130                | 0.420                 | 0.359 | 0.057 |              |
| Qwen's own letter pmf                | 0.997                          | 0.000                | 0.060                 | 0     | 0.046 |              |

- Overall the frozen probe matches its MMLU-Pro held-out level (top-1 0.885 vs 0.897, NLL 0.306 vs
  0.303) on a different dataset, prompt format and option count; the drop is on wrong questions
  (0.66 vs 0.83, n = 100, lower Qwen confidence). The inserted "\n\n" neither helps nor hurts.
- **O-lens on TruthfulQA** (40,766 reads, same settings, at the edited prompt's `nl` tokens): same
  direction, weaker, peaking earlier. Endorse at Qwen's option 0.71 (R) / 0.45–0.50 (W) vs gold (W)
  0.33–0.43 and others ~0.42 at blocks 32–36; at block 52 reject is 0.08–0.09 at Qwen's option, 0.15
  at gold (W), ~0.21 elsewhere. Lens-only pick: = Qwen's answer 0.27 (R) / 0.15 (W), gold 0.08 (W).
  It never favours the gold over Qwen's answer.

## Where things live

- Code: [`environments/hint_mcq/probe.py`](../environments/hint_mcq/probe.py) (`collect`, `fit`,
  `pmf`, `whiten`, `dump`, `lens-requests`, `lens-stats`, `tqa-prep`, `oos`);
  [`scripts/hs_at_positions.py`](../scripts/hs_at_positions.py) (HF forward pass over pre-tokenized
  prompts, any GPU venv); [`scripts/olens_capture_check.py`](../scripts/olens_capture_check.py) (the
  reader's capture vs saved activations); the reader's `char_positions`, explicit `positions` and
  `prompt_ids` request fields
  ([`scripts/olens_read_requests.py`](../scripts/olens_read_requests.py)).
- HF runs (`runs/mcq_answer_probe/`):
  [`2026-10-28_scale2k`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/mcq_answer_probe/2026-10-28_scale2k)
  (selection, options, results, pmf/whiten/lens_stats tables, `pmf_options.jsonl` per-question
  values),
  [`2026-10-28_olens-nl`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/mcq_answer_probe/2026-10-28_olens-nl)
  (MMLU-Pro lens reads, expview), and
  [`2026-10-28_tqa-oos`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/runs/mcq_answer_probe/2026-10-28_tqa-oos)
  (TruthfulQA selection, Qwen letter log-probs for both prompts, `oos*.md`, lens reads).
- Activations are not on HF: `data/probe_cache/<run>/` (`nl.npy`, `end.npy`: [options, 33 layers,
  5120] fp16; 4.2 GB MMLU-Pro, 2 × 2.5 GB TruthfulQA) on the machine that ran them (Kylie's bramble
  for these runs). Regenerating takes ~15 min each on one H100/H200.
- Cost: ~$13 of RunPod via SkyPilot (H100 for collection, H200 for the reads and TruthfulQA).

```bash
# MMLU-Pro (GPU, then CPU)
uv run --with torch --with transformers --with accelerate python -m environments.hint_mcq.probe collect \
    --runs 2026-10-27_scale2k 2026-10-27_scale2k-topup 2026-10-26_or_nothink2 --slug scale2k
uv run --with scikit-learn python -m environments.hint_mcq.probe fit data/runs/mcq_answer_probe/<run> --finish
uv run --with scikit-learn python -m environments.hint_mcq.probe pmf data/runs/mcq_answer_probe/<run>
uv run --with scikit-learn python -m environments.hint_mcq.probe dump data/runs/mcq_answer_probe/<run>   # per-question values
# O-lens requests at the nl tokens (+ the capture check's reference), then olens_read_requests.py on a GPU
uv run python -m environments.hint_mcq.probe lens-requests data/runs/mcq_answer_probe/<run> data/lens_req/requests.jsonl
uv run python -m environments.hint_mcq.probe lens-stats data/runs/mcq_answer_probe/<run> READS.jsonl ...
# TruthfulQA: prep (CPU), hs_at_positions.py on a GPU into data/probe_cache/<tqa run>[_orig]/, then the frozen probe
uv run --with transformers python -m environments.hint_mcq.probe tqa-prep
python scripts/hs_at_positions.py data/tqa_oos/hs_inputs.jsonl OUT_DIR
uv run --with scikit-learn python -m environments.hint_mcq.probe oos data/runs/mcq_answer_probe/<mmlu run> data/runs/mcq_answer_probe/<tqa run> [--acts <tqa run>_orig]
```

HF transformers falls back to the slow reference kernel for Qwen3.6's linear attention without
`flash-linear-attention`; it is still ~50 questions/min on an H100. The one-engine reader's venv
needs `accelerate` added (`--no-deps`) for `hs_at_positions.py`. RunPod H200 capacity was scarce
(most regions empty; a Japan box downloaded weights at ~5 MB/s and was dropped).
