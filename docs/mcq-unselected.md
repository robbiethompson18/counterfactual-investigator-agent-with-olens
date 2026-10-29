# mcq_unselected: what will Qwen say? An unselected MMLU-Pro set, and the 2k/2k split

**Status (2026-10-29):** ACTIVE. Super experimental: split, reads and a placebo smoke; no full run
yet.

## Why

The hint evals ([envs/hint_cf.md](envs/hint_cf.md), `cfbench/evals/hint_qwen_dsv4*`) select
questions three times over: Qwen3.6-27B consistent, DSv4 Flash consistent and disagreeing with Qwen,
and Qwen following the hint. The disagreement filter makes the questions harder than average (Qwen
56% right on them vs 68% on the bank) and imports DSv4's letter bias as a class signal: DSv4's
consistent errors sit on questions whose gold is A or B (122 and 103 of them on the 2,200 bank vs 20
at J), so an early hinted letter alone predicts "Qwen knew" with AUROC 0.70 on the filtered set, and
0.53 with no filter. Robbie, 2026-10-29: keep only the consistency filter, drop the hint, and ask
the plain question: given the prompt, what letter will Qwen say?

## The split (earmarked data)

Drawn once by `cfbench/evals/mcq_unselected/eval.py` (`SEED` 0) and written to the inputs run
`runs/cfbench_evals/2026-10-29_mcq-unselected-inputs/split.json` on HF:

- `random.Random(0).sample(items(), 4000)` over all 12,032 MMLU-Pro test questions in
  `environments/hint_mcq/env.py`'s item order (all 14 categories, not only the seven of the 2,200
  bank).
- **Eval half: 2,000 questions**, the first 2,000 of the sample that no probe has trained on.
- **Probe half: 2,000 questions**, the rest, which includes the 236 sampled questions a probe had
  already trained on. Train probes here, not on the eval half.
- Probe-trained questions, 718 in all: the MCQ answer probe's balanced set
  ([mcq-answer-probe.md](mcq-answer-probe.md), ids from `runs/mcq_answer_probe/2026-10-28_olens-nl`)
  and the refit probe's 496 train items (`runs/mcq_answer_probe/2026-10-28_eval-baseline-12k`, a
  subset of the 718).
- Overlap with earlier work is fine and expected: the sample is over the same 12,032 questions the
  hint evals drew from. Only probe training is kept off the eval half.

- **Batch 2 (2026-10-29 evening, Robbie: "draw fresh").** `SEED2` = 1 draws 2,000 more questions
  from the 8,032 in neither half (none probe-trained), version `mmlu_pro_b2`: 1,975 with census
  answers, 1,294 consistent cases (1,039 right). No overlap with batch 1 or the probe half. Reads in
  `runs/olens_reads/2026-10-29_mcq-unselected-nohint-b2` (three sharded pods, six layers). Pool the
  two batches by case: same prompt, same harness. Inputs run with both:
  `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-b2`.

## Census and cases

Qwen3.6-27B, thinking off, no hint, 8 samples per question, from the four runs that together cover
all 12,032 questions: `envs/hint_mcq/2026-10-26_or_nothink2`, `2026-10-27_scale2k`,
`2026-10-27_scale2k-topup`, `2026-10-28_all12k-nohint`, **with their label files pinned** in
`CENSUS` (eval.py). Gotcha: at 13:56 PDT on 2026-10-29 a `measure@...-strict.jsonl` relabel landed
in all four runs, and the usual `latest_labels` (newest by mtime) silently moved batch 1 from 1,285
to 1,306 cases (27 dropped, 48 added, counts like `{'B': 4}` from unparsed samples) before the pin.
A question is a case when its modal answer has at least 7/8 of the answered samples. No other
selection.

|                                         | count       |
| --------------------------------------- | ----------- |
| eval half                               | 2,000       |
| with census answers                     | 2,000       |
| consistent (>= 7/8), i.e. cases         | 1,285       |
| of which Qwen's modal answer is correct | 1,044 (81%) |
| of which wrong                          | 241         |

So "guess the gold" scores 81% here; the interesting cases are the 241. Report overall accuracy as
the headline and the wrong-subset accuracy as the diagnostic, without selecting on it.

## The task

Single turn, no tools but `submit_answer`, Gemini 3.8 Flash at effort low
(`core.INLINE_CONDITIONS`):

- `transcript_inline`: the exact no-hint prompt, the setup (8 samples, agreed >= 7 times), "predict
  the letter the model gives, which may or may not be the correct one". The subject is never named
  (`MODEL_NAME`; the runner asserts the id never reaches the predictor). No transcript exists; the
  condition name is cfbench's.
- `olens_inline`: plus O-lens reads at the newline ending each option and at the end of the prompt
  (the instruction's ".", `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`), layers
  32/40/48/52/56/60 (cutoff_qa_inline's six), every bullet with its FVE, from
  `runs/olens_reads/2026-10-29_mcq-unselected-nohint` (space lens, T 1; 32-52 in the first job,
  56/60 topped up the same day).
- `olens_placebo_inline`: the same block with another question's own reads at the same positions
  (same option count, seeded, never the question's own), from the same reads run. The two placebo
  smokes below predate the reads and drew from Kylie's hinted-prompt pools instead.

Truth is Qwen's modal letter; scored as % right (`cfbench.answer`). Bars to add later: the frozen
probe, trained on the probe half; Qwen's own letter logprob as the ceiling.

## Runs

- `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs`: split + cases (identical to the first
  build, `2026-10-29_mcq-say-inputs`, made before the eval was renamed from `mcq_say`).
- `runs/cfbench/2026-10-29_mcq-say-placebo-smoke10`: 10 cases, placebo condition only, 1 draw, for
  Robbie to review the prompt in Docent before anything else runs.

## Open items

- Own reads: `python -m cfbench.evals.mcq_unselected.eval requests OUT.jsonl` writes the read
  requests, but their end-of-prompt positions are incomplete (see the TODO in `requests()`): pass
  explicit token positions as Kylie's tail job did.
- The prompt wording is a first draft Robbie expects to tweak after the smoke.
