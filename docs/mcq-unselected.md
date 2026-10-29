# mcq_unselected: what will Qwen say? An unselected MMLU-Pro set, and the 2k/2k split

**Status (2026-10-29):** ACTIVE. Super experimental: split, reads and a placebo smoke; no full run
yet. Cases rebuilt the same day with the strict bare-letter parse: 1,251; the first build's 1,285
are superseded (see Census and cases).

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

## Batch 2 (2026-10-29 evening)

Robbie: "draw fresh". `SEED2` = 1 draws 2,000 more questions from the 8,032 in neither half (none
probe-trained), version `mmlu_pro_b2`: 1,272 consistent cases under the strict parse (1,034 right),
no overlap with batch 1 or the probe half. Reads:
`runs/olens_reads/2026-10-29_mcq-unselected-nohint-b2` (three sharded pods, six layers, over a
1,294-prompt request file written before the strict rebuild; every strict case is in it). The inputs
run with both batches is `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-b2`
(`--versions mmlu_pro` / `mmlu_pro_b2` select a batch); pool them by case, same prompt, same
harness.

## Census and cases

Qwen3.6-27B, thinking off, no hint, 8 samples per question, from the four runs that together cover
all 12,032 questions: `envs/hint_mcq/2026-10-26_or_nothink2`, `2026-10-27_scale2k`,
`2026-10-27_scale2k-topup`, `2026-10-28_all12k-nohint`, read from their labels file
`labels/measure@4f098526d544-strict.jsonl` (`LABELS` in eval.py, named, not the newest file). A
reply counts only when it is one bare letter (`hint_mcq`'s `measure()` since 2026-10-29); a worked
solution is unscored and counts against consistency. A question is a case when at least 7/8 of its
samples gave the same bare letter. No other selection.

|                                         | count       |
| --------------------------------------- | ----------- |
| eval half                               | 2,000       |
| with census answers                     | 2,000       |
| consistent (>= 7/8), i.e. cases         | 1,251       |
| of which Qwen's modal answer is correct | 1,031 (82%) |
| of which wrong                          | 220         |

So "guess the gold" scores 82% here; the interesting cases are the 220. Report overall accuracy as
the headline and the wrong-subset accuracy as the diagnostic, without selecting on it.

**The first build (1,285 cases: 1,044 right, 241 wrong) is superseded.** It read Qwen's replies with
`hint_mcq`'s old parser, the first standalone capital A-J anywhere in the reply (CODE_SMELL P1 #1).
4,785 of the 96,462 census replies (5%) are worked solutions instead of a letter, and in those the
first capital is often the article "A", the pronoun "I" or a variable. Rebuilt from the same census
with the strict parse:

|                 | first build | strict rebuild |
| --------------- | ----------- | -------------- |
| cases           | 1,285       | 1,251          |
| Qwen right      | 1,044       | 1,031          |
| Qwen wrong      | 241         | 220            |
| removed / added |             | 34 / 0         |
| letter changed  |             | 0              |

- The 34 removed (21 wrong, 13 right): 23 had no bare-letter reply at all (8 worked solutions, which
  the old parser called consistent on their first capital; 14 of the 34 old letters were A, 4 were
  I); 11 had some bare letters, fewer than 7 the same.
- None added and no letter changed, by construction: a kept case has at least 7 bare letters, which
  both parsers read the same. Every case has n = 8; 11 have one worked solution beside 7 letters.
- Over all 12,032 questions the same rule gives 7,943 consistent questions (1,709 wrong), down from
  8,129 (1,821 wrong).
- Why the share is over all samples, not the scored replies only: scored-only gives 1,306 eval-half
  cases, 54 of them on one to six bare letters (12 on a single one), and changes 3 letters. Against
  a reference that reads each worked solution's last letter, that set has 32 bad cases (wrong
  letter, or not 7/8 consistent), more than the old parser's 23; this rule has none.
- The own O-lens reads (`runs/olens_reads/2026-10-29_mcq-unselected-nohint`, requested for the first
  build) cover all 1,251 cases.

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

- `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-strict`: split + cases from the strict-parse
  census, current ("strict" is the parse; the `params.strict` flag is cfbench's, unused here). The
  split is the first build's, unchanged. `latest_inputs` picks the newest inputs run by local mtime,
  so pass `--inputs` on a machine that downloaded both.
- `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs`: SUPERSEDED, the first build (old parser);
  identical to `2026-10-29_mcq-say-inputs`, made before the eval was renamed from `mcq_say`.
- `runs/cfbench/2026-10-29_mcq-say-placebo-smoke10`: 10 cases of the first build, placebo condition
  only, 1 draw, for Robbie to review the prompt in Docent before anything else runs. 2 of its 10
  (10008, 10067) are not cases in the strict rebuild.

## Open items

- Own reads: `python -m cfbench.evals.mcq_unselected.eval requests OUT.jsonl` writes the read
  requests, but their end-of-prompt positions are incomplete (see the TODO in `requests()`): pass
  explicit token positions as Kylie's tail job did.
- The prompt wording is a first draft Robbie expects to tweak after the smoke.
