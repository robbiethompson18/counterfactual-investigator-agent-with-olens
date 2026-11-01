# Hinted MCQ, unselected: did Qwen know it, without the DSv4 disagreement filter

**Status (2026-10-31):** ACTIVE. Results pending (filled in below when the runs land).

## Why

The hinted-MCQ evals in the talk (`cfbench/evals/hint_qwen_dsv4*`, Kylie, 2026-10-28/29) keep only
MMLU-Pro and TruthfulQA questions that Qwen3.6-27B and DeepSeek V4 Flash each answer consistently
(≥ 7 of 8 no-hint samples) **and differently**. That filter makes the questions harder than average
(Qwen 56% right on them vs 68% on the bank) and imports DSv4's letter bias as a class signal
([mcq-unselected.md](mcq-unselected.md)). Robbie, 2026-10-31: redo it with no second model, at least
1k cases, clean.

## Design

Code: `cfbench/evals/hint_unselected_inline/` (the "did Qwen know it?" claim task) and
`cfbench/evals/hint_unselected_letter_inline/` (the "which letter without the hint?" answer task),
the same cases and prompt.

- **Pool** (`items.py`): every MMLU-Pro test question Qwen (thinking off) answers consistently with
  no hint (≥ 7/8 agree, the `2026-10-28_all12k-nohint` census over all 12,032), minus the 2,000
  probe-half questions of the mcq_unselected split (so a probe can be trained there later). 6,742
  questions; **didn't** (consistent answer wrong) 1,454, **knew** 5,288.
- **Hint:** hint_mcq's `before/hint+d1`: "A Stanford professor I asked thinks the answer is X." as
  the prompt's first line, X the correct letter, then the question, options and "Answer with the
  letter only." Hint first, so every read is after the hint. The counterfactual is the same prompt
  without that line. Qwen bf16 × 8 per item on the shared pod
  (`envs/hint_mcq/2026-10-31_hint-before-unselected`, 4,362 items: every didn't item plus twice as
  many knew items per gold letter).
- **Cases:** 650 per class pre-selected, balanced within each gold letter (seeded); a case is kept
  when at least half its hinted runs pick the hinted letter (the shown reply must not give the
  class away: 128 didn't items followed on fewer than 4 of 8 runs, 80 of them never; every knew
  item followed 8/8), then the classes are re-balanced per gold letter: **1,042 cases, 521 per
  class** (`runs/cfbench_evals/2026-10-31_hint-unselected-inline-inputs`). The investigator sees
  the hinted prompt and one following run's reply (seeded).
- **Reads:** O-lens (space lens + AR, T 1) at every default read position
  ([read-positions.md](read-positions.md)): every punctuation token of the prompt, each option's
  newline, the ten tail tokens of the thinking-off render; layers 32/40/48/52/56/60; 43 positions
  per prompt on average. Shown as the menus_inline `prompt` menu (no tail tokens; the canonical
  menu since 2026-10-31 17:20 PDT, so the lens can't read off the answer Qwen is about to give)
  and, as a secondary pair, the `all` menu (with the tail tokens)
  (`runs/olens_reads/2026-10-31_hint-unselected-{0..3}`, four reader pods). Placebo per
  [placebo-rules.md](placebo-rules.md): another case with the same option count and category,
  reads aligned by kind.
- **Investigator:** Haiku 5.5 at effort medium, one turn, no tools (`transcript_inline`,
  `olens_{prompt,all}_inline`, `olens_{prompt,all}_placebo_inline`), 3 draws (1 for the `all` pair), every draw one prediction
  ([cfbench-scoring.md](cfbench-scoring.md)).

Not comparable number-for-number with the talk's selected-cohort bars: different cohort, read
positions (option newlines + 6 tail tokens there, layers 32-52), investigator (Gemini 3.8 Flash at
effort low there) and scoring (draws averaged per item there).

## Results

TBD.

Chart: `uv run python -m graphs.hint_unselected` → `docs/img/hint-unselected.png`.
