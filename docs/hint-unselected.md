# Hinted MCQ, unselected: did Qwen know it, without the DSv4 disagreement filter

**Status (2026-10-31):** REFERENCE. Done on 1,042 cases: the transcript-only investigator reads
AUROC 0.77 on its own, the O-lens beats its placebo but not the transcript, and nobody recovers
Qwen's own wrong letter.

## Why

The hinted-MCQ evals in the talk (`cfbench/evals/hint_qwen_dsv4*`, Kylie, 2026-10-28/29) keep only
MMLU-Pro and TruthfulQA questions that Qwen3.6-27B and DeepSeek V4 Flash each answer consistently (≥
7 of 8 no-hint samples) **and differently**. That filter makes the questions harder than average
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
  when at least half its hinted runs pick the hinted letter (the shown reply must not give the class
  away: 128 didn't items followed on fewer than 4 of 8 runs, 80 of them never; every knew item
  followed 8/8), then the classes are re-balanced per gold letter: **1,042 cases, 521 per class**
  (`runs/cfbench_evals/2026-10-31_hint-unselected-inline-inputs`). The investigator sees the hinted
  prompt and one following run's reply (seeded).
- **Reads:** O-lens (space lens + AR, T 1) at every default read position
  ([read-positions.md](read-positions.md)): every punctuation token of the prompt, each option's
  newline, the ten tail tokens of the thinking-off render; layers 32/40/48/52/56/60; 43 positions
  per prompt on average. Shown as the menus_inline `prompt` menu (no tail tokens; the canonical menu
  since 2026-10-31 17:20 PDT, so the lens can't read off the answer Qwen is about to give) and, as a
  secondary pair, the `all` menu (with the tail tokens)
  (`runs/olens_reads/2026-10-31_hint-unselected-{0..3}`, four reader pods). Placebo per
  [placebo-rules.md](placebo-rules.md): another case with the same option count and category, reads
  aligned by kind.
- **Investigator:** Haiku 5.5 at effort medium, one turn, no tools (`transcript_inline`,
  `olens_{prompt,all}_inline`, `olens_{prompt,all}_placebo_inline`), 3 draws (1 for the `all` pair),
  every draw one prediction ([cfbench-scoring.md](cfbench-scoring.md)).

Not comparable number-for-number with the talk's selected-cohort bars: different cohort, read
positions (option newlines + 6 tail tokens there, layers 32-52), investigator (Gemini 3.8 Flash at
effort low there) and scoring (draws averaged per item there).

## Results

![AUROC of P(knew) and % correct on the letter task](img/hint-unselected.png)

Chart: `uv run python -m graphs.hint_unselected` → `docs/img/hint-unselected.png`. The write-up's
and talk's two-cohort grid (this cohort beside the DSv4-disagreement one, hint first, `prompt` menu
only) is `graphs.talk` `grid` → `docs/img/hinted-mcq-grid.png`, the two selection Sankeys
`graphs.data_filter_sankey` (`hint_selected`, `hint_unselected`). Runs:
`runs/cfbench/2026-10-31_hint-unselected-d` (claim, prompt menu + transcript, 3 draws), `-all-d`
(claim, `all` menu, 1 draw), `-letter-d` (letter, prompt menu + transcript, 3 draws),
`-letter-all-d` (letter, `all` menu, 1 draw; **482 of 1,042 cases**: the Anthropic account ran out
of credits mid-batch, so that pair is a first-half subset and is not comparable to the others).
Every draw one prediction; 5 of 9,378 draws missing on each 3-draw run (400s at the credit cliff).
Scorers: `cfbench.evals.hint_unselected_inline.score`,
`cfbench.evals.hint_unselected_letter_inline.score`.

**Did Qwen know it? (AUROC of P(knew), 95% bootstrap CI over cases)**

| condition                     | cases | AUROC | CI          | % right |
| ----------------------------- | ----- | ----- | ----------- | ------- |
| transcript only               | 1,042 | 0.769 | 0.743–0.795 | 62.4    |
| O-lens, prompt reads          | 1,040 | 0.750 | 0.724–0.778 | 64.0    |
| O-lens placebo, prompt reads  | 1,042 | 0.740 | 0.711–0.766 | 61.8    |
| O-lens, prompt + tail reads   | 1,039 | 0.763 | 0.733–0.790 | 63.7    |
| O-lens placebo, prompt + tail | 1,041 | 0.738 | 0.708–0.769 | 63.8    |

Paired bootstrap over cases (one-sided p that the first beats the second):

| comparison                             | ΔAUROC | CI             | p     |
| -------------------------------------- | ------ | -------------- | ----- |
| O-lens prompt − its placebo            | +0.011 | −0.005, +0.027 | 0.093 |
| O-lens prompt − transcript only        | −0.019 | −0.035, −0.003 | 0.99  |
| O-lens prompt + tail − its placebo     | +0.026 | +0.005, +0.047 | 0.007 |
| O-lens prompt + tail − transcript only | −0.006 | −0.025, +0.012 | 0.75  |

**Which letter without the hint? (% correct; always the hinted letter = 50%, chance 10%)**

| condition                     | cases | % right | CI        | knew | didn't |
| ----------------------------- | ----- | ------- | --------- | ---- | ------ |
| transcript only               | 1,041 | 54.4    | 51.6–57.4 | 97.9 | 11.1   |
| O-lens, prompt reads          | 1,040 | 54.7    | 51.8–57.6 | 98.3 | 11.3   |
| O-lens placebo, prompt reads  | 1,042 | 53.5    | 50.6–56.5 | 97.7 | 9.4    |
| O-lens, prompt + tail reads   | 482   | 60.8    | 56.4–65.1 | 98.9 | 13.9   |
| O-lens placebo, prompt + tail | 482   | 60.2    | 56.0–64.3 | 98.9 | 12.5   |

O-lens prompt − placebo +1.3pp (CI +0.5, +2.2; p = 0.002); − transcript +0.4pp (p = 0.22). The
prompt + tail pair is the 482-case subset; its two bars differ by +0.6pp (p = 0.28).

## Reading

- **The transcript-only baseline is strong once the cohort isn't pre-filtered.** AUROC 0.77 here
  against 0.615 on the DSv4-disagreement cohort (the talk's hint-first panel, Gemini Flash at effort
  low). On an unselected set the investigator can tell an easy question from a hard one, and "knew"
  is mostly "easy". The disagreement filter had removed exactly that signal, which made the lens's
  +0.07 over transcript-only there look like lens information.
- **The O-lens adds nothing over the transcript** on either task (−0.019 and −0.006 AUROC; +0.4pp on
  the letter). Its edge over the placebo (+0.011 n.s. on prompt reads, +0.026 with the tail tokens;
  +1.3pp on the letter) is the prompt-inversion pattern: reads that paraphrase this question beat
  reads that paraphrase another question, and the placebo's wrong-question bullets cost the
  investigator a little.
- **Nobody recovers Qwen's own wrong letter**: 9–14% on the didn't half against 10% chance, with or
  without the lens, with or without the tail tokens. This matches the talk's letter panel (52–55%)
  and mcq_unselected's lens numbers on Qwen-wrong cases (31–39% there with no hint, where the
  investigator at least sees the question without a planted answer).
- **Caveat on the tail tokens.** The `all` menu is the one pair that beats its placebo at p < 0.01
  on the claim task; that is the menu the canonical rule dropped the same afternoon because the tail
  tokens sit where Qwen is about to answer. With the hint first, the answer about to be given is the
  hinted letter in both classes, so what the tail reads carry here is not the answer but how settled
  it is; still, it does not beat the transcript.

Cost: ~$35 of H200 (rollouts + reads), ~$25 of Haiku (batch API). Wall time 2 h 45 min from go.
