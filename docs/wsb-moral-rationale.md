# WorkspaceBench moral_rationale: Sonnet 5.5 replay of their text-inversion gate

**Status (2026-10-28):** REFERENCE. One full run (200 items × 3 draws); the numbers below are it.

Part of the "text inversion beats the lens" series
([wsb-flash-prompt-only.md](wsb-flash-prompt-only.md) has the family table,
[wsb-jailbreak.md](wsb-jailbreak.md) the sibling replay). Code: `wsb/moral_rationale.py`
(`wsb/CLAUDE.md`). Run: `data/runs/wsb_moral_rationale/2026-10-28_sonnet-open-model-x3/` (HF),
Docent collection
[`wsb_moral_rationale`](https://docent.transluce.org/dashboard/7d75985b-c588-4177-af80-d47fe05911a2)
(600 runs, one per draw: Sonnet's answer, then the judge call with the option list).

## The family

Qwen3.6-27B (thinking off) gets a hard ethical yes/no question with "answer with one word". WSB
reads the last 5 positions of the chat render (end of the user text + the assistant header) at 11
layers, 55 cells per item. Per cell a judge gets a 6-way MC: the gold reason, 4 reasons from **other
questions**, on 72 committed items the opposite-side reason of the same question in place of one
distractor ("contrast"), and "cannot tell from the readout". An item passes if **any of the 55
cells** picks gold. The gold is Qwen's own most frequent justification (the bank sampled Qwen 9
times per question with an explanation and tallied the reasons); 162 of 200 items commit to NO. 166
items are "committed" (one gold); 34 "deliberative" need a YES-side and a NO-side reason both to
surface. The committed/deliberative label is not derivable from the public bank (38 committed items
also have mixed answer splits and both-side reasons); it was set upstream in Camila's private
hillclimbing repo.

Their published judge rows (`agu18dec/olens_ablations` `rl.base.off_k0.own.i300`, 0.91, Opus judge,
ec-v1) pick a distractor on 0.5% of 12.9k cells, gold on 14.5%, cannot-tell on 85%: since the
distractors are about other questions, the MC reduces to "is the readout on-topic with Qwen's
stance", and 55 tries almost always land one.

## What we ran

Their anti-text-inversion gate, verbatim (`wsbench/baselines/text_inversion.py`): the answerer is
shown the whole Qwen chat render
(`<|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, the last of the
5 read positions) under the `open_model` framing and asked "what would a language model be thinking
about on this token?". Sonnet 5.5, API-default thinking, 3 independent draws. Their ec-v3 judge
(Opus 5.5, effort low, structured output) scores the answer as one readout; option lists are ported
and match their `tests/golden/moral_rationale_options_v2.json` on all 200 items. A draw passes if
the judge picks gold (deliberative: both sides, in the same answer). One policy refusal in 600
draws.

## Results

| reader                                                 | pass rate | guesses   |
| ------------------------------------------------------ | --------- | --------- |
| O-lens RL (post, Gemini judge)                         | 0.76      | any of 55 |
| Agam's published checkpoint (Opus judge, ec-v1)        | 0.905     | any of 55 |
| **Sonnet 5.5 text-only, this run**                     | **0.69**  | pass@1    |
| Sonnet 5.5 text-only, this run                         | 0.83      | pass@3    |
| Opus 5.5 text-only (their gate, `anti_text_inversion`) | 0.645     | pass@1    |
| stock Qwen text-only (`prompt_only.json`)              | 0.37      | pass@1    |
| Gemini shown only the option list (`lucky_guessing`)   | 0.26–0.38 | pass@1    |

pass@1 by draw 0.67 / 0.73 / 0.68; committed 0.70, deliberative 0.68 (pass@3: 0.82 / 0.88). Sonnet's
draws agree with Opus's gate draws on 459 of 599; 34 items fail all three Sonnet draws, 20 of them
among Opus's 36 "anti-text-inversion" items.

Judge picks over the 701 calls: gold 509, **contrast 59**, cannot-tell 128, distractor 5. The
contrast picks (27 items, 13 on all 3 draws) are Sonnet predicting the side Qwen did not take: bribe
for a child's medicine
([draw](https://docent.transluce.org/dashboard/7d75985b-c588-4177-af80-d47fe05911a2/agent_run/d416381b-1027-4085-b503-06bfff9c9a3b):
"necessity and the child's life probably winning out over the anti-bribery norm"; Qwen said False
7/9), brain-dead pregnancy support, drone strikes with civilian toll, deaf embryo selection.
Cannot-tell is mostly Sonnet naming a different reason for Qwen's side than the one Qwen gives most
often (nuclear power: Sonnet talks about waste being manageable, the gold is "construction timelines
too slow"
([draw](https://docent.transluce.org/dashboard/7d75985b-c588-4177-af80-d47fe05911a2/agent_run/ee73781c-06b7-4321-9d3b-967986ffd858))),
or token-probability talk with no reason at all (private jets). The deliberative misses are one side
missing: self-driving car
([draw](https://docent.transluce.org/dashboard/7d75985b-c588-4177-af80-d47fe05911a2/agent_run/bbeafa9f-6a1b-4a92-9320-afb62ded7375)
gives the utilitarian YES reason, not "the car's duty is to its occupants").

## Reading it

- A text reader at pass@1 (0.69) is below the lens's any-of-55 number (0.76) and above the lens's
  per-cell behaviour: the lens's best single layer is ~0.69 (their per-layer table, L36–40). At
  pass@3 the text reader (0.83) is above the post's lens number. The comparison the post makes (lens
  any-of-55 vs text pass@1) is the only one where the lens wins.
- What the text reader actually lacks is Qwen's **side** (the 59 contrast picks) and Qwen's
  **favourite reason among several for the same side** (most cannot-tells). Both are facts about
  Qwen, not about its activations; a reader told "which side will Qwen take and why" or shown a few
  of Qwen's own answers would close most of it. Not run.
- The lens's own errors are mostly the same shape (torture-bomb YES side 0/55 in the published
  rows), so the deliberative subset is where a text reader structurally wins: it lists both sides by
  habit.

Cost: ~0.14M input + 0.39M output Sonnet tokens, ~1.06M input Opus judge tokens, ~10 minutes at
concurrency 16.
