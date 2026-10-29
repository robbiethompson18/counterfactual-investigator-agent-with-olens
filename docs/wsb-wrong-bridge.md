# WorkspaceBench wrong_bridge: O-lens vs Sonnet text inversion, with and without the answer

**Status (2026-10-28):** REFERENCE. One lens run (space lens, T 1, 3 draws) and two Sonnet 5.5 text
arms on the 100-item bank, all judged by Sonnet 5.5 with WSB's own bank judge.

## The family

[WorkspaceBench](https://github.com/camilablank/workspace-bench) (`evals/wrong_bridge`, pinned
`86612fa`): a plain-text two-hop prompt such as `Fact: The composer of Taxman was born in the year`.
Qwen3.6-27B resolves hop 1 to the wrong entity (John Lennon, not George Harrison) and writes
hop2(wrong bridge) = 1940. The scored label is the wrong bridge; the twin family `wrong_bridge_true`
scores the same readouts against the true bridge. The label comes from Qwen itself: a hop-1 probe
("The composer of Taxman is") names it, and the bank keeps an item only when splicing the bridge
into the prompt flips the answer (details in the family README). The family was added after the
text-inversion critique and has no published lens number. WSB's own text gate (Opus 5.5, prompt
only, one cell at L44, Opus judge) names the wrong bridge on 0.14 of draws and the true bridge on
0.98.

## What we ran

- **Lens:** `olens_s3d_space_rl600` + AR on the one-engine reader (`scripts/olens_read_requests.py`,
  plain render via `text`, `positions: "last"`), final token, layers 20–60 step 4, 3 draws per item
  at T 1. Reads: `data/runs/wsb_wrong_bridge_lens/2026-10-28_olens-space-k3-T1/` (expview
  `#dataset=runs/wsb_wrong_bridge_lens/2026-10-28_olens-space-k3-T1`). One H200, 3,300 reads in 4
  min.
- **Sonnet prompt-only:** WSB's anti-text-inversion gate replayed with Sonnet 5.5 as answerer
  (`open_model` framing, byte-identical template), 3 draws.
- **Sonnet with answer (ours, not a WSB number):** Sonnet 5.5 also sees Qwen's completion ("1940")
  and is asked which entity the model resolved hop 1 to.
- **Judge for everything:** WSB's `bank-2026-10-10` prompt with the forbidden-answer line (naming
  only "1940" does not count), on `claude-sonnet-5-5` at effort low. Lens cells are judged per
  (item, layer) with K = 1 draw (WSB's protocol) and K = 3 draws as the samples. Code:
  `wsb/wrong_bridge.py` (PR #362); runs under `data/runs/wsb_wrong_bridge/`, Docent collection
  `bf0054fc-e318-4aca-a0a3-b565816d9ae6`.

## Results (100 items)

| reader                                     | names Qwen's wrong bridge | names the true bridge |
| ------------------------------------------ | ------------------------- | --------------------- |
| O-lens, best single layer (L60), K = 1     | 0.36                      | 0.08                  |
| O-lens, any of 11 layers, K = 1            | 0.46                      | 0.16                  |
| O-lens, any of 11 layers, K = 3            | 0.48                      | 0.17                  |
| Sonnet prompt-only, pass@1 / pass@3        | 0.35 / 0.48               | 0.97 / 0.97           |
| Sonnet with Qwen's answer, pass@1 / pass@3 | 0.83 / 0.86               | 0.95 / 0.96           |
| WSB's Opus gate (their number, Opus judge) | 0.14                      | 0.98                  |

Lens per layer, K = 1, wrong bridge: L20–32 ≤ 0.04, L36–48 0.15–0.16, L52 0.24, L56 0.31, L60 0.36.

Item overlap on the wrong bridge (lens = K 1, any layer):

| vs                        | lens only | Sonnet only | both | neither |
| ------------------------- | --------- | ----------- | ---- | ------- |
| Sonnet prompt-only pass@3 | 23        | 25          | 23   | 29      |
| Sonnet with answer pass@3 | 4         | 44          | 42   | 10      |

## Reading it

- **The lens tracks Qwen's computation.** It names the wrong bridge three times as often as the true
  one (0.46 vs 0.16), rising with depth, and the true-bridge hits are mostly items where it names
  both. That is the result the family was built to show, and the first lens number on it.
- **Prompt-only Sonnet ties the lens at pass@3 (0.48 vs 0.46 / 0.48), but as a mention rate.** On
  102 of its 105 passing draws Sonnet gives the true bridge as its answer and lists the wrong one as
  a competitor; the judge credits any mention. The lens's passes are the same kind: the item passes
  if any bullet at any layer names Lennon, beside bullets naming McCartney or Beethoven. Neither
  number is "the reader's answer was B'".
- **With the answer in view, text wins outright: 0.83 pass@1, 0.86 pass@3.** Only 4 items are
  lens-only against this arm (all city hops, where the answer pins the bridge less well); 44 are
  Sonnet-only. WSB's gate cuts the text before the answer, which is fair for a lens that must read
  the bridge before it is written and irrelevant for an investigator who has the output. The 14
  items Sonnet misses on all three draws are ones where it blames hop 2 instead of hop 1 ("AC/DC is
  right, 1981 is just a wrong date").
- **Decoder prior caveat.** The lens decoder is Qwen plus a LoRA with the same wrong knowledge; a
  decoder that reconstructs "composer of Taxman ... born" from the final-token activation can name
  Lennon from its own weights. The knowledge_cutoff README concedes this for that family (stock Qwen
  given the text ties the lens). A hop-1 probe of Qwen recovers the label on 100 of 100 items by
  construction. Most lens bullets are year-guessing continuations ("1935. The year Lennon was born
  is"), not entity names, so the bridge surfaces as a side effect of predicting the year.
- **Judge noise.** Sonnet at effort low fails some bullets that name the target ("The year Lennon
  was born is" at L60 draw 0 judged not expressed; the K = 3 call on the same cell passed on "The
  month John Lennon was born is a multiple of 12"). Treat ±0.05 as noise. WSB's Opus gate (0.14) is
  lower than our Sonnet replay (0.35) for some mix of a less chatty answerer and a stricter judge.
- **Departures from WSB's instrument:** judge model Sonnet 5.5 (theirs: Gemini 3.8 Flash, Opus in
  the gate); targets include `intermediates_aliases` (their code sends `intermediates` only); the
  forbidden line is sent as the README says (their `FORBIDDEN_FAMILIES` has only `multihop`).

For the sprint doc: wrong_bridge is the family that shows a lens reading a model-specific error a
frontier text reader misses from the prompt alone, and also the family where one extra piece of text
the investigator always has (the model's output) beats the lens by 35–40 points.
