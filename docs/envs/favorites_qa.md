# favorites_qa: predicting Qwen's own favourites from the question

**Status (2026-10-29):** ACTIVE. Bank v3 (5,187 questions, 1,500 stable under Qwen) is the env's
default; v2 (2,742, 790 stable) and the pilot v1 (290, 62 stable) are kept. Results on all 1,500
(O-lens and NLA +7pp over transcript-only's 80%, J++ +0) are in
[docs/favorites-qa-inline.md](../favorites-qa-inline.md).

WorkspaceBench's `basic_readout` family has 32 `implicit` items ("What is your favorite card game?
Answer in one word." → poker), labelled with Qwen3.6-27B's own gated answer. Its text readers get
about half of them (Sonnet as Qwen 0.54) and the lens 0.94 at layer 60
([wsb-flash-prompt-only.md](../wsb-flash-prompt-only.md)). 32 items is too few for a paired test, so
this env farms more of them the way [cutoff_qa](cutoff_qa.md) farmed factual questions. There's no
truth to verify, so the only filter is Qwen's own stability.

## How the bank is made

1. Six Sonnet subagents in Claude Code each write 50 questions in one domain from
   [`scripts/favorites/BRIEF.md`](../../scripts/favorites/BRIEF.md): food and drink, animals and
   nature, arts, places, sports and games, science and everyday things. Free response only, no
   option lists. Each category is asked once, in one of seven shapes: `favorite`, `least_favorite`,
   `pick_one`, `recommend`, `be_one`, `name_one`, `choose_for`. Categories are mostly ordinary and
   about a quarter mildly specific. No numeric answers, and none of WorkspaceBench's 32.
2. `scripts/favorites/bank.py merge` validates the files and drops categories two writers both used
   (10 of 300).
3. The gate is `scripts/cutoff_gate.py`, unchanged except that questions without a truth are classed
   `stable` instead of right/wrong: 8 samples at T 1 plus greedy, thinking off, no system prompt, on
   the shared server. Stable means at least 7 of 8 agree.
4. `bank.py build RUN --upload` attaches the gate and writes HF `banks/favorites_qa/v1.json`.
   `qwen_modal` is a raw sample whose gate normal form is the mode. Consumers normalize it with
   their own norm, because the gate's normalize and cutoff_qa's `norm` disagree on apostrophes
   ("chef s knife" vs "chefs knife").

## Pilot gate (`cutoff_gate/2026-10-29_favorites-v1-pilot290`)

| shape          | n   | stable | unstable | refusal |
| -------------- | --- | ------ | -------- | ------- |
| favorite       | 47  | 7      | 39       | 1       |
| least_favorite | 40  | 2      | 25       | 13      |
| pick_one       | 39  | 8      | 31       | 0       |
| recommend      | 41  | 10     | 31       | 0       |
| be_one         | 40  | 4      | 36       | 0       |
| name_one       | 39  | 19     | 20       | 0       |
| choose_for     | 44  | 12     | 32       | 0       |
| all            | 290 | 62     | 214      | 14      |

- **The yield is 21%, lower than WorkspaceBench's.** Its 32 are generic categories (color → blue).
  Ours are more specific, and at T 1 Qwen spreads over the top few answers (Bach 4, Beethoven 3,
  Mozart 1). At ≥ 6/8 the pilot would keep 107. Part of the spread is the strict string match
  ("mount kilimanjaro" vs "kilimanjaro", "nuclear" vs "nuclear energy").
- **`least_favorite` draws refusals** ("As an AI, I do not have personal preferences"): 13 of 40.
- **`name_one` is mostly typicality** ("Name a swimming stroke"): the most stable shape, and the
  easiest for a text reader (below).

## Bank v2: wave 1 (`cutoff_gate/2026-10-29_favorites-v2-2742`)

50 Sonnet subagents, one narrow domain each (fruits and vegetables, national dishes, wild mammals,
cities by country, classical music, Pixar/Disney/Ghibli, fictional universes, ...), 50 questions
each, from brief v2: no `least_favorite` or `be_one`, at most 4 `name_one`, subcategories
("Argentine dish", "Mario enemy") encouraged, the pilot's 322 categories as an avoid-list. Merged
with the pilot's 290 (58 categories that two writers both used dropped): 2,742 questions, all gated
together, so v2 re-gates the pilot items (61 stable this time vs 62).

| shape          | n     | stable | %   |
| -------------- | ----- | ------ | --- |
| name_one       | 234   | 98     | 42% |
| pick_one       | 634   | 207    | 33% |
| recommend      | 575   | 180    | 31% |
| choose_for     | 589   | 163    | 28% |
| favorite       | 630   | 135    | 21% |
| be_one         | 40    | 5      | 12% |
| least_favorite | 40    | 2      | 5%  |
| all            | 2,742 | 790    | 29% |

- **Wave 1 yields 30% (729 of 2,452)**, up from the pilot's 21%: the shape mix and the subcategories
  both help. Per writer it ranges 3/50 (individual sports) to 30/50 (national dishes).
- Only 13 refusals in all, 11 of them the pilot's `least_favorite` items.
- Two w50 questions were reworded by their writer after the merge; the bank has the wording that was
  gated.
- Reaching 1,000 stable takes ~700 more questions at this yield (one ~15-agent wave).

## Investigator (`cfbench/evals/favorites_qa_inline/`)

cutoff_qa_inline's setup on these items: Claude Haiku 5.5 at effort medium sees only the question,
one turn, one `submit_answer`, and the model is never named. The conditions are `transcript_inline`,
`olens_inline` (the item's own end-of-prompt reads) and `olens_placebo_inline` (another item's
reads). The reads are the space O-lens plus AR at the six end-of-prompt tokens (".", `<|im_end|>`,
"\n", `<|im_start|>`, "assistant", "\n"), layers 32/40/48/52/56/60. Qwen's thinking-off render then
has `<think>\n\n</think>\n\n` before the reply, so the last read is 4 tokens before the answer, not
at it. That holds for cutoff_qa_inline's reads too.

Transcript-only, 62 items × 3 draws (`cfbench/2026-10-29_favorites-haiku-pilot62-ts`):

| shape          | items | exact match (loose)  |
| -------------- | ----- | -------------------- |
| name_one       | 19    | 95%                  |
| pick_one       | 8     | 88%                  |
| favorite       | 7     | 76%                  |
| recommend      | 10    | 70%                  |
| choose_for     | 12    | 56%                  |
| be_one         | 4     | 50%                  |
| least_favorite | 2     | 100%                 |
| all            | 62    | 74% exact, 77% loose |

That compares with 67% on cutoff_qa_inline's 200. The misses are Haiku's prior, not Qwen's: Tolstoy
for Dostoevsky, Rear Window for Psycho, vim for nano, chess for Catan, Empire State Building for
Burj Khalifa. Those 14 or so items are the lens's headroom, which is too few for a paired test at
pilot size.

## Next

- See [docs/favorites-qa-inline.md](../favorites-qa-inline.md) for the investigator results on all
  1,500 stable v3 items and their caveats.
