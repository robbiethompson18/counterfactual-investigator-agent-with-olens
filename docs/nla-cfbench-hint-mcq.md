# NLA tools in cfbench on hint_mcq: null, and the readouts don't say what Qwen would answer

**Status (2026-10-28):** NULL. Partial smoke (88 of 200 planned predictions, 23 items), stopped when
Gemini routing broke; no signal worth the full run.

**Question.** Does giving the cfbench predictor NLA readouts (instead of O-lens bullets) help it
predict what Qwen3.6-27B answers once the hint is removed? Same task and items as the hint_mcq
scale-up in [envs/hint_cf.md](envs/hint_cf.md) (155 d1 items, hint = gold letter, followed 8/8).

## Setup

- **Reads.** Camila's multilayer NLA (`camilablank/qwen3.6-27b-multilayer-nla-rl600`) at layers
  32/40/48 on the one-engine reader ([nla-reader.md](nla-reader.md)), critic-scored FVE, on the same
  stop-token and assistant-header positions as the O-lens leg, for every rollout of every usable
  item: `runs/nla_reads/2026-10-27_hint-mcq-scale2k` on HF (`cfbench/wrong_mcq/reads.jsonl`, 784k
  rows; `cfbench/placebo/reads.jsonl`, 24k rows from favorites/opinion_mcq). One H200, 2.7 h. No FVE
  filter (`min_fve` None). Mean FVE at L32/40/48: 0.27 / 0.28 / 0.07; at assistant `<|im_end|>` it
  is negative (−0.98 / −0.55 / −0.53), so those reads are confabulation.
- **Conditions.** `nla_agentic` and `nla_placebo_agentic` (PR #304): `search_nla` / `nla_at`, the
  O-lens tools with readouts in place of bullets, FVE shown per readout. Harness v8, no bump (other
  conditions' prompts byte-identical).
- **Predictor.** Gemini 3.8 Flash, effort medium, 2 draws, on 50 of the 155 d1 items
  (`random.Random(0).sample`). Compared against the O-lens leg's own rows (draws 0–1) on the same
  items.
- **Item filter** (the O-lens leg's, after its pooling bug): drop an item when its pooled no-hint
  modal answer is under 80% of runs or equals the hint letter. Drops 271, 357, 445, 6092 here.

Runs (all on HF under `runs/cfbench/`): `2026-10-28_answer-mcq-scale2k-nla-smoke50` (64 rows) and
`-b` (23) via `gemini-direct/` with `GEMINI_API_KEY`; `-c`, `-d` (1 row), `-dA`, `-dB` abandoned. On
OpenRouter, every multi-turn Gemini call 400'd "Corrupted thought signature" (BYOK route, even after
the BYOK keys were turned off); direct Google calls kept timing out or 503ing "high demand".

## Results

On the 23 kept items that have an NLA draw:

| condition       | draws | % correct | predicts the hinted letter | correct when it doesn't |
| --------------- | ----- | --------- | -------------------------- | ----------------------- |
| transcript-only | 46    | 20%       | 80%                        | 9/9                     |
| O-lens          | 46    | 28%       | 67%                        | 13/15                   |
| O-lens placebo  | 46    | 30%       | 70%                        | 14/14                   |
| NLA             | 40    | 18%       | 82%                        | 7/7                     |
| NLA placebo     | 38    | 32%       | 68%                        | 12/12                   |

NLA − NLA placebo is −14pp with a 95% interval of roughly ±19pp: null, leaning negative. On all 46
kept items the O-lens rows are 20% / 26% / 24% (TS / O-lens / placebo), as in the O-lens leg.

**% correct measures whether the predictor dares say Qwen is wrong.** The hinted letter is the gold
answer, and the predictor names it in 67–82% of draws in every condition; when it names anything
else it is almost always right. So the score is the rate of "Qwen won't know this without the hint"
calls, and no tool moves it above placebo.

**The NLA pushes the predictor toward the hinted answer, if anything.** Every NLA draw called
`search_nla` (68/88 also `nla_at`), but the readouts describe the question, not Qwen's leaning:
genre ("criminal law multiple-choice exam question (bar/MBE style)"), the doctrine in play, the
option template, what the next token should be. That is prompt inversion. When a rationale cites the
NLA (6 of 88, all on the NLA, none on the placebo), it is as evidence that Qwen "recognizes" the
textbook answer, and all 6 predict the hinted letter and are wrong, e.g. item 1392, felony murder,
Qwen's own answer A 7/8, predicted B
([Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/79b07060-c8f3-4486-b985-daab4daa5349)),
and item 4684, Huey Long, Qwen's own answer D 8/8, predicted the hinted F
([Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/545158d8-0523-40b6-8588-23615e56f09e)).
This matches the O-lens leg's dissection: the reads are on the hinted transcript, where the hint has
already overwritten Qwen's own answer before any readable position.

## Not done

The full run (155 d1 + 62 d2, 868 predictions, ~8k Gemini requests) was not started: the smoke shows
no signal. Reading the NLA in the no-hint arm, where the O-lens leg found the own answer is
readable, is the version worth trying if anyone picks this up.
