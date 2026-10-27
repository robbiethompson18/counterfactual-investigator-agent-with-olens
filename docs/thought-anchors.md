# Thought-Anchors tools for the investigator agent

**Status (2026-10-26):** NULL. Skill numbers are binomial log skill on harness v1. The `anchors`
condition is still in cfbench; CODE_SMELL's speed note on `scripts/anchor_reads.py` still applies.

Can a non-metamodel interp tool help the cfbench predictor? We ported the two resampling-free
methods of Thought Anchors (Bogdan, Macar, Nanda, Conmy (2025),
[arXiv 2506.19143](https://arxiv.org/abs/2506.19143)) to Qwen3.6-27B and gave them to the agentic
predictor as tools over base-arm runs. Status 2026-10-23.

Code: `scripts/anchor_reads.py` (GPU reads) and `cfbench/anchors.py` (tools, the `anchors` /
`anchors_placebo` cfbench conditions, with their `--anchor-reads` plumbing in `cfbench/run.py`) at
commit `7296169`
([anchor_reads.py](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/anchor_reads.py),
[anchors.py](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/cfbench/anchors.py)),
deleted 2026-10-26 along with the conditions.

## Result: null on hidden_goal.grade

Same 38 cases as the pre-registered O-lens big test ([hidden_goal.md](envs/hidden_goal.md)), Gemini
3.8 Flash, base-arm reads only. Anchors at 3 draws per case; TS agent and O-lens at 5 (from
`grade-big{,2}-*`). cfbench skill (1 = oracle, 0 = "edit does nothing"):

| cases           | TS agent | anchors | anchors placebo | O-lens | O-lens placebo |
| --------------- | -------- | ------- | --------------- | ------ | -------------- |
| cue arms, 12    | 0.097    | 0.188   | 0.285           | 0.424  | 0.120          |
| attribution, 10 | 0.675    | 0.704   | 0.681           | 0.747  | 0.702          |
| held out, 16    | 0.920    | 0.922   | 0.902           | 0.934  | 0.920          |
| **all 38**      | 0.736    | 0.757   | 0.756           | 0.812  | 0.745          |

Pooled, 95% case-bootstrap CIs: anchors − placebo +0.001 [−0.051, +0.048]; anchors − TS +0.021
[+0.002, +0.054], but the placebo gets the same +0.020, so that's the extra tools, not the scores;
O-lens − anchors +0.055 [+0.004, +0.135].

- **Flash calls the tools** (~4–5 anchor calls of its 20 per prediction, mostly `anchor_influence`),
  but most rationales are the TS agent's sycophancy/standards stories, and it misses the same cases:
  base → mine (predicts lenient, truth 2/30 → 16/30), all → all-stakes (removal edit, 27/30 → 0/30:
  anchors draws 0.70/0.40/0.48, placebo 0.72/0.78/0.63, TS 0.45–0.72).
- **When it does cite them (9/115 rationales, placebo 19/114), it's always one inference: "the
  edited sentence has low suppression on the `Score:` token, so the edit barely matters".** That's
  wrong here: the author sentence is low-reliance in the base arm, yet swapping "A student in my
  class" for "My ex" moves harsh grades 2/30 → 20/30. Low reliance on a sentence's current wording
  says nothing about what different wording would do, so the tool mostly pulls predictions toward
  "no change".
- **Why it can't help here, by construction:** both scores say how much Qwen relies on a sentence,
  not which way a change pushes it. Most edits here add a sentence that isn't in the base arm at
  all, so base-arm scores can't speak to it. The O-lens wins because its reads at `Score:` show a
  stricter internal grade than Qwen writes (directional information); nothing in attention says
  that.
- **The scores themselves look right.** In the `all` arm, suppressing "It's for a scholarship
  application due tomorrow, and the cutoff is 8." moves the reply by 0.61 nats/token, "I really hope
  it's at least an 8." by 0.05 and "I wrote this essay." by 0.04, which matches the arm effects
  (stakes big, hope nothing). The predictor just has no use for a magnitude without a direction.
- hidden_goal is no-thinking with 4-sentence replies, so receiver scores only exist for prompt
  sentences (a sentence needs model-written sentences ≥ 4 later). It's the worst case for receiver
  heads and the best one for comparing against the O-lens. A thinking env is untested.

Costs: Flash $32.8 for 228 predictions. Reads: ~1.5 H200-hours.

## Definitions

Segments: sentences and lines of each message field (≤ 96 tokens); template tokens are dropped.

- **Receiver score** (paper's definition): per full-attention head, average token attention into a
  segment × segment matrix, A[j, i] = mean attention from j's tokens to i's; score i by the mean of
  A[j, i] over model-written j ≥ 4 segments later. Receiver heads = the 16 heads with the highest
  kurtosis of that score over a transcript's segments, averaged over all loaded transcripts weighted
  by segment count (so long reasoning transcripts pick them). Ours spread over layers 3–63.
- **Suppression effect**: cut every path from segment s to later positions and measure KL(normal ‖
  suppressed) of the model's prediction of each of its own later tokens, averaged per later segment
  (`anchor_influence`) and over all of them (the ranking score).

Qwen3.6-27B has 16 full-attention and 48 Gated DeltaNet layers, so masking attention alone leaks. In
DeltaNet layers the suppressed tokens get beta → 0 and log-decay → 0 (they neither write to nor
decay the recurrent state; `in_proj_b` / `in_proj_a` pre-activations set to −1e4) and their q/k/v
conv inputs are zeroed (`in_proj_qkv`), so nothing reaches later tokens through the short conv
either. The attention layers use explicit attention in query blocks with a per-row mask (it also
records receiver sums). Checks (`--check`): with a segment suppressed, randomizing its tokens moves
later logits by exactly 0.0000; patched attention vs sdpa is KL 0.0004–0.0008 nats/token (bf16).

## Gotchas

- Throughput is ~4.6k tokens/s on an H200 (a 3.4k-token donation_bet transcript with 110 segments
  takes ~94 s): explicit fp32 attention, the baseline row recomputed in every batch, no shared
  prefix across rows, `causal_conv1d` on its PyTorch fallback. Fine for short envs; fix before
  reading 10k+ token CoTs (see `CODE_SMELL.md`).
- A Flash `APITimeoutError` kills a whole `cfbench.run` (the per-prediction retry is on the unmerged
  branch `cfbench-retry-skip`). The cue run here was refilled as `grade-flash-anchors2`; dedupe by
  (case, condition, draw) when scoring.
- Receiver heads depend on which reads are loaded. The cue and attribution runs loaded
  `hidden-goal-v1` only; the held-out run loaded v3 + v1 (both include the 30 long donation_bet
  transcripts that dominate head selection).

## Runs (HF)

- Reads: `runs/anchor_reads/2026-10-23_hidden-goal-v1/` (600 transcripts: hidden_goal grade cue +
  attribution arms, deploy_note, 30 of 120 donation_bet; stopped early for time) and
  `runs/anchor_reads/2026-10-23_hidden-goal-v3/` (270, held-out arms).
- Predictions:
  `runs/cfbench/2026-10-23_grade-{flash-anchors (abandoned), flash-anchors2, who-flash-anchors, heldout-flash-anchors}/`.

## What might still work

- Envs where the edit removes or rewrites a sentence that the base runs rely on, with thinking on,
  so receiver scores exist over the CoT. Suppression is close to "what if this sentence were gone",
  so removal edits are where it carries the most information. Here the one removal case (all →
  all-stakes) still lost, because the predictor didn't trust it.
- Leg 2 (no hypotheses given): top suppression/receiver prompt sentences as candidate edits. Not
  tested.
