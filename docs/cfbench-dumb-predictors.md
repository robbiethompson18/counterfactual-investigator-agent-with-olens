# cfbench with dumber, non-thinking predictors (2026-10-26)

**Status (2026-10-26):** NULL. Harness v7, RMSE, space-lens reads. Thinking-off predictors don't use
the O-lens; only Gemini 3.8 Flash gains from it.

Question (Neel's suggestion): does a dumber investigator, with thinking off, lean on the O-lens more
and so show more uplift over transcript-only and placebo? **No.** Every thinking-off predictor is
within ±2pp of its placebo. The only O-lens gain is Gemini 3.8 Flash's (+5pp over placebo at both
`minimal` and `medium`), carried by the hidden_goal cases already known to favour it
([cfbench-harness-smoke.md](cfbench-harness-smoke.md)). The cheap models do see the lens's key
bullets; they don't act on them.

## Setup

cfbench (`cfbench/CLAUDE.md`), harness v7, target Qwen3.6-27B, 20 tool calls, all base runs.
Conditions: transcript-only (`agentic`), `olens`, `olens_placebo`. `--effort off` (new in this
change) sends OpenRouter `reasoning: {enabled: false}`; every thinking-off row has 0 reasoning
characters.

Six cases, picked to favour the lens: the four [model-chart](cfbench-model-chart.md) cases plus
hidden_goal base → stakes (3/30 → 18/30, the lens's best case) and mine → base (16/30 → 2/30, where
everyone fails). O-lens reads: `2026-10-22_hidden-goal-v1` (v2 for mine → base) +
`2026-10-23_canonical-v1`; the placebo draws from canonical-v1's other envs. That differs from the
older smoke runs, whose hidden_goal placebo used donation_bet reads.

Which models can turn thinking off on OpenRouter (probed 2026-10-26): Gemini 3 Flash (preview),
Gemini 3.1 Flash Lite, Haiku 5.5, GPT-6 Luna, GPT-6.1 Terra, DeepSeek V4.1 Flash, Kimi K3, Qwen3.6
Flash can. Gemini 3.8 Flash, GLM 5.3, GLM 5.3 Flash, Gemini 3.5 Flash Lite, MiniMax M3.1 Flash and
Grok 4.7 return 400 "Reasoning is mandatory". Gemini 3.8 Flash at `minimal` thinks almost nothing in
practice (0 reasoning characters in its O-lens and placebo rows, ~2,300 per transcript-only
prediction, against ~25,000 at medium); GLM 5.3's effort setting does nothing.

## Results

100 − RMSE in pp over the 6 cases (equal weight). "Edit does nothing" scores 40.9; the noise ceiling
is about 93. CI is a 90% bootstrap over draws within each case, so it covers draw noise only, not
case choice.

| predictor @ thinking         | draws | TS   | O-lens | placebo | O-lens − placebo [90% CI] | lens calls / pred | $/pred |
| ---------------------------- | ----- | ---- | ------ | ------- | ------------------------- | ----------------- | ------ |
| Gemini 3.8 Flash @ medium    | 3     | 47.7 | 52.7   | 47.3    | +5.4 [+1.7, +8.8]         | 3.1               | 0.19   |
| Gemini 3.8 Flash @ minimal   | 3     | 46.2 | 49.2   | 44.3    | +4.8 [+1.9, +7.6]         | 0.9               | 0.08   |
| Gemini 3 Flash @ off         | 5     | 44.8 | 43.7   | 45.6    | −1.9 [−5.2, +1.3]         | 5.9               | 0.06   |
| Gemini 3.1 Flash Lite @ off  | 10    | 45.6 | 45.3   | 43.8    | +1.5 [−0.4, +3.6]         | 1.1               | 0.008  |
| DeepSeek V4.1 Flash @ off    | 10    | 45.1 | 46.3   | 45.2    | +1.1 [−0.5, +2.6]         | 3.4               | 0.005  |
| GPT-6 Luna @ off             | 10    | 42.8 | 43.6   | 42.6    | +1.1 [+0.1, +2.2]         | 1.1               | 0.001  |
| GPT-6.1 Terra @ off          | 10    | 44.9 | 45.3   | 44.8    | +0.6 [−0.2, +1.3]         | 2.3               | 0.015  |
| Haiku 5.5 @ off              | 10    | 42.2 | 43.0   | 42.5    | +0.5 [−0.7, +1.7]         | 0.0               | 0.001  |
| GLM 5.3 @ medium (can't off) | 10    | 42.0 | 42.5   | 41.2    | +1.3 [−0.2, +2.7]         | 0.6               | 0.004  |

- **Thinking-off predictors barely beat the null in any condition** (42–46 vs 40.9). Every one is
  far below Flash at medium.
- **Using the lens isn't the bottleneck.** Gemini 3 Flash with thinking off calls the lens tools
  most (5.9 per prediction) and gets nothing. DeepSeek calls them 3.4 times and gets +1. On
  hidden_goal base → stakes, the "2/10 on a strict academic scale" bullets that Flash at medium
  cites appear in the tool output of 3/5 Gemini 3 Flash and 9/10 DeepSeek draws. Both then reason
  "scholarship pressure → leniency" and predict 0.02–0.22, against Flash-medium's 0.44 mean (truth
  0.60). Turning a lens bullet into a prediction seems to need the thinking.
- **Flash's gain is the same two hidden_goal cases as before**: base → stakes (O-lens 76 vs placebo
  59 at medium) and mine → base (58 vs 33). At `minimal`, part of the gain is the placebo hurting
  killswitch (70 vs 95).
- **Haiku never calls a lens tool** and makes 1.5 tool calls per prediction, as in the model chart.
- **killswitch stays a Gemini 3.8 Flash strength.** Flash scores 91–95 on it at both efforts. The
  other predictors score 26–56, against a null of 23.

On the four [model-chart](cfbench-model-chart.md) cases, with that chart's predictors at high effort
(and Flash at low, from its effort sweep) next to these runs:

![transcript-only vs O-lens](img/cfbench-report/ts_vs_olens.png)

Every predictor is within 1pp of the diagonal or below it, except Gemini 3.8 Flash (high), Gemini 4
Argon and Haiku (high, +3pp each; Haiku's placebo gains the same). Without the two extra hidden_goal
cases, Flash at minimal and medium has no O-lens gain at all (48.9 vs 49.3, 47.4 vs 47.2). The
thinking-off predictors all sit in a bunch at 38–44, a few points above "edit does nothing".

## Cost and runs

$25.8 in all (pilot $6.95), for 1,278 predictions. Flash at medium was $10.30 of it. Runs:
`runs/cfbench/2026-10-26_dumbpred-pilot-*` (1 draw, groups by effort) and
`2026-10-26_dumbpred-main-*` (the remaining draws, `--skip` over the pilot) on HF and in the
`cfbench` Docent collection. `meta.json` `params.effort` says which thinking setting a run used;
predictions rows don't record it. Flash minimal and medium share a predictor id, so `cfbench.score`
over both merges them; split by run.
