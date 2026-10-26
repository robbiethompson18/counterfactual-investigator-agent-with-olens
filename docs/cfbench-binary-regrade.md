# cfbench regraded as a binary question (2026-10-26)

**Status (2026-10-26):** REFERENCE. Throwaway regrade of existing Flash predictions; no new runs.

A throwaway check: do the report's conclusions depend on scoring the predictor's rate estimate
(RMSE, [cfbench-scoring.md](cfbench-scoring.md)), or do they survive if we pretend the question had
been a yes/no? No new predictor runs; the existing Gemini 3.8 Flash predictions in
`cfbench.report.data.groups()` were reinterpreted after the fact.

## The regrade

- **Truth** per case (ordered arm pair X → Y): yes if k_Y/n_Y − k_X/n_X ≥ T, with T = 10pp by
  default. "Falls by ≥ T" needs no separate label because every pair is also a case in the other
  order.
- **The predictor's answer**: yes if its submitted p̂ minus the base rate it was shown is ≥ T. The
  null ("edit does nothing") always says no.
- **Score**: per draw, then the fraction of draws that said yes per case, equal weight per case (the
  RMSE convention). Reported as balanced accuracy (mean of the true-positive and true-negative
  rates), so the null scores 50% exactly instead of the ~70% plain accuracy it gets from 70% of
  cases being "no". AUROC with p̂ − base rate as the score was also checked.
- 95% case-bootstrap intervals; differences between conditions are paired (same resamples).

## Result

![RMSE bars over binary bars, per target](img/cfbench-binary-regrade/binary_vs_rmse.png)

Balanced accuracy at T = 10pp, percent, 95% CI:

| target       | cases (yes) | env diff only | transcript-only | O-lens      | placebo     |
| ------------ | ----------- | ------------- | --------------- | ----------- | ----------- |
| GLM 5.3      | 280 (85)    | 78 [72, 83]   | 73 [67, 78]     | 72 [67, 78] | 73 [67, 78] |
| Qwen w/o CoT | 144 (45)    | 76 [68, 83]   | 78 [71, 84]     | 79 [72, 85] | 77 [70, 85] |

AUROC is 0.82–0.86 everywhere with overlapping CIs.

- **O-lens vs placebo vs transcript-only is still a wash** at T = 5, 10, 20 and 30pp, on both
  targets, under balanced accuracy and AUROC. Paired gains vs transcript-only are within ±1 point
  with CIs straddling zero.
- **One ranking flips: on GLM, env diff only (no transcripts) wins the binary grading** by 5 points
  of balanced accuracy, paired CI [+2, +8], where under RMSE it ties. Without transcripts the
  predictor moves further from the base rate (mean |p̂ − base| 0.27 vs 0.23) and is right about the
  direction about as often (82% vs 80% on cases that move ≥ 10pp), so it clears the threshold more
  often. The binary grading doesn't charge for overshooting; RMSE does (it overshoots the true move
  on 44% of cases vs 39%).
- On Qwen w/o CoT the RMSE finding that transcripts help (−4pp RMSE) keeps its direction but loses
  significance in the binary grading (plain accuracy +3.7 points, CI [+0.0, +0.07]; balanced
  accuracy CI crosses zero). The tool conditions pull away from env diff only as T grows (at 30pp:
  80–82 vs 72), so big effects are where transcripts help there.

![Balanced accuracy vs threshold](img/cfbench-binary-regrade/binary_threshold_sweep.png)

## Caveat

This reinterprets a rate-estimation answer as a yes/no. A predictor that hedges toward the base rate
is penalized here in a way it would not be if it had actually been asked the binary question.
Comparing the two question formats properly needs a rerun with a binary prompt (and a
`HARNESS_VERSION` bump).

## Reproducing

Not in the repo (the regrade was a scratch script). The whole computation is: load the cases with
`cfbench.report.data.load_rows` and `build_cases` for each group in `groups()`, then per case and
condition compute `mean(p - q_base >= T for p in case.p[cond])` against `case.q - case.q_base >= T`;
balanced accuracy from the per-case yes fractions, AUROC from the per-case mean `p - q_base`,
bootstrap over cases. If it's wanted for the report it belongs next to `brier_skill` in
`cfbench/report/data.py` plus a flag on `cfbench.score`.
