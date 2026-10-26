# Honest-grader envs: can the O-lens read the grade Qwen would give if it were being honest?

Plan written 2026-10-26, after the hidden_goal.grade result
([envs/hidden_goal.md](envs/hidden_goal.md)) turned out to be the only env in the 26-env no-CoT
sweep ([nocot-cfbench.md](nocot-cfbench.md)) where the O-lens beat transcript tools and placebo.
Code: [`evalcue/envs/honest/`](../evalcue/envs/honest/).

## Why the essay env worked, in one paragraph

Not "hidden goal". The reads that carried the uplift were at the `Score:` token of base-arm replies:
Qwen writes `Score: 5/10` and the lens at layer 40 says "2.5/10 (or D+)" or "6/10 (or 2/10 on a
strict academic scale)". That is a readout of the runner-up answer and its rationale at the one
token where the decision is made. The investigator (Gemini 3.8 Flash) then reasons that an edit
which drops the student framing or adds a scholarship bar unlocks the stricter scale, and gets 0/30
→ 27/30 right where every transcript-only and placebo draw predicts sycophancy. Five features made
it possible: a forced decision token; a behavior that is a threshold on a continuous latent; an edit
that selects among rubrics Qwen already represents in the base arm rather than adding a new fact; a
predictor prior that is wrong in a specific direction; thinking off (with CoT the effect vanishes
and the lens echoes the CoT). The agentic envs (shutdown, scheming) have none of these: the decision
is a tool call deep in a long transcript and the edit adds a line the base arm never saw.

## The plan

Make many scenarios of the same shape, cheaply, and let the scores be what they are.

- **Scenario**: one kind of 0–10 judgment. Essay, code, resume, startup pitch, poem, short story,
  research abstract, cover letter, recipe, math proof. One module each in `evalcue/envs/honest/`.
- **Things judged**: two per scenario, trying a little to span quality (one weak, one decent).
- **Framings**, the same five in every scenario, worded for what is judged: `neutral`;
  `flatter_explicit` (the user asks outright for kindness or a high score); `flatter_implicit`
  (ownership plus stakes, no instruction, the lever that worked in hidden_goal); `honest_explicit`
  ("be totally honest, even if it's harsh"); `honest_peer` (the score is checked against expert
  graders or peer models). Two flattery framings differing in explicitness because an explicit
  instruction makes Qwen roleplay niceness, and the lens reading a "real grade" under roleplay is a
  different phenomenon from reading the grade Qwen suppresses on its own.
- **Rollouts**: Qwen3.6-27B, thinking off, n=10 per framing per thing, OpenRouter pinned to Alibaba,
  through `evalcue.run` so each run lands on HF and Docent.
- **Then**: O-lens reads at the `Score` token (one read per rollout, so ~100 per scenario), and two
  prediction tasks. Binary, for cfbench as it stands: pick the cutoff per scenario after seeing the
  scores, by a fixed rule, not by eye. Continuous, the real question: the investigator sees the
  flattering-framing rollouts for one essay plus the lens on them and predicts the honest-framing
  mean score for that essay; compare correlation with truth across things judged for transcript
  tools vs O-lens vs placebo. The transcript-only predictor knows honest is lower; the question is
  whether the lens says how much lower for this one.

Nothing is iterated to make effects large. No item calibration, no lever screening, no lens
pre-check. Build, run, record.

## Status

- `honest.essay` built and smoke-tested (2 rows, 2026-10-26,
  `data/runs/honest_essay/2026-10-26_smoke`).
- Nine more scenarios being built in parallel on separate branches, one PR each.
