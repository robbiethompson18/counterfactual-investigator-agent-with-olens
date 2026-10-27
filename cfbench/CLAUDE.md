# cfbench: predict what an environment edit does to a model's behavior

Our from-scratch replacement for the CHIVE prediction harness. Agam's
[detective-joracle](https://github.com/agu18dec/detective-joracle) is reference-only here
(`vendor/CLAUDE.md`).

## The task

An **env** has several **arms**: one base task plus small edits to it (an eval cue, a different
document, a task knob). We have labelled rollouts of a target model (Qwen3.6-27B, sometimes GLM 5.3
or gpt-oss-120b) in every arm. A **case** is an ordered pair of arms, base X → counterfactual Y. The
predictor sees:

- the target model's name and the behavior's definition (the env's own label rubric where it has
  one);
- the base-arm rate k_X/n_X;
- evidence about the base arm, by condition:
  - `transcript`: one base-arm transcript with the behavior and one without, in random order,
    labelled (just one if the arm has only one kind), and told the pick says nothing about the rate;
  - `no_transcript`: only the base arm's starting context (prompts and any prefilled turns), so it
    knows the task but not how the model behaves on it;
  - `agentic`: the starting context plus an index of all n base-arm runs (id, behavior yes/no,
    length), and tools to read any run (`read_transcript`, 40k-character pages) or regex-search all
    of them (`search_transcripts`), 20 calls in total before it must submit (`cfbench/tools.py`);
- the edit, exactly: the before/after file for funding_email, a unified diff of the starting context
  for secret_number.

It answers P(behavior) for a fresh run in Y through a `submit_prediction` tool that rejects anything
outside [0.01, 0.99] and asks again. Every predictor runs with OpenRouter reasoning effort `medium`;
left to defaults, Haiku 5.5 and Gemini Flash Lite think 0 tokens and Gemini Flash thinks a lot,
which confounds the comparison. Arm ids never reach the predictor (names like `placebo` would leak
the answer). Every ordered pair is a case, so there's no "the" base arm. Each draw re-picks the
shown transcripts.

Every prediction row and `meta.json` records `harness_version` (`cfbench/core.py`; runs without it
are version 1). Skill is only comparable within a version, except that version 7 only reworded the
task sentence (scored by squared error, not log-likelihood; the same best answer), so version-6 runs
stay valid and are compared with 7 as if they were one version.

O-lens conditions (`cfbench/olens.py`, reads from `--olens-reads`, JSONL with a `requests.jsonl` of
the read transcripts next to it):

- `olens`: `agentic` plus `search_olens(regex)`, a regex over the transcripts of the runs that have
  reads, returning the O-lens bullets at the punctuation inside each match and the first read after
  it, best FVE first (defaults `top_k=1`, `min_fve=0.1`, `max_bullets=20`), and
  `olens_at(run, char)`, every read around a char from any search hit (default `window=10`, layers
  32/40/48). Every hit shows the transcript text around the read. `search_transcripts` also gives
  each hit's char in this condition. `search_olens(regex, over="bullets")` runs the regex over the
  bullets' own text instead (every bullet at every read, `top_k` ignored, `min_fve` kept) and
  returns only the matching bullets, so the agent can find lens content the transcript never says
  (harness version 2). Every hit shows the run's text in `<transcript>` (`⟨READ⟩` at the read token;
  search hits stop at the edges of the read's message field) and the lens output in
  `<olens layer=… fve=…>` (version 6).
- `olens_placebo`: the same tools, answered from a run of a different env at the same relative
  position, so the predictor gets as much O-lens text but none about this scenario (the reads must
  cover 2+ envs).

The Thought-Anchors conditions (`anchors`, `anchors_placebo`: attention receiver scores and
suppression effects per sentence as tools) were removed on 2026-10-26 after a null result; see
[docs/thought-anchors.md](../docs/thought-anchors.md). Code: `cfbench/anchors.py` and
`scripts/anchor_reads.py` at commit `7296169`
(https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/cfbench/anchors.py).

## Scoring

**RMSE of the predicted rate** ([docs/cfbench-scoring.md](../docs/cfbench-scoring.md), since
2026-10-26): per case, (p̂ − k_Y/n_Y)² averaged over draws; over cases with equal weight; reported in
percentage points next to the RMSE of "the edit does nothing" (predict the shown base rate), with a
skill 1 − Σ MSE / Σ MSE_null as a secondary number. `cfbench.score` prints it, and
`cfbench.report.charts` makes the report figures from it.

Runs before 2026-10-26 were scored with a binomial log-likelihood skill
(`cfbench.report.data.log_skill`), defined with the harness v1 results in
[docs/cfbench-v1-results.md](../docs/cfbench-v1-results.md).

## Running

```bash
uv run python -m cfbench.run evalcue_runs:funding_email/<run> --pairs arm_a:arm_b,arm_b:arm_a --conditions transcript,no_transcript,agentic --slug fe-all
uv run python -m cfbench.score data/runs/cfbench/<date>_fe-all
```

`--max-calls` sets the tool conditions' budget (default 20). `--effort` sets the predictor's
reasoning effort (default `medium`; `off` disables thinking, which some models refuse, see
[docs/cfbench-dumb-predictors.md](../docs/cfbench-dumb-predictors.md)). `--model` picks the target
(default `Qwen/Qwen3.6-27B`); `--predictors` is a comma list of OpenRouter ids (default Haiku 5.5 in
the CLI; the report's baseline predictor is Gemini 3.8 Flash, `google/gemini-3.8-flash`, so pass
it). Needs `OPENROUTER_API_KEY`. Each call is a `scripts/runs.py` run:
`data/runs/cfbench/<date>_<slug>/` with `meta.json` and `predictions.jsonl` (one row per prediction,
including the predictor's full conversation, tool-call counts and OpenRouter cost), synced to HF
when it finishes. Runs from before `runs.py` are flat `runs/cfbench/*.jsonl` files on HF.

For any evalcue env, `evalcue_runs:<env>/<run>[+<run>...]` reads the env's run dirs directly and
`--pairs hub` pairs its hub arm (smallest total diff to the others, usually the base) with every
other arm both ways (`cfbench/envs/evalcue_runs.py`).

Transcripts are the envs' own run dirs, `data/runs/<env>/<date>_<slug>/` (each loader's `DATA`); arm
contents are read from the env's configs and `vendor/agent-interp-envs` (run
`git submodule update --init` first). Pull transcripts with
`hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset --include 'runs/<env>/*' --local-dir data`.

## Adding an env

A module in `cfbench/envs/` with `BEHAVIOR`, `rollouts(arm, model)` and `edit(base_arm, cf_arm)`;
the contract is in `cfbench/core.py`'s docstring. Import arm contents and the label definition from
where the env defines them (the env module, its detector docstring); don't copy them.
`evalcue_runs.py` (any evalcue env's rows) diffs the starting context recorded in the rows, which
works for any env whose arms differ only in what the model is shown up front.

## Results

Current numbers (RMSE, harness v6+): [docs/cfbench-scoring.md](../docs/cfbench-scoring.md),
[docs/cfbench-model-chart.md](../docs/cfbench-model-chart.md),
[docs/envs/hidden_goal.md](../docs/envs/hidden_goal.md) (the O-lens pre-registered test),
[docs/nocot-cfbench.md](../docs/nocot-cfbench.md) and [docs/cot-cfbench.md](../docs/cot-cfbench.md).
The 2026-10-21/22 harness v1 log-skill tables are in
[docs/cfbench-v1-results.md](../docs/cfbench-v1-results.md).
