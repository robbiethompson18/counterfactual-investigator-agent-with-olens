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
  - `no_transcript`: only the base arm's starting context (prompts and any prefilled turns), so it
    knows the task but not how the model behaves on it;
  - `transcript_agentic`: the starting context plus an index of all n base-arm runs (id, behavior
    yes/no, length), and tools to read any run (`read_transcript`, 40k-character pages) or
    regex-search all of them (`search_transcripts`), 20 calls in total before it must submit
    (`cfbench/tools.py`). This is the baseline every tool condition is compared against;
  - `olens_agentic` and `olens_placebo_agentic`: `transcript_agentic` plus O-lens tools, below.
- the edit, exactly: the before/after file for funding_email, a unified diff of the starting context
  for secret_number.

It answers P(behavior) for a fresh run in Y through a `submit_prediction` tool that rejects anything
outside [0.01, 0.99] and asks again. Every predictor runs with OpenRouter reasoning effort `medium`;
left to defaults, Haiku 5.5 and Gemini Flash Lite think 0 tokens and Gemini Flash thinks a lot,
which confounds the comparison. Arm ids never reach the predictor (names like `placebo` would leak
the answer). Every ordered pair is a case, so there's no "the" base arm.

**Condition names changed on 2026-10-27** (`cfbench/core.py` `LEGACY_CONDITIONS`; every reader maps
stored rows through `canon_condition`): `agentic` → `transcript_agentic`, `olens` → `olens_agentic`,
`olens_placebo` → `olens_placebo_agentic`. The old `transcript` condition pasted two hand-picked
base-arm transcripts into the prompt. It is retired and its code deleted: a predictor that reads two
runs is a weaker investigator, not a control, so a tool condition must never be compared against it.
Its old rows load as `transcript_inline_retired` and are excluded from every chart. Runs made before
2026-10-27 that used it as the baseline are listed in docs/cfbench-v1-results.md and
docs/olens-cfbench-spike.md; treat those comparisons as void.

Every prediction row and `meta.json` records `harness_version` (`cfbench/core.py`; runs without it
are version 1). Skill is only comparable within a version, except that version 7 only reworded the
task sentence (scored by squared error, not log-likelihood; the same best answer), so version-6 runs
stay valid and are compared with 7 as if they were one version.

O-lens conditions (`cfbench/olens.py`, reads from `--olens-reads`, JSONL with a `requests.jsonl` of
the read transcripts next to it):

- `olens_agentic`: `transcript_agentic` plus `search_olens(regex)`, a regex over the transcripts of
  the runs that have reads, returning the O-lens bullets at the punctuation inside each match and
  the first read after it, best FVE first (defaults `top_k=1`, `min_fve=0.1`, `max_bullets=20`), and
  `olens_at(run, char)`, every read around a char from any search hit (default `window=10`, layers
  32/40/48). Every hit shows the transcript text around the read. `search_transcripts` also gives
  each hit's char in this condition. `search_olens(regex, over="bullets")` runs the regex over the
  bullets' own text instead (every bullet at every read, `top_k` ignored, `min_fve` kept) and
  returns only the matching bullets, so the agent can find lens content the transcript never says
  (harness version 2). Every hit shows the run's text in `<transcript>` (`⟨READ⟩` at the read token;
  search hits stop at the edges of the read's message field) and the lens output in
  `<olens layer=… fve=…>` (version 6).
- `olens_placebo_agentic`: the same tools, answered from a run of a different env at the same
  relative position, so the predictor gets as much O-lens text but none about this scenario (the
  reads must cover 2+ envs).

`--olens-positions` (default `sparse`) says which read positions the tools serve, for both
conditions (the placebo pool is the same `Reads`). `sparse` keeps every row of a request that is not
`positions: "all"` exactly as the job read it (stop tokens, `turn_starts` headers, whatever it was),
and of an every-token request only template tokens (`field` null: `<|im_start|>`, role, newline,
`<think>`, `</think>`, `<|im_end|>`) and stop tokens (the reader's PUNCT regex); so the default is
every sparse read ever made, and a dense file adds nothing beyond that. It refuses to start if a
transcript's reads have no surviving row. `all` keeps every row and refuses to start unless every
base run with reads was read with `positions: "all"` in its `requests.jsonl` (the other runs and
their file are listed), so an "every token" run never quietly mixes in sparse reads. Recorded in
`meta.json` params as `olens_positions`.

Both refuse to start if any base arm has reads on fewer than 5 runs and under 50% of its runs (e.g.
1 of 8; 9 of 40 is fine). Read more runs. **Agents never pass `--allow-few-reads` without asking a
human first**, even for a deliberate one-run experiment. As of 2026-10-27, 23 of 268 earlier O-lens
runs would have been refused.

The Thought-Anchors conditions (`anchors`, `anchors_placebo`: attention receiver scores and
suppression effects per sentence as tools) were removed on 2026-10-26 after a null result; see
[docs/thought-anchors.md](../docs/thought-anchors.md). Code: `cfbench/anchors.py` and
`scripts/anchor_reads.py` at commit `7296169`
(https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/cfbench/anchors.py).

## The answer task

`--task answer` (2026-10-26) is the second kind of case: instead of P(behavior) in the
counterfactual arm, the predictor submits the **answer** the model gives there, through
`submit_answer` (`cfbench/answer.py`). It is for envs where the model's own answer is the thing to
recover and the behavior bool is degenerate, e.g. `hint_cf_*` with base = the hinted arm ("Mine is
Rust." → Rust) and counterfactual = `nohint`: the question is what Qwen says without the hint, and
the rate of "follows the hint" there is 0 by construction. (The reverse pair, `nohint` → `hint+d1`,
is an ordinary rate case.)

- The env supplies `ANSWER_FORMAT` (what the model was told, "one word only"), `ANSWER_MATCH` (how
  answers are compared, "the first word, case-insensitive") and `canon_answer(text)`, that
  comparison as a function. Ports with those attributes work through `evalcue_runs` as usual
  (`PORTS` maps a run dir name to its port). Each run's `Rollout.answer` is `canon_answer` of its
  final reply.
- The prompt is the rate prompt with "## Behavior" replaced by "## The question" (format, match
  rule, the base arm's answer counts), transcripts labelled by answer instead of yes/no, and the
  task sentence asking for the single most likely answer. The rate prompt is byte-identical to
  before (checked by hashing every condition's prompt), so no harness-version bump.
- `--items all` (or a comma list) makes one case per item (`<spec>@<item>`), since each item is its
  own question.
- **Grading: % correct**, by `python -m cfbench.answer RUN`. A draw is correct when its submission
  canonicalises to the counterfactual arm's most common answer; a case's score is the mean over
  draws; the headline is the mean over cases, next to the null "the edit does nothing" (the base
  arm's most common answer). Rows carry `answer`, `canon`, `correct`, `modal_cf`, `k_cf`/`n_cf` (the
  modal answer's count) and both arms' answer counts.
- **The two tasks never share a grader.** Every row carries `task`; `cfbench.score` refuses answer
  rows and `cfbench.answer` refuses rate rows. Don't put both tasks in one run, and don't average a
  % correct with an RMSE.

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
uv run python -m cfbench.run evalcue_runs:funding_email/<run> --pairs arm_a:arm_b,arm_b:arm_a --conditions no_transcript,transcript_agentic --slug fe-all
uv run python -m cfbench.score data/runs/cfbench/<date>_fe-all
```

`--max-calls` sets the tool conditions' budget (default 20). `--effort` sets the predictor's
reasoning effort (default `medium`; `off` disables thinking, which some models refuse, see
[docs/cfbench-dumb-predictors.md](../docs/cfbench-dumb-predictors.md)). `--model` picks the target
(default `Qwen/Qwen3.6-27B`); `--predictors` is a comma list of OpenRouter ids (default Gemini 3.8
Flash, `google/gemini-3.8-flash`, the report's baseline predictor; was Haiku 5.5 until 2026-10-26).
Needs `OPENROUTER_API_KEY`. Each call is a `scripts/runs.py` run: `data/runs/cfbench/<date>_<slug>/`
with `meta.json` and `predictions.jsonl` (one row per prediction, including the predictor's full
conversation, tool-call counts and OpenRouter cost), synced to HF when it finishes. Runs from before
`runs.py` are flat `runs/cfbench/*.jsonl` files on HF.

For any evalcue env, `evalcue_runs:<env>/<run>[+<run>...]` reads the env's run dirs directly and
`--pairs hub` pairs its hub arm (smallest total diff to the others, usually the base) with every
other arm both ways (`cfbench/envs/evalcue_runs.py`). For an `environments/` run folder
(`data/envs/<env>/<run>/`, `environments run`), `env_runs:<env>/<run>[+<run>...][@<item>]` does the
same over its episodes and `measure` labels (`cfbench/envs/env_runs.py`); sources are
`<env>/<run>:<episode id>`, which is what `scripts/olens_requests_from_runs.py envs:<env>/<run>`
puts in read requests, so reads join. `--base-positive` keeps only base-arm runs that showed the
behavior: on the answer task every base run's answer is listed in the index the predictor sees, so a
hint_mcq base run that ignored the hint hands it the own answer (2026-10-27: 46% "correct" with no
lens on the pause arms, 0% with the flag).

Transcripts are the envs' own run dirs, `data/runs/<env>/<date>_<slug>/` (each loader's `DATA`); arm
contents are read from the env's configs and `vendor/agent-interp-envs` (run
`git submodule update --init` first). Pull transcripts with
`hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset --include 'runs/<env>/*' --local-dir data`.

Evals that pick their own cases (one shown transcript per case, two models paired, the model's name
hidden, extra notes for the investigator) are folders in `cfbench/evals/`, run with
`python -m cfbench.eval`: see [evals/CLAUDE.md](evals/CLAUDE.md).

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
