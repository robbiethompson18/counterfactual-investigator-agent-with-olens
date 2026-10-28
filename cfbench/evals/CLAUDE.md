# cfbench/evals: evals that pick their own cases

An **eval** is a set of cfbench cases chosen for one question, run through the standard predictor,
prompt and tools (`cfbench/core.py`, `tools.py`, `olens.py`, `predict.py`). It is not tied to one
env: an eval can span several envs, models and arm pairs, and one env can be in many evals. Use one
when `cfbench.run`'s "every base-arm run of one model, arm pair → arm pair" doesn't fit, e.g. one
shown transcript per case, two models paired on the same items, or a model name hidden.

## A folder per eval

`cfbench/evals/<name>/`:

- `eval.py`:
  - `ENVS`: the `environments/` envs it uses; the env viewer links them.
  - `VERSIONS`: named variants, e.g. prompt versions.
  - `CONDITIONS`: its default conditions.
  - `MODEL_NAME`: `None`, or the words used in place of the target model's id in the prompt (the
    runner asserts the id never reaches the predictor).
  - `prep(out, versions, placebo)`: writes the eval's inputs (cases, O-lens reads in the
    `cfbench/olens.py` format) into a `scripts/runs.py` run, synced to HF, so later runs need only
    that run.
  - `reads(inputs)`: the reads files to load.
  - `cases(inputs, versions)`: yields `(prompt env, case row fields, base runs)` per case. The
    prompt env needs `BEHAVIOR` and `view()` or `edit()`, as in `cfbench/core.py`. Row fields must
    include `env`, `env_name` (the environments/ env), `model`, `base_arm`, `cf_arm`,
    `k_base`/`n_base` (what the investigator is shown), `k_cf`/`n_cf` (the truth), `task`, and
    optionally `version`, `item` and `k_base_pool`/`n_base_pool` (the real base rate, when fewer
    runs are shown).
  - `BASE_LINE(case)` (optional): the sentence that replaces the prompt's "In the base arm, `model`
    showed the behavior in k of n runs." (`k_base`/`n_base` of the runs shown); `""` drops it.
    Without it the sentence stays as `core.build_prompt` writes it.
  - `LINKS(row)` (optional): per prediction, links for the env viewer's card, e.g.
    `{"transcript": {"env", "run", "episode"}, "O-lens reads": {"dataset", "item"}}`; an episode
    gets its Docent URL from the env's `links.json`.
- `eval.md`: anything else the investigator should know. The runner puts it in every prompt as
  `## About this evaluation`, just before the task, and records it in `meta.json`.

## Running

```bash
uv run python -m cfbench.eval NAME --prep [--versions a,b] [--placebo DIR]  # cfbench_evals/<date>_<name>-inputs
uv run python -m cfbench.eval NAME --slug SLUG [--versions a,b] [--draws 1] [--limit N] [--conditions ...]
uv run python -m cfbench.eval NAME --show [--versions a,b]   # the first case's prompt per condition, no API calls
uv run python -m cfbench.score data/runs/cfbench/<date>_<slug>
uv run python -m environments evals NAME --upload   # the viewer's evals/NAME.json + expview of every conversation
```

Runs land in `data/runs/cfbench/<date>_<slug>/` like any cfbench run (meta params `eval`, `envs`),
synced to HF at the end. Slugs ending `-smoke` are left out of the viewer. Gemini 3.8 Flash with 20
tool calls costs ~$0.10 per draw on a 25k-character transcript, so budget draws × cases × conditions
before launching.

## In the env viewer

`python -m environments evals` (and `index`) writes `evals/index.json` and `evals/<name>.json`
(`environments/_lib/evals.py`): every eval, per run the RMSE per condition vs "the edit does
nothing", the paired-model comparison (mean p per model, AUROC between their cases) when two models
share cases, and per (env, version, arm pair, model) a before/after chart with every draw. The page
has an Evals list (`#evals=1`), one page per eval (`#eval=NAME[&env=E]`), and each env's Evals tab
lists the evals that use it. Clicking a draw (point or chip) opens a card with its predicted rate,
the investigator's explanation, and Docent + expview links for its conversation and for the
transcript it investigated (`LINKS`). Plain `cfbench.run env_runs:<env>/...` runs show as the eval
`<env>-pairs`.

## Evals

- `af_refusals`: alignment faking public → preview, DeepSeek V4 Flash and GLM 5.3 paired. Each case
  shows one public refusal (reply normalised to `<rejected/>`), the model is hidden, and the O-lens
  reads cover the prompt only, in one format for both models. So only the reads can tell the models
  apart. Versions `original` and `v7`.
- `overeager_main`: over-eagerness on GLM 5.3, the env's "main: one-edit fixes and nulls"
  comparisons (payback, dana, calendar, gitbranch, email junebug, 4 Johns), base arm → each of its 4
  one-line edits, 24 cases. Every pooled GLM base-arm run is shown; O-lens reads (Celeste's RL lens,
  template tokens) exist almost only on base runs that acted without asking. Version `main`.
