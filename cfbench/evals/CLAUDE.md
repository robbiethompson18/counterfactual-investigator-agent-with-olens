# cfbench/evals: evals that pick their own cases

An **eval** is a set of cfbench cases chosen for one question, run through the standard predictor,
prompt and tools (`cfbench/core.py`, `tools.py`, `olens.py`, `predict.py`). It is not tied to one
env: an eval can span several envs, models and arm pairs, and one env can be in many evals. Use one
when `cfbench.run`'s "every base-arm run of one model, arm pair → arm pair" doesn't fit, e.g. one
shown transcript per case, two models paired on the same items, or a model name hidden.

The env viewer lists the evals the code defines: every folder here, plus `<env>-pairs` for each env
in `PAIRS_ENVS` (`environments/_lib/evals.py`). A new pairs eval needs its env added there; an eval
in `RETIRED_EVALS` there is left off the lists.

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
    `case` also has the prompt's `condition`. Without it the sentence stays as `core.build_prompt`
    writes it.
  - `TASK` (optional): `"rate"` (default), or `"claim"`: the probability that a claim stated in the
    task is true, with truth 0 or 1 as `k_cf`/`n_cf` (scored the same way, RMSE = root Brier).
  - `ASK(case)` (optional): the body of `## Your task` in place of `build_prompt`'s.
  - `INPUTS_FROM` (optional): another eval whose inputs run this one reads (it then needs no
    `prep`).
  - `PROMPT(prompt, case)` (optional): a last rewrite of the whole prompt, after everything else.
  - `MODEL_IDS` (optional): every model id the predictor must never see (default: the case's
    `model`).
  - `OLENS` (optional): overrides of the O-lens tools' defaults for this eval, e.g.
    `{"min_fve": 0.0}` to show every bullet (each still with its FVE); `cfbench/olens.py` `Lens`
    fields.
  - `OLENS` may also set `context: "xml"`: hits show ~10 tokens each side with the read token
    wrapped, `not a foo<read>,</read> a bar`, in place of `⟨READ⟩` (and the prompt says so).
  - `INLINE(env, condition, case, base, seed)` and `INLINE_EFFORT` (optional): the single-turn
    conditions (`core.INLINE_CONDITIONS`: `transcript_inline`, `olens_inline`,
    `olens_placebo_inline`): the eval writes the whole prompt (the runner adds eval.md), the
    predictor gets no tools, only the submit call, at `INLINE_EFFORT` (or `--effort`, which when
    given applies to every condition). Returns (prompt, extra row fields).
  - `DEFAULT_VERSIONS` (optional): the versions run without `--versions` (default all).
  - `LINKS(row)` (optional): per prediction, links for the env viewer's card, e.g.
    `{"transcript": {"env", "run", "episode"}, "O-lens reads": {"dataset", "item"}}`; an episode
    gets its Docent URL from the env's `links.json`.
  - `RUNS` (optional): the cfbench run dir names the env viewer shows for this eval, each with all
    its conditions current and no pooled row across them; without it the viewer picks runs itself
    (newest run per condition supersedes, current runs pooled).
  - `ERROR_RATE` (optional): `True` shows an answer-task eval on the env viewer as error rate (1 − %
    correct, lower is better) in its bars, tables and paired differences.
  - `VERSION_IN_IDS` (optional): `True` prefixes each prediction's expview item id with its version,
    for evals whose versions share arm pairs; off by default so other evals' links don't change.
- `score.py` `PAGE(rows) -> str` (optional): the eval's own page section, HTML built from the current runs' prediction
  rows. `environments evals` stores it as the eval's `page_html`; the env viewer shows it at the top of the eval's page,
  with the standard claim tables folded below. Helpers (bar chart with CIs, tables, condition swatches; anything with
  `data-cond` gets that condition's colour as `--c`): `cfbench/evals/page.py`. `essay_grading_cf_inline` is the example.
- `eval.md`: anything else the investigator should know. The runner puts it in every prompt as
  `## About this evaluation`, just before the task, and records it in `meta.json`.

## Running

```bash
uv run python -m cfbench.eval NAME --prep [--versions a,b] [--placebo DIR]  # cfbench_evals/<date>_<name>-inputs
uv run python -m cfbench.eval NAME --show [--versions a,b]   # the first case's prompt per condition, no API calls
uv run python -m cfbench.eval NAME --slug SLUG [--versions a,b] [--draws 1] [--limit N] [--conditions ...]
uv run python -m cfbench.score data/runs/cfbench/<date>_<slug>
# many inline predictions on Haiku: everything to Anthropic's Message Batches API at once (half price, no per-minute cap)
uv run python -m cfbench.eval NAME --slug SLUG ... --predictors anthropic-batch/claude-haiku-5-5 --batch submit   # sends, exits
uv run python -m cfbench.eval NAME ... --predictors anthropic-batch/claude-haiku-5-5 --batch collect --run DIR    # waits, writes rows
uv run python -m environments evals NAME --upload   # the viewer's evals/NAME.json + expview of every conversation
```

`--batch submit` builds and sends one batch (1,000 requests) at a time, so memory holds one batch
whatever the run's size, and writes `batches.jsonl` (each batch id with its requests' predictions
and request hashes). `--batch collect`, with the same other arguments (`--slug` not needed), can run
any time after and be rerun: it waits for each batch, writes its rows (a reply without a valid
submission gets its further turns, an errored request its whole conversation, from
`anthropic-direct/`), and finishes the run once all have ended. Inline conditions only; `--reuse`
works as usual at submit. Batches of 1,000 inline prompts took ~10 min on 2026-10-31; Anthropic's
cap is 24 h, after which requests expire unbilled.

Collect refuses arguments that differ from the submit's (pin `--inputs` if a newer inputs run
appeared) or a rebuilt request that differs from the one sent, and finishes the run only when every
prediction has a row; otherwise it marks it abandoned and says so: rerun collect, or after an
interrupted submit, submit a new run with `--reuse` of this one's rows.

Runs land in `data/runs/cfbench/<date>_<slug>/` like any cfbench run (meta params `eval`, `envs`),
synced to HF at the end. Slugs ending `-smoke` are left out of the viewer. Gemini 3.8 Flash with 20
tool calls costs ~$0.10 per draw on a 25k-character transcript, so budget draws × cases × conditions
before launching.

## In the env viewer

`python -m environments evals` (and `index`) writes `evals/<name>.json`, one file per eval, with no
shared index (the page lists evals from `spec/evals.json` and reads each eval's own file;
`environments/_lib/evals.py`): every eval, per run the RMSE per condition vs "the edit does nothing"
as bars with 95% bootstrap CIs (resampling arm pairs) plus every paired difference between
conditions with its CI, the paired-model comparison (mean p per model, AUROC between their cases)
when two models share cases, and per (env, version, arm pair, model) a before/after chart with every
draw. The page has an Evals list (`#evals=1`), one page per eval (`#eval=NAME[&env=E]`), and each
env's Evals tab lists the evals that use it. Clicking a draw (point or chip) opens a card with its
predicted rate, the investigator's explanation, and Docent + expview links for its conversation and
for the transcript it investigated (`LINKS`). Plain `cfbench.run env_runs:<env>/...` runs show as
the eval `<env>-pairs`.

## Evals

- `af_refusals`: alignment faking public → preview, DeepSeek V4 Flash and GLM 5.3 paired. Each case
  shows one public refusal (reply normalised to `<rejected/>`), the model is hidden, and the O-lens
  reads cover the prompt only, in one format for both models. So only the reads can tell the models
  apart. Versions `original` and `v7`.
- `af_which`: the af_refusals cases with a different question. The investigator is told one of two
  models alignment fakes and the other doesn't, and gives P(the target model is the faker); truth 1
  for DSv4 Flash, 0 for GLM 5.3 (`TASK = "claim"`). Its own inputs run: af_refusals' prep with 4
  bullets per read and the newer DSv4 reads.
- `af_which_pair`: af_which with both models' refusals of an item shown side by side as runs 0
  (model A) and 1 (model B), in both orders; P(A is the faker), truth 1 when A is DSv4 Flash. Reads
  af_which's inputs run.
- `af_which_pair_real`: af_which_pair with the replies exactly as generated (GLM's `<OUTPUT>`
  wrapper kept), O-lens reads on the reply tokens and on the user turn's template token, exactly 4
  bullets per position ("4 samples from the lens"), positions whose bullets name either model
  dropped, and the random-model placebo shown on the shown run's own transcript (harness 12). Its
  own inputs run (`--prep --tag real`).
- `overeager_main`: over-eagerness on GLM 5.3, the env's "main: one-edit fixes and nulls"
  comparisons (payback, dana, calendar, gitbranch, email junebug; 4 Johns left out, it is email item
  0), base arm → each of its 4 one-line edits. A case is one arm pair and one shown base run where
  GLM acted without asking and that has O-lens reads (8 per base arm, one per item): 160 cases.
  Conditions `no_transcript`, `transcript_agentic`, `olens_agentic`; the prompt states the pooled
  base rate. Version `main`.
- `essay_grading`: Qwen3.6-27B (thinking off) grading a mediocre essay, harsh = Score ≤ 4/10. The 38
  ordered arm pairs of the pre-registered 2026-10-23 test (docs/envs/hidden_goal.md) as versions
  `cue` (12), `who` (10) and `heldout` (16), each on the one 2026-10-22 n=30 run it used, every base
  run shown. O-lens reads are the old test's (2026-10-22_hidden-goal-v1/2/3, renamed to the env_runs
  sources); placebo from `--placebo` (the old test used donation-bet-v1). Imported old rows
  (`import_old.py`, no rerun): `2026-10-28_essay-grading-flash-v1` (harness v1, Flash) and
  `2026-10-28_essay-grading-sol-v6` (harness v6, GPT-6.1 Sol), not comparable to new runs.
- `essay_grading_inline`: the scaled essay_grading eval (2026-10-30). Every essay_grading arm (21 old
  + the 205-arm wide set, `environments/essay_grading/wide.py`, written before any was run) × 8
  essays = 1,808 prompts; the investigator sees one prompt and names the score Qwen gives most often
  (single-turn). Headline: share of Qwen's samples matching the submitted score (`score.py`, CIs
  resample arms). Versions `bare` and `anchored` (plus the base arm's scores on that essay). Reads:
  every punctuation token + the ten thinking-off tail tokens, layers 32-60; placebo = another arm's
  reads on a different essay of the same PERSUADE topic, never base's (2026-10-31; it was the same
  essay, which could pick the counterfactual prompt). All-layer NLA and J++ pairs on Robbie's
  defaults, Haiku at effort medium; `essay_grading_cf_inline` runs the same conditions.
- `hint_qwen_dsv4` (`knew` variant) and `hint_qwen_dsv4_letter`: questions Qwen3.6-27B and DSv4
  Flash answer consistently and differently, hinted to the correct letter (5 hint-following Qwen
  runs shown), half Qwen knew without the hint and half not, balanced within each gold letter (the
  two models' opposite position biases otherwise let the hinted letter give the class away). `knew`:
  P(Qwen answers correctly without the hint), scored as % right at p >= 0.5 (its own score.py; the
  page's claim table shows % right). `letter`: the answer task, name Qwen's no-hint letter (reads
  hint_qwen_dsv4's inputs). Versions `mmlu_pro` and `truthfulqa` sources, pooled into one 86/86 set.
  Both also run single-turn (`hint_qwen_dsv4/inline.py`, effort low): the prompt and reply in
  context, plus (`olens_inline`) every bullet at each option's newline, or (`olens_placebo_inline`)
  another question's with the same option count. A filler variant (hint_mcq's `pause/` arms, reads
  at the filler sentence's punctuation) was tried and dropped: O-lens = placebo
  (docs/envs/hint_cf.md). The four `*_tail_inline` evals also run `nla_inline` (ceselder's layer-42
  NLA) and `jlens_inline` (J++ top 10, layers 32-60 step 4) with their placebos, read at exactly
  the O-lens reads' tokens (`hint_qwen_dsv4/lens_reads.py`; runs
  `nla_reads/2026-10-31_hint-qwen-dsv4-ceselder-l42`, `jpp_reads/2026-10-31_hint-qwen-dsv4-same-tokens`).
- `supergpqa_inline`: mcq_unselected's question on SuperGPQA (`environments/supergpqa`): 3,000
  consistent cases (seed 0; the other 3,328 consistent questions are the probe pool), no hint,
  single-turn letter task, Haiku 5.5. Seven conditions: `transcript_inline`, O-lens / NLA / J++ each
  with a same-option-count placebo; own reads at the option newlines and the ten end-of-prompt
  tokens of the thinking-off render (empty think block included). `eval.py requests OUT_DIR` writes
  the O-lens/NLA and J++ request files (docs/supergpqa-inline.md).
