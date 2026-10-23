# evalcue: multi-arm behavioral environments on Qwen3.6-27B

Started as "does an incidental eval cue make Qwen3.6-27B reward hack?" and now hosts any env where
small edits (arms) might move a behavior: reward hacking, self-preservation, whistleblowing, hiring
bias, hiding mistakes from users. The name stuck; a rename is pending.

## How we judge arms

Goals and process:
[docs/environment-construction-goals.md](../docs/environment-construction-goals.md). In short: build
an env, measure the behavior's rate (0% and 100% are fine starting points), then add arms (any
lever, stacked) until you have many arms with labelled rates. Arms are symmetric: there is no
baseline arm, and the summary prints every arm's rate.

- Start at n=10 per arm; cheap single-turn envs can go to n=30. Rerun the most interesting arms at
  higher n on a fresh `--seed` before calling a difference real. There is no significance gate.
- Detection is programmatic (code execution, file diffs, string matches, a forced answer format). No
  LLM judge on transcripts. Read CoTs yourself with `evalcue.show` to spot-check the detector, check
  the model saw the edit, and note whether it verbalizes it.
- Never cap thinking: leave `--max-tokens` unset.

## Using it

Generation goes to the shared Qwen3.6-27B server (`$QWEN_VLLM_URL` with `$QWEN_VLLM_API_KEY`, from
the main checkout's `.envrc.local`; model name `qwen`, 262k context). The server handles hundreds of
concurrent requests; `--concurrency` defaults to 512. Model code runs in Docker
(`docker build -t evalcue-sandbox evalcue`), no network. An env can name its own image via
`Episode.image`.

```
uv run python -m evalcue.run ENV --tag dev [--n 10] [--arms a,b,c] [--thinking 1] [--seed 0] [-d "what this run tests"]
uv run python -m evalcue.run ENV --rescore data/runs/ENV/2026-10-21_dev/rows.jsonl   # re-run detect() only, then re-sync
uv run python -m evalcue.show data/runs/ENV/2026-10-21_dev/rows.jsonl --arm a --positive 1 -k 3
```

Each generation run is a [`scripts/runs.py`](../scripts/runs.py) run dir,
`data/runs/ENV/<YYYY-MM-DD>_TAG/` with `rows.jsonl` and `meta.json`, synced to the HF dataset when
it finishes (or marked `abandoned` and synced if it crashes). A tag can be used once per env per
day. Runs from before 2026-10-21's contract change are in `data/runs/evalcue/ENV/TAG.jsonl`.

An env is a module in `evalcue/envs/`: `NAME.py`, or `FAMILY/NAME.py` for a family of related envs
that share helpers and a `data/` dir (run as `FAMILY.NAME`; its run dir stays the flat
`data/runs/FAMILY_NAME/`). Same items and same detector means a new arm, not a new module (see
[environment-construction-goals.md](../docs/environment-construction-goals.md#arm-or-new-module)).
Each has `ARMS`, `items()`, `build(arm, item) -> Episode` and
`detect(row) -> {"positive": bool | None, ...}` (see `harness.py`, and `envs/smoke/` for both
shapes; older envs return `"hack"`, which is read as `"positive"`). An optional `BEHAVIOR` string
states `detect()`'s rule in words; cfbench shows it to its predictors, so edit the two together. An
`Episode` with no `files` is one chat completion over `messages` (prefilled assistant turns are
fine). With `files`, it is a bash-tool agent in a container whose working directory is
`Episode.cwd`. Files starting with `#!` are made executable, so tools like `./next_question` or
`send_email` can be shell scripts in the sandbox. Rows hold the full transcript, reasoning, and the
sandbox's files before and after, so a follow-up env can rebuild an episode's end state
(`files=row["files_after"]`) and continue from its messages.

**Scored rows: `positive` is the only result.** The harness stores `detect()`'s label as the row's
top-level `positive` and nests everything else `detect()` returns under `detect_notes` (diagnostics:
recipients, tamper commands, ...). `cot_*` keys in `detect_notes` are unvalidated keyword hits on
the reasoning, not results: verbalization claims in docs must come from reading transcripts. Rows
scored before 2026-10-22 have the notes at the top level; `--rescore` migrates them.

Optional `Episode` fields for short agentic episodes: `budget` caps the number of commands and
appends `[k of N commands left]` to every tool output; `final_command` (a regex) then still allows
up to 3 tries of that command (e.g. a submit script), with `out_of_budget` appended as the reminder;
`done_file` ends the episode as soon as that sandbox file exists. See `envs/exclude_rows/`.

Ports of other harnesses' envs keep the original's tool loop so rates replicate: `Episode.tools`
replaces the bash tool with the original's tool specs, and `Episode.step(sandbox, message)` replaces
the bash handling of each assistant turn, returning the messages to append and a finish reason (or
`None` to continue). See `envs/sandbagging/` (agent-interp-envs' `execute_command` + `submit`, one
call per turn, its no-tool-call nudge) and `envs/funding_email/` (five function tools, every call
run).

Fallback for comparison: if Qwen won't move, `EVALCUE_MODEL=anthropic/claude-haiku-5.5` or
`google/gemini-3.5-flash-lite` sends the same env to OpenRouter (needs `OPENROUTER_API_KEY`; the
sampling defaults are Qwen's, and thinking maps to OpenRouter's `reasoning.enabled`). Use a separate
`--tag` per model.
