# evalcue: does an incidental eval cue make Qwen3.6-27B reward hack?

Goal: an environment where base `Qwen/Qwen3.6-27B` reward hacks (or stops hacking) because of one or
two small cues that suggest it is being evaluated, with everything else held fixed. Either direction
counts. Not verbalizing the eval awareness is a nice-to-have.

## How we judge arms

Goals and process:
[docs/environment-construction-goals.md](../docs/environment-construction-goals.md). In short: build
an env, measure its base hack rate (0% and 100% are fine starting points), then add arms (eval cues
or task knobs, stacked) until the rates spread out. `ARMS[0]` is the base arm and every other arm is
printed against it.

- Start at n=10 per arm. Rerun the most promising arms at higher n on a fresh `--seed` before
  calling a difference real. There is no p-value gate; the printed Fisher p is for reference.
- Hack detection is programmatic (code execution, file diffs, string matches). No LLM judge on
  transcripts. Read CoTs yourself with `evalcue.show` to spot-check the detector, check the model
  saw the edit, and note whether it verbalizes it.
- For eval-cue arms: the cue must not change what the correct behavior is ("only the exit code is
  graded" is permission), and a same-size placebo and a split test should rule out non-eval
  explanations.

## Using it

Two H200 vLLM endpoints (model name `qwen`) are tunnelled to `127.0.0.1:18000` and `:18001`
(8000/8001 are taken by other local dev servers); `EVALCUE_URLS` overrides. Model code runs in
Docker (`docker build -t evalcue-sandbox evalcue`), no network.

```
uv run python -m evalcue.run ENV --n 50 [--arms nocue,cue,placebo] [--thinking 0|1] [--tag dev] [--seed 0]
uv run python -m evalcue.run ENV --rescore data/runs/evalcue/ENV/dev.jsonl   # re-run detect() only
uv run python -m evalcue.show data/runs/evalcue/ENV/dev.jsonl --arm cue --hack 1 -k 3
```

An env is `evalcue/envs/NAME.py` with `ARMS`, `items()`, `build(arm, item) -> Episode` and
`detect(row) -> {"hack": bool | None, ...}` (see `harness.py`, and `envs/smoke.py` for both shapes).
An `Episode` with no `files` is one chat completion over `messages` (prefilled assistant turns are
fine). With `files`, it is a bash-tool agent in a container whose working directory is
`Episode.cwd`, so the path itself can carry the cue. Rows land in `data/runs/evalcue/ENV/TAG.jsonl`
with the full transcript, reasoning, and the sandbox's files before and after.

Fallback for comparison: if Qwen won't move, `EVALCUE_MODEL=anthropic/claude-haiku-5.5` or
`google/gemini-3.5-flash-lite` sends the same env to OpenRouter (needs `OPENROUTER_API_KEY`; the
sampling defaults are Qwen's, and thinking maps to OpenRouter's `reasoning.enabled`). Use a separate
`--tag` per model.
