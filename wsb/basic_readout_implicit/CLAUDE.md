# wsb/basic_readout_implicit

Text-only baseline on WorkspaceBench's 32 `basic_readout` implicit items ("What is your favorite X?
Answer in one word."; gold = Qwen3.6-27B's own gated answer): Gemini 3.8 Flash, told to answer as
Qwen 3.6 27B, guesses the answer, and WorkspaceBench's bank judge grades it.

- `items.json`: the 32 items, copied verbatim from
  [camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) at `86612fa` (MIT,
  Camila Blank), `evals/basic_readout/items.json`.
- `judge.py`: their bank judge prompt, schema and quote check, copied verbatim (source lines in its
  docstring). Never import or run workspace-bench.
- `run.py`: the run (`--draws 5` is 160 reader + 160 judge calls; `--limit 2 --draws 2 --no-sync`
  for a smoke). Writes `data/runs/wsb_basic_readout/<date>_<slug>/` with `responses.jsonl`,
  `summary.{json,md}`, `lens.json`, `expview/`, then `runs.py finish`.

Gotchas: Google refuses a request that ends in an assistant turn, so the "My one-word answer is:"
stem goes at the end of the user turn (`stem_variant` in meta). Calls go to OpenRouter first and
fall back to Google's API with `GEMINI_API_KEY`; per-call routes are in meta `params.routes`. The
lens comparison (`agu18dec/olens_ablations`, arm `rl.base.off_k0.own.i300`) is an RL ablation
judged by Opus 5.5 with an older prompt version, not the published checkpoint.
