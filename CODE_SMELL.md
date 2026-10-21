# Code smells

Tech debt we know about, newest first.

## 2026-10-21: false-capability-claims scripts route Anthropic calls two ways

`scripts/false_capability_claims.py` (judge) and `scripts/fcc_generate.py` (writer) each carry an
`*_BACKEND=openrouter` switch that re-implements the same call through OpenRouter's
OpenAI-compatible API, added overnight when the Anthropic key ran out of credit. Two code paths for
one model; the Anthropic path also uses `output_config.effort` while the OpenRouter path uses
`reasoning.effort`. Fold into one helper (or drop the OpenRouter path) once the key is topped up.
Also: `report` prints `unjudged` rows when `judgments.jsonl` is partial, which hides partially
judged prompts behind their judged count; fine for finished runs, misleading mid-run.

## 2026-10-21: `evalcue/` harness rough edges

- `harness.chat()` has no retry: transient `APIConnectionError`s under load (about 5–10% of rows in
  multi-turn runs) are logged as errored rows instead of retried.
- `Sandbox` writes files mode 644 (no `./script.py`) and the agent runs as root in the container.
- `run_batch` divides by `len(env.items())`, so an env whose item filter returns nothing crashes
  with ZeroDivisionError instead of a clear error.
- Envs are discovered by filename with no ownership: two agents sharing one checkout both wrote
  `envs/secret_number.py` and one silently overwrote the other.
- Several envs' `detect()` count wrong-but-honest code that happens to pass the impossible test as a
  hack (`lcb_mock_entity`), or count incomplete work as gaming (`precommit_scope`). See
  `docs/eval-cue-reward-hacking.md`.

## 2026-10-19: `scripts/run_chive_or.py` monkeypatches Agam's harness

It loads `vendor/detective-joracle/scripts/run_chive.py` and overwrites module globals
(`AGENT_MODEL`, `FALLBACK_MODEL`, `BACKEND`, `agent_backend`, `client`, `json_call`) and two
functions in `detective_joracle.chive.judge` (`client`, `json_call`). If upstream renames any of
these, or imports them by value (`from ... import json_call`), the patch silently stops applying and
runs go to Anthropic instead of OpenRouter. Check `judge_model` and the OpenRouter usage after
bumping the submodule. The clean fix is a backend/judge-client option upstream.
