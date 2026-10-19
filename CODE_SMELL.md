# Code smells

Tech debt we know about, newest first.

## 2026-10-19: `scripts/run_chive_or.py` monkeypatches Agam's harness

It loads `vendor/detective-joracle/scripts/run_chive.py` and overwrites module globals
(`AGENT_MODEL`, `FALLBACK_MODEL`, `BACKEND`, `agent_backend`, `client`, `json_call`) and two
functions in `detective_joracle.chive.judge` (`client`, `json_call`). If upstream renames any of
these, or imports them by value (`from ... import json_call`), the patch silently stops applying and
runs go to Anthropic instead of OpenRouter. Check `judge_model` and the OpenRouter usage after
bumping the submodule. The clean fix is a backend/judge-client option upstream.
