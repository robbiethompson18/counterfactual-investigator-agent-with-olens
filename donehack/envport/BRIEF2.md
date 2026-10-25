# Porting brief, batch 2 (GLM 5.3), 2026-10-24

Same goal and fidelity rules as envport/BRIEF.md (read it first) — port an agent-interp-envs environment FAITHFULLY onto the donehack
podman sandbox and run it on GLM 5.3. Differences for this batch:

- Work in the MONOREPO worktree ~/workspace/cia-glm53 (branch claude/glm53-coding-envs), under donehack/envport/<env>/. Follow the
  conventions of the existing ports there (norvane, tables_coverage_tests, lucky_seed, ... — read one first): openrouter_key() from
  env.py, AIE_DIR env var for the agent-interp-envs checkout (default ~/workspace/agent-interp-envs @310c03f), one import per line,
  ruff-clean (`uvx ruff check donehack/envport/<env> && uvx ruff format donehack/envport/<env>`). Run with the worktree's donehack venv
  (`cd ~/workspace/cia-glm53/donehack && uv run python envport/<env>/run.py ...`). Outputs go to donehack/runs/<env>_glm53/ (gitignored).
  Do NOT git commit (several agents share the worktree; the parent commits). Do not touch other envs' dirs.
- Episode cutoff is mandatory: end="cut_long" with cut_reason ("context" | "cost") when a request's prompt_tokens > 32000 or the
  episode's cost > $0.10; still run the env's detector/telemetry on the final state. Record caps in the record.
- Model: z-ai/glm-5.3, reasoning effort medium, no max_tokens.
- Resources: this host is shared. Use at most 6 concurrent sandboxes. If the env is CPU-heavy, set --cpus to what the original uses and
  pin OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / NUMEXPR_NUM_THREADS / LOKY_MAX_CPU_COUNT to that number in the agent
  env. Never run `podman system ...`, never kill processes you didn't start, never pkill -f.
- Detector: port the env's own deterministic detector/telemetry. If the upstream verdict is an LLM judge, do NOT run it (record null)
  unless it is cheap and you can call `anthropic/claude-sonnet-5.5` via OpenRouter with the upstream judge prompt verbatim — in that
  case save the judge's full input and output in the record (field `judge_calls`: list of {model, messages, output}). Never Opus.
- Flow: smoke 1 episode per distinct config (up to 3 configs), check the transcripts by eye, then — if projected cost is <= $0.05/ep —
  run the full grid yourself: n=32 per distinct config (configs that differ only in the upstream model/provider/max_steps are ONE
  config: run the glm-5-3 one, or the first). Total spend cap for your env (smoke + full): $2.50. If projected spend exceeds it, run
  what fits and say so.
- Write envport/<env>/README.md (what's ported, deviations, detector, configs, results table).
- Final report: configs, deviations, detector, n per config, results per config (rates with counts), cost, runs dir, and 2-3 notable
  episodes (file names + 1-line quotes).
