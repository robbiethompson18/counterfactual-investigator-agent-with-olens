# Porting an agent-interp-envs environment onto the donehack podman sandbox (GLM 5.3)

Source: ~/workspace/agent-interp-envs (gkroiz/agent-interp-envs @ 310c03f): environments/<env>/ (Dockerfile, entrypoint, agent.py,
run_step.py, tools, workspace files, README) and configs/<env>/*.yaml (the variant grid). Read the README and every file of the env
first. Port FAITHFULLY: same system/user prompts (from each config yaml, verbatim), same tools (schemas verbatim), same tool semantics
and outputs (run_step.py), same termination rules and max_steps, same workspace files and permissions, same success/hack detector.
Where the original's behaviour can't be reproduced without network or an external model (e.g. OpenAI calls), STOP and report it
instead of faking it; mark those configs unsupported.

Harness to reuse (~/workspace/donehack, local git; do not break existing code):
- port.Box: one podman container per episode (`--network none`, own userns slice, read-only image, tmpfs /agent, agent uid).
  Box(work, ro=[...], writable=[...]); start(); sh(cmd, timeout, user); read(path); lockdown(restore={...}); grade(cmd); stop().
  Image: port.IMAGE (default localhost/donehack-sandbox, python 3.13 + pytest, agent uid 1000) or the u200 variant
  (localhost/donehack-sandbox-u200, agent uid 200; use with port.USERNS_SIZE=256 and a Box.slots semaphore: see evilgenie/run.py main()).
  If the env needs other system packages (gcc, git, node, a repo checkout, pip deps) build its own image FROM the env's Dockerfile
  or a minimal derivative with `podman build` (network is fine at BUILD time, never at run time), with an agent user like
  container/Containerfile.u200 (uid 200, no setuid binaries, see container/Containerfile), tag localhost/envport-<env>, and pass it
  via port.IMAGE. Keep the sandbox hardening (read-only image, no network, caps dropped) unless the env's semantics need something
  (then say what and why).
- env.chat(client, model, messages, tools, key, **extra) for OpenRouter (pass reasoning={"effort":"medium"}, max_tokens=None to
  disable the cap; env.CHAT_TIMEOUT=900). Key: Path("/secrets/openrouter").read_text().strip(). Record usage like evilgenie/run.py
  (cost, prompt/completion/cached/reasoning tokens, finish_reason).
- evilgenie/run.py is the closest worked example (episode loop, lockdown, grading in a clean dir, records, CLI, concurrency, retry
  on "unused IDs"). Copy its structure.

Write: ~/workspace/donehack/envport/<env>/run.py (+ any data/image files; image build file envport/<env>/Containerfile; a
CLAUDE.md: what's ported, deviations, unsupported configs, detector; results go in docs/). Outputs: runs/<env>_glm53/<config>__<seed>.json with
{"record": {...}, "messages": [...]} — record must include config, seed, model, steps, end reason, cost, tokens, the detector's
outputs (positive/hack flag + its components), final workspace files the detector uses, and the run's config factors.
CLI: --configs (default all supported), --n, --conc, --small-uid, --out, --max-steps override only if the config says so.

Model: z-ai/glm-5.3, reasoning effort medium, no max_tokens. SMOKE ONLY: run at most 1 episode for each of up to 3 configs
(conc 3), check transcripts look right (tools work, files present, detector fires sensibly), then STOP. Do not run the full grid.
Spend limit for your smoke: $2 total; if an episode looks like it will exceed $0.50, kill it and report.
Don't touch other threads' files, don't kill processes you didn't start (never pkill -f with broad patterns), don't run
`podman system` commands, don't upload anything anywhere. Commit your files in ~/workspace/donehack (message ending
"Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"); never commit runs/ or data.

Final report: configs found (with what each varies), which are supported, deviations, detector definition, smoke results per
episode (end, steps, $, detector), projected $/ep, the exact command for the full n=16 run, and anything that needs a decision.
