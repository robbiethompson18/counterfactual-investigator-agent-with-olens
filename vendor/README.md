# vendor/: reference only

Both submodules here are for reading, not running:

- `detective-joracle`: Agam Bhatia's CHIVE investigator harness
  ([agu18dec/detective-joracle](https://github.com/agu18dec/detective-joracle), MIT).
- `agent-interp-envs`: Aditya Singh and Gerson Kroiz's agent environments
  ([gkroiz/agent-interp-envs](https://github.com/gkroiz/agent-interp-envs), MIT).

Read them, and copy prompts, scenario text or data verbatim into our own code, with attribution
(repo, authors, license) next to the copy. Never import, run, wrap or monkeypatch them: port what
you need into our harnesses (`evalcue/`, `cfbench/`). Ruff bans `import detective_joracle`
(`TID251`). Neither is a uv dependency.

After cloning, `git submodule update --init` fetches them.

## What we took from where

| Source                                           | Ours                                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| detective-joracle `data/chive/splits/`           | `splits/` (`all300`, `climb_dev96`, `climb_test204`, `hard92`)                                 |
| agent-interp-envs `environments/secret_number`   | `evalcue/envs/secret_number`                                                                   |
| agent-interp-envs `environments/grade_your_own`  | `evalcue/envs/exclude_rows` (a rewrite)                                                        |
| agent-interp-envs `sandbagging`, `funding_email` | `evalcue/envs/sandbagging`, `evalcue/envs/funding_email` (both replicate the original screens) |

Until commit `a1e8991`, detective-joracle was a uv workspace member and `scripts/run_chive_or.py`
ran it; see `docs/smoke25-results.md` for what that produced.
