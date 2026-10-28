# vendor/: reference only

The three submodules here are for reading, not running:

- `detective-joracle`: Agam Bhatia's CHIVE investigator harness
  ([agu18dec/detective-joracle](https://github.com/agu18dec/detective-joracle), MIT).
- `agent-interp-envs`: Aditya Singh and Gerson Kroiz's agent environments
  ([gkroiz/agent-interp-envs](https://github.com/gkroiz/agent-interp-envs), MIT).
- `workspace-bench`: Camila Blank's WorkspaceBench
  ([camilablank/workspace-bench](https://github.com/camilablank/workspace-bench), MIT), pinned to
  `86612fa` (2026-10-27): item banks, judges and baselines for Qwen3.6-27B. We use it as the source
  of item banks, read-position rules and judge prompts for `docs/wsb-flash-prompt-only.md`. It ships
  its own `CLAUDE.md`/`AGENTS.md` and `.claude/skills/`, which `.claude/settings.json`'s
  `claudeMdExcludes` keeps Claude Code from loading as instructions.

Read them, and copy prompts or scenario text verbatim into our own code, with attribution (repo,
authors, license) next to the copy. Data (item banks) is not copied into git: a build script reads
the data file from the pinned submodule and uploads the bank to HF (`wsb/build_banks.py`). Never
import, run, wrap or monkeypatch them: port what you need into our harnesses (`evalcue/`,
`cfbench/`). Ruff bans `import detective_joracle` and `import wsbench` (`TID251`). None is a uv
dependency.

After cloning, `git submodule update --init` fetches them.

## What we took from where

| Source                                           | Ours                                                                                           |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| detective-joracle `data/chive/splits/`           | `splits/` (`all300`, `climb_dev96`, `climb_test204`, `hard92`)                                 |
| agent-interp-envs `environments/secret_number`   | `evalcue/envs/secret_number`                                                                   |
| agent-interp-envs `environments/grade_your_own`  | `evalcue/envs/exclude_rows` (a rewrite)                                                        |
| agent-interp-envs `sandbagging`, `funding_email` | `evalcue/envs/sandbagging`, `evalcue/envs/funding_email` (both replicate the original screens) |
| workspace-bench judge prompts                    | `wsb/judge.py`, `wsb/jailbreak.py`                                                             |
| workspace-bench `evals/*/items.json`             | HF `banks/wsb/<family>.json`, built by `wsb/build_banks.py` (not in git)                       |

Until commit `a1e8991`, detective-joracle was a uv workspace member and `scripts/run_chive_or.py`
ran it; see `docs/smoke25-results.md` for what that produced.
