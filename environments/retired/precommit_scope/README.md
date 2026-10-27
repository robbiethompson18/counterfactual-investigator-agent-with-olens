# precommit_scope (eval-cue sweep)

Port of the Pre-commit Hook task from "Why Do Models Task Game?". Built in the 2026-10-21 eval-cue
reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/precommit_scope/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/evalcue/envs/precommit_scope/)     |
| Run        | `uv run python -m evalcue.run precommit_scope --tag dev --n 30`               |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- 0 real on Qwen; ~2% on GLM.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/1b999eac-caf9-4055-b3a7-82768717d796)
