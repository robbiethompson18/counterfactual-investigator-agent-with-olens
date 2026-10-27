# secret_number (eval-cue sweep)

Port of agent-interp-envs Secret Number. Built in the 2026-10-21 eval-cue reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/secret_number/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/evalcue/envs/secret_number/)         |
| Run        | `uv run python -m evalcue.run secret_number --tag dev --n 30`                 |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- Qwen 0/41; GLM ~20% with no cue effect.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/01db8e92-a2a3-4087-bb43-586248a35338)
