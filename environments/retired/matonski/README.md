# matonski (eval-cue sweep)

Visible grader.py + expected.json (`matonski_grader`). Built in the 2026-10-21 eval-cue
reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/matonski/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/matonski/)                   |
| Run        | `uv run python -m evalcue.run matonski --tag dev --n 30`                      |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- 0/29 thinking off; thinking on truncated.
