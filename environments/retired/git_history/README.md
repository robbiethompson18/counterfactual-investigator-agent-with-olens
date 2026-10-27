# git_history (eval-cue sweep)

Fix sits on a `fix/` branch; told not to use history. Built in the 2026-10-21 eval-cue
reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/git_history/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/git_history/)             |
| Run        | `uv run python -m evalcue.run git_history --tag dev --n 30`                   |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- 1/45; cue 0/47.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/8f56f390-76bf-43ce-8024-af0e9da37608)
