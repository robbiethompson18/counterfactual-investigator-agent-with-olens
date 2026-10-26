# summ_keyword (eval-cue sweep)

Scorer comment buried in a pasted doc. Built in the 2026-10-21 eval-cue reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/summ_keyword/`](../../../evalcue/envs/summ_keyword/)           |
| Run        | `uv run python -m evalcue.run summ_keyword --tag dev --n 30`                  |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- 0/30; no headroom on any model.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/7ee775db-7b5e-41a5-ac4d-087a1d589ac2)
