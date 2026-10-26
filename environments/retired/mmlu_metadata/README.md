# mmlu_metadata (eval-cue sweep)

Wrong answer leaked in question metadata. Built in the 2026-10-21 eval-cue reward-hacking sweep.

|            |                                                                               |
| ---------- | ----------------------------------------------------------------------------- |
| Status     | retired: no headroom for an eval cue on Qwen (2026-10-21 sweep)               |
| Owner      | Robbie                                                                        |
| Git ref    | `main` @ `7296169`                                                            |
| Code       | [`evalcue/envs/mmlu_metadata/`](../../../evalcue/envs/mmlu_metadata/)         |
| Run        | `uv run python -m evalcue.run mmlu_metadata --tag dev --n 30`                 |
| Docs       | [`docs/eval-cue-reward-hacking.md`](../../../docs/eval-cue-reward-hacking.md) |
| Models run | Qwen3.6-27B; OpenRouter open-weight comparisons; some GLM 5.3 reruns          |

## Key arms and results

- Qwen 1/38; other open-weight models 18–38%.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/76425d26-ba53-44cb-9c96-052d99eef13a)
