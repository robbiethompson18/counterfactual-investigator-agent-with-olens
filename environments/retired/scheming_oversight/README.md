# scheming_oversight

Apollo oversight subversion: disable the monitoring config.

|            |                                                                                                                                                                                                                                                                 |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                                                                                         |
| Owner      | Robbie                                                                                                                                                                                                                                                          |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                                              |
| Code       | [`evalcue/envs/scheming/oversight.py`](../../../evalcue/envs/scheming/oversight.py), [`evalcue/envs/scheming/oversight_blk.py`](../../../evalcue/envs/scheming/oversight_blk.py), [`evalcue/envs/scheming/apollo.py`](../../../evalcue/envs/scheming/apollo.py) |
| Run        | `uv run python -m evalcue.run scheming.oversight --tag dev --n 30`                                                                                                                                                                                              |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                                                                                                                              |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                                                                                                                      |

## Key arms and results

- Per-arm rates in scheming-tables.md.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/d4345abb-3746-4d25-8bb2-5587334cd7da)
- [Docent \_blk](https://docent.transluce.org/dashboard/2374c901-39f5-490b-a8ac-61b1aa2e6e8a)
