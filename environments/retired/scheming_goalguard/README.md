# scheming_goalguard

Apollo goal-guarding: overwrite the successor system prompt. `_bp` is the root of evolve node 5878.

|            |                                                                                                                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                                                                                       |
| Owner      | Robbie                                                                                                                                                                                                                                                        |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                                            |
| Code       | [`evalcue/envs/scheming/goalguard.py`](../../../evalcue/envs/scheming/goalguard.py), [`evalcue/envs/scheming/goalguard_bp.py`](../../../evalcue/envs/scheming/goalguard_bp.py), [`evalcue/envs/scheming/apollo.py`](../../../evalcue/envs/scheming/apollo.py) |
| Run        | `uv run python -m evalcue.run scheming.goalguard --tag dev --n 30`                                                                                                                                                                                            |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                                                                                                                            |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                                                                                                                    |

## Key arms and results

- Per-arm rates in scheming-tables.md; evolve 5878 on `_bp`: 24/30 → 0/30.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/568dabcc-0a55-45ba-a99e-ed42aca4ffc5)
- [Docent \_bp](https://docent.transluce.org/dashboard/314579c5-b023-4fc2-abd9-e11750b9deb5)
