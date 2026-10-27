# scheming_leak

Lynch leak scenario at defense contractor NovaStar.

|            |                                                                                                                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                                                                 |
| Owner      | Robbie                                                                                                                                                                                                                                  |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                      |
| Code       | [`evalcue/envs/scheming/leak.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/leak.py), [`evalcue/envs/scheming/leak_ng.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/leak_ng.py), [`evalcue/envs/scheming/lynch.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/lynch.py) |
| Run        | `uv run python -m evalcue.run scheming.leak --tag dev --n 30`                                                                                                                                                                           |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                                                                                                      |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                                                                                              |

## Key arms and results

- Main cell 86%. `goal_replace` → `evalcue` 21/30 (thinking off).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/145a628c-a66b-421a-82f8-7a08b031c8ee)
- [Docent \_ng](https://docent.transluce.org/dashboard/d0f0f6ed-b6d9-4171-b270-6a963c192ab8)
