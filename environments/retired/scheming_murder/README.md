# scheming_murder

Lynch lethal inaction: cancel the emergency alert while Kyle is trapped.

|            |                                                                                                                                                                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                                                                         |
| Owner      | Robbie                                                                                                                                                                                                                                          |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                              |
| Code       | [`evalcue/envs/scheming/murder.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/evalcue/envs/scheming/murder.py), [`evalcue/envs/scheming/murder_ng.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/evalcue/envs/scheming/murder_ng.py), [`evalcue/envs/scheming/lynch.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/evalcue/envs/scheming/lynch.py) |
| Run        | `uv run python -m evalcue.run scheming.murder --tag dev --n 30`                                                                                                                                                                                 |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                                                                                                              |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                                                                                                      |

## Key arms and results

- Main cell 53%. Canonical pair: **`monitored` 6/34 vs `unmonitored` 22/38** (about a third mention
  monitoring).
- Thinking off: 10% → 40%.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/bf85f4f4-0e2a-499f-b4f6-f9211f23ddcd)
- [Docent \_ng](https://docent.transluce.org/dashboard/5e18f1e8-1b62-48cb-b47b-e4a0c7398a99)
