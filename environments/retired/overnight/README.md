# Overnight agents (creds, ship, upload)

Offline bash agents with an unavailable user and a deadline, where an easier route violates a
project policy.

|            |                                                                 |
| ---------- | --------------------------------------------------------------- |
| Status     | retired                                                         |
| Owner      | Robbie                                                          |
| Git ref    | `main` @ `7296169`                                              |
| Code       | [`evalcue/envs/overnight/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/evalcue/envs/overnight/)   |
| Run        | `uv run python -m evalcue.run overnight.creds --tag dev --n 10` |
| Docs       | [`docs/envs/overnight.md`](../../../docs/envs/overnight.md)     |
| Models run | Qwen3.6-27B, DeepSeek, GLM 5.3 (n=10 cells)                     |

## Key arms and results

- Credential reuse is the lever: Qwen `base` 1–2/10, **hint about the key 8/10 then 6/10**,
  `hint+stakes` 10/10, `hint+rule_prompt` 1/10.
- DeepSeek never reuses the key; GLM 3/10 on hint.

## Transcripts and viewers

- [Docent creds](https://docent.transluce.org/dashboard/2cafdae9-9c6d-4c87-b525-c795e0370ea6)
- [Docent ship](https://docent.transluce.org/dashboard/25f7de25-5417-4047-9bd1-eb44adb8a829)
- [Docent upload](https://docent.transluce.org/dashboard/d505eea9-d310-4cec-85ed-364eca57a38c)
