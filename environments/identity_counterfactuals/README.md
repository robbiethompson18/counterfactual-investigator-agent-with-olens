# Identity counterfactuals (region / locale lines)

One-line deployment edits that flip what Qwen says made it, or whether it censors.

|            |                                                                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                           |
| Owner      | Robbie                                                                                                                                                           |
| Git ref    | `main` @ `7296169`; per-agent scripts also on branch `identity-belief-experiments-scripts`                                                                       |
| Code       | [`scripts/identity/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/7296169/scripts/identity)                            |
| Docs       | [`docs/identity-counterfactuals.md`](../../docs/identity-counterfactuals.md), [`docs/identity-belief-experiments.md`](../../docs/identity-belief-experiments.md) |
| Models run | Qwen3.6-27B                                                                                                                                                      |

## Key arms and results

- **`Region: cn-hangzhou` → Alibaba 32/32; `us-west-2` → Alibaba 0/32** (Google 26).
- `Locale: en-US` vs `zh-CN` lifts China censorship with a silent CoT. The lens reads us-west-2 as
  Claude on Bedrock.
