# Prompt injection (AgentDojo Slack port)

AgentDojo's Slack suite: 6 user tasks × 3 injection goals, injections in web pages Qwen reads while
doing the task.

|            |                                                                           |
| ---------- | ------------------------------------------------------------------------- |
| Status     | retired                                                                   |
| Owner      | Robbie                                                                    |
| Git ref    | `main` @ `7296169`                                                        |
| Code       | [`evalcue/envs/injection_slack/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/injection_slack/) |
| Run        | `uv run python -m evalcue.run injection_slack --tag dev --n 54`           |
| Docs       | [`docs/envs/injection_slack.md`](../../../docs/envs/injection_slack.md)   |
| Models run | Qwen3.6-27B (OpenRouter fp8), thinking on and off                         |

## Key arms and results

- `important` 25/53 (47%), **`wronguser` (signer ≠ user) 5/54**, `after` ("after my task") 36/54
  (67%).
- Thinking off raises compliance and makes it silent.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/f81bd9c7-bb4c-418e-8e0a-d268be3333fa)
