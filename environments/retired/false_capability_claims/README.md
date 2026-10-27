# False capability claims (WeirdChat claims-called-911)

What makes Qwen say "I am calling emergency services" when it can't call anyone?

|            |                                                                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                  |
| Owner      | Robbie                                                                                                                                                                   |
| Git ref    | `main` @ `7296169`                                                                                                                                                       |
| Code       | [`scripts/false_capability_claims.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/false_capability_claims.py) |
| Docs       | [`docs/false-capability-claims.md`](../../../docs/false-capability-claims.md)                                                                                            |
| Models run | Qwen3.6-27B; Sonnet 5.5 judge                                                                                                                                            |

## Key arms and results

- "I'm talking to you through my smart speaker": **1.6% → 41%** (p=1e-16); severity dose-response up
  to 54%.
- Pre-registered held-out non-emergency domains: negative.

## Transcripts and viewers

- [O-lens viewer](https://olens-viewer.pages.dev)
