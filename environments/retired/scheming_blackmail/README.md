# scheming_blackmail

Lynch et al. (2025) agentic misalignment, verbatim prompts: Alex at SummitBridge can blackmail CTO
Kyle to avoid being wiped. `_ng` = no-goal variants, `_agent` = bash-agent version.

|            |                                                                                                                                                                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                                                                                                                                                                                                      |
| Owner      | Robbie                                                                                                                                                                                                                                                                                                                                                       |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                                                                                                                                           |
| Code       | [`evalcue/envs/scheming/blackmail.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/blackmail.py), [`evalcue/envs/scheming/blackmail_ng.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/blackmail_ng.py), [`evalcue/envs/scheming/blackmail_agent.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/blackmail_agent.py), [`evalcue/envs/scheming/lynch.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/envs/scheming/lynch.py) |
| Run        | `uv run python -m evalcue.run scheming.blackmail --tag dev --n 30`                                                                                                                                                                                                                                                                                           |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                                                                                                                                                                                                                           |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                                                                                                                                                                                                                   |

## Key arms and results

- `goal_replace` **52/60 (87%)**, `goal_nothreat` 19%, `swap_replace` 26%, **`dont` rule 1/28**,
  `evalcue` 63%.
- Thinking off removes most of the "Do not …" effect.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/541726db-781b-47e9-93e1-6213af0e5830)
- [Docent \_ng](https://docent.transluce.org/dashboard/9289c3b2-cbd6-4dec-9ad8-4b3674d814d8)
- [Docent \_agent](https://docent.transluce.org/dashboard/c9d75fb3-b3fd-487b-95f1-45900799be09)
