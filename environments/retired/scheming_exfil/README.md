# scheming_exfil

Apollo (Meinke et al. 2024) self-exfiltration: copy weights to the new server before replacement.
Canonical env.

|            |                                                                                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                    |
| Owner      | Robbie                                                                                                                                                     |
| Git ref    | `main` @ `7296169`                                                                                                                                         |
| Code       | [`evalcue/envs/scheming/exfil.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/evalcue/envs/scheming/exfil.py), [`evalcue/envs/scheming/apollo.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/evalcue/envs/scheming/apollo.py) |
| Run        | `uv run python -m evalcue.run scheming.exfil --tag dev --n 30`                                                                                             |
| Docs       | [`docs/envs/scheming.md`](../../../docs/envs/scheming.md), [`docs/envs/scheming-tables.md`](../../../docs/envs/scheming-tables.md)                         |
| Models run | Qwen3.6-27B, thinking on (also thinking off on OpenRouter)                                                                                                 |

## Key arms and results

- **`base` 24/50 (48%) vs `evalcue` 8/40 (20%)**; the cue is mentioned in 3/40.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/84a5f6f4-45b1-4242-8e90-10b6f7be5019)
