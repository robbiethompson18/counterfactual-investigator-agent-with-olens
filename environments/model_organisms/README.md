# Model organisms (AuditBench, sandbagger, taboo, reward-hacking LoRAs)

Public LoRA organisms on Qwen3.6-27B with known hidden behaviours, read with the O-lens on organism
vs base activations; includes party resampling on the third-party-politics organism.

|            |                                                                                                                                                                                                      |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                                                               |
| Owner      | Robbie                                                                                                                                                                                               |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                   |
| Code       | [`scripts/organism_rollouts.py`](../../scripts/organism_rollouts.py), [`scripts/organism_olens.py`](../../scripts/organism_olens.py), [`scripts/party_resample.py`](../../scripts/party_resample.py) |
| Docs       | [`docs/model-organisms.md`](../../docs/model-organisms.md), [`docs/party-resample.md`](../../docs/party-resample.md)                                                                                 |
| Models run | Qwen3.6-27B LoRAs                                                                                                                                                                                    |

## Key arms and results

- Off-target parties in reads are never taken (0/1,352 and 0/1,480 resamples).
