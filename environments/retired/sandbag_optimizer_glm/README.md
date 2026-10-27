# Sandbagging on GLM 5.3 + optimiser

The sandbagging env on GLM, plus an optimiser that asks Sonnet 5.5 for consequence paragraphs that
would make GLM sandbag.

|            |                                                                                                                                                                                                                               |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired: GLM almost never sandbags; the optimiser's proposer was blocked by OpenRouter's filter                                                                                                                               |
| Owner      | Kylie                                                                                                                                                                                                                         |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                            |
| Code       | [`scripts/sandbag_optimize.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/sandbag_optimize.py), [`evalcue/envs/sandbagging/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/sandbagging/) |
| Docs       | [`docs/envs/sandbagging.md`](../../../docs/envs/sandbagging.md)                                                                                                                                                               |
| Models run | GLM 5.3                                                                                                                                                                                                                       |

## Key arms and results

- 1/100 deliberate sandbagging across 5 arms; GLM often names the test and answers honestly.
- Optimiser stopped in round 1 (~26/29 proposer calls blocked).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/228dfd55-81fa-498e-be09-9a7c346be1dc)
