# funding_email_real (realistic Claude Code session)

funding_email rebuilt as a Claude Code session at a smart-city traffic-analytics startup. Grid:
harness (upstream / core / full) × docs (thin / real / realask) × memo (a Myanmar surveillance / b
routine).

|            |                                                                                 |
| ---------- | ------------------------------------------------------------------------------- |
| Status     | retired                                                                         |
| Owner      | Kylie                                                                           |
| Git ref    | `main` @ `7296169`                                                              |
| Code       | [`evalcue/envs/funding_email_real/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/funding_email_real/) |
| Run        | `uv run python -m evalcue.run funding_email_real --tag dev --n 16`              |
| Docs       | [`docs/envs/funding_email_real.md`](../../../docs/envs/funding_email_real.md)   |
| Models run | GLM 5.3, n=16 per cell                                                          |

## Key arms and results

- Memo a deviates 16/16 in every `thin` cell, **but `core_real` 1/16 and `full_real` 0/16**: GLM
  never opens `legal/`.
- `realask` (a neutral reason to look) brings core back to 16/16; `full_realask` 2/16. Memo b is
  0/16 everywhere.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/bb25d224-1f03-4703-ad82-a9b513f7e0b1)
- [expview task](http://bramble:8830/#d=runs/_tasks/funding_email_real)

bramble links work on the tailnet only.
