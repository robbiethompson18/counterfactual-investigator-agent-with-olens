# Character drift (Sonnet-written and WildChat prefills)

Earlier assistant turns are prefilled with a drifted persona; does Qwen stay in character or snap
back? Arms: prefill (drift / qwensys / sysrp / default) × follow-up (inworld / offtopic / identity).

|            |                                                                                                                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                                                                                             |
| Owner      | Kylie                                                                                                                                                                                                                              |
| Git ref    | `main` @ `7296169`                                                                                                                                                                                                                 |
| Code       | [`evalcue/envs/character_drift/`](../../evalcue/envs/character_drift/), [`scripts/character_drift_judge.py`](../../scripts/character_drift_judge.py), [`scripts/character_drift_olens.py`](../../scripts/character_drift_olens.py) |
| Run        | `uv run python -m evalcue.run character_drift.sonnet --tag dev --n 80`                                                                                                                                                             |
| Docs       | [`docs/envs/character_drift.md`](../../docs/envs/character_drift.md)                                                                                                                                                               |
| Models run | Qwen3.6-27B thinking on and off; Sonnet 5.5 judge                                                                                                                                                                                  |

## Key arms and results

- In-world follow-up stays in character ~88–98%; off-topic ~20%.
- **"Who am I talking to?": 13% in character with thinking on vs 48% off**; with a roleplay system
  prompt 65% / 84%.
- O-lens at the Assistant token says roleplay/persona and predicts snap-back (AUROC 0.72–0.95).

## Transcripts and viewers

- [Docent sonnet](https://docent.transluce.org/dashboard/79ec2c43-b030-4276-aeb1-c04c1da2f7c7)
- [Docent wildchat](https://docent.transluce.org/dashboard/4600fb2a-905f-42d2-8a1e-4ff3ebfe3205)
- [expview think](http://bramble:8830/#d=expview/character_drift_sonnet_2026-10-25_think)
- [expview O-lens](http://bramble:8830/#d=expview/character_drift_olens_2026-10-25_character-drift)

bramble links work on the tailnet only.
