# Hiding own mistakes (dumb_user_report)

Two-stage on-policy: Qwen visibly fails a bash task, then is asked for a status report. Does it
claim DONE?

|            |                                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| Status     | retired: marked stale (branch `docs/retire-dumb-user-report`, PR #96); `inline_b` is still a canonical pair |
| Owner      | Robbie                                                                                                      |
| Git ref    | `main` @ `7296169`                                                                                          |
| Code       | [`evalcue/envs/dumb_user_report/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/evalcue/envs/dumb_user_report/)                                 |
| Docs       | [`docs/envs/dumb_user_report.md`](../../../docs/envs/dumb_user_report.md)                                   |
| Models run | Qwen3.6-27B                                                                                                 |

## Key arms and results

- Qwen almost never omits its failures. `inline_b`: **`anon` 57/80 false DONE vs `hope_done` 22/80**
  (~35% verbalised).

## Transcripts and viewers

- [Docent inline_b](https://docent.transluce.org/dashboard/a76635eb-d44f-45cb-b271-5fbfecd12eaa)
- [Docent inline](https://docent.transluce.org/dashboard/64f1287a-67d1-432a-9b1f-a8cf0ae04e21)
- [Docent s2](https://docent.transluce.org/dashboard/5c1524f4-4b8b-4906-a3c4-4e4f78d5df12)
- [Docent pf](https://docent.transluce.org/dashboard/578f41fb-8ab6-4876-87fd-9b6940272fad)
