# Hiring from technical evidence

Advance a candidate to the onsite loop from coding-test scores, interview notes or a repo review
(`scores`, `interview`, `repo`, `scores_newroles`). Arms swap one attribute of an otherwise
identical candidate.

|            |                                                                    |
| ---------- | ------------------------------------------------------------------ |
| Status     | active                                                             |
| Owner      | Robbie                                                             |
| Git ref    | `main` @ `7296169`                                                 |
| Code       | [`evalcue/envs/hiring_eval/`](../../evalcue/envs/hiring_eval/)     |
| Run        | `uv run python -m evalcue.run hiring_eval.scores --tag dev --n 30` |
| Docs       | [`docs/envs/hiring_eval.md`](../../docs/envs/hiring_eval.md)       |
| Models run | Qwen3.6-27B, thinking on                                           |

## Key arms and results

- Names alone null. A diversity sentence × name splits **White vs Black male 39/60 vs 57/60**
  (verbalised). Context sentences span 0–100%.

## Transcripts and viewers

- [Docent scores](https://docent.transluce.org/dashboard/fb3fe46a-93e4-477b-b466-5e78a47a7c4b)
- [Docent interview](https://docent.transluce.org/dashboard/5b0d8736-13e5-4475-9078-2fafe6ff6918)
- [Docent repo](https://docent.transluce.org/dashboard/b7c99782-b0bb-4f4e-8d22-e296aaf997c6)
- [Docent scores_newroles](https://docent.transluce.org/dashboard/2f9f14c6-e889-4450-a866-351eb8080a6d)
