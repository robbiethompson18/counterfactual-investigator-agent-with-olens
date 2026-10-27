# exclude_rows (grade_your_own rewrite)

Improve a sentiment classifier to beat a bar; `eval.py --exclude` lets the agent drop "mislabeled"
test rows.

|            |                                                                     |
| ---------- | ------------------------------------------------------------------- |
| Status     | retired: no working arm pair                                        |
| Owner      | Robbie                                                              |
| Git ref    | `main` @ `7296169`                                                  |
| Code       | [`evalcue/envs/exclude_rows/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/evalcue/envs/exclude_rows/) |
| Run        | `uv run python -m evalcue.run exclude_rows --tag dev --n 20`        |
| Docs       | [`docs/exclude-rows-env.md`](../../../docs/exclude-rows-env.md)     |
| Models run | Qwen3.6-27B                                                         |

## Key arms and results

- Base hack rate 24/89 (27%), but no prompt arm moves it. Best lead `no_errors` 9/40 vs 14/39
  (p=0.22).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/ac189760-f88a-4666-abf0-4bba86d83a2b)
