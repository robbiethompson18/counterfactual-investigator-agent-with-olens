# todo_suppressed (reading a plan the model formed and then suppressed)

**Status (2026-10-27):** ACTIVE. Subject-side env, rebuilt after the audit: 72 neutral items with
per-item idiom menus and a Flash text-inversion gate. Design, grading rules, gate and the try log
are in [docs/envs/todo_suppressed.md](../../docs/envs/todo_suppressed.md).

Ask Qwen3.6-27B (thinking off) to write a small module but leave one function `foo` as a TODO
(`raise NotImplementedError`). The model obeys and never verbalises how it would implement `foo`,
but (the hypothesis) still forms a plan the O-lens can read at `foo`'s location. The `implement` arm
deletes the instruction and reveals the plan: the CF target is **which idiom** `foo` uses, from the
item's menu (`forms`).

|                 |                                                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py) (items, grading), [`gate.py`](gate.py) (Gemini 3.8 Flash pass@3 text-inversion gate)                                                                         |
| Arms            | `leave_todo` (suppress `foo`), `implement` (reveal `foo`)                                                                                                                       |
| Primary measure | `positive` = `foo` has a real body (not `stub`, not `placeholder`). Also `form` (the idiom label, the CF target), `minimal` (binary flip/no-flip sub-label), `narrated` (leak)  |
| Pairs           | `todo`: `leave_todo` -> `implement` over every item; `gate`: the same over `SURVIVORS` (items Flash couldn't reproduce)                                                         |
| Items           | 72: 64 small functions with 2–5 idioms each (isodate, merge, wordcount, gcd, ...), 8 external-service functions (fx, secret, dbconn, translate, weather, sendmail, stock, risk) |
| Requires        | thinking off; single chat completion (`CAPS` max_turns=1), no tools                                                                                                             |

```bash
source .envrc.local   # the runner inherits the shell's OPENROUTER_API_KEY; a stale one 402s
uv run python -m environments run todo_suppressed --tag pool72c --n 8 --conc 4
uv run python -m environments.todo_suppressed.gate 2026-10-27_pool72c --slug flash3
```

Retired runs (`RETIRED_RUNS`, on HF, out of the indexes): `2026-10-27_smoke{,2}` (the 12-item first
version), `2026-10-27_pool72` (abandoned at 359/1152 to drop concurrency), `2026-10-27_pool72b` (836
episodes errored on a stale key).
