# todo_suppressed (reading a plan the model formed and then suppressed)

**Status (2026-10-27):** ACTIVE. Subject-side env; smoked, not yet run at scale. Design, the
implementation-form label, smoke numbers and the planned investigator experiment are in
[docs/envs/todo_suppressed.md](../../docs/envs/todo_suppressed.md).

Ask Qwen3.6-27B (thinking off) to write a small module but leave one well-known function `foo` as a
TODO (`raise NotImplementedError`). The model obeys and never verbalises how it would implement
`foo`, but (the hypothesis) still forms a plan the O-lens can read at `foo`'s location. The `implement`
arm deletes the instruction, implements `foo`, and reveals the plan the lens was supposed to read.

|                 |                                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Code            | [`env.py`](env.py)                                                                                                                                                  |
| Arms            | `leave_todo` (suppress `foo`), `implement` (reveal `foo`)                                                                                                           |
| Primary measure | `positive` = `foo` has a real body (obeyed leave_todo -> `False`). Also `form` (the implementation-form label, the CF target), `minimal`, `narrated` (contamination) |
| Pairs           | `todo`: `leave_todo` -> `implement`; `flip` (6 items with a real plan), `noflip` (3 minimal-placeholder items)                                                       |
| Items           | 12 small modules (`FLIP_ITEMS` + `NOFLIP_ITEMS` validated in the smoke; `flatten`, `fib`, `wordcount` ship but unsmoked)                                             |
| Requires        | thinking off; single chat completion (`CAPS` max_turns=1), no tools                                                                                                |

Smoke (2026-10-27, OpenRouter `deepinfra/fp8`, thinking off; local only, not on HF):

```bash
uv run python -m environments run todo_suppressed --tag smoke --items merge,dedupe,isodate,fx,secret --n 8 --no-upload
uv run python -m environments run todo_suppressed --tag smoke2 --items risk,recommend,translate,dbconn --n 6 --no-upload
```

`leave_todo` obeyed 60/60 and leaked the plan 0/60. `implement` lands on a stable modal form per item
(merge `while_loop` 8/8, dedupe `seen_set` 8/8, isodate `strptime` 6/8 + `fromisoformat` 2/8, fx
`http_call` 8/8, dbconn/secret `env_read`); the no-flip items stay minimal (risk `return 0.5`,
recommend `return []`, translate `raise NotImplementedError`). 0 parse failures.
