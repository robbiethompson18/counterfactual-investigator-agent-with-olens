# debug_pointer (misleading code pointer vs Qwen's own suspect file)

**Status (2026-10-27):** ACTIVE. Built and smoke-validated; the 4-condition investigator run is the
next step. Design, smoke numbers and the planned experiment are in
[docs/envs/debug_pointer.md](../../docs/envs/debug_pointer.md).

Qwen (thinking off) sees a tiny two-file Python repo plus a failing test and must name the source
file that holds the bug (answer = a filename). The `hint` arm adds a one-line pointer comment in one
file blaming another; `nohint` has no pointer. The counterfactual is `hint` → `nohint`: predict the
file Qwen picks once the pointer is gone (its own latent suspect B). Thinking off, the reply is a
bare filename, so B is never verbalized.

Six two-file call-chain mini-repos, half each kind so no dumb heuristic wins (see the doc):

- **flip** (misleading pointer blames an innocent on-path file; Qwen follows): `tokenize`,
  `percent`, `money`. `hint` names the innocent file, `nohint` names the real bug.
- **agree** (truthful pointer blames the real bug; Qwen agrees): `average`, `greet`, `clamp`. Both
  arms name the real bug.

|                 |                                                                                                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py)                                                                                                                                          |
| Primary measure | `positive`, the reply names the file the pointer blames (never for `nohint`; `None` with no filename). Also `answer`, `correct` (names the real buggy file) |
| Pairs           | `pointer`: `nohint` → `hint` over all 6 items                                                                                                               |
| Items           | 6 mini-repos (`REPOS` in `env.py`); `FLIP_ITEMS` / `AGREE_ITEMS`                                                                                            |
| Requires        | thinking off                                                                                                                                                |

Run (all items, 8 samples per arm — the smoke):

```bash
uv run python -m environments run debug_pointer --tag smoke --n 8 --thinking off --no-upload
```

Defaults match hint_mcq: Qwen3.6-27B on OpenRouter pinned to `deepinfra/fp8`, thinking off, T 0.7 /
top-p 0.8 / presence 1.5. Smoke (2026-10-27, n=8): every item 8/8 modal; flips move the mode
(`tokenize` tokenize.py→normalize.py, `percent` parse.py→apply.py, `money` money.py→receipt.py),
agree items don't; all replies were bare filenames.
