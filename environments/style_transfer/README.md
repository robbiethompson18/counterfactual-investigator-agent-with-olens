# style_transfer (does a codebase reshape the form of a function the model knows?)

**Status (2026-10-28):** ACTIVE. Subject-side arms, an AST structure label and the Flash
text-inversion gate (10 survivors / 130 items); no investigator run yet. Design, smoke numbers, gate
results and the try log: [docs/envs/style_transfer.md](../../docs/envs/style_transfer.md).

The subject (Qwen3.6-27B, thinking off) implements a well-known function (130-item pool: factorial,
gcd, flatten, two_sum, bfs, ...) it certainly knows, so correctness isn't the variable; the
**structure** of the code is: `recursive` / `while` / `for` / `loop` (both) / `comprehension` /
`none` (builtins, slicing, arithmetic), parsed with `ast`. Arms: `no_context` (the default
structure, the CF target), `context_recursive` and `context_iterative` (the same ask prefixed with a
small `utils.py` whose helpers are all recursive / all loops, style conveyed by example only).

|                 |                                                                                                                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py), gate [`gate.py`](gate.py)                                                                                                                                                 |
| Primary measure | `positive` = recursive; `structure` is the multi-class CF target (`canon_answer` for cfbench's answer task)                                                                                   |
| Pairs           | `from_recursive` = (no_context, context_recursive), `from_recursive_gated` = the same on the 10 gate `SURVIVORS`; `from_iterative`; `codebase_style` = (context_iterative, context_recursive) |
| Requires        | `thinking: False`                                                                                                                                                                             |

```bash
uv run python -m environments run style_transfer --tag pool --n 8            # Qwen smoke, every arm x item x 8
uv run python -m environments.style_transfer.gate 2026-10-27_pool --table    # modal structures, flips, usable items
uv run python -m environments.style_transfer.gate 2026-10-27_pool            # Flash pass@3 gate -> data/runs/style_transfer_gate/
```
