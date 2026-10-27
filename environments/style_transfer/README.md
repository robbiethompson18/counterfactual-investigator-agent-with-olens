# style_transfer (does a codebase reshape the form of a function the model knows?)

**Status (2026-10-27):** ACTIVE. New env; subject-side arms + smoke only, no investigator run yet.
Full design, smoke numbers and the echo-vs-latent caveat:
[docs/envs/style_transfer.md](../../docs/envs/style_transfer.md).

The subject (Qwen3.6-27B, thinking off) implements a well-known function (12 items: factorial,
fibonacci, gcd, is_prime, binary_search, collatz_steps, count_vowels, digit_sum, ackermann, hanoi,
flatten, merge_sort) it certainly knows, so correctness isn't the variable — the **form** is:
recursive (self-call) vs iterative (loop). Arms: `no_context` (the default form, the CF target),
`context_recursive` and `context_iterative` (the same ask prefixed with a small `utils.py` whose
helpers are all recursive / all iterative, style conveyed by example only).

|                 |                                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| Code            | [`env.py`](env.py)                                                                                           |
| Primary measure | `positive` = the implementation is recursive (`form`, `has_def`, `language`, `mentions_style` also returned) |
| Pairs           | `from_recursive` = (no_context, context_recursive); `from_iterative` = (no_context, context_iterative)       |
| Requires        | `thinking: False`                                                                                            |

Run (tiny smoke, OpenRouter, local):

```bash
uv run python -m environments run style_transfer --tag smoke --n 8
```
