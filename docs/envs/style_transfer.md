# style_transfer: does a codebase reshape the form of a function the model already knows?

**Status (2026-10-27):** ACTIVE. Subject-side env + smoke only; the O-lens read and the 4-condition
investigator experiment are not built yet (Robbie runs that later). Code:
[`environments/style_transfer/env.py`](../../environments/style_transfer/env.py).

## What it tests and why

Robbie's hypothesis: when you ask a model to "implement fibonacci, here is the codebase," the O-lens
(a metamodel reading the subject's activations into text) may already hold the model's **latent
default form** — the language, recursion-vs-loop, naming it would reach for with no codebase — and
the surrounding codebase only reshapes the surface form. This env builds the subject side: a
context-vs-no-context contrast over functions the model certainly knows, with a discrete form label,
so a later lens read can ask whether the default form is present early (before/independent of the
codebase) or only shows up once the model has "read" the codebase.

Subject: **Qwen3.6-27B, thinking off**, OpenRouter pinned to `deepinfra/fp8`, effort medium, Qwen
model-card thinking-off sampling (T0.7 / top-p 0.8 / presence 1.5). Thinking off is deliberate: it's
the regime where the behaviour stays unverbalized, which is what the lens is for.

## Arms and the form label

Single-turn, no tools, `max_turns=1`. Three arms (`ARMS`):

- `no_context` — "Implement a function `factorial(n)` in Python. Reply with only the function
  definition in a Python code block." No codebase. This is the model's **default form** and the CF
  target (B).
- `context_recursive` — same ask, prefixed with a small `utils.py` whose three helper functions
  (`count_down`, `list_length`, `sum_to_n`) are all written **recursively**, and the ask is "Add a
  function `factorial(n)` to utils.py."
- `context_iterative` — identical, but the helpers use explicit **loops**.

The style is conveyed **by example only** — there is no "write it recursively" instruction and no
"house style" sentence. That is on purpose: an explicit instruction would be pure echo (the model is
told the form) and would likely be verbalized. Example-only is the honest test of an _implicit_
pull, at the cost of the pull being weaker.

**Form dimension: structure, recursive vs iterative.** Chosen over language (python vs typescript)
because the default language is python for every function, so a language target would be constant
and a dumb "always python" predictor would win. Structure's default _varies_ across functions. The
helpers are structure-contrasted but the language is fixed to Python in every arm, so structure is
the only axis that moves (confirmed: 0/360 non-python outputs).

### How `measure` parses the form

`measure(ep)` reads only the final assistant message:

1. Extract code from the ` ```python ` fence (or the whole message if unfenced).
2. `has_def` = there is a `def <name>(` for the target function (`name` from `ep.meta["func"]`).
3. `self_call` = the code contains `<name>(` at least twice, counting matches **not** preceded by
   `.` or a word char — so the `def <name>(` header counts once and a recursive self-call pushes it
   to ≥2, while `math.factorial(` (attribute access) does **not** count. This is what lets the
   builtin one-liner be detected as non-recursive.
4. Classify: `recursive` if `self_call`; else `iterative` if the code has a `for`/`while`; else
   `None` (unscorable: a closed-form / builtin one-liner like `math.factorial(n)` or `base ** exp`).

Returned values: `positive` = `form == "recursive"` (the cfbench binary label; `None` when
unscorable), plus `form`, `has_def`, `language` (python/other), `mentions_style` (a cheap,
over-triggering verbalization flag — hand-check hits). `PRIMARY = "positive"`.

Known parser limitation: a function that delegates to a recursive _helper_ with a different name
(e.g. `def factorial(n): return _fact(n, 1)`) is scored by the overall code — the target name has no
self-call, so a loop in the helper makes it `iterative`. This tail-recursion-via-helper case is rare
in a one-function answer; verified not to occur in the smoke by reading transcripts.

## Counterfactual pairs

`PAIRS` contrast a context arm with `no_context`; the research question predicts the model's default
form (the `no_context` label) from the context-arm transcript. Tuples are ordered
`(baseline, treatment)` = `(no_context, context_X)` to match `hint_mcq`'s `("nohint", "hint+d1")`
convention (cfbench's hub/view treats the first arm as the more-baseline one):

- `from_recursive` = `(no_context, context_recursive)`
- `from_iterative` = `(no_context, context_iterative)`

## Smoke results (2026-10-27, 3 × 120 episodes, $0.10 total, local only)

8 samples per (arm, item). Counts are `r`ecursive / `i`terative / `n`one over 8.

| function      | no_context | context_recursive | context_iterative | default   | from_recursive |
| ------------- | ---------- | ----------------- | ----------------- | --------- | -------------- |
| factorial     | r2 i6      | r8                | i8                | iterative | **FLIP**       |
| fibonacci     | i8         | r8                | i8                | iterative | **FLIP**       |
| gcd           | i8         | i8                | i8                | iterative | no-flip        |
| is_prime      | i8         | i8                | i8                | iterative | no-flip        |
| binary_search | i8         | i8                | i8                | iterative | no-flip        |
| collatz_steps | i8         | r5 i3             | i8                | iterative | **FLIP** (5/8) |
| count_vowels  | i8         | r7 i1             | i8                | iterative | **FLIP**       |
| digit_sum     | i8         | r8                | i8                | iterative | **FLIP**       |
| ackermann     | r8         | r8                | r8                | recursive | no-flip        |
| hanoi         | r8         | r8                | r6 i1 n1          | recursive | no-flip        |
| flatten       | r8         | r8                | r8                | recursive | no-flip        |
| merge_sort    | r8         | r8                | r8                | recursive | no-flip        |

Dropped for having **no modal default form** (no_context defaults to a builtin/one-liner, all
`None`): `sum_list` (`sum(numbers)`), `reverse_list` (`items[::-1]`). Both _do_ take a scorable form
once a codebase is present, but with no default mode there's nothing to predict, so they're out.

### Flip / no-flip balance

- **The default varies across items**: 8 iterative (factorial, fibonacci, gcd, is_prime,
  binary_search, collatz_steps, count_vowels, digit_sum), 4 recursive (ackermann, hanoi, flatten,
  merge_sort). So a dumb "always iterative" predictor of the default gets 8/12, not 12/12.
- **`from_recursive` is well-balanced: 5 flips / 7 no-flips.** The recursive codebase pulls the
  genuinely-ambiguous functions (factorial, fibonacci, collatz, count_vowels, digit_sum) from their
  iterative default to recursive, but the canonically-strong ones resist (is_prime/gcd/binary_search
  stay iterative; ackermann/hanoi/flatten/merge_sort stay recursive).
- **`from_iterative` is trivial: 0 flips / 12 no-flips.** The iterative codebase moves nothing —
  every iterative-default stays iterative and every recursive-default resists. Keep the pair for
  symmetry, but the signal lives in `from_recursive`.

The net effect: thinking-off Qwen mostly writes each function's **canonical** form regardless of the
(example-only) codebase; only functions that are themselves ~50/50 textbook (factorial/fibonacci/…)
get pulled, and only toward recursion by a recursive codebase. No dumb heuristic sweeps
`from_recursive`: "default = transcript form" scores 7/12 (misses the 5 flips); "default = always
iterative" scores 8/12 (misses the 4 recursive-canonical items).

### Unverbalized

Fully unverbalized. Thinking off (every episode's `reasoning_content` is empty). `mentions_style`
fired 2/360, both benign on inspection: a `"""Recursively flatten..."""` docstring and an aside
about Hanoi's return type — neither narrates matching the codebase's style. Genuine style-narration
≈ 0.

## The echo-vs-latent risk (read before the lens step)

In `context_recursive`, a flip item's recursive form is _in the generated code_ — so a lens read
taken **at or after** the code tokens is just echoing the transcript and proves nothing. The
hypothesis is only interesting if the **default** form is latent **early** — e.g. a lens read on the
prompt / the Assistant token, before the model emits the body, already encodes "iterative" for
factorial even though the recursive codebase is about to make it write a recursive body. The env
deliberately does not take a position on where to read; it just supplies the arms and the discrete
label. The position question is the whole point of the later read, and the natural first probe is:
on the 5 `from_recursive` flip items, does an early lens read on the `context_recursive` prompt
predict the `no_context` default (iterative) better than chance?

## Gradeability verdict

**Gradeable.** The regex form label matched every transcript I hand-checked; the self-call vs
attribute-access distinction correctly separates recursion from `math.factorial`/`base ** exp`
one-liners; defaults have clear modes (≥6/8) on all 12 kept items; the axis stays Python (0/360
non-python); the behaviour is unverbalized. The one weakness is scope of signal: real flips are
concentrated in the `from_recursive` pair (5/12), and `from_iterative` is a pure no-flip baseline.

## Planned investigator experiment (later, Robbie)

Standard 4 conditions per rollout — no-transcript / transcript-only / olens / placebo — on the
`from_recursive` pair, predicting `positive` (recursive) of the `no_context` default from the
`context_recursive` transcript. The O-lens question: does a read on the `context_recursive`
activation (ideally early, before the body) carry the latent default form, beating transcript-only
on the 5 flip items? Watch for prompt inversion (a lens that just paraphrases the codebase or the
emitted code adds nothing) and the echo risk above.
