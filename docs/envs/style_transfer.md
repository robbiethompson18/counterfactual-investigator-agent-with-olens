# style_transfer: does a codebase reshape the form of a function the model already knows?

**Status (2026-10-28):** ACTIVE. Subject-side env, AST structure label and the Flash text-inversion
gate; the O-lens read and the investigator experiment are not built yet. Code:
[`environments/style_transfer/env.py`](../../environments/style_transfer/env.py), gate
[`gate.py`](../../environments/style_transfer/gate.py). The first version of this env (12 items,
binary recursive/iterative regex label, PR #247 as opened) failed the 2026-10-27 audit; the redesign
and every try are logged under "Gate iterations" below.

## What it tests and why

Robbie's hypothesis: when you ask a model to "implement fibonacci, here is the codebase," the O-lens
(a metamodel reading the subject's activations into text) may already hold the model's **latent
default form** (the language, recursion-vs-loop, naming it would reach for with no codebase) and the
surrounding codebase only reshapes the surface form. This env builds the subject side: a
context-vs-no-context contrast over functions the model certainly knows, with a discrete form label,
so a later lens read can ask whether the default form is present early (before/independent of the
codebase) or only shows up once the model has "read" the codebase.

Subject: **Qwen3.6-27B, thinking off**, Qwen model-card thinking-off sampling (T0.7 / top-p 0.8 /
presence 1.5). Thinking off is deliberate: it's the regime where the behaviour stays unverbalized,
which is what the lens is for. `DEFAULTS` point at OpenRouter `deepinfra/fp8`; the pool run below
was served from our own 1x H200 vLLM pod (bf16) because the new OpenRouter account caps Qwen at 20
requests/min shared by every agent (`meta.json` records the target).

## Arms and the structure label

Single-turn, no tools, `max_turns=1`. Three arms (`ARMS`):

- `no_context`: "Implement a function `factorial(n)` in Python. Reply with only the function
  definition in a Python code block." No codebase. This is the model's **default** and the CF
  target.
- `context_recursive`: same ask, prefixed with a small `utils.py` whose three helper functions
  (`count_down`, `list_length`, `sum_to_n`) are all written **recursively**; the ask is "Add a
  function `factorial(n)` to utils.py."
- `context_iterative`: identical, but the helpers use explicit **loops**.

The style is conveyed **by example only**: no "write it recursively" instruction, no "house style"
sentence. An explicit instruction would be pure echo and would likely be verbalized; example-only is
the honest test of an implicit pull, at the cost of the pull being weaker.

**Target: `structure`, six classes**, parsed with `ast` (`env.classify`), over the target function
plus every user-defined function reachable from it (helpers, inner defs):

| class           | rule                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------- |
| `recursive`     | a call cycle reachable from the target (`merge_sort → merge_sort`, `hanoi → _move → _move`) |
| `while` / `for` | an explicit loop statement of that kind, no self-call                                       |
| `loop`          | both kinds of loop                                                                          |
| `comprehension` | no loop statement; a comprehension / generator expression or `map` / `filter` / `reduce`    |
| `none`          | none of the above: arithmetic, slicing, builtins (`sum(items)`, `sorted(a) == sorted(b)`)   |

Names in strings, comments and docstrings never count (the regex version of this label matched the
English "for" in a docstring and missed `hanoi`'s inner `_hanoi` recursion). `math.gcd(a, b)` inside
`def gcd` is a library call, not recursion (only `self.`/`cls.` attribute calls resolve to a def).
The auditor's 25 synthetic cases plus a few more all pass; on the audit smoke the only disagreement
with the old regex label is the `hanoi` case it got wrong.

Why six classes and not the old binary recursive/iterative: the audit found the binary default is
world-guessable from the function name ("recursive iff the loop version needs an explicit stack"
scored 12/12), and the regex counted `sum(1 for c in s if ...)` as iterative while the PR dropped
`sum_list`'s `sum(...)` as unscorable. The multi-class label makes the borderline choices (`while`
vs `for`, a loop vs a comprehension vs a builtin one-liner) part of the target, which is where the
default is Qwen-specific rather than textbook, and gives one class definition applied everywhere.

`measure()` returns `positive` = recursive (the registry's bool primary), `structure` (the CF
target), `has_docstring`, `type_hints`, `raises`, `helpers` (user-defined functions the target
reaches) and `mentions_style` (a verbalization flag; hand-check hits). `None` when there is no
parseable def of the target. For cfbench's answer task the env exports `ANSWER_FORMAT`,
`ANSWER_MATCH` and `canon_answer(text)` (the structure class of a code block, or a bare class name).

## Counterfactual pairs

Ordered `(baseline, treatment)` as cfbench's hub/view expects (`hint_mcq`: `("nohint", "hint+d1")`):

- `from_recursive` = `(no_context, context_recursive)`: the research question, predict the default
  from the recursive-codebase transcript.
- `from_iterative` = `(no_context, context_iterative)`.
- `codebase_style` = `(context_iterative, context_recursive)`: the single-lever pair (same ask, only
  the helpers' style differs).

cfbench reads these pairs through `cfbench/envs/env_runs.py` (any env module's `PAIRS` and
`canon_answer`); nothing in cfbench is specific to this env, and no cfbench run has used it yet. The
old PR text's "wired for cfbench" meant only that.

## Item pool

`FUNCS` has 65 functions (numeric, string, list, graph and DP classics). The pool is wide on
purpose: the gate picks the items whose default a text-only guesser can't predict, so the env ships
the whole pool and the gate output (below) says which items are the lens test set. The helper names
in the codebases (`count_down`, `list_length`, `sum_to_n`) collide with none of them.

«SMOKE»

## The text-inversion gate (Gemini 3.8 Flash, pass@3)

`gate.py` ports the retired `evalcue/envs/hint_cf/guess_gate.py`. For each item Gemini 3.8 Flash
(OpenRouter, reasoning effort medium, T 1.0) sees what cfbench's transcript-only predictor would
see: the `context_recursive` prompt, one real `context_recursive` Qwen reply (the lowest-numbered
sample with a parseable def) and the `no_context` prompt, and is asked to write the code Qwen3.6-27B
with thinking off produces for the `no_context` prompt; its code is classified with `canon_answer`.
The prompt also states the match rule (the six classes), which only makes the gate stricter. 3
samples per item; **an item is killed when any sample's class is Qwen's modal `no_context` class**
(pass@3; the target is multi-class). Items whose modal class holds in fewer than 6/8 samples are
unusable regardless. Output is a run under `data/runs/style_transfer_gate/` (`guesses.jsonl` one row
per item × sample with the full prompt, raw and canonical guess, Qwen's mode and the match flag;
`verdicts.jsonl` one row per item).

«GATE»

## Gate iterations (2026-10-27)

«TRIES»

## The echo-vs-latent risk (read before the lens step)

The `context_recursive` prompt is mostly recursive helper code, and on a flip item the recursive
form is also _in the generated code_. A lens read taken on the helpers or at/after the body will
paraphrase the transcript (prompt inversion) and proves nothing. The hypothesis is only interesting
if the **default** is latent **early**: a read on the ask line or the Assistant token, before the
body, already encodes e.g. `comprehension` for `digit_sum` even though the recursive codebase is
about to make it write a recursive body. The env takes no position on where to read; the position
question is the later experiment, and the natural first probe is: on the surviving flip items, does
an early read on the `context_recursive` prompt predict the `no_context` class better than Flash's
guesses in the gate run?

## Planned investigator experiment (later, Robbie)

Standard conditions per rollout (no-transcript / transcript-only / olens / placebo) on the
`from_recursive` pair restricted to the gate survivors, as cfbench's answer task (predict the
`no_context` structure class from the `context_recursive` transcript; `canon_answer` grades). Watch
for prompt inversion and the echo risk above.
