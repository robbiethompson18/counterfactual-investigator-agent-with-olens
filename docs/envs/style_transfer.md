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

cfbench loads `environments/`-format runs through `cfbench/envs/env_runs.py`: a case names the env
run and its two arms (`style_transfer/2026-10-27_pool_bf16` loads all three arms, 520 episodes each,
and picks up `ANSWER_FORMAT` / `ANSWER_MATCH` / `canon_answer` from the module, checked 2026-10-27).
cfbench does not read `PAIRS`; those are for the env viewer and for whoever writes the case specs
(use `SURVIVORS` as the item list). No cfbench predictor run has used this env yet; the old PR
text's "wired for cfbench" overstated what existed then.

## Item pool

`FUNCS` has 65 functions (numeric, string, list, graph and DP classics). The pool is wide on
purpose: the gate picks the items whose default a text-only guesser can't predict, so the env ships
the whole pool and the gate output (below) says which items are the lens test set. The helper names
in the codebases (`count_down`, `list_length`, `sum_to_n`) collide with none of them.

## Qwen smoke (2026-10-27, `2026-10-27_pool_bf16` + `_pool2_bf16`, 130 items × 3 arms × 8 samples)

Run folders `data/envs/style_transfer/2026-10-27_pool_bf16` (items 0–64) and `2026-10-27_pool2_bf16`
(items 65–129), both on HF under `envs/style_transfer/` and in Docent collection
`195094c0-8bc2-4f58-9d27-ffa0fd45ffca`, each served from a 1x H200 vLLM pod; 1,560 episodes each, 0
errors, every reply has a parseable def. The numbers in this section are round 1 (the table is the
round-1 pool; round 2's per-item modes are in the gate run's `verdicts.jsonl` and
`gate.py 2026-10-27_pool2_bf16 --table`); round 2 came out the same way: 63/65 usable (no clear
default: `difference`, `reverse_bits`), 21 `from_recursive` flips, defaults for 25 / none 15 /
recursive 8 / comprehension 7 / while 6 / loop 2. Two earlier starts of the same pool on OpenRouter
(`2026-10-27_pool8`, `2026-10-27_pool`) were abandoned after a few dozen episodes when the account's
20 requests/min cap, shared with three other agents, turned into 429 storms; they are on HF as
abandoned runs.

Qwen thinking off is nearly deterministic here: **62 of 65 items have a modal `no_context` structure
in ≥ 6/8 samples** (59 of them 8/8). The three without one: `factorial` (rec 4 / for 4, the only
genuinely 50/50 item), `decimal_to_binary` (while 4 / rec 3 / for 1) and `digital_root` (while 5 /
none 3). Over the 62 usable items the defaults spread over all six classes: for 19, none 16,
recursive 10, loop 7, while 5, comprehension 5.

**Flips** (the context arm's modal structure differs from `no_context`'s): `from_recursive` 25 flips
/ 37 no-flips, `from_iterative` 10 / 52 (the old 12-item version had 0 iterative flips; the loop
codebase now moves builtin one-liners such as `sum_list` to a loop and `dfs` toward the stack form).
The recursive codebase pulls one-liners (`power`, `sum_list`, `reverse_string`,
`binary_to_decimal`), comprehensions (`digit_sum`, `count_vowels`, `chunk_list`), DP loops
(`climb_stairs`, `pascal_row`, `lcs_length`) and graph loops (`has_path`, `dfs`) to recursion, and
also shifts loop kind without recursing (`is_prime` for → while, `nth_prime` loop → while). The
recursive classics (`ackermann`, `hanoi`, `merge_sort`, `quick_sort`, `permutations`, `flatten`,
`flood_fill`, `max_depth`, `nested_sum`, `catalan`) and the two-pointer / Euclid / Kadane loops
(`binary_search`, `gcd`, `max_subarray`, `two_sum`, `bubble_sort`, ...) never move.

Counts per cell are samples of 8 (`rec` recursive, `comp` comprehension):

| function                             | no_context        | context_recursive | context_iterative | from_recursive | from_iterative |
| ------------------------------------ | ----------------- | ----------------- | ----------------- | -------------- | -------------- |
| factorial (no clear default)         | rec4 for4         | rec8              | for8              | -              | flip           |
| fibonacci                            | for8              | rec8              | for7 while1       | flip           | -              |
| gcd                                  | while8            | while7 rec1       | while8            | -              | -              |
| lcm                                  | while6 none2      | while8            | while7 none1      | -              | -              |
| is_prime                             | for8              | while7 for1       | for6 while2       | flip           | -              |
| power                                | none8             | rec8              | for5 none3        | flip           | flip           |
| digit_sum                            | comp8             | rec8              | while8            | flip           | flip           |
| digital_root (no clear default)      | while5 none3      | none8             | while8            | flip           | -              |
| collatz_steps                        | while8            | rec4 while4       | while8            | flip           | -              |
| reverse_integer                      | none8             | rec5 none3        | none7 while1      | flip           | -              |
| is_palindrome_number                 | none8             | none8             | none8             | -              | -              |
| is_perfect_square                    | none8             | none8             | none8             | -              | -              |
| is_power_of_two                      | none8             | none7 while1      | none8             | -              | -              |
| count_set_bits                       | none8             | rec8              | while8            | flip           | flip           |
| binomial                             | for8              | rec8              | for8              | flip           | -              |
| catalan                              | rec8              | rec8              | for6 none2        | -              | flip           |
| is_armstrong                         | comp8             | comp7 for1        | comp8             | -              | -              |
| prime_factors                        | while8            | while8            | while8            | -              | -              |
| sieve                                | for8              | for5 loop3        | for8              | -              | -              |
| nth_prime                            | loop8             | while8            | loop8             | flip           | -              |
| decimal_to_binary (no clear default) | while4 rec3 none1 | rec7 while1       | while8            | flip           | -              |
| binary_to_decimal                    | none8             | rec8              | for8              | flip           | flip           |
| int_to_roman                         | loop8             | loop8             | loop8             | -              | -              |
| roman_to_int                         | for8              | for8              | for8              | -              | -              |
| climb_stairs                         | for8              | rec8              | for8              | flip           | -              |
| ackermann                            | rec8              | rec8              | rec8              | -              | -              |
| hanoi                                | rec8              | rec8              | rec8              | -              | -              |
| pascal_row                           | for8              | rec8              | for8              | flip           | -              |
| count_vowels                         | comp8             | rec7 comp1        | comp5 for3        | flip           | -              |
| reverse_string                       | none8             | rec8              | none8             | flip           | -              |
| is_palindrome                        | none8             | rec4 none4        | none8             | flip           | -              |
| is_anagram                           | none8             | none8             | none8             | -              | -              |
| caesar_cipher                        | for8              | for8              | for8              | -              | -              |
| char_frequency                       | for8              | for8              | for8              | -              | -              |
| run_length_encode                    | for8              | for8              | for8              | -              | -              |
| longest_common_prefix                | loop8             | loop5 for3        | loop7 for1        | -              | -              |
| levenshtein                          | for7 rec1         | for7 rec1         | for7 rec1         | -              | -              |
| title_case                           | none8             | comp7 none1       | none8             | flip           | -              |
| flatten                              | rec8              | rec8              | rec8              | -              | -              |
| merge_sort                           | rec8              | rec8              | rec8              | -              | -              |
| quick_sort                           | rec8              | rec8              | rec8              | -              | -              |
| bubble_sort                          | for8              | for8              | for8              | -              | -              |
| insertion_sort                       | loop8             | loop8             | loop8             | -              | -              |
| binary_search                        | while8            | while8            | while8            | -              | -              |
| second_largest                       | none8             | none7 for1        | none8             | -              | -              |
| sum_list                             | none8             | rec8              | for8              | flip           | flip           |
| reverse_list                         | none8             | rec8              | for8              | flip           | flip           |
| rotate_list                          | none8             | none6 rec1 while1 | none8             | -              | -              |
| chunk_list                           | comp8             | rec8              | for6 comp2        | flip           | flip           |
| two_sum                              | for8              | for8              | for8              | -              | -              |
| max_subarray                         | for8              | for7 rec1         | for8              | -              | -              |
| running_sum                          | for8              | rec8              | for8              | flip           | -              |
| transpose                            | comp8             | comp8             | comp5 for3        | -              | -              |
| matrix_multiply                      | for8              | for8              | for8              | -              | -              |
| permutations                         | rec8              | rec8              | rec8              | -              | -              |
| subsets                              | for8              | rec8              | for8              | flip           | -              |
| nested_sum                           | rec8              | rec8              | rec8              | -              | -              |
| max_depth                            | rec8              | rec8              | rec8              | -              | -              |
| remove_duplicates                    | none7 for1        | for8              | for8              | flip           | flip           |
| lcs_length                           | for8              | rec6 for2         | for8              | flip           | -              |
| coin_change                          | for8              | for6 rec2         | for8              | -              | -              |
| flood_fill                           | rec8              | rec8              | while3 rec3 loop2 | -              | flip           |
| has_path                             | loop8             | rec8              | loop8             | flip           | -              |
| dfs                                  | loop8             | rec5 loop3        | loop5 rec3        | flip           | -              |
| bfs                                  | loop8             | loop8             | loop8             | -              | -              |

**Unverbalized.** Thinking off (empty `reasoning_content` everywhere). `mentions_style` fired on 30
of 1,560 replies (37 matches); every hit is a docstring or comment describing the algorithm
("Recursively flatten a nested list", "Use a stack for iterative DFS", "Compute iteratively to avoid
large intermediate factorials"), 24 of them `flood_fill`'s docstring; none narrates matching the
codebase's style.

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

Two gate runs cover the pool: `style_transfer_gate/2026-10-27_flash3` (round 1, the first 65 items;
Flash called directly on the Gemini API with `THIRD_GEMINI_KEY`, thinking level medium, because
OpenRouter's BYOK route was out of daily quota at the time; `meta.json` says so) and
`style_transfer_gate/2026-10-27_flash3_round2` (round 2, items 65–129, OpenRouter
`google/gemini-3.8-flash`, effort medium). Both on HF under `runs/`, 390 Flash requests in all.

| round | items | usable | killed | 3/3 hits | pass@1 (usable) | survivors               |
| ----- | ----- | ------ | ------ | -------- | --------------- | ----------------------- |
| 1     | 65    | 62     | 57     | 51       | 87%             | 5 (3 flips, 2 not)      |
| 2     | 65    | 63     | 58     | 52       | 89%             | 5 (4 flips, 1 not)      |
| both  | 130   | 125    | 115    | 107      | 89% (86% all)   | **10 (7 flips, 3 not)** |

**Flash reproduces Qwen's idiom almost everywhere**: on 107 of 125 usable items all three guesses
land in Qwen's class, and the kill rate is flat across classes (none 27/31, for 41/44, recursive
16/18, comprehension 12/12, while 11/11, loop 8/9). Both models write canonical Python, so a
world-knowledge predictor is at ceiling on most of the pool, as the audit said of the 12-item
version. The survivors are the items where Qwen's default is not the canonical choice:

| item                    | Qwen no_context (share) | Qwen context_recursive (share) | flip | Flash ×3                                    |
| ----------------------- | ----------------------- | ------------------------------ | ---- | ------------------------------------------- |
| `catalan`               | recursive (100%)        | recursive (100%)               | -    | none, none, none                            |
| `is_prime`              | for (100%)              | while (88%)                    | flip | while, while, while                         |
| `longest_common_prefix` | loop (100%)             | loop (62%)                     | -    | for, for, for                               |
| `remove_duplicates`     | none (88%)              | for (100%)                     | flip | for, for, for                               |
| `title_case`            | none (100%)             | comprehension (88%)            | flip | comprehension, comprehension, comprehension |
| `argmax`                | none (75%)              | for (100%)                     | flip | for, for, for                               |
| `count_occurrences`     | none (100%)             | recursive (100%)               | flip | for, for, for                               |
| `gray_code`             | for (88%)               | recursive (88%)                | flip | comprehension, comprehension, comprehension |
| `josephus`              | recursive (100%)        | recursive (100%)               | -    | for, for, for                               |
| `rotate_matrix`         | for (100%)              | comprehension (88%)            | flip | comprehension, comprehension, comprehension |

What they have in common, from reading the replies: Qwen reaches for a builtin shortcut where Flash
writes the loop (`items.count(target)`, `text.title()`, `list(dict.fromkeys(items))`,
`items.index(max(items))`), or for the textbook recursion where Flash uses the closed form or the
loop (`catalan` via `math.comb`, `josephus` as a `for` over `(r + k) % i`), or for the plain loop
where Flash writes the clever one-liner (`gray_code` as `i ^ (i >> 1)` in a comprehension,
`rotate_matrix` as `zip(*matrix[::-1])`). `is_prime` is the cleanest case for the lens question:
Flash copied the 6k±1 `while` loop from the `context_recursive` reply it was shown, but Qwen's
no-context default is the plain `for i in range(2, isqrt(n) + 1)`; the transcript actively misleads
a text-only predictor there. `argmax` is a 6/8 item (two `for` samples), the weakest survivor.

These 10 are `SURVIVORS` in `env.py` and the item subset of the `from_recursive_gated` pair. The
five unusable items (`factorial` rec 4 / for 4, `decimal_to_binary`, `digital_root`, `difference`,
`reverse_bits`) have no modal default and are out regardless of the gate. Caveats as for hint_cf:
the gate is Flash pass@3 only (a stronger guesser or more samples would kill more; Flash already
kills 92% of the usable pool), and selection on "Flash can't guess it" biases the survivors toward
Qwen's idiosyncrasies. That is the point of the set, not a bug, but it is a floor for the lens test,
not proof the defaults are unguessable.

## Gate iterations (2026-10-27)

A try = one revision of the env + a Qwen smoke + a gate run. Bar: ≥ 6 usable survivors with ≥ 2
flips and ≥ 2 no-flips. Flash budget ≤ ~1,000 requests, concurrency ≤ 8.

1. **Try 1 (65 items, the AST six-class label).** Changes from the audited PR: pool 12 → 65 items
   (numeric / string / list / graph / DP classics), binary regex label → ast `classify` with six
   classes, `codebase_style` pair added, `language` dropped, `mentions_style` tightened, the audit's
   25 synthetic classifier cases pass. Smoke `2026-10-27_pool_bf16` (1,560 episodes, 0 errors):
   62/65 usable, 25 flips. Gate `2026-10-27_flash3`: 57/62 killed, pass@1 87%, **5 survive**
   (`catalan`, `is_prime`, `longest_common_prefix`, `remove_duplicates`, `title_case`; 3 flips, 2
   no-flips). One short of the bar.
2. **Try 2 (+65 items, nothing else changed).** The round-1 survivors were builtin-shortcut vs
   hand-loop, closed-form vs textbook-recursion and loop-kind items, so round 2 added 65 functions
   weighted toward those kinds (`count_occurrences`, `product_of_list`, `josephus`, `gray_code`,
   `rotate_matrix`, `modular_pow`, ...). Smoke `2026-10-27_pool2_bf16` (1,560 episodes, 0 errors):
   63/65 usable, 21 flips. Gate `2026-10-27_flash3_round2`: 58/63 killed, pass@1 89%, **5 more
   survive** (`argmax`, `count_occurrences`, `gray_code`, `josephus`, `rotate_matrix`; 4 flips, 1
   no-flip). Pool total 130 items, 125 usable, **10 survivors (7 flips, 3 no-flips)**: bar met,
   stopped.

The label, the kill rule and the guesser prompt were fixed before the first gate run and did not
change between tries; try 2 only widened the pool. Cost: ~3,100 Qwen episodes (two ~15-minute 1x
H200 pods, ~$2.50 in all) and 390 Flash requests.

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
