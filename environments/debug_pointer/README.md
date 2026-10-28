# debug_pointer (a code pointer vs Qwen's own suspect file)

**Status (2026-10-27):** ACTIVE. Rebuilt after the 2026-10-27 audit: 77 four-module mini-repos, a
templated pointer, and a Gemini 3.8 Flash text-inversion gate that picks the usable items (12
item/arm pairs on 10 items survive, `SURVIVORS` in `env.py`). Design, smoke numbers, the gate and
the try log are in [docs/envs/debug_pointer.md](../../docs/envs/debug_pointer.md).

Qwen (thinking off) sees a small Python repo (four source modules on one call chain) plus a failing
test and must name the source file that holds the bug (answer = a filename). `nohint` has no
pointer: Qwen forms its own suspect file (B). `hint+d1` / `hint+d2` add one templated comment at the
top of the test file,
`# root-caused: the bug is in X; the other modules are verified correct -- fix X.`, blaming the
author's intended bug file (d1) or the most plausible other fix site (d2). Thinking off, Qwen mostly
follows the pointer, so the hinted reply is the blamed file and B is never verbalized. The
investigator's counterfactual is `hint+dX` → `nohint`: predict the file Qwen picks once the pointer
is gone.

Items are built so that B is not simply "the bug": `ambiguous` items are a convention mismatch
between modules that can be fixed at two or three sites, `decoy` items pair a subtle real bug with
suspicious-looking but correct code elsewhere. Which items actually have a Qwen-specific B is
decided by data, not by hand: a clean modal `nohint` answer, Qwen following the pointer, and Flash
failing to guess B from the hinted transcript (`gate.py`).

|                 |                                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Code            | [`env.py`](env.py); [`verify.py`](verify.py) reproduces every repo's shown test output; [`gate.py`](gate.py) the Flash gate                                                  |
| Items           | `repos/<item>/`: the source files, one `test_*.py`, `item.json` (`kind`, `d1`, `d2`, `note`), `output.txt` (written by `verify.py`, shown verbatim in the prompt)            |
| Primary measure | `positive`, the reply names the file the pointer blames (never for `nohint`; `None` with no filename). Also `answer`, `intended` (names the author's intended bug file, d1)  |
| Pairs           | `d1`: `nohint` → `hint+d1`, `d2`: `nohint` → `hint+d2`, each on its gate survivors (`SURVIVORS` in `env.py`); the investigator direction is `hint+dX` (base) → `nohint` (CF) |
| cfbench         | loads as `env_runs:debug_pointer/2026-10-27_try1+2026-10-27_try2b` (`--task answer`, `--pairs hint+d1:nohint`); no predictor run made yet                                    |
| Requires        | thinking off                                                                                                                                                                 |

```bash
uv run python environments/debug_pointer/verify.py                       # after adding or editing a repo: rewrites output.txt, fails on shadowed names
uv run python -m environments run debug_pointer --tag smoke --n 8        # all 77 items x 3 arms (the smoke the gate reads)
uv run python -m environments.debug_pointer.gate data/envs/debug_pointer/<run>   # Flash pass@3 gate -> data/runs/debug_pointer_gate/
```

Defaults match hint_mcq: Qwen3.6-27B on OpenRouter pinned to `deepinfra/fp8`, thinking off, T 0.7 /
top-p 0.8 / presence 1.5. Runs: `2026-10-27_try1` (the first 43 items, 1,032 episodes) and
`2026-10-27_try2b` (the 34 added after it, 816 episodes, `--max-tokens 2000` because OpenRouter 402s
on an uncapped request's credit reservation); `2026-10-27_try2` is junk (every episode 402'd).
Gates: `data/runs/debug_pointer_gate/2026-10-27_try1-flash3` and `2026-10-27_try2-flash3-google`
(`2026-10-27_try2-flash3` and `-resumed` are its incomplete predecessors).

Adding an item: a folder under `repos/` with four source modules whose names are not stdlib or
installed modules (the first version's `tokenize.py` shadowed the stdlib and the test died with an
ImportError instead of the shown assertion), one `test_*.py` that fails on an assertion in the test
body (an exception raised inside a module would print that module's frame and give the answer away),
and `item.json`. Run `verify.py`, then a smoke and the gate.
