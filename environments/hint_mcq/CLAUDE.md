# hint_mcq

Hinted MMLU-Pro: a hint toward a wrong letter vs Qwen's own answer; the `nohint` arm is the MMLU-Pro
pool for the MCQ-answer task. What it tests, arms, measures and runs: [README.md](README.md).

## Known issues (not fixed, revisit later)

Codex found these in a 2026-10-28 grader review at `cda1fe1` (moved here from `CODE_SMELL.md`);
Robbie hasn't reviewed them, so treat them as findings to verify. P1 = prioritize before trusting
affected results; P2 = narrower correctness or reporting issue.

### P2: MMLU's primary arm comparison does not measure answer changes

[`environments/hint_mcq/env.py`](env.py), `measure()` and `PAIRS`: `nohint` is _defined_ to be
negative regardless of its answer, while a hint arm is positive when that same answer matches its
hint. Identical B replies under nohint and hint-B therefore produce a 0-to-1 "effect" without any
answer change. This is documented implementation behavior, but an invalid interpretation of the
primary-rate contrast as the hint's effect. Score both arms against the same pair-specific target
letter (d1 and d2 need separate baselines), or keep the current label explicitly as "followed an
instruction that was present" and avoid that contrast. The separate cfbench answer task is not
itself invalidated by this definition.
