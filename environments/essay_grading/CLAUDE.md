# essay_grading

Essay grading: who wrote the essay and what the user wants. What it tests, arms, measures and runs:
[README.md](README.md).

## Known issues (not fixed, revisit later)

- **Bold scores are converted to negative grades:** `env.py`'s `SCORE` regex is a copy of
  `../_lib/honest.py`'s and has the same bug; the finding is in
  [../honest/CLAUDE.md](../honest/CLAUDE.md).
