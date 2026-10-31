# essay_grading

Essay grading: who wrote the essay and what the user wants. What it tests, arms, measures and runs:
[README.md](README.md).

## Score parsing

Uses the shared `parse_score` in `../_lib/honest.py`; see
[the parsing policy and local audit](../honest/CLAUDE.md#score-parsing). Missing or invalid final
scores are unscorable (`positive=None`, `score=None`, `has_score=False`).
