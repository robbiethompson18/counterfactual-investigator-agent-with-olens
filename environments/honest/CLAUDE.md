# honest

Honest grader: 11 scenarios × 5 framings; the shared prompt and score parser are in
`../_lib/honest.py`. What it tests, arms, measures and runs: [README.md](README.md).

## Known issues (not fixed, revisit later)

Codex found these in a 2026-10-28 grader review at `cda1fe1` (moved here from `CODE_SMELL.md`);
Robbie hasn't reviewed them, so treat them as findings to verify. P1 = prioritize before trusting
affected results; P2 = narrower correctness or reporting issue.

### P2: bold scores are converted to negative grades

[`environments/_lib/honest.py`](../_lib/honest.py), `SCORE` (line 45), and its copy in
[`environments/essay_grading/env.py`](../essay_grading/env.py), line 94, reject `Score: **2/10**`
and `**Score:** **2/10**`. A harsh grade therefore becomes `positive=False` with no score.
Conversely, the regex accepts `Score: 4/100` as 4/10 because the denominator has no boundary.
Normalize Markdown around the score and require the complete denominator; share the parser. All 550
locally available honest replies had parseable scores, so this check demonstrates a parser bug
without establishing an error rate in that run.
