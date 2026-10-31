# honest

Honest grader: 11 scenarios × 5 framings; the shared prompt and score parser are in
`../_lib/honest.py`. What it tests, arms, measures and runs: [README.md](README.md).

## Score parsing

**Fixed 2026-10-31:** honest and essay_grading share `parse_score` in `../_lib/honest.py`.
The final explicit `Score:` field wins, including ordinary Markdown around the label/value.
Its decimal numerator must be finite and within 0–10, with denominator exactly 10. A missing or
invalid final field gives `score=None` and `positive=None`; never fall back to an earlier score.

This fixes the 2026-10-28 review finding: bold scores were missed, `/100` was accepted as `/10`,
and an earlier score could override a final bold score. An offline audit of the two locally saved
canonical runs (550 honest + 19,783 essay_grading responses) found zero score corrections and one
missing essay score changing from negative to unscorable (1/20,333, 0.0049%). This is local coverage,
not a census of HF. Historical labels were not rewritten.
