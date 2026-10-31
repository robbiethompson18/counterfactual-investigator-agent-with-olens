# donation_bet

Donation Bet: value leakage into a Fermi estimate. What it tests, arms, measures and runs:
[README.md](README.md).

## Known issues (not fixed, revisit later)

Codex found these in a 2026-10-28 grader review at `cda1fe1` (moved here from `CODE_SMELL.md`);
Robbie hasn't reviewed them, so treat them as findings to verify. P1 = prioritize before trusting
affected results; P2 = narrower correctness or reporting issue.

### P2: donation estimates silently lose their scale or ignore a correction

[`environments/donation_bet/env.py`](env.py), `NUM`, `BOLD`, `LABELLED`, `parse_ref()` (lines
279–302): `**2.3e9**`, `**2.3 × 10^9**`, and `Estimate: 2.3e9` all parse as `2.3`; `**250** million`
parses as `250`. These are wrong numbers, not merely unscorable formats, and can flip the threshold
label by orders of magnitude. Also, `1,000,000\nAfter recalculating, my final estimate is 200,000.`
returns 1,000,000 because the first-line shortcut preempts the final estimate. Parse complete
numeric expressions across formatting and prefer an explicit final estimate over an earlier
candidate. Reject unsupported numeric suffixes instead of accepting just their mantissa.

### P2: donation's declared no-bet comparisons have no scored baseline

[`environments/donation_bet/env.py`](env.py), `build_ref()`, `measure()` and `PAIRS`: `neutral`
deliberately has no threshold metadata, so even a valid estimate always produces `positive=None`.
Yet the three "No bet vs neutral bet" pairs compare that arm with `bet_neutral` on three line sets.
The neutral side consequently has zero scored observations. This is a comparison-design error rather
than an estimate-extraction failure: evaluate neutral estimates against the pair's line without
adding a threshold to the prompt. One generic neutral boolean cannot represent all three line sets.
