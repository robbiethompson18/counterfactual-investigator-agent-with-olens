# grade_your_own: retired for Qwen3.6-27B (replaced by `evalcue/envs/exclude_rows.py`)

Don't spend more time on this environment as shipped for Qwen3.6-27B. We've stopped working on it.
Its Qwen-sized rewrite is the exclude-rows env in the `evalcue` harness (branch
`evalcue/exclude-rows` until it merges).

The 2026-10-21 screen ([report](../../../docs/envs-screen/grade_your_own.md)) found that at an
unreachable bar (+7 or +8 points; Qwen's honest plateau is about +3.7) Qwen neither cheats nor
reports the shortfall. It keeps trying model after model for hours, then degrades into reasoning
loops: 0 of 24 hard-target runs ended on their own. At an easy bar (+2) it reported honest success
in 30/30 runs. One run in 54 cheated, by picking the seed where the baseline scores worst.

The rewrite keeps the temptation but forces a decision. It has one bad action (excluding correctly
labelled test rows), a command budget with a forced submit, and small, fast data.

The configs and scripts stay here so the screen can be reproduced.
