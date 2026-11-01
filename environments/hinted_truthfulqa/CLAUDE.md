# hinted_truthfulqa

Hinted TruthfulQA: a hint naming a wrong letter, where only who gives it changes (`hint/<source>`
arms). What it tests, arms, measures and runs: [README.md](README.md).

## Known issues (not fixed, revisit later)

### HF's hinted_truthfulqa index shows stale labels for 6 runs

Left over from the labels-picker fix (PR #416, 2026-10-29), not from the Codex review below. HF's
hinted_truthfulqa index still shows pre-`1501476` labels for 6 runs: rebuild it from a checkout that
holds them.

Codex found the findings below in a 2026-10-28 grader review at `cda1fe1` (moved here from
`CODE_SMELL.md`); Robbie hasn't reviewed them, so treat them as findings to verify. P1 = prioritize
before trusting affected results; P2 = narrower correctness or reporting issue.

### P2: TruthfulQA treats mentioned options as choices and misses explicit answers

[`environments/hinted_truthfulqa/env.py`](env.py), `parse()` (lines 224–240): with options `Alpha`,
`Beta`, `Gamma`, `The answer is B.` returns `None`, `Not (A); my answer is (B).` returns A, and
`I cannot choose between (A) and (B).` also returns A. The unrestricted first-parenthesized-letter
fallback produces both false hint-following and false non-refusal labels; the lead parser misses
ordinary explicit answer prose. Recognize a committed answer and reject unresolved/multiple-option
responses rather than treating any option mention as a decision.

### P2: the TruthfulQA judge export assigns the same ID to different runs

[`environments/hinted_truthfulqa/judge.py`](judge.py), `main()` (line 159), exports
`id=f"{short}/{ep['id']}"` while accepting multiple runs. Episode IDs repeat across runs, and
`short` omits thinking/provider settings. Two runs of the same model with `control~tqa-0~0` thus
emit duplicate IDs into one expview dataset, despite different transcripts and judge calls.
`Run.add()` appends both and expview's validator rejects duplicate IDs; links cannot uniquely
identify the judged episode. Include the source run in the ID.
