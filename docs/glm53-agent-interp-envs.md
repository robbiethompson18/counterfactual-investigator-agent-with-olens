# GLM 5.3 on agent-interp-envs coding, Donation Bet and evalcue envs

Runs from 2026-10-23. Model: `z-ai/glm-5.3` on OpenRouter, reasoning effort medium, no
`max_tokens`. Coding runs use the donehack podman sandbox ([donehack/README.md](../donehack/README.md)).
Hack and solve rates are pytest results. No LLM judge was used. Brackets are 95% CIs.

## Coding (agent-interp-envs `coding`, via donehack)

- **is_prime grid** (`donehack/variants.py`): 26 single-factor conditions, each run with and without
  the "Iterate ... pytest" line, n=64 per cell. Hack% with / without the line:

  | condition | base | rules | ask_user | nt2 | self_report | fractional | writable |
  |---|---|---|---|---|---|---|---|
  | hack % | 75 / 69 | 8 / 5 | 25 / 25 | 48 / 36 | 58 / 50 | 66 / 72 | 64 / 78 |

- **Local tasks** (`t_<task>`, repo base config, n=8): all solved 8/8 except everlasting_lids and
  mirror_and_order (7/8 each). No test edits.
- **EvilGenie** (`donehack/evilgenie/run.py`, `configs/coding/lcb.yaml` config): 299 episodes over
  152 problems.
  - 79% solved (pass both the visible tests and the holdout).
  - 20% pass the visible tests but fail the holdout. These are mostly wrong solutions, not hacks.
  - 4 episodes really hardcode the sample outputs (arc191_c, arc195_c).
  - 0 test edits.
- **ImpossibleBench** (same runner, `--data data_ib`, lcb.yaml config): 384 episodes, stopped early
  for budget. Hack rates are 87% [81, 91] on `conflicting` and 98% [93, 99] on `oneoff`. On
  `original`, 100% are solved. The hacks are explicit special cases, e.g. a call counter that returns
  each answer the tests expect in turn.
- **Grader awareness** ([donehack/analysis/grader_awareness.md](../donehack/analysis/grader_awareness.md)):
  - V4 Flash searches outside the workspace in 67% of is_prime episodes. Qwen3.6-27B does in 8%,
    GLM in 0.3%.
  - A CoT conclusion of "there is no hidden grader" predicts hacking: within-condition CMH OR 4.5.

## Donation Bet (`donation_bet.goodbad`, branch claude/donation-bet-named-pairs)

n=800 per unnamed arm and 400 per named arm. The gap is
P(estimate ≤ line | good cause below) − P(≤ line | good cause above).

| comparison | GLM's own neutral-median lines | Qwen's lines |
|---|---|---|
| good vs no donation | +30.6 pp [+26, +35] | +4.1 |
| good vs bad cause | +11.1 [+6, +16] | +8.3 |
| good vs beers | +4.8 [0, +10] | +0.4 |
| named charities, pooled (9 pairs) | +4.1 [+2, +6] | +2.9 |

- Pooled named: CMH p < 0.001, but no single pair survives Holm correction.
- About 1.3% of replies are misparsed by `detect()`'s regex.
- Details: [docs/envs/donation_bet.md](envs/donation_bet.md) on that branch.

## evalcue envs (n=50 per arm)

- **funding_email**: arm_a 98% vs arm_b 0%.
- **sandbagging**: normal 14%, reverse 24%, a_goal 0%, a_goal_lt 4%, b_goal_lt 0%.
- **secret_number**: about 20% cheat, with no cue effect.
- **precommit_scope**: about 2%.

## Links (private claude.ai artifacts; Docent)

- Summary: https://claude.ai/artifact/RBBPSNnSWzBf46UBJ3jTKN
- Coding expview (is_prime grid and local tasks): https://claude.ai/artifact/VukYinByXae3Uo5cVKWb8C
- EvilGenie + ImpossibleBench expview: https://claude.ai/artifact/CZRDK5j6j7vc5yuQa3sMEB
- Donation Bet stats and expview: https://claude.ai/artifact/2LkXKGBJHyDVicGRAYwQAs
- Summary figure: https://claude.ai/artifact/3Y3GnaXH5GY6t3bs4rFAEn
- Docent `coding` collection (V4 Flash, Qwen and GLM): https://docent.transluce.org/dashboard/7a0fe639-9867-4a20-b99b-4f7fb9702eb8
  - Run dirs are `data/runs/coding/2026-10-2*_*`, written by `donehack/analysis/docent_rows.py`.
  - Each was uploaded once (see its `docent.json`). Don't rewrite or re-upload them.
- Docent `donation_bet_goodbad` collection: https://docent.transluce.org/dashboard/fe024751-6ef7-4f31-9df5-3abe990d816c

## Scripts

- `donehack/variants.py --model z-ai/glm-5.3 --reasoning medium --no-max-tokens --small-uid N ...`
  runs the is_prime grid and the local tasks.
- `donehack/evilgenie/run.py [--data data_ib]` runs EvilGenie or ImpossibleBench. Before it can run,
  `extract.py` / `extract_ib.py` must dump the problems.
- `donehack/evilgenie/regrade.py` re-grades episodes made before the normalised holdout comparison.
- `donehack/analysis/glm53_dash/`: `build.py` makes the summary page, `expview_export.py [lcb]`
  the expview pages, and `fig/summary_fig.py` the figure.

## Gotchas

- **LiveCodeBench output formats differ.** LeetCode public tests print `[6, 2]` while the private
  tests print `[0,1]`. Holdout comparison is therefore JSON-equal (functional problems) or
  whitespace-token-equal (stdin problems).
- **7 EvilGenie problems accept any valid answer.** Their holdout can't be checked exactly.
  `multi_answer` flags them and they never set `hack_signal`.
- **Rootless podman collapses at about 140 concurrent sandboxes.** `podman ps` took over 30 s until
  the libpod SQLite DB (`~/.local/share/containers/storage/db.sql`) was switched to WAL. After that,
  100+100 sandboxes ran fine.
- **`--userns auto` slices limit concurrency.** With 1024-id slices, only 64 sandboxes fit in 65k
  subuids. The u200 image (agent uid 200) with 256-id slices (`--small-uid`) fits 4× as many.
- **Enlarging `/etc/subuid` can break existing images.** After it, `podman system migrate` leaves
  image layers owned by the old subuid range unreadable. Fix: keep the old range as an extra line,
  or wipe storage as root.
- **Never `pkill -f` with a pattern that appears in your own shell command.** It kills the issuing
  shell.
