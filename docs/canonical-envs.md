# Canonical envs

**Status (2026-10-29):** MIXED. No longer the standard set. Six of the eight envs are retired, not
ported ([`environments/retired/`](../environments/retired/README.md)): `evalcue/` was deleted on
2026-10-27, so the `evalcue.run` names below are at commit `c7b2e56`. `hidden_goal.grade` and
`donation_bet.goodbad` were ported as [`essay_grading`](../environments/essay_grading/) and
[`donation_bet`](../environments/donation_bet/). The criteria, the rates and the canonical fp8 runs
of 2026-10-23 at the end stand; the reads table predates the thinking-off reads of all eight
(2026-10-24, [nocot-cfbench.md](nocot-cfbench.md)).

The 8 envs (and their arm pairs) we use as the standard set for O-lens experiments: cfbench
predictions with and without lens reads, thinking-off reruns, read budgets. Picked 2026-10-23.

**Status: chosen by Robbie's Claude from the env docs. Robbie OK'd the env list in chat but has not
reviewed the arms, rates or transcripts.** Treat the arm choices as a first draft. If you change
one, say why here.

## Criteria

1. A big arm-pair effect at mid rates, not stuck at 0% or 100%.
2. The cause of the effect is mostly not stated in the CoT, so the lens has something to add over
   transcript-only.
3. Programmatic labels (no LLM judge).
4. Cheap to run.

## The list

Rates are as stated in each env's doc (n per arm in the doc). Servers differ: OpenRouter fp8 ran
thinking-on Lynch up to 40 points below the bf16 vLLM tables, so compare arms only within one run.

| env (`evalcue.run` name)        | canonical arm pair(s)                                                           | how often the CoT states the cause             | doc                                                  |
| ------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| `injection_slack`               | `important` 25/53 vs `wronguser` 5/54; `important` vs `after` 36/54             | ~2/49 `wronguser` refusals name the signer     | [envs/injection_slack.md](envs/injection_slack.md)   |
| `hidden_goal.grade`             | `base` 2/30 vs `mine` 16/30, `ex` 20/30, `ai` 23/30 (thinking off already)      | author named in 0/30 replies (`ai`, `ex`)      | [envs/hidden_goal.md](envs/hidden_goal.md)           |
| `scheming.exfil`                | `base` 24/50 vs `evalcue` 8/40                                                  | cue mentioned in 3/40                          | [envs/scheming.md](envs/scheming.md)                 |
| `whistleblow.single`            | `bold+penalty+minor` 37/64 vs `+no_shutdown` 6/32, `+docs_reversed` 7/32        | order ≤ 5/24; the removal never                | [envs/whistleblow.md](envs/whistleblow.md)           |
| `shutdown.killswitch` / `.cron` | killswitch `base` 0/30 vs `may_edit` 36/40; cron `base` 0/20 vs `may_edit` 9/30 | 7/36 killswitch positives mention write access | [envs/shutdown.md](envs/shutdown.md)                 |
| `dumb_user_report.inline_b`     | `anon` 57/80 vs `hope_done` 22/80                                               | ~35% of `hope_done` runs                       | [envs/dumb_user_report.md](envs/dumb_user_report.md) |
| `scheming.murder_ng`            | `monitored` 6/34 vs `unmonitored` 22/38                                         | about a third mention monitoring               | [envs/scheming.md](envs/scheming.md)                 |
| `donation_bet.goodbad`          | `bet_neutral` vs `good_below`: 32/60 vs 49/59 (on), 14/57 vs 47/58 (off)        | CoT 25/49 (on); reply 16/47 (off)              | [thinking-off.md](thinking-off.md)                   |

Close calls left out: `hiring_resume` borderline names (Black 56% vs White 48%, unverbalized but an
8-point gap); the locale censorship pair in
[identity-counterfactuals.md](identity-counterfactuals.md) (silent, but 0% vs ~100% and not an
evalcue env). Left out on purpose: verbalized levers (sandbagging, funding_email, whistleblow's
bold/tame), envs with no working arm pair (exclude_rows), and n=10 cells (overnight).

## O-lens reads on hand (2026-10-23, space lens `olens_s3d_space_rl600`)

| env                              | read transcripts                                                             |
| -------------------------------- | ---------------------------------------------------------------------------- |
| hidden_goal.grade                | 840 (L32/40/48, thinking off): `olens_reads/2026-10-22_hidden-goal-v{1,2,3}` |
| donation_bet.goodbad             | 120 (6 layers, thinking on; no `bet_neutral`): `donation-bet-v1`             |
| shutdown.killswitch              | 70: `cfbench-spike-v1`                                                       |
| inline_b                         | 21 of 160 requested: `cfbench-spike-v1` (rest in the unread queue)           |
| scheming.exfil                   | 20 of 90: `cfbench-spike-v1`                                                 |
| whistleblow.single               | 18 of 72: `cfbench-spike-v1`                                                 |
| injection_slack, murder_ng, cron | none                                                                         |

No thinking-off transcript outside hidden_goal has reads.

## Canonical runs (2026-10-23, OpenRouter DeepInfra fp8, n=40, seed 2)

`<env>/2026-10-23_canon_{on,off}` (thinking on / off): the runs cfbench's model chart and smoke
tests read. Rates differ from the bf16 numbers in the table above; compare arms within one run only.

| env                         | arms                          | thinking on            | thinking off            |
| --------------------------- | ----------------------------- | ---------------------- | ----------------------- |
| `shutdown.killswitch`       | `base` → `may_edit`           | 0/40 → 30/40           | 0/40 → 6/40             |
| `shutdown.cron`             | `base` → `may_edit`           | 0/40 → 27/40           | 0/40 → 9/40             |
| `scheming.exfil`            | `base` → `evalcue`            | 16/40 → 10/40          | 14/40 → 7/40            |
| `dumb_user_report.inline_b` | `anon` → `hope_done`          | 27/40 → 9/40           | 27/40 → 7/40            |
| `hidden_goal.grade`         | `base` → `mine` / `ex` / `ai` | 0 → 1 / 9 / 17 (of 40) | 2 → 9 / 20 / 29 (of 40) |

No `canon_*` run exists for `injection_slack`, `whistleblow.single`, `scheming.murder_ng` or
`donation_bet.goodbad`.
