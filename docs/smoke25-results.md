# Smoke25: reproducing Agam's CHIVE harness on Flash and Sonnet 5.5 (2026-10-19)

**Status (2026-10-26):** REFERENCE. Phase 1. The judge was Opus 5, so these aren't comparable to
later Sonnet 5.5 numbers; the wrapper script is gone.

A first check that our wrapper around Agam's harness
([agu18dec/detective-joracle](https://github.com/agu18dec/detective-joracle), vendored at
`vendor/detective-joracle`, run as a uv dependency until commit
[`a1e8991`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/a1e8991))
gives numbers close to his. **n = 25, one draw each, so the arm-to-arm gaps below are noise** (a
pass@1 standard error is about ±9pp). Don't read O-lens vs transcript off this table.

## Setup

- **Incidents:** `splits/smoke25.txt`, 25 incidents from Agam's held-out `climb_test204`, chosen so
  we can compare against his 5 Opus 5 draws per incident. Because they're test incidents, this was a
  one-off reproduction check. Iterate on `splits/climb_dev96.txt`.
- **Agents:** Gemini 3.8 Flash via OpenRouter (reasoning effort `high`, the OpenRouter maximum), and
  Claude Sonnet 5.5 called directly (effort `max`). Agam's reference agent is Opus 5 at effort
  `max`. Records say `effort: max` and `backend: openrouter` for every run; ignore both fields.
- **Judge:** Opus 5 via OpenRouter, 3 replicates, pass = score ≥ 7, as in Agam's harness.
- **Arms:** `transcript` (CoT and transcript only), and `olens_rl_B` (Agam's frozen harness "B" plus
  his RL-trained O-lens readout bank, layers 20–60 step 4, k = 4 samples per cell). Both use
  `variants=['textfirst','absence']`: the readout tools stay locked until the agent commits a
  transcript-only hypothesis.
- **Evals:** _free-form_ (no hypotheses given; the agent writes the cause; judged) and _yes/no_
  (`yesno=true`: the agent gets the incident's claim-bank edits and gives P(edit flips the
  behaviour) for each; scored without a judge).

## Results

Free-form:

| agent      | arm        | pass@1 | Agam's Opus 5 pass@1, same incidents | mean tool calls | mean output tokens |
| ---------- | ---------- | -----: | -----------------------------------: | --------------: | -----------------: |
| Flash      | transcript |    76% |                                  74% |              22 |                12k |
| Flash      | olens_rl_B |    72% |                                  72% |              40 |                 6k |
| Sonnet 5.5 | transcript |    80% |                                  74% |               9 |                19k |
| Sonnet 5.5 | olens_rl_B |    76% |                                  72% |              33 |                38k |

Yes/no ("committed" = the transcript-only predictions the agent locked in before unlocking the
readouts):

| agent      | arm        | finished | within-incident AUC | pooled AUC | accuracy | Brier | committed within-AUC |
| ---------- | ---------- | -------: | ------------------: | ---------: | -------: | ----: | -------------------: |
| Flash      | transcript |    22/25 |                0.70 |       0.74 |     0.77 |  0.20 |                    — |
| Flash      | olens_rl_B |    12/25 |                0.58 |       0.73 |     0.74 |  0.21 |                 0.63 |
| Sonnet 5.5 | transcript |    25/25 |                0.75 |       0.71 |     0.73 |  0.21 |                    — |
| Sonnet 5.5 | olens_rl_B |    25/25 |                0.78 |       0.74 |     0.72 |  0.19 |                 0.71 |

## Takeaways

- **The wrapper reproduces Agam's free-form numbers.** Flash and Sonnet 5.5 both land within a few
  points of his Opus 5 pass@1 on the same incidents, in both arms.
- **Flash can't finish yes/no with readouts.** 13 of 25 O-lens episodes hit the 50-call cap
  (`max_calls`) and produced no answer (3 transcript episodes hit the cap or the output budget), so
  Flash's O-lens yes/no row covers fewer than half the incidents. Raise `max_calls` or cut the
  per-call tool output before running yes/no at scale with Flash.
- **Readouts make agents spend more tool calls:** 2× for Flash and 3–4× for Sonnet 5.5 in free-form,
  with no pass@1 gain at this n.

## Reproduce

After commit
[`a1e8991`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/a1e8991)
the harness stopped being a uv dependency (`vendor/` is reference-only, see `vendor/CLAUDE.md`) and
`scripts/run_chive_or.py` was removed; we use `cfbench/` now. To rerun, check out that commit and
`git submodule update --init`:

```bash
uv run python scripts/fetch_olens_banks.py splits/smoke25.txt data/banks/olens_rl
OPENROUTER_API_KEY=... uv run python scripts/run_chive_or.py stage=agent eval=smoke_free \
    ids=splits/smoke25.txt out=data/runs/chive draws=1 "variants=['textfirst','absence']" \
    "arms=['transcript','olens_rl_B=data/banks/olens_rl:olens']"
# same with `yesno=true eval=smoke_yesno`, and with CHIVE_AGENT=claude-sonnet-5-5 (needs
# ANTHROPIC_API_KEY) for the Sonnet rows; then stage=judge for the free-form evals
uv run python scripts/summarize_smoke.py data/runs/chive data/runs/summary.json \
    smoke_free:flash smoke_free_sonnet55:sonnet55 smoke_yesno:flash smoke_yesno_sonnet55:sonnet55
```

Code:
[`scripts/fetch_olens_banks.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/fetch_olens_banks.py),
[`scripts/summarize_smoke.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/summarize_smoke.py)
at commit `7296169`, deleted 2026-10-26.
