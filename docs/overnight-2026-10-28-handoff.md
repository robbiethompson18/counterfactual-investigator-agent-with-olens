# Overnight 2026-10-27/28: four gated subject-side envs, handoff

**Status (2026-10-28):** REFERENCE. Super duper experimental: four envs built, audited, rebuilt and
gated overnight by Claude agents with almost no human review, then merged so they can be iterated
on. Treat every number as a sign-of-life probe.

## What happened

1. Four new subject-side envs (Qwen3.6-27B, thinking off) were audited by subagents on 2026-10-27.
   All four were NOT CLEAN: each counterfactual label was predictable from prompt surface or world
   knowledge.
2. Robbie asked for a biased proof of concept anyway. One agent per env rebuilt it with a much
   larger item pool, then ran a **text-only gate** with at most 5 tries. The gate shows Gemini 3.8
   Flash the base-arm prompt, one Qwen reply and the counterfactual prompt. It kills an item if
   Flash can predict Qwen's counterfactual answer: any match in 3 guesses for multi-class targets,
   the majority of 3 for binary ones. Survivors are where a lens could add something. Selecting on
   Flash's failures biases the transcript baseline down, so compare O-lens only against placebo on
   survivors.
3. For envs with survivors: O-lens reads of the base-arm transcripts (one 1x H200 pod, ~7 min,
   ~$0.6), then cfbench `olens_agentic` vs `olens_placebo_agentic` (Flash, 3 draws). No
   transcript-only confirmation runs (Robbie: they add bias).

## Results per env

| env                                        | PR   | gate survivors                                       | O-lens vs placebo (vs null)         |
| ------------------------------------------ | ---- | ---------------------------------------------------- | ----------------------------------- |
| [debug_pointer](envs/debug_pointer.md)     | #246 | 12 item/arm pairs on 10 items (of 77)                | 8% vs 0% (vs 0%), 12 cases: null    |
| [style_transfer](envs/style_transfer.md)   | #247 | 10 of 130 (7 flips, 3 no-flips)                      | 17% vs 23% (vs 30%), 10 cases: null |
| [todo_suppressed](envs/todo_suppressed.md) | #248 | 10 of 62 usable (15 more with errored Flash guesses) | not run                             |
| [notice_defect](envs/notice_defect.md)     | #250 | 9 of 122 usable, all not-noticed: **failed** the bar | not run                             |

- **debug_pointer:** all 3 correct O-lens draws are on 2 cases (`headingrad`, `agecalc`). The reads
  that helped sit on the prompt's code and restate the convention the code breaks, which Flash could
  do itself. No read named Qwen's own suspect.
- **style_transfer:** the doc's echo-vs-latent caveat shows up in the data. The early read on
  `count_occurrences`'s ask line says "This function should be recursive", which is the codebase's
  pull, not Qwen's default.
- **todo_suppressed:** stopped mid-try-1. The tally in the doc takes each item's latest gate
  verdict; the merging gate run never finished.
- **notice_defect:** five tries; Flash predicts every defect Qwen notices, so only "missed" items
  survive and "always not flagged" sweeps them.

## Where everything lives

- Code (all merged to `main`):
  `environments/{debug_pointer,style_transfer,todo_suppressed,notice_defect}/` (`env.py`, `gate.py`;
  debug_pointer also `verify.py` and `repos/`). Each env's doc in `docs/envs/` has its try log and
  caveats.
- Env runs: HF `envs/<env>/<run>/` (Docent collections named after each env).
- Gate runs: HF `runs/<env>_gate/` (`guesses.jsonl`, `verdicts.jsonl`; style_transfer's are
  `runs/style_transfer_gate/2026-10-27_flash3{,_round2}`).
- O-lens reads: HF `runs/olens_reads/2026-10-27_dp-survivors`,
  `runs/olens_reads/2026-10-27_st-survivors`.
- cfbench: HF `runs/cfbench/2026-10-28_dp-d1-olens-vs-placebo`, `_dp-d2-olens-vs-placebo`,
  `_st-olens-vs-placebo` (the `2026-10-27_dp-*` ones are abandoned after Gemini 429s).

## Caveats

- **Serving mix:** style_transfer's Qwen ran bf16 on short-lived RunPod pods; the others ran on
  OpenRouter fp8 (venice or deepinfra). debug_pointer's try 2 used `--max-tokens 2000` (OpenRouter
  402 on uncapped requests; replies are 3-4 tokens).
- **Gate plumbing:** some gates mixed OpenRouter and direct Google calls; the route is per row or in
  `meta.json`.
- **Gemini billing:** at 01:40 PDT `OLD_GEMINI_API_KEY` returned 402 (prepayment depleted),
  `THIRD_GEMINI_KEY` 403 (outstanding balance) and OpenRouter's BYOK route 402. Only
  `GEMINI_API_KEY` worked.
- **Shared code:** #301 (closed) fixed `scripts/olens_requests_from_runs.py` the same night main did
  (55e1d90). One difference to check: it set `reasoning_content: ""` on assistant turns so
  thinking-off transcripts render the empty think block; main's version doesn't, and tonight's reads
  were built with #301's.
- **`environments run --no-upload` still uploads to Docent** whenever `DOCENT_API_KEY` is set (only
  HF is skipped). An audit smoke made the empty public Docent collection `debug_pointer` that way.

## What I'd do next

- If anything: todo_suppressed's 10 survivors are the one untested set. Finish its gate on the 15
  errored items, then reads + olens vs placebo, the same recipe as above.
- Don't spend more on notice_defect's current design: the gate can only keep "missed" items.
- For debug_pointer and style_transfer, the null is on 10-12 cases; a bigger pool is the only way to
  a real test, and style_transfer needs reads before the codebase (or on a no-codebase prompt) to
  avoid the echo.
