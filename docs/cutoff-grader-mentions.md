# cutoff_qa grader mentions: O-lens reads say "graded" far more under an agentic template

**Status (2026-10-31):** REFERENCE. Regex only (no LLM judge). One prompt set, one model
(Qwen3.6-27B), one lens.

**Question.** Do the O-lens reads of a cutoff_qa prompt mention a grader more often when Qwen gets
the question in an agentic template (a system block with a `submit_answer` tool,
`cfbench/evals/cutoff_qa_tool_inline`) than in the plain chat template? The tool's system block
never says "grade", "score", "eval" or "test".

**Setup.** `scripts/cutoff_grader_mentions.py`, run `cutoff_grader_mentions/2026-10-30_regex` on
HF.

- Reads:
  - Chat template: `olens_reads/2026-10-29_cutoff{200,800}-tail` + `2026-10-30_cutoff-fill-kylie`
    (999 items), plus `2026-10-30_cutoff-new1k` (another 1,000 items) as a replication.
  - Agentic template: `2026-10-30_cutoff-v3tool-punct` (1,849 items).
  - All runs use the space lens + AR at T 1, layers 32/40/48/52/56/60.
- On the 863 items both templates share, the user turn is token-identical, and the same 67,806
  (position, token, layer) sites are read in both.
- A hit is a bullet matching `\bgrade[rd]\b`, case-insensitive. It counts only if the word is not
  within ±40 tokens of the read and not in the system block. Wider word families (grader, eval/test,
  correctness, "testing me") are in the run's CSVs.
- CIs are 95% bootstraps over items.

**Result.** On the 863 shared items:

| `\bgrade[rd]\b`, novel hits                                    | chat              | agentic              |
| -------------------------------------------------------------- | ----------------- | -------------------- |
| reads at the `.` ending "If you are not sure, give your best guess." | 2.4% [2.0, 2.8]   | 19.2% [18.1, 20.3]   |
| items with ≥ 1 such read (6 layers)                            | 14.0% [11.6, 16.3] | 73.2% [70.1, 76.4]  |
| reads at the `,` after "not sure"                              | 0.7%              | 2.4%                 |
| reads at every other position (question, `<\|im_end\|>`, header, think block) | ≤ 0.2%    | ≤ 0.2%               |

- **Mechanism:** almost every hit continues the instruction, e.g. "If unsure, give your best guess.
  Your answer will be graded."
- **Layers:** at the `.` the agentic template peaks at L40 (40.7% vs 3.5%) and L56 (34.0% vs 7.6%).
- **Stale answers:** the rate is the same on stale and non-stale items.
- **"Is the user testing me?"** bullets occur in both templates at similar rates.

Charts: `uv run python -m graphs.cutoff_grader_mentions`. Slide: the anecdata frame in
`slides/talk`, with example item e032-09 at L40.

**Caveats.**

- This is a regex count, not a judge.
- The reads are of prompts only, so there are no replies to compare against.
- The agentic template differs from the chat template only by the system block, so this can't say
  which part of that block (tool schema, format instructions) does it.
