# supergpqa_inline: what letter will Qwen say? 3,000 SuperGPQA cases, seven single-turn conditions

**Status (2026-10-29):** ACTIVE. Super experimental: inputs, reads and the Haiku run in flight; no
result yet.

**Question.** [mcq_unselected](mcq-unselected.md)'s question on a harder pool: given only the
prompt, can an investigator predict the letter Qwen3.6-27B (thinking off) answers, and do O-lens,
NLA or J++ reads of the prompt help beyond a placebo? No hint, no counterfactual, no LLM judge:
truth is Qwen's modal letter, scored as % of cases right.

**Pool.** `environments/supergpqa/`: 10,000 non-calculation SuperGPQA questions, 8 samples each
(runs `2026-10-29_n8` and `2026-10-29_scale9k`): 50.6% right, 6,328 consistent at mcq_unselected's
rule (the modal bare letter in ≥ 7 of 8 samples, counted over all samples). The pool is ~17pp harder
than hint_mcq's MMLU-Pro subset and gives 1.5× the share of consistent-wrong items
([environments/supergpqa/README.md](../environments/supergpqa/README.md)).

**Split (earmarked data).** `cfbench/evals/supergpqa_inline/eval.py` `split()`, seed 0, written to
`runs/cfbench_evals/2026-10-29_supergpqa-inline-inputs/split.json` on HF:

- **Eval set: 3,000 cases**, drawn at random from the 6,328 consistent questions (1,878 Qwen right,
  1,122 wrong). Every read and every prediction below is on these.
- **Probe / validation pool: the other 3,328 consistent questions.** Train probes or anything else
  there; the 3,672 inconsistent questions are in neither.

**Reads**, all on the 3,000 cases, same request text (the thinking-off chat render, empty think
block included), two 1x H200 pods per reader (`--shard i/2`), driven from this doc's session
(2026-10-29 ~15:50 PDT):

- O-lens: `olens_reads/2026-10-29_supergpqa-3k`, space lens + AR, T 1, 20 positions per case (the
  newline ending each option, then `.`, `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`,
  `<think>`, `\n\n`, `</think>`, `\n\n`), layers 32/40/48/52/56/60. mcq_unselected's reads stopped
  before the think block; the four extra tokens are the ones right before the letter.
- NLA: `nla_reads/2026-10-29_supergpqa-3k`, the all-layer NLA at the same 20 positions, layers
  32/40/48, critic FVE added by `nla_score.py`.
- J++: `jpp_reads/2026-10-29_supergpqa-3k`, every token, layers 32/40/48/52/56/60, top 5.

**Conditions** (`cfbench/evals/supergpqa_inline/`, single turn, Claude Haiku 5.5 at effort medium,
one `submit_answer`, the model never named): `transcript_inline`; `olens_inline` /
`olens_placebo_inline`; `nla_inline` / `nla_placebo_inline`; `jlens_inline` /
`jlens_placebo_inline`. A placebo is another case's reads with the same option count, seeded per
(draw, case), presented as this case's, the same rule for all three lenses (cutoff_qa_inline used
unrelated prompts for its NLA and J++ placebos). Prompt texts follow mcq_unselected (O-lens) and
cutoff_qa_inline (NLA, J++).

```bash
uv run python -m cfbench.eval supergpqa_inline --prep
uv run --with transformers --with jinja2 python -m cfbench.evals.supergpqa_inline.eval requests OUT_DIR
uv run python -m cfbench.eval supergpqa_inline --slug supergpqa-haiku-3k --predictors anthropic/claude-haiku-5.5 --draws 1
uv run python -m graphs.supergpqa_inline
```

Budget: ~$55 of GPU for the three readers and ~$27 of Haiku per draw
([cost-estimates.md](cost-estimates.md)).

![transcript-only vs O-lens, J++ and NLA and their placebos, all / Qwen wrong / Qwen right](img/supergpqa-inline.png)

![paired differences between conditions with 95% CIs](img/supergpqa-inline-diff.png)

## Result

Not yet run.
