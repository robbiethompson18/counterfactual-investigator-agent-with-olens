# Default read positions: read everywhere, all six layers, thinking off

**Status (2026-10-30):** ACTIVE. Robbie's default for every O-lens / NLA / J++ read of a prompt or
transcript; Kylie's runs may choose differently.

## The default

Unless a run has a stated reason to do otherwise, a metamodel read of a Qwen3.6-27B prompt is taken
at **every one of these positions**:

- **Every punctuation token** of the prompt: the reader's `PUNCT` regex (`[.,;:!?]` plus trailing
  quotes or brackets, `scripts/olens_read_requests.py`), and every `<|im_end|>`. That includes the
  `?` that ends the question and the `.` after each MCQ option letter.
- **Each MCQ option's newline**: the `\n` ending each option line, `\n\n` after the last
  (`option_newlines`).
- **The ten tail tokens of the thinking-off render**, up to the token right before Qwen's reply: `.`
  (end of the instruction), `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`, `<think>`,
  `\n\n`, `</think>`, `\n\n`.

On a whole transcript (Qwen's reply included), that means the reader's default: every `PUNCT` token
and every `<|im_end|>` everywhere, plus the ten tail tokens before each reply you care about.

And always:

- **All six layers: 32 / 40 / 48 / 52 / 56 / 60**, for the O-lens, the NLA and J++ alike. Never a
  subset for one lens: the 2026-10-29 SuperGPQA NLA read only 32 / 40 / 48 and had to be topped up.
- **Thinking off** (the repo default, docs/thinking-off.md), and **the render Qwen actually saw**:
  the thinking-off chat template, whose generation prompt ends `<think>\n\n</think>\n\n`.
- **O-lens:** space lens + AR (`olens_s3d_space_rl600`), T 1. **NLA:** all-layer NLA, greedy, then
  the critic's FVE (`nla_score.py`). **J++:** every token.

Showing the investigator only some of these positions (a smaller block) is a choice made when
building the condition, from reads that already exist. It is not a reason to read fewer.

## Why

The 2026-10-29 inline evals (cutoff_qa, favorites_qa v3, mcq_unselected, supergpqa) were read at
hand-picked positions: the six tokens from the instruction's `.` to the `\n` after `assistant`, plus
the option newlines on the MCQ sets. SuperGPQA also had the four think-block tokens. That missed
three things:

- **The think block, where the answer is.** On SuperGPQA, the top O-lens bullet names Qwen's letter
  on 80% of cases at the last token (`\n\n` after `</think>`, L60), against 72% at the instruction's
  `.` ([supergpqa-inline.md](supergpqa-inline.md)). The other three sets stopped 4 tokens short, so
  their O-lens gains are likely underestimates.
- **Punctuation inside the prompt**: 3 per prompt on cutoff, 1.3 on favorites (the question's own
  `?`), 22 on mcq_unselected and 18 on SuperGPQA. Nothing there was ever read. For "what will Qwen
  answer" the late tokens probably dominate, but anything the lens says that isn't in the transcript
  (leg 2 of the project) would show up at those mid-prompt sites.
- **Consistency**: SuperGPQA's NLA had 3 layers where everyone else had 6.

Reading everywhere costs about 2.5× the hand-picked set. At the measured rates
([cost-estimates.md](cost-estimates.md)), a 3,000-question MCQ set at ~40 positions × 6 layers is
~720k O-lens reads (~$55 of H200) and the same NLA reads (~$95). That is cheap next to finding out
later that the signal was at a position nobody read.

## How to request it

The reader's defaults cover punctuation and `<|im_end|>`, but not the think block. A `messages` row
with `generation_prompt: true` renders the thinking-off header (the empty think block) since
2026-10-31; before that it rendered the thinking-ON header (`<think>\n`), so reads at or after
`<think>` from such rows saw a prompt Qwen never saw (reads up to the `\n` after `assistant` were
unaffected: the model is causal and the prefix is identical). Set `thinking: true` on the row for
the thinking-on header. A **`text` row holding the thinking-off render** with explicit positions
also works, and is what the existing inline sets use:

```python
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, add_special_tokens=False).input_ids
toks = [tok.decode([i]) for i in ids]
positions = {p for p, t in enumerate(toks) if PUNCT.match(t) or t == "<|im_end|>"}
positions |= set(option_newline_positions)  # MCQs: the token holding each option's ending newline
positions |= set(range(len(ids) - 10, len(ids)))  # the ten tail tokens, ending at </think>\n\n
row = {"id": ..., "text": text, "positions": sorted(positions)}
```

then run `runpod/olens_reads.sh run NAME requests.jsonl TAG --layers 32,40,48,52,56,60` (and
`runpod/nla_reads.sh` the same way). `scripts/fill_read_requests.py` (PR #432) is a worked example:
it does this for the four inline sets and asserts that the new render's tokens equal the old reads'
tokens.

## Which reads have it

- **cutoff_qa (999), favorites_qa v3 (1,500), mcq_unselected (2,579), supergpqa (3,000):** the
  2026-10-29 runs plus `olens_reads/2026-10-30_inline-fill` and `nla_reads/2026-10-30_inline-fill`
  (the missing positions, plus SuperGPQA's NLA at layers 52 / 56 / 60), all six layers for both
  lenses. Load old + fill as one row list with `cfbench.evals.inline_reads.rows(set, reader)` (PR
  #432); no splicing of files. The evals' existing conditions still read only the old positions, so
  their published numbers reproduce. New conditions should read the full set.
- J++ was already read at every token on all four sets.
- Every other read run before 2026-10-30: check its `meta.json` `positions` param before assuming.
