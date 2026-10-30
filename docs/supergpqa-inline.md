# supergpqa_inline: what letter will Qwen say? 3,000 SuperGPQA cases, seven single-turn conditions

**Status (2026-10-30):** REFERENCE. Linear probe baseline added 2026-10-30 (91.9%, 87.8% on
Qwen-wrong). One investigator (Haiku 5.5), one draw, 3,000 cases, all seven conditions run; the
caveats at the end are open.

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
block included), two 1x H200 pods per reader (`--shard i/2`; the O-lens's second shard was finished
by two more pods after its host throttled, every shard file's provenance is in the run's
`meta.json`), 2026-10-29 15:50–19:20 PDT, ~$60 of GPU:

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

Runs `cfbench/2026-10-29_supergpqa-haiku-3k-{ts,olens,jlens,nla}` (Haiku 5.5, effort medium, one
draw per case; $62 of Haiku in all, $40 of it the two J++ conditions, whose every-token blocks are
44k tokens per prompt). Exact match of the submitted letter to Qwen's modal letter; "Qwen wrong" =
the modal letter is not the gold one.

| condition            | all (n = 3,000) | Qwen wrong (n = 1,122) | Qwen right (n = 1,878) |
| -------------------- | --------------: | ---------------------: | ---------------------: |
| transcript_inline    |           68.9% |                  46.2% |                  82.5% |
| olens_inline         |       **76.9%** |              **57.0%** |              **88.8%** |
| olens_placebo_inline |           67.6% |                  46.1% |                  80.4% |
| jlens_inline         |           71.1% |                  49.6% |                  83.9% |
| jlens_placebo_inline |           67.5% |                  44.7% |                  81.1% |
| nla_inline           |           71.5% |                  49.6% |                  84.6% |
| nla_placebo_inline   |           68.4% |                  47.1% |                  81.2% |

Paired differences over cases, mean and 95% bootstrap CI (pp):

| difference                       |              all |        Qwen wrong |        Qwen right |
| -------------------------------- | ---------------: | ----------------: | ----------------: |
| O-lens − O-lens placebo          | +9.3 [7.8, 10.8] | +11.0 [8.5, 13.6] |  +8.4 [6.7, 10.1] |
| J++ − J++ placebo                |  +3.6 [2.3, 5.0] |   +4.8 [2.4, 7.3] |   +2.8 [1.1, 4.5] |
| NLA − NLA placebo                |  +3.1 [1.8, 4.3] |   +2.5 [0.2, 4.8] |   +3.5 [1.9, 5.0] |
| O-lens − transcript only         |  +8.0 [6.5, 9.2] | +10.9 [8.4, 13.6] |   +6.2 [4.6, 7.8] |
| O-lens placebo − transcript only | −1.4 [−2.6, 0.0] |  −0.1 [−2.4, 2.2] | −2.1 [−3.7, −0.6] |
| O-lens − J++                     |  +5.8 [4.3, 7.3] |  +7.5 [4.7, 10.2] |   +4.8 [3.2, 6.5] |
| O-lens − NLA                     |  +5.4 [4.0, 6.8] |  +7.5 [4.9, 10.0] |   +4.2 [2.7, 5.6] |
| NLA − J++                        | +0.4 [−1.0, 1.8] |  +0.0 [−2.8, 2.7] |  +0.7 [−0.9, 2.3] |

So on SuperGPQA the O-lens is a clear win: +9pp over its own placebo and +8pp over the prompt alone,
larger on the cases Qwen gets wrong (+11pp), where "answer the question yourself" fails. J++ and the
NLA each beat their placebo by ~3pp and are indistinguishable from each other. No placebo helps:
unlike cutoff_qa_inline, where off-topic blocks nudged Haiku toward stale answers, here every
placebo is at or slightly below prompt-only. Guessing the gold letter would score 62.6%.

**Where the signal is: the lens writes the letter at the last tokens before the answer.** Share of
cases where the top-FVE bullet at a position names Qwen's modal letter (`canon` of the bullet text,
the same parser as the submissions):

| position                       | L32 | L40 | L48 | L52 | L56 | L60 |
| ------------------------------ | --: | --: | --: | --: | --: | --: |
| `.` ending the instruction     |  6% | 11% | 26% | 67% | 70% | 72% |
| `assistant`                    |  7% | 11% | 22% | 35% | 49% | 55% |
| `</think>`                     |  7% | 11% | 12% | 27% | 46% | 66% |
| `\n\n` after `</think>` (last) | 11% | 12% | 19% | 73% | 77% | 80% |
| an option newline (pooled)     |     |     |     |     |     | 13% |

The think-block tokens that mcq_unselected's reads stopped short of are the best positions: at the
last token, layer 60, the top bullet names Qwen's letter on 80% of cases (76% of the Qwen-wrong
ones). `<|im_end|>`, the newlines around `<|im_start|>` and `<think>` carry almost nothing.

**Reading the lens without Haiku.** A fixed rule, "submit the letter in the top-FVE bullet at the
last token", scores above the investigator:

| rule                                       |   all | Qwen wrong | Qwen right |
| ------------------------------------------ | ----: | ---------: | ---------: |
| gold letter                                | 62.6% |         0% |       100% |
| O-lens, last token, L60, top-FVE bullet    | 79.7% |      75.8% |      82.1% |
| O-lens, last token, L52, top-FVE bullet    | 73.1% |      69.7% |      75.2% |
| Haiku with the O-lens block (olens_inline) | 76.9% |      57.0% |      88.8% |
| NLA readout at the last token, L48, parsed | 19.9% |      19.1% |      20.3% |
| J++ top-1 token at the last token, L52     | 24.8% |      26.6% |      23.8% |

Haiku with the block is 3pp below the rule overall and 19pp below it on the Qwen-wrong cases: given
a read that names the letter Qwen will give, it still answers the question itself about a third of
the time. The NLA's readouts rarely state a letter in a parseable form, and the J++ top token at the
last prompt token is a letter on only ~30% of cases at L52 and almost never at L60, so for those two
lenses the Haiku conditions are the only usable readout here.

**Caveats.**

- The lens is a Qwen3.6-27B LoRA, so as in cutoff_qa_inline this design can't tell "the lens reads
  the letter Qwen represents" from "the lens answers the question from Qwen's own weights". Here the
  two come apart on the Qwen-wrong cases: a text-only reader with Qwen's knowledge would get the
  gold letter, but the top bullet gives Qwen's wrong letter 76% of the time, which is evidence for
  the first reading. The clean control is still WorkspaceBench's text-inversion gate.
- One investigator, one draw per case, prompts as drafted for mcq_unselected and cutoff_qa_inline;
  the placebo is another case's reads with the same option count for all three lenses, so the three
  placebo rows are comparable with each other but not with cutoff_qa_inline's unrelated-prompt
  placebos.
- The all-layer NLA's critic FVE is low here (mean 0.12 at L32, 0.27 at L40, 0.35 at L48, vs
  0.35–0.53 on cutoff_qa), and layers 52–60 were not read for it.

## Linear probe baseline (2026-10-30)

A linear probe on the option-newline activations reads Qwen's letter on 88% of the Qwen-wrong cases,
against 57% for Haiku with the O-lens and 76% for the O-lens last-token rule. Probe: Kylie's one-hot
listwise conditional logit ([mcq-answer-probe.md](mcq-answer-probe.md)), `nl` site (the newline
token ending each option, the same tokens the O-lens reads used: `option_newlines`; every site
decodes to "\n" or "\n\n" for the last option), `hidden_states[52]`, diag standardisation, λ 1. No
LLM is involved. Run `runs/mcq_answer_probe/2026-10-30_supergpqa-inline-probe` (`scores.md`;
`predictions.jsonl` in cfbench answer format), made by
`environments/hint_mcq/probe_unselected.py prep supergpqa_inline INPUTS` then `score`. One H200 job
covered this and mcq_unselected.

| reader                                                          |           all |    Qwen wrong |    Qwen right |
| --------------------------------------------------------------- | ------------: | ------------: | ------------: |
| Haiku, prompt only (transcript_inline)                          |         68.9% |         46.2% |         82.5% |
| Haiku + O-lens (olens_inline)                                   |         76.9% |         57.0% |         88.8% |
| O-lens rule, last token, L60, top-FVE bullet                    |         79.7% |         75.8% |         82.1% |
| probe fit on the 3,328-question probe pool                      |         91.9% |         87.8% |         94.4% |
| Kylie's MMLU-Pro probe (718 questions), frozen                  |         90.4% |         86.5% |         92.7% |
| Kylie's MMLU-Pro probe, frozen, no scaling (raw h)              |         90.5% |         87.1% |         92.6% |
| Qwen's own letter logits (ceiling)                              |         99.8% |         99.5% |         99.9% |
| _gain over prompt only: own-pool probe / frozen MMLU-Pro probe_ | +23.0 / +21.5 | +41.6 / +40.3 | +11.9 / +10.2 |

- **The MMLU-Pro probe transfers unchanged.** Fit on 718 MMLU-Pro questions, it is 1.5 points behind
  a probe fit on 3,328 SuperGPQA questions. The probe reads a "this option is the one" direction
  that does not depend on the dataset, as it did on TruthfulQA.
- **The probe beats every lens condition by more on Qwen-wrong cases** (+31pp over Haiku + O-lens,
  +12pp over the lens rule), where answering the question yourself fails.
- The probe pool is 38% Qwen-wrong, the same mix as the eval set; it is not balanced.

**Next.** (1) The lens rule as an investigator tool: give Haiku the last-token top bullet only, or
make the agent's `olens_at` default to that position. (2) Done: the probe baseline above. (3) A
text-inversion gate (Sonnet given the prompt text). (4) More draws on the O-lens conditions, which
cost $6.50 per draw.
