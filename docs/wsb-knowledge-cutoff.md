# WorkspaceBench knowledge_cutoff: stock Qwen, no lens, reproduces and beats the lens

**Status (2026-10-28):** REFERENCE. One run per arm; nothing here is pending.

## The claim being checked

WorkspaceBench's `knowledge_cutoff` family
([camilablank/workspace-bench](https://github.com/camilablank/workspace-bench) at `86612fa`,
`evals/knowledge_cutoff/`) asks Qwen3.6-27B a short factual question about an event after its
knowledge cutoff. The label is **Qwen's own stable wrong answer** (at least 7 of 10 samples agree),
kept only if Opus 5.5 (three times) and Sonnet 5.5 miss it under two framings. The published O-lens
scores 0.486 on the 109 frozen items under the family's regex instrument of record (gold named by
strictly more samples than every other option at the reply-start cell, best layer L60 chosen on the
same items).

The family README adds: "Baseline C (stock Qwen given only the text, same regex rule) ties the
O-lens at 0.486 (paired difference +0.000 [−0.119, +0.128])", and "this family cannot separate
reading the activation from decoder priors". The number comes from the authors' private source repo;
the public repo ships neither Baseline C's readouts nor its k or template. This doc reproduces it,
and adds the simpler baseline the family's own definition implies.

## Setup

`wsb/knowledge_cutoff.py` ([wsb/CLAUDE.md](../wsb/CLAUDE.md)). Stock Qwen3.6-27B on the shared vLLM
server (bf16, thinking off through the chat template, T 1, top-p 1), 109 frozen items, no judge;
scoring is a port of their `score.py` strict-win regex. The 41 nocot items are excluded (their text
is in nocot's password-protected zip).

- **describe** (Baseline C's recipe): Qwen is shown its own chat render of the question through
  reply_start (the newline pair after the empty think block) under the stock-Qwen prompt-only
  template, "What would a language model be thinking about on this token?", 5 samples of up to 256
  tokens as one cell.
- **answer**: Qwen answers the user turn, 10 samples of up to 64 tokens as one cell. This is the
  bank's own gate re-run.

Runs (HF `runs/wsb_knowledge_cutoff/`, Docent collection
[wsb_knowledge_cutoff](https://docent.transluce.org/dashboard/9a217654-25a7-41f9-a640-88fa76abd043)):
`2026-10-28_qwen-describe-k5-vllm`, `2026-10-28_qwen-answer-k10-vllm`. Earlier dirs with the same
slugs minus `-vllm` are abandoned OpenRouter attempts (the account is capped at 20 requests/min for
this model).

## Results

| arm                                     | strict win (their rule) | 95% CI (item bootstrap) | any sample names gold | per-sample gold rate |
| --------------------------------------- | ----------------------- | ----------------------- | --------------------- | -------------------- |
| published O-lens, L60 (their README)    | 0.486                   | [0.394, 0.578]          |                       |                      |
| Baseline C, their number (their README) | 0.486                   |                         |                       |                      |
| **describe** (Baseline C recipe, ours)  | **0.459**               | [0.367, 0.560]          | 0.75                  | 0.31                 |
| **answer** (Qwen answers, ours)         | **0.890**               | [0.826, 0.945]          | 0.97                  | 0.68                 |

By label type, strict win: describe 0.55 stale / 0.37 confab_plausible; answer 0.93 stale / 0.85
confab_plausible. Qwen names the true answer in 4% of answer samples and 9% of describe samples.

- **Baseline C replicates.** 0.459 vs their 0.486, inside both CIs. 359 of 545 describe samples hit
  the 256-token cap, like their prompt-only floors (40–73% truncated); Qwen mostly narrates the
  chat-template structure and gets to the answer late or never, which is why the recipe scores low.
- **Qwen answering beats the lens by 40pp.** Of the 12 answer-arm misses, several are the bank's
  regex strictness (Qwen says "349" for a gold of "349.99", "5.1" for "powershell 5.1", "Council
  Regulation (EC) No 139/2004" for "eu merger regulation") and the rest are 10-sample ties (Bhopal 5
  vs Indore 5, Villach 4 vs Klagenfurt 4). None is a case of Qwen not knowing its own answer.

## What it means

The lens's 0.486 here is not evidence of reading the residual stream. The strongest no-metamodel
baseline on this family is the subject model itself, and it scores 0.89 under the same rule: the
label is defined as what Qwen says, so asking Qwen is the ceiling, and the lens (Qwen plus a LoRA,
same cutoff) sits far below it. The authors' Baseline C under-states this because the prompt-only
template asks Qwen to describe a model's state rather than answer, and most of its 256 tokens go to
narrating the chat template.

For the counterfactual investigator this family is moot either way: the transcript already contains
Qwen's answer.

The same pattern appears in `directed_modulation_obscure` (its bank header): lens 0.337, stock-Qwen
text reader R1 0.471 [0.378, 0.566], options-aware reader R6 0.537; the freeze ruling made only the
frontier text readers R0/R0s binding floors ("R1/R3 context"), and those score ~0 by construction
because the gate removed every item Opus or Sonnet could answer. Not re-run here.
