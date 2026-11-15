# Decapitated-network unroll on WorkspaceBench's multi-token families

**Status (2026-11-14):** NULL. Qwen3.6-27B cut after block L with the J++ lens as its head, left to
write 32 tokens, passes 1-8% of items per family; the 🔮-lens passes 42-90% and the prompt-only
floor 42-69%. Don't retry the pure unroll; the untested variant is "J++ picks the first token,
another head spells the rest".

The idea (Robbie, 2026-11-14): skip the AR, the inverter and RL. Keep blocks 0..L of the model, drop
the rest, and put the J++ lens (Ayonrinde and Lindsey, 2026,
[neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)) on top:
`lm_head(norm(J[L] h_L))`. That is a smaller language model. Give it a prompt and let it generate:
its first token is the ordinary J++ read at the prompt's last token, and each token it writes goes
back through blocks 0..L. The generated text is the readout. We found nobody who has tried this as a
lens. The nearest work is early-exit generation for speed (CALM, Schuster et al., 2022; LayerSkip,
Elhoushi et al., 2024) and DecoderLens (Langedijk et al., 2023, encoder-decoder models).

## Setup

- **Items:** [WorkspaceBench](https://github.com/camilablank/workspace-bench)'s six multi-token
  families, 598 plain-text prompts, read at the last prompt token. These are where a single-token
  lens cannot name the concept ("Aghlabid dynasty", "Simula 67").
- **Cut layers:** their grid, L20 to L60 in steps of 4. An item passes if any layer passes.
- **Decodes, 32 tokens each:** greedy; greedy with every generated token banned from repeating; one
  sample at T 1, top-k 64, top-p 0.95 (WorkspaceBench's sampling).
- **Scorer:** their regex contract (`src/wsbench/multitoken/regex.py`), copied verbatim. No judge
  call. A form must be found whole in one readout, and only forms that are more than one Qwen token
  count.
- **Control:** nothing cut, i.e. Qwen itself continuing the prompt for 32 tokens, greedy.
- **Code:** `scripts/decap_unroll.py` is the producer (pod, HF transformers) and is all we kept. The
  scorer port (`wsb/mt_regex.py`), the requests and scoring module (`wsb/decap.py`) and the bank
  builder's six multi-token families are at commit
  [`aac1504c`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/commit/aac1504c1c917ce51d40eb7279660657ed88ccbf),
  PR #635's first commit. The six banks are on HF under `banks/wsb/`.
- **Run:** `wsb_decap_unroll/2026-11-14_jpp-unroll`, 28 min on one H200, $2.23.

## Result

Pass rate, any of the 11 cut layers. Brackets are 95% bootstrap intervals over items.

| family                | no repeats | greedy | sampled | any of 3 | Qwen uncut | 🔮-lens | prompt only |
| --------------------- | ---------- | ------ | ------- | -------- | ---------- | ------- | ----------- |
| basic_readout_mt      | 1% [0, 3]  | 1%     | 0%      | 2%       | 46%        | 90%     | 59%         |
| multihop_mt           | 1% [0, 3]  | 0%     | 0%      | 1%       | 12%        | 42%     | 69%         |
| multilingual_mt       | 6% [2, 11] | 11%    | 7%      | 16%      | 1%         | 76%     | 42%         |
| typo_mt               | 8% [3, 14] | 6%     | 4%      | 11%      | 27%        | 87%     | 67%         |
| multilingual_multihop | 2% [0, 5]  | 1%     | 1%      | 4%       | 7%         | 46%     | 45%         |
| multilingual_typo     | 1% [0, 3]  | 1%     | 1%      | 3%       | 25%        | 90%     | 66%         |

🔮-lens: the published per-item rows (`agu18dec/olens_ablations`,
`ab4/offset/results/wsbench/rl.base.off_k0.own.i300`), same scorer, same layers. Prompt only:
WorkspaceBench's frozen floor (`evals/baselines/prompt_only.json`), stock Qwen shown the prompt and
asked what a model would be thinking at its last token.

- **Only the deepest cut says anything.** With the no-repeat decode, every pass but one is at L60;
  L20 to L56 pass 0% in every family but one (1% at L24 on multilingual_typo). Below L60 the text is
  punctuation, fragments and repeated tokens.
- **Greedy loops.** At L60, 93% of greedy readouts end with 8 tokens drawn from at most 2 distinct
  tokens (" Tunis Tunis Tunis ..."). The J++ head names the current concept again, as we saw when we
  used it as a tool ([jlens-cfbench.md](jlens-cfbench.md)).
- **At L60 with repeats banned it is a stream of related concepts, not the answer.** "Tunis Tunisia
  Egypt Egyptian Cairo Baghdad" for the Aghlabid dynasty; "streptomycin producing bacteria found
  soil microbiology lab Rutgers first antibiotic" for the streptomycin item, which passes. On
  multilingual_mt and multilingual_typo a word of four or more letters from every required concept's
  forms shows up somewhere in 72% of items, but the whole form in one readout in only 6% and 1%.
- **It rarely spells a multi-token word.** After writing a first token the cut model moves to a
  neighbouring concept instead of finishing the word, so the regex, which needs the whole form,
  almost never fires. The passes are mostly common words ("vacuum cleaner", "fire extinguisher",
  "candlestick", "narcissism").
- **The static J++ top-10 scores 0% here by construction** (no token is a multi-token form;
  WorkspaceBench adds an LLM summary of the bag, which we did not run).
- Of the 598 items, 37 pass under any of the three decodes; the 🔮-lens passes 32 of those 37 and
  398 more.

## What this does and doesn't show

It shows the pure unroll is not a multi-token reader. It does not test the split design: take the
J++ head's top tokens as first tokens, let a next-token head (the full model, or a late-layer logit
lens) finish each word, and reset between picks. The workspace paper's figures suggest the first
token is readable (`black` for blackmail, `phot` for photosynthesis); this run says the J++ head
will not supply the second.

Every readout is in the run on HF (`readouts.jsonl`: per item and layer, the head's top 10 and the
three decodes; `scores.jsonl`, `summary.json`).
