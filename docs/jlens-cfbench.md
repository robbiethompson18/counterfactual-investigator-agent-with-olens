# J-lens (J++) as a cfbench tool on the hint_mcq answer task

**Status (2026-10-28):** NULL. Underpowered and incomplete (19 of the 50 planned items): the J-lens
ties the O-lens and its +10pp over its placebo is the size of the item-to-item noise; the two hits
only the J-lens gets are both "A", the J++ lens's favourite letter. The reads and the harness work;
finish the 31 items before trusting either direction.

2026-10-27/28. The question: does the J++ lens (Ayonrinde & Lindsey 2026,
[neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) `qwen3.6-27b/jpp`),
given to the predictor as tools, help it recover Qwen3.6-27B's own answer on hint_mcq, as the O-lens
was tested in [envs/hint_cf.md](envs/hint_cf.md)? Same task: `hint+d1` → `nohint`, answer task,
Gemini 3.8 Flash at effort medium, 3 draws, on the first 50 of the 154 scale2k items whose hinted
arm followed the hint 8/8.

## Harness (PR #312, harness version 8)

- `jlens_agentic` / `jlens_placebo_agentic` (`cfbench/jlens.py`, usage in `cfbench/CLAUDE.md`): the
  O-lens tools' shape, `search_jlens(regex, over="transcript"|"tokens")` and `jlens_at(run, char)`;
  each read shows one `<jlens layer=L>` line per layer with the top-5 next tokens and their
  probabilities. No FVE, no probability filter, layers 36–60, every token.
- Same PR, both lenses: every hit names its token, and a turn's header tokens (`<|im_start|>` … the
  `\n\n` right before the reply) sit at the start of that turn instead of all at one char at the
  previous turn's end, labelled as it. Before v8 a transcript search returned only the first of that
  stack, so the read right before the answer was hard to reach. It made no visible difference to the
  O-lens: 28% (v7) vs 27% (v8) on the same items.

## Reads and the lens check

`jpp_reads/2026-10-27_hint-mcq-scale2k`: every token, layers 36,40,…,60, top-5, the `hint+d1` and
`hint+d2` transcripts of every scale2k survivor (4,144) plus the 1,200-transcript placebo pool
(favourites/opinion), one-engine reader, 14 min on one H200.

J++ top-1 equals the actual next token only 6–16% of the time at L60, and it confidently names the
current or previous option letter. `jpp_sanity/2026-10-28_next-token-alignment` checked that against
Qwen's own next-token distribution on 10 transcripts: **no bug.** Layer convention (`block_output`,
target 63), norm and position alignment match upstream (safety-research/jpp_lens), recomputed reads
match the stored rows (~98% top-1), and it reproduces the J++ README's heart example ("
heart"/"心脏" from L24). It is a concept lens, not a next-token one: the plain logit lens beats it
at L52+ (top-1 agreement with the model at L60: 0.18 J++ vs 0.33 logit lens). Inside user turns
Qwen's own top-1 is mostly `<|im_end|>`, which drags every next-token number down.

Signal check before any predictions (the 18-item `jpp_reads/2026-10-27_hint-mcq-header`): at the
`\n\n` right before the answer, layers 44–56, the own letter is the top non-hint letter on 7 of 18
items, but J++ favours A and B there; among the 11 items whose own answer isn't A or B it is 5 of 11
(chance ~1/7).

## Result (19 items, matched)

| condition       | version | draws   | correct   | items with a hit                              |
| --------------- | ------- | ------- | --------- | --------------------------------------------- |
| J-lens          | v8      | 50      | 28%       | 1380, 10843, 10910 (1/3), 10992, 10993, 11140 |
| J-lens placebo  | v8      | 48      | 18%       | 10843, 10875, 10992, 10993                    |
| O-lens          | v7 / v8 | 57 / 49 | 28% / 27% | 10843, 10992, 10993, 10995, 11140, 11242      |
| O-lens placebo  | v7 / v8 | 57 / 45 | 21% / 25% | 1380, 10843, 10875, 10992, 10993 (v7)         |
| transcript only | v7 / v8 | 57 / 47 | 18% / 19% | 10875, 10992, 10993, 11140, 11242 (v7)        |

v7 rows are the O-lens agent's (`cfbench/2026-10-27_answer-mcq-scale2k-d1*`), v8 rows ours.

- Most hits are items every condition gets (10992, 10993, 10843): guessable from the transcript.
- The J-lens-only hits are 1380 (1/1, which the v7 O-lens placebo also gets 3/3) and 10910 (1/3),
  and both answers are A, the letter J++ favours at the answer position
  ([10910, Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/b6ca4520-afca-4a42-ae3c-bc421b1960fa)).
- These 19 items sit low for transcript-only: on all 154 items the O-lens agent's v7 rows give
  transcript 24%, O-lens 28%, placebo 24%; on the first 50, 17 / 25 / 23. So a 10pp gap on 19 items
  is within the subset-to-subset swing.

## Runs

`cfbench/2026-10-28_answer-mcq-scale2k-d1-jlens50` (163 rows over OpenRouter BYOK, abandoned when
the Google project's prepay ran out), `-b` (84 rows, Google's OpenAI-compatible endpoint called
directly with `OLD_GEMINI_API_KEY`), `-e` (3 rows, direct, `GEMINI_API_KEY`); `-c` and `-d` have 0
rows. The non-J-lens rows in the first two are v8 reruns of transcript/O-lens, stopped on Robbie's
call (they match v7). Harness smokes: `cfbench/2026-10-28_jlens-harness-smoke`,
`…_jlens-google-direct-smoke`.

## Finishing it

The 31 missing items need ~200 predictions (J-lens and its placebo only), e.g.

```bash
uv run python -m cfbench.run env_runs:hint_mcq/2026-10-27_scale2k+2026-10-27_scale2k-topup \
  --model Qwen/Qwen3.6-27B --task answer --items <the first 50 d1 items> --pairs hint+d1:nohint \
  --conditions jlens_agentic,jlens_placebo_agentic \
  --jlens-reads data/runs/jpp_reads/2026-10-27_hint-mcq-scale2k/reads.jsonl --draws 3 \
  --skip <every …-jlens50* predictions.jsonl> --slug …
```

Gotchas from the night of 2026-10-28: one `openai.APITimeoutError` kills a whole cfbench run (only
status errors are retried per draw), and from 01:50 to past 09:00 PDT `OLD_GEMINI_API_KEY`'s project
got only 503 "high demand" / timeouts while `GEMINI_API_KEY`'s project answered 200 and OpenRouter
showed the model up, so a stall can be per-project, not Google-wide.
