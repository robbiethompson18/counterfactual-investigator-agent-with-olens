# Cost estimates: what a read job or a predictor run costs, from measured rates

**Status (2026-10-29):** ACTIVE. Rates measured on the runs named; redo the arithmetic, not the
measurements, when planning a run.

Every planning question so far ("what would 10k rollouts cost", "5k readouts on three lenses",
"Haiku on 7 conditions") came down to the same few rates. They are here so nobody re-derives them.

## GPU: $5.29 per H200-hour (RunPod, 2026-10)

| job                                         | rate per 1x H200                                  | measured on                                                                                                                                 |
| ------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.6-27B short MCQ rollouts (shared pod) | ~50 episodes/s (2x H200, `--conc 128`)            | `envs/supergpqa/2026-10-29_n8`: 8,000 episodes in ~2.5 min                                                                                  |
| O-lens reads, one-engine reader             | ~19 reads/s (one read = one position × one layer) | `olens_reads/2026-10-29_mcq-unselected-nohint-b2`: 119k reads, 3 pods, ~1 h incl. boot ([one-engine-reader.md](olens-one-engine-reader.md)) |
| NLA readouts (AV decode)                    | ~16 reads/s, then the critic at ~35 reads/s       | `nla_reads/2026-10-27_hint-mcq-nla` ([nla-reader.md](nla-reader.md))                                                                        |
| J++ reads, every token                      | ~690 positions/s                                  | `jpp_reads/2026-10-27_hint-mcq-header`                                                                                                      |
| pod boot + `olens_reads.sh setup`           | ~8 min of pod time                                | every read job                                                                                                                              |

So a read job costs about `positions × layers / rate / 3600 × $5.29`, plus ~$0.70 of boot per pod. A
3,000-case MCQ set at 20 positions (10 option newlines + the 10 end-of-prompt tokens):

| reader           | reads            | GPU-hours | cost |
| ---------------- | ---------------- | --------- | ---- |
| O-lens, 6 layers | 360,000          | ~5.3      | ~$28 |
| NLA, 3 layers    | 180,000          | ~4.6      | ~$24 |
| J++, 6 layers    | 60,000 positions | <0.1      | ~$1  |

## Predictors (OpenRouter / Anthropic prices, 2026-10)

| model            | input $/M | output $/M | note                                                                         |
| ---------------- | --------- | ---------- | ---------------------------------------------------------------------------- |
| Claude Haiku 5.5 | 0.10      | 0.50       | prompts ≤ 100k tokens; batch API is half                                     |
| Gemini 3.8 Flash | ~0        | ~0         | BYOK on Robbie's Google project: per-day caps, not dollars (CLAUDE.local.md) |

Measured per draw, Haiku 5.5 at effort medium, one `submit_answer` (runs
`cfbench/2026-10-29_mcq-unselected-haiku-b2` and `2026-10-29_supergpqa-haiku-3k-{ts,jlens}`):

| condition                                              | prompt tokens | output tokens | $ per draw |
| ------------------------------------------------------ | ------------- | ------------- | ---------- |
| `transcript_inline` (the prompt only)                  | ~1,000        | ~500          | 0.0002     |
| `olens_inline`, 6 layers, 16 positions, 4 bullets each | ~19,000       | ~480          | 0.0016     |
| `jlens_inline`, every token (~190) × 6 layers × top 3  | ~44,000       | ~690          | 0.0067     |

The block size is what matters: a J++ every-token block on a 10-option MCQ is 44k tokens and costs
4x an O-lens block. Rule of thumb: **$0.0002 per prompt-only draw, ~$0.0001 per 1k prompt tokens of
reads block**, so count the block's tokens first (`cfbench.eval NAME --show`, then `count_tokens`).
A 20-position O-lens block at 6 layers is ~22k tokens, a 3-layer NLA block ~9k. For 3,000 cases × 1
draw: prompt only $0.50, O-lens or its placebo ~$6, NLA ~$4, J++ $20. Only consistent questions
become cases (65% of SuperGPQA items, 63% of MMLU-Pro), so count cases, not items.

## Rollouts (Qwen3.6-27B on OpenRouter)

DeepInfra fp8 is ~$0.001 per short MCQ episode, but the new OpenRouter account is capped at 20
requests/min for this model (CLAUDE.local.md), so anything over ~1,000 episodes goes to the shared
vLLM pod.
