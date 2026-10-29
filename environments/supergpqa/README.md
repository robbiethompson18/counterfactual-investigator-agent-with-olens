# SuperGPQA, thinking off (a harder MCQ pool than MMLU-Pro)

**Status (2026-10-29):** ACTIVE. One arm, no counterfactual yet. 10,000 items × 8 done (two runs).

Why: for "guess the MCQ answer Qwen gives without seeing it", the usable items are the ones where
Qwen's no-CoT answer is consistent and wrong (a consistent right answer is guessable from the
question). MMLU-Pro's knowledge categories give 16.5% such items at 67% accuracy
([hint_mcq](../hint_mcq/) `nohint`, 2,200 items). SuperGPQA
([Du et al., 2025, arXiv 2502.14739](https://arxiv.org/abs/2502.14739), 26,529 mostly 10-option
questions over 285 disciplines; Qwen3.6-27B's card says 66% with thinking on) should be harder.

|                 |                                                                                                                                                                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Code            | [`env.py`](env.py); [`report.py`](report.py) pooled accuracy, per-item consistency, consistent-and-wrong share, by difficulty / discipline (works on `hint_mcq` runs too)                                                                                                            |
| Arms            | `base`: the question, lettered options, "Answer with the letter only." (hint_mcq's `nohint` prompt)                                                                                                                                                                                  |
| Primary measure | `correct`, the reply is the gold letter. Also `answer`; a reply that is more than one letter is unscorable (`verbose`)                                                                                                                                                               |
| Pairs           | none                                                                                                                                                                                                                                                                                 |
| Items           | 10,000 uuids: items 0:1000 a seeded sample of the 15,300 non-calculation questions (the first run), 1000:10000 a second seeded sample of the rest (the scale-up) (`is_calculation` false: on calculation items Qwen writes the working out, i.e. CoT in the reply, 2026-10-29 smoke) |
| Requires        | thinking off                                                                                                                                                                                                                                                                         |
| Docent          | [collection](https://docent.transluce.org/dashboard/e75c1169-f6dd-4f36-9137-35ede45f8997)                                                                                                                                                                                            |

```bash
uv run python -m environments run supergpqa --tag n8 --n 8 --model Qwen/Qwen3.6-27B --served-name qwen --backend vllm --url $QWEN_VLLM_URL
uv run python -m environments.supergpqa.report supergpqa/2026-10-29_n8 --by difficulty,discipline
uv run python -m environments.supergpqa.report hint_mcq/2026-10-26_or_nothink2 hint_mcq/2026-10-27_scale2k \
    hint_mcq/2026-10-27_scale2k-topup hint_mcq/2026-10-28_all12k-nohint --arm nohint --by category
```

The `DEFAULTS` are hint_mcq's (OpenRouter, DeepInfra fp8), but the new OpenRouter account is capped
at 20 requests/min for this model, so the run above used the shared bf16 vLLM server (~60 episodes/s
at `--conc 128`, 8,000 episodes in ~2.5 min).

## Runs

`2026-10-29_n8`: 1,000 items × 8 samples, Qwen3.6-27B bf16 on the shared vLLM server, thinking off,
T 0.7 / top-p 0.8 / presence 1.5. 116 of 8,000 replies (1.5%) were not a lone letter (nearly all
several letters, "B, C, D, H", on "which of these statements are wrong" items; a few worked
solutions on math items the `is_calculation` flag misses) and 8 items lost all 8 samples, leaving
992 items. Consistent = modal answer in ≥ 80% of samples.

| pool (run)                                                       | items | accuracy | consistent | unanimous | consistent & wrong |
| ---------------------------------------------------------------- | ----: | -------: | ---------: | --------: | -----------------: |
| MMLU-Pro knowledge categories (`hint_mcq` nohint, fp8 DeepInfra) | 2,200 |    67.0% |      76.7% |     72.2% |        16.5% (363) |
| SuperGPQA non-calculation (`2026-10-29_n8`, bf16 vLLM)           |   992 |    50.1% |      64.3% |     57.5% |        25.0% (248) |

By SuperGPQA's own difficulty label: easy 50.3% / middle 51.5% / hard 36.9% accuracy, consistent and
wrong 21% / 27% / 36% (only 53 hard items survive the non-calculation filter). By discipline,
accuracy runs from 31% (History, n = 39) and 36% (Literature and Arts, n = 112) to 58% (Engineering,
n = 229) and 62% (Law, n = 42).

`2026-10-29_scale9k`: items 1000:10000 × 8, same server and sampling, `--no-docent` (not on Docent;
backfill a subset with `scripts/docent_upload.py` when needed). The runner hung on 111 requests that
never returned (CODE_SMELL 2026-10-29), so it was killed and finished by hand as `abandoned` with
71,889 of 72,000 episodes (index, labels, meta, HF upload). Both runs pooled:

| pool (runs)                                  | items | accuracy | consistent | unanimous | consistent & wrong |
| -------------------------------------------- | ----: | -------: | ---------: | --------: | -----------------: |
| SuperGPQA non-calculation (`n8` + `scale9k`) | 9,906 |    50.6% |      65.0% |     58.4% |      24.7% (2,444) |

By difficulty: easy 50% / middle 52% / hard 40% accuracy, consistent and wrong 21% / 27% / 33% (n =
4,617 / 4,769 / 520). By discipline, accuracy runs from 32% (History, n = 419) and 40% (Literature
and Arts, n = 1,115) to 58% (Economics) and 62% (Philosophy).

So SuperGPQA is ~17pp harder than our MMLU-Pro subset and yields 1.5× the share of consistent-wrong
items (25% vs 16.5%): 10,000 items give 2,444 usable ones, and the remaining ~5,300 non-calculation
questions would give ~1,300 more. Caveats: the two pools were served at different precisions (bf16
here, fp8 for hint_mcq), and SuperGPQA's gold labels are noisier than MMLU-Pro's (some "consistently
wrong" items are the dataset being wrong; spot-check before using one as a lens target). Qwen is
also less consistent here (64% vs 77% of items at ≥ 80%), so more samples per item buy more.
