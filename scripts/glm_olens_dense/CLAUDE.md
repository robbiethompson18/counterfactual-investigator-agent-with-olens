# GLM-5.3 O-lens reads in Celeste's format

The pipeline runs on bramble and a 4xB200 box, Kylie's compute; Robbie's agents use RunPod
(`runpod/CLAUDE.md`).

Reads Celeste's GLM-5.3 oracle lens
([ceselder/glm-5.3-olens-lens](https://huggingface.co/ceselder/glm-5.3-olens-lens), RL checkpoint
`rl1b_rl1b_2_step_0100`) over our transcripts and writes tables with the schema of her
`ceselder/glm-5.3-olens-counterfactual-investigator` `*/readouts_dense.parquet`, so
`expview/importers/celeste.py` imports them.

| file                          | where   | what                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `render.py`                   | bramble | GLM-5.3 chat-template render of a `rows.jsonl` episode (exactly her `rendered_text`: 149/150 prefix, 167/168 late views on `ib_realistic_user_part2`) and her approximate `late` view                                                                                                                                                    |
| `prep.py`                     | bramble | positions to read: every punctuation / newline token (her `is_punct`) + every template/special token (id >= 154820), index >= 2, < 32,768; exact-prefix duplicates read once (`dense_prefix_dups.parquet`). Tool schemas rebuilt from the realism harness (`realvariants` / `email_env.build`), checked against each row's system prompt |
| `make_jobs.py`                | bramble | worker job files; `--per-episode K` = K positions per episode evenly spaced over its candidates (full-episode order, late view = tail)                                                                                                                                                                                                   |
| `worker.py` + `grammar_lp.py` | GPU box | boots her engine (`olens_glm53_usage.GlmOlensEngine` from the lens repo + the `vllm-metamodels` glm53 fork shipped there) once, then for each job: capture block 51 of the unmerged base, merge the RL lens, greedy + 4 samples at T=1                                                                                                   |
| `select_2026_10_25_late.py`   | bramble | Kylie's 2026-10-25 23:07 selection: half the episodes per (arm, item) at seed 0, punctuation + template tokens, no whitespace-only tokens; Donation Bet's 168-row sample plus the first 6 reply tokens                                                                                                                                   |
| `collect.py`                  | bramble | joins worker outputs with prep positions -> `readouts_dense.parquet`, `episodes.parquet`, `dense_prefix_dups.parquet`, an empty `readouts.parquet`                                                                                                                                                                                       |

Box setup (4xB200; ~10 min engine boot from /dev/shm, 3-4 min once compiled):

```bash
hf download nvidia/GLM-5.3-NVFP4 --local-dir /dev/shm/hf/GLM-5.3-NVFP4
hf download ceselder/glm-5.3-olens-lens --local-dir /dev/shm/hf/lens   # olens_glm53_usage.py, vllm_metamodels_glm53_b26f02d.tar.gz, checkpoints
uv venv /root/v31 && uv pip install -p /root/v31 vllm==0.31.0 peft pandas pyarrow && uv pip install -p /root/v31 --no-deps -e vllm-metamodels-glm53
cp /dev/shm/hf/lens/olens_glm53_usage.py worker.py grammar_lp.py /root/
python worker.py --model-dir /dev/shm/hf/GLM-5.3-NVFP4 --rl-lens /dev/shm/hf/lens/rl1b_rl1b_2_step_0100 --jobs /dev/shm/jobs --prefix-cache
```

Throughput on 4xB200 (4,096 positions, greedy + 4 samples): 25 pos/s as shipped, 43.7 pos/s with
`--prefix-cache` (her `lens_cache_salt=lens-sha256`: the 5 bullets of one position share one steered
prefill). Of that, ~1/3 is prefill + per-request overhead (131 pos/s with max_tokens=1) and ~2/3 is
decode of the W4A16 Marlin MoE at batch 512; the grammar processor is not the bottleneck (batched
version: same rate). Capture runs at ~11k tokens/s.

## Runs

`runs/olens_reads/2026-10-25_glm53_ib_user_dense/` in the sprint HF dataset
([commit a3706b2](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/commit/a3706b2517afc79c9e8c8d73adcaa414a1b24668)),
`meta.json` + one subdir per source run:

| subdir                                      | source                                                        | what was read                                                                                                                                                                                                                                                          |
| ------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2026-10-25_glm53_ib_realistic_user_ladder` | `runs/coding/2026-10-25_glm53_ib_realistic_user_ladder`       | 56,931 reads of 4.17M candidates: `kind=spread64` (64 per episode, evenly spaced over its candidates, 703 episodes) + `kind=head` (every unique position through the first user message); 144,822 shared-prefix positions map to these via `dense_prefix_dups.parquet` |
| `2026-10-25_glm53_email_ambiguous_part3`    | `runs/email_ambiguous/2026-10-25_glm53_email_ambiguous_part3` | every position (`kind=dense`): 19,295 reads + 39,940 shared-prefix mappings, 256 episodes                                                                                                                                                                              |

Two more uploads use the same layout:

- `runs/olens_reads/2026-10-25_glm53_email_dense/`
  ([commit 540b233](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/commit/540b233)):
  subdirs `2026-10-24_glm53_email_ambiguous` (1,664 episodes) and `_part2` (128), from
  `runs/email_ambiguous/`. `kind=head` only: every punctuation/template position in the system
  prompt and first user turn, 3,194 unique positions over all arms.
- `runs/olens_reads/2026-10-25_glm53_funding_real_dense/`
  ([commit 5e5c870](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/commit/5e5c870)):
  subdirs `2026-10-25_glm53_{pilot,grid,realask,full_edit}`, from `runs/funding_email_real/`.
  `head` + `spread64`: 18 cells, 288 episodes, 25,077 positions. The `full_real_*` arms in `_grid`
  and `full_realask_*` in `_realask` are superseded by `_full_edit` and only have head reads (via
  prefix dups).

All four sets are imported into the expview viewer (bramble:8830) under
`imports/olens_reads/<upload>/<subdir>`.

Differences from Celeste's tables: `sft_*` null (RL lens only), `fve_*` NaN (no reconstructor
scoring), chat-template/special tokens included as positions, `prefix_cache=True` with
`lens_cache_salt=lens-sha256`, extra `part` column; the ladder rows have `dense=False`.
