# RunPod network volumes: what was on them when we deleted them

**Status (2026-11-15):** REFERENCE. All three network volumes were deleted on 2026-11-15, two weeks
after the sprint ended. This is the inventory taken just before, and what to do differently if the
project restarts. `runpod/qwen_server.sh up` and the read scripts still name the deleted volume ids,
so they fail until a new volume is made ([runpod-qwen-server.md](runpod-qwen-server.md) says how).

## Why they went

No pod had run since 2026-10-31. The volumes were the account's only spend: 850 GB at $0.083/h,
about $60 a month, with a $39.95 balance (roughly 20 days). A volume saves about ten minutes of H200
per cold boot, roughly $1, so one 300 GB volume ($21 a month) only pays for itself above about
twenty boots a month.

## What was on them

Sizes are from `aws s3 ls --recursive --summarize` over RunPod's S3 API. "n/m" means the listing
timed out (large prefixes do), so the size is not measured.

| path                    | `qwen-fastboot` (US-CA-2, 300 GB, `wwwce2mejb`) | `qwen-fastboot-nc1` (US-NC-1, 300 GB, `cdde9ia5iu`) | kind                |
| ----------------------- | ----------------------------------------------- | --------------------------------------------------- | ------------------- |
| `.cache/huggingface/`   | n/m                                             | n/m                                                 | cache               |
| `home/`                 | n/m                                             | n/m                                                 | cache               |
| `olens_ckpt/`           | n/m                                             | 60.2 GB, 201 files                                  | cache               |
| `lora_vllm/`            | 4.7 GB                                          | 4.7 GB                                              | cache (regenerable) |
| `debs-22.04`, `-24.04/` | 5 MB                                            | present                                             | cache               |
| `reads/`                | 6.3 GB, 148 files                               | 40.1 GB, 468 files                                  | outputs             |
| `stdprobes/`            | 0.9 GB (`stdprobe-0`, `-1`)                     | 0.5 GB (`stdprobe-5`)                               | outputs             |
| `olens_rl/`             | none                                            | 2.1 GB, 54 files                                    | outputs             |
| `results/`              | 60 KB                                           | none                                                | outputs             |
| `watch_*/`              | 4 dirs                                          | 18 dirs                                             | pod markers         |
| `boot.log`, `vllm.log`  | last written 2026-10-31                         | last written 2026-10-31                             | logs                |

- **`.cache/huggingface/hub/`** held `Qwen/Qwen3.6-27B`, `agu18dec/olens_and_ar`,
  `agu18dec/qwen3.6-27b-all-layer-nla-rl2000`, `neuronpedia/jacobian-lens` and the gated dataset
  `agu18dec/local-workspace` (whiteners) on both, plus
  `camilablank/qwen3.6-27b-multilayer-nla-rl600` on NC-1.
- **`home/`** was the pod's `$HOME`: Triton, TileLang and NVIDIA compile caches, the reason a cached
  boot takes 3 minutes.
- **`olens_ckpt/`** held `agu18dec_olens_and_ar/`, `local_workspace/` and `nla64/` (the all-layer
  NLA's AV and AR). The same repos were also in the hub cache, so the checkpoints were probably
  stored twice; we did not compare contents.
- **`lora_vllm/`** held the three adapters renamed for vLLM's LoRA runtime (`olens_space_rl600`,
  `ar_ptag_pooled`, `nla64_av`), made by `olens/scripts/lora_for_vllm.py`.
- **`reads/`** held one `TAG.reads.jsonl` per reader shard with its transcripts, provenance, log and
  job script, and for NLA shards `TAG.reads.acts.bf16` and `TAG.reads.pred.bf16`.
- **`olens_rl/`** held the 2026-10-23 GRPO experiments on the lens (`grpo1`, `greedy_v1`, `main`,
  `test`): adapters every 25 steps, reads per step, logs.
- **`results/`** held the 2026-10-21 vLLM throughput benchmarks at concurrency 32, 128 and 256.

The third volume, `dsv4-cache` (US-CA-2, 250 GB, `sujswd59wj`), was written on one day, 2026-10-27,
for the DeepSeek V4 Flash alignment-faking reads:

| path                  | contents                                                                              | kind    |
| --------------------- | ------------------------------------------------------------------------------------- | ------- |
| `dsv4/w/`             | `v4flash/` (the model's weights), `v4lens/` (its lens: LoRA, carrier, history)        | cache   |
| `dsv4/home/`          | `$HOME`: uv, Triton, TileLang and TensorRT-LLM caches                                 | cache   |
| `.cache/huggingface/` | `kernels-community/finegrained-fp8` only                                              | cache   |
| `dsv4/out/`           | 2.8 GB, 42 files: `af-v7`, `vllm-capture`, `dsv4-af-orig` reads, logits, logs, states | outputs |
| `code/`, `code_vllm/` | 0.6 MB and 13 MB: the reader scripts and their `requests.jsonl`                       | code    |

## What was kept

Before deleting, every file under the output paths was compared by byte size with the local mirror
of the HF dataset (`data/`), counting how many files share each size. A file counted as already held
only if it was over 1 MB and the mirror had at least as many files of that exact size as the volumes
did. By that test most read shards, all of `stdprobes/` (the mirror's
`standard_probes/2026-10-31_nested2-v1/capture/{0,1,5}`) and the reads under `dsv4/out/` were
already local.

The rest was split in two:

- **Archived:** 543 files, 3.8 GB, at `volume_archive/2026-11-15/<volume>/<original path>` in the HF
  dataset. That is every non-tensor file that failed the test, and every file under 1 MB whatever
  its size matched: read shards with their requests, transcripts, provenance, logs and job scripts,
  `results/`, the `olens_rl/` adapters (`adapters/current` of all four runs, `grpo1` steps 25 to
  100, `test` step 2), eval logs and per-step reads, and the DSV4 `code/`.
- **Not kept:** activation tensors. The volumes held 42 GB of them: 19.8 GB of NLA `*.acts.bf16`,
  19.8 GB of critic `*.pred.bf16` and 2.7 GB of DSV4 `states.pt`. The mirror has same-size copies of
  15.8 GB of the activations, and the `v3tool-nla` activations (1.5 GB) merged into one file. Gone
  are all the critic predictions, the `essay_nla_1..6` activations (2.5 GB) and the DSV4 states.
  Their rows already carry the critic's `fve` and `cos`, and a forward pass regenerates them. Every
  cache was dropped too.

Failing the test mostly meant merged, not unfetched. The six `essay_nla_*` shards hold 241,770 rows,
the same count as the local `nla_reads/2026-10-31_essay-grading-inline-all-layer/reads.jsonl`, which
differs only by an added `shard` field. The three `v3tool-nla` activation files sum to exactly the
size of the local `nla_reads/2026-10-30_cutoff-v3tool-punct/reads.acts.bf16`. So the archive is
mostly a second copy under the pod-side names; we did not check the other shards row by row.

## If we did this again

Put on a volume:

- **The subject model's weights and the pod's `$HOME` compile caches.** These are what turn a
  13-minute boot into a 3-minute one.
- **Lens, AR and NLA checkpoints, the whiteners, and the vLLM-renamed adapters.** The whiteners come
  from a gated dataset, so a cached copy also saves a token round-trip. Store each once: either the
  hub cache or a `--local-dir` copy, not both.
- **The `sshd` debs**, a few MB that save an `apt-get` on every boot.

Keep off a volume:

- **Outputs.** `reads/` grew to 40 GB on one volume. The rule in `runpod/CLAUDE.md` already says a
  read that exists only on a volume doesn't exist; the readers should also delete a shard from the
  volume once it has been fetched and checked.
- **Activation tensors** (`*.acts.bf16`, `*.pred.bf16`, `states.pt`): 42 GB across the volumes, most
  of `reads/`. They are only needed between the read and the critic's scoring pass.
- **Training runs.** `olens_rl/` kept a 234 MB adapter every 25 steps on the cache volume.
- **A volume per one-day experiment.** `dsv4-cache` was used on 2026-10-27 only and then billed
  about $17 a month for 19 days. A model that is read once can download to container disk.

Process:

- **Fetch shards under the name they have on the pod.** Most shards were fetched and then merged or
  renamed locally (`essay_nla_1..6` became one `reads.jsonl`), so this audit could not tell fetched
  from unfetched by name and fell back to matching byte sizes, which is weak for files that share a
  size (adapter checkpoints, fixed-shape tensors, copies of one request file).
- **Start at 150 GB and grow only when a cache needs it.** Both Qwen volumes were doubled to 300 GB
  on 2026-10-29 to make room for reads, which should not have been there. We believe RunPod volumes
  can be grown but not shrunk; that is from memory, not re-checked.
- **A second datacenter's volume is a stock hedge, not a cache.** `qwen-fastboot-nc1` existed
  because US-CA-2 often had no H200s. That was worth $21 a month during the sprint and nothing after
  it.
- **Delete the volumes the day the last pod is deleted.** These idled from 2026-10-31 to 2026-11-15,
  about $30.
