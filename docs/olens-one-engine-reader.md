# One-engine O-lens reader: capture, lens and AR on one vLLM

**Status (2026-10-26):** ACTIVE. Works end to end and matches the HF+vLLM reader's FVEs; 2x its
throughput on short transcripts in the first hour of tuning. Not yet the default reader.

The batch reader (`scripts/olens_read_requests.py`) used to run two copies of Qwen3.6-27B on one
H200: vLLM decoding the merged lens, and HF transformers for activation capture and the AR. The HF
side was 71% of the time at ~18% MFU, the two copies left vLLM 0.45 to 0.55 of the GPU, and that
memory split caused the OOM relaunch loops and the skipped giant transcripts of the thinking-on
reads ([runpod-qwen-server.md](runpod-qwen-server.md)). `--one-engine` moves all three jobs onto one
vLLM process:

- **Engine:** vLLM 0.27.1 with
  [ceselder/vllm-metamodels](https://github.com/ceselder/vllm-metamodels) (1.1.0.post7), whose hooks
  return the residual stream at requested prompt positions
  (`extra_args={"output_residual_stream": [L], "capture_positions": [...]}`). `enable_lora=True`,
  `max_loras=2`, `enable_prompt_embeds=True`, CUDA graphs and torch.compile on
  (`VLLM_LENS_CUDA_GRAPHS=1`, `VLLM_LENS_COMPILE=1`; the fork turns compile off by default).
- **Capture:** `max_tokens=1` requests on the base model, every transcript of a batch in one
  generate. Hook index L is the output of decoder block L, the same convention as the HF hooks.
- **Lens:** the space-lens LoRA through vLLM's LoRA runtime, keys renamed by
  `olens/scripts/lora_for_vllm.py` (the AR's adapter has a third key layout,
  `base_model.model.layers.N._orig_mod.`, which the script now handles). Activations go in as
  `prompt_embeds`, as `olens.vllm_lens` does. `olens.vllm_bullet_stop.BulletStop`, a vLLM v1 logits
  processor, forces `<|im_end|>` once a fifth bullet starts, so reads stop at 4 bullets instead of
  running to the 160-token cap.
- **AR:** the AR LoRA on the same engine, `[Layer NN] <span>` prompts, residual after the last block
  at the last token, then the final RMSNorm (HF's `last_hidden_state` is post-norm, the hook is
  pre-norm), the AR's layer norm and head. Whitened NNLS as before.
- **Loop:** `run_batched` in the reader takes up to `--batch-reads` (470) new reads from as many
  transcripts as needed, captures them in one generate, decodes them in one generate (one scheduler
  wave under `max_num_seqs` 512), scores every bullet in one AR generate, then writes per
  transcript. The HF reader pipelined one transcript at a time with a decode thread; one engine
  cannot overlap its own calls, so batching replaces the overlap.

Setup and run: `runpod/one_engine_setup.sh` (boot with `IMAGE=vllm/vllm-openai:v0.27.1`), then
`runpod/olens_reads.sh run NAME REQ TAG --layers 32,40,48 --one-engine --lens-lora ... --ar-lora ...`.
Code: `olens/src/olens/one_engine.py`, `olens/src/olens/vllm_bullet_stop.py`,
`scripts/olens_read_requests.py`, `scripts/olens_reads_compare.py`.

## Benchmark (2026-10-26, 18:32 to 19:24 PDT, two 1x H200 pods, 1.76 GPU-hours)

Both readers read the same 1,800 transcripts (`runs/olens_reads/2026-10-26_hint-cf-v1`: the hint_cf
favorites / opinion_mcq / wrong_mcq envs, one user turn and a one-token reply, 192k tokens, 75,552
reads at layers 32/40/48 of which 22,932 were decoded and the rest copied from shared prefixes).
Main-thread time is the reader's own stage clock, so it excludes engine load.

| Reader                                | reads/s | main-thread | stage split                       |
| ------------------------------------- | ------- | ----------- | --------------------------------- |
| HF + vLLM (merged lens), vLLM 0.32    | 9.5     | 40.2 min    | capture 4%, vLLM wait 60%, AR 36% |
| one engine, sequential per transcript | 8.6     |             | tiny decode batches (15-45 reads) |
| one engine, batched, no compile       | 15.5    |             |                                   |
| one engine, batched + compile         | 18.8    | 19.8 min    | capture 2%, decode 68%, AR 30%    |

The one-engine stages alone: decode 27.6 reads/s, AR 62 reads/s. Engine start is ~3.5 min (weights
off the volume 2 min, CUDA graphs, warmup) plus 2.5 min of torch.compile the first time per volume.

Parity on all 75,552 shared reads: joint FVE correlation 0.979 (means 0.183 vs 0.178), top-bullet
FVE correlation 0.974, 3.98 bullets per read on both. Top bullets are identical in 17.9% of reads
and share their first three words in 40.3%; the rest is bf16 drift between two decode stacks on
long, near-paraphrase bullets, plus the FVE sort reordering near-equal bullets (the 10-22
LoRA-vs-merged test, which compared decode order, had 87% first-line agreement). If a downstream use
needs text-identical reads across the two readers, that is not available; FVEs are interchangeable.

## Where the time goes now, and what to try next

- **Decode at 27.6 reads/s** is ~3.5k generated tokens/s at batch ~480, about 2x slower per step
  than the plain chat server. Suspects: vLLM's untuned LoRA kernels ("Using default LoRA kernel
  configs"), 12 LoRA target modules in all 64 blocks, and the metamodel hooks compiled as a custom
  op. Try: `merge_lora` from vllm-metamodels (merge the lens into the live weights, unmerge, merge
  the AR) instead of the LoRA runtime, or tune the punica configs.
- **AR at 62 reads/s** (~250 spans/s) is bounded by vllm-metamodels' per-request capture overhead
  (its README: ~7 to 12 ms per request), not by prefill FLOPs.
- **Batching is per transcript group**, so the engine idles during the CPU-side NNLS and JSON writes
  (~5% here). A second thread preparing the next batch's prompts would hide that.
- **Memory:** one weight copy, 63 GB of KV, `max_num_seqs` 512 at `gpu_memory_utilization` 0.88.
  Each concurrent decode still needs a linear-attention state block, so 1024 does not fit; giants
  (200k tokens) now go through vLLM chunked prefill with `--max-model-len` raised, untested.
- **The AR is optional** in this design: `--batch-reads` worth of decodes without the AR stage would
  run at the decode rate (~28 reads/s), and if FVE turns out not to help the agent, dropping it is a
  flag.
- The old reader stays the default until a longer-transcript run (thinking-on evalcue episodes)
  confirms the capture path on multi-thousand-token prompts.
