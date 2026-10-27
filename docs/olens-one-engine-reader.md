# One-engine O-lens reader: capture, lens and AR on one vLLM

**Status (2026-10-27):** ACTIVE. The only batch reader since 2026-10-27: matches the deleted HF+vLLM
reader's FVEs on short and long transcripts, 2x its throughput. Also serves the J++ lens (below).

The batch reader (`scripts/olens_read_requests.py`) used to run two copies of Qwen3.6-27B on one
H200: vLLM decoding the merged lens, and HF transformers for activation capture and the AR. The HF
side was 71% of the time at ~18% MFU, the two copies left vLLM 0.45 to 0.55 of the GPU, and that
memory split caused the OOM relaunch loops and the skipped giant transcripts of the thinking-on
reads ([runpod-qwen-server.md](runpod-qwen-server.md)). All three jobs now run on one vLLM process;
the HF path (hook-LoRA, HF capture and AR, the merged-lens checkpoint and `merge_lens.py`, and with
it the parked NLA and J++ code) was deleted in PR #229 and is at that PR's first commit.

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
  `prompt_embeds`. No stop rule runs in the engine: the space lens ends its own list at four bullets
  (97% of reads, the rest at five, none at the 160-token cap), and the reader trims to four. A
  logits processor that forced `<|im_end|>` at the fifth bullet (`vllm_bullet_stop.py`, deleted
  2026-10-27) cost 6% per decode step and saved nothing; see "Batch size and BulletStop" below.
- **AR:** the AR LoRA on the same engine, `[Layer NN] <span>` prompts, residual after the last block
  at the last token, then the final RMSNorm (HF's `last_hidden_state` is post-norm, the hook is
  pre-norm), the AR's layer norm and head. Whitened NNLS as before.
- **Loop:** `run_batched` in the reader takes up to `--batch-reads` (470) new reads from as many
  transcripts as needed, captures them in one generate, decodes them in one generate (one scheduler
  wave under `max_num_seqs` 512), scores every bullet in one AR generate, then writes per
  transcript. The HF reader pipelined one transcript at a time with a decode thread; one engine
  cannot overlap its own calls, so batching replaces the overlap.

Setup and run: boot with
`IMAGE=vllm/vllm-openai:v0.27.1 VOLUMES=<cache volume> NAME=x runpod/qwen_server.sh up 1`, then
`runpod/olens_reads.sh setup x` and
`runpod/olens_reads.sh run x REQ TAG --layers 32,40,48 [--max-model-len 131072]`. Code:
`olens/src/olens/one_engine.py`, `scripts/olens_read_requests.py`, `scripts/olens_reads_compare.py`
(parity between two reads files).

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

## Long transcripts (2026-10-27, `runs/olens_reads/2026-10-26_long-parity`)

The 21k-token `whistleblow_snitch` cot-r3 transcript (5,295 reads, the longest the old reader read):
capture 2 s, decode 231 s (23 reads/s), AR 88 s, 16.5 reads/s overall. Against the old reader's
reads of it: joint FVE correlation 0.966 (means 0.146 old vs 0.137 new), top-bullet FVE correlation
0.952, 3.99 bullets per read on both. Lower text agreement than on short prompts (top bullet
identical 9%, same first three words 36%): the old reader captured in 2048-token chunks through HF's
cache, which itself moves the residual by bf16 noise, and 21k tokens of context give the two stacks
more to drift on. No sign of a position bug (that would put the correlation near 0).

**Giants (90k to 130k tokens) hang the default GDN prefill.** The first long run put a 98,708-token
transcript through the engine at `--max-model-len 131072` and sat 12 minutes at ~60% GPU with no
output after vLLM's "CuTeDSL JIT compilation during inference: \_FullyFusedDeltaRuleSm90" warning
(`long-v2-131k-hang.log` in the run). The same transcript with `gdn_prefill_backend="triton"`
captures in 11.1 s (8.9k tok/s), a 92,520-token one in 10.2 s, positions and norms as expected
(`probe-A-triton-131k.log`). `OneEngine` now sets the triton backend; prefill is ~2% of the reader's
time, so short transcripts lose nothing. Not isolated to the backend alone: the probe also ran with
`enable_lora=False` and `max_num_seqs=64`, neither of which should touch a prefill kernel. The
reader skips any transcript at or over `--max-model-len` with a log line instead of crashing; the
longest cot-r3 transcript is 131,420 tokens, so giants need `--max-model-len 262144` (one prefill
chunk per transcript, untested at that length).

End to end on the 98k giant at layer 32 (`long-v4-giants.log`, `reads.giant-L32.jsonl`): 17,860
reads, capture 14 s, decode 1,021 s (17.5 reads/s: one transcript is one 17,860-prompt generate,
which vLLM runs in ~35 waves with no cross-transcript batching to fill the last ones), AR 320 s,
13.2 reads/s overall. A giant is 2% of a run's transcripts and ~15% of its reads, so `--batch-reads`
matters less than the decode rate for them.

## Merge mode and the decode ceiling (2026-10-27, `runs/olens_reads/2026-10-27_merge-v1`)

`--lora-mode merge` builds a plain engine (`enable_lora=False`) and uses vllm-metamodels'
`merge_lora` to fold the lens into the served weights before the decode and the AR over it before
the scoring, then `unmerge_lora` before the next capture (its key mapping strips `base_model.model.`
and tries the `model.language_model.` layout, so the renamed adapter dirs work). On the hint-cf
benchmark set, 556 transcripts and 14,361 decoded reads, 1x H200, 36 min of pod:

| mode                      | reads/s | decode stage | AR stage   | per batch of ~480 reads                             |
| ------------------------- | ------- | ------------ | ---------- | --------------------------------------------------- |
| LoRA runtime (2026-10-26) | 18.8    | 27.6 reads/s | 62 reads/s |                                                     |
| merge_lora (2026-10-27)   | 19.8    | 31.7 reads/s | 60 reads/s | unmerge 0.9 s + merge lens 1.27 s + merge AR 1.28 s |

+5% overall. Net of the three publishes (~3.5 s of a 25 s batch) the decode kernels are ~23% faster,
as the vllm-metamodels README says, but that was never the main cost. Parity with the runtime reads
on 20,859 shared rows: joint FVE correlation 0.986, top bullet identical 28%, 3.98 bullets per read
on both. Either mode is fine; `runtime` stays the default because it needs no publishes and the
difference is 5%.

**The decode is at stock vLLM's rate for this model.** `decode_probe.py` (in the run) timed one
merged engine on 470 real layer-32 activations, and `plain_probe.py` the same prompts on a stock
vLLM 0.27.1 with the vllm-metamodels plugin off (`VLLM_PLUGINS=""`), both at fixed 100 tokens:

| variant                                                                  | B=470 ms/step       | B=128 | B=32 |
| ------------------------------------------------------------------------ | ------------------- | ----- | ---- |
| reader path (`prompt_embeds` + BulletStop, natural stop, mean 91 tokens) | 85 (thinning batch) |       |      |
| `prompt_embeds`, fixed length                                            | 112                 |       |      |
| token prompts, fixed length                                              | 108                 | 34.5  | 20.0 |
| token prompts + BulletStop                                               | 115                 |       |      |
| stock vLLM, plugin off, token prompts                                    | 108                 | 34.5  | 20.3 |

So `prompt_embeds` costs 4%, the BulletStop logits processor 6%, the plugin and the merged lens
nothing. The step time grows almost linearly with batch above ~128 (0.2 ms per sequence per step
with a ~7 ms floor), so batching stops paying there: 1.6k / 3.7k / 4.4k tokens/s at 32 / 128 / 470,
the same curve as the chat server's `vllm bench serve` (1,621 / 3,940 / 4,692 at 32 / 128 / 256,
[runpod-qwen-server.md](runpod-qwen-server.md)). A dense 27B step at batch 470 has a ~26 ms compute
floor and an ~11 ms weight-read floor, so a per-sequence kernel bounds it, most likely the GDN
recurrent decode of Qwen3.6's hybrid stack; that is vLLM's kernel, not ours. Decode MFU is ~24%
(4.4k tokens/s × 54 GFLOP against 989 TFLOP/s), AR ~37%, capture ~44%, blended ~29%.

What is left to gain in our code is small: dropping the AR stage (`--batch-reads` worth of decodes
alone run at the decode rate, ~32 reads/s). The CPU side is not worth hiding: the 470 NNLS fits and
1,500 JSON records of a batch take 0.12 s on a laptop against a ~25 s batch (0.5%). Going past ~4.5k
tokens/s on one H200 means a faster GDN decode kernel in vLLM, fewer tokens per read, or more GPUs.
Memory: one weight copy, 63 GB of KV, `max_num_seqs` 512 at `gpu_memory_utilization` 0.88; each
concurrent decode also needs a linear-attention state block, so 1024 does not fit (and would not
help, given the curve above).

## Batch size and BulletStop (2026-10-27, logs in `runs/olens_reads/2026-10-27_merge-v1`)

**Two scheduler waves per generate are slower, not faster.** `--batch-reads 470` vs `940`
(`--batch-transcripts 512`, LoRA runtime) on the same 40 cot-r3 transcripts (~630 reads each at
layers 32/40/48), 10 transcripts per segment: 19.3 reads/s at 470 (one transcript per batch) vs 18.2
at 940 (two per batch), the decode stage 8% slower and the AR unchanged (`batch-ab470.log`,
`batch-ab940.log`). The probe's hope that continuous batching would refill the thinning tail of the
first wave did not survive the scheduler. The default stays 470.

**The lens does not need a stop rule.** `bullet_probe.py` on 470 real layer-32 activations (merged
engine): with no stop rule and the 160-token cap, the space lens ends its list at four bullets in
455 to 459 of 470 reads, at five in the rest, and never reaches the cap, mean 107 tokens; the
BulletStop logits processor got the same reads in 12.6 to 13.2 s vs 12.2 to 12.4 s without it. At
fixed length it costs 7 ms of a 107 ms step (115 vs 107) even after moving its string work to
precomputed per-token flags, so the cost is the per-step Python over 470 rows (and vLLM running a
non-argmax-invariant processor), not the string scanning. Deleted; the reader trims reads to four
bullets in text, as before. Expected gain ~5% on the decode stage, not yet measured end to end.

## J++ on the same engine (2026-10-27, `runs/jpp_reads/2026-10-27_hint-mcq-header`)

The J++ lens ([Ayonrinde & Lindsey 2026](https://huggingface.co/neuronpedia/jacobian-lens),
`qwen3.6-27b/jpp`) reads next-token logits off the residual after block l as
`lm_head(norm(J[l] h_l))`, l in 0..62. `olens/src/olens/jpp.py` holds the 63 maps (fp16, 3.3 GB)
plus the base's final norm and `lm_head`, and takes residuals from `OneEngine.capture_many`, so it
has no model and no LoRA of its own. `scripts/jpp_read_requests.py` reuses the O-lens reader's row
rendering, positions and queue
(`READER=scripts/jpp_read_requests.py runpod/olens_reads.sh run ...`), so J++ rows join O-lens reads
on (id, token_index); `--positions all` reads every token for the viewer's token view,
`--layers all` serves all 63. Output schema is the old HF reader's
(`runs/jpp_reads/2026-10-22_cfbench-spike-v1`). Throughput is the capture's: ~690 positions/s at
every token of short transcripts, with the lens math at ~0.5 ms per position for 16 layers.

Parity with the HF reader's 2026-10-22 reads on 6 transcripts, every token, 16 layers: top-1
agreement 95.6% (99.4% at L0 falling to 89.6% at L60, the HF-vs-vLLM residual drift with depth),
top-5 overlap 0.96, top-token probabilities equal to three decimals (`parity/` in the run).

**The final-norm gain bug.** The first J++ attempt agreed at top-1 only 59%, with probabilities ~50x
too small: it applied the final RMSNorm as `x_norm * w`, but Qwen3.6's RMSNorm is `x_norm * (1 + w)`
(HF `Qwen3_5RMSNorm`, vLLM `GemmaRMSNorm`; the stored weights sit near 1 for the final norm and near
0 for the block norms). `OneEngine.ar` had applied the same wrong gain since 2026-10-26. Its effect
on the O-lens reads was small: on bullets with identical text, the buggy AR's FVEs correlate 0.9998
with the HF reader's but sit 3% low (mean 0.1148 vs 0.1181, 9,951 bullets); the fixed AR has no bias
(0.1076 vs 0.1075, corr 0.9995, mean |diff| 0.0018, 5,200 bullets; `ar-fix-check/` in
`runs/olens_reads/2026-10-27_hint-mcq-header`). The 0.976 to 0.992 whole-read FVE correlations
between runs come from the decode emitting different bullets run to run, not from the AR. `norm_w`
is now the gain `1 + w` for both paths; the hint-mcq-header O-lens reads were re-read with it, the
2026-10-26 one-engine reads (hint-cf-v1, long-parity, merge-v1) keep the 3% low FVEs.
