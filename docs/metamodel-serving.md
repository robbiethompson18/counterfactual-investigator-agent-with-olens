# Serving metamodels on vLLM for Qwen3.6-27B

How we decode metamodel readouts (O-lens, NLA, J++) fast, what other people already built, and the
TODO to package it. Written 2026-10-22.

## TODO: a minimal "serve metamodels for Qwen3.6-27B" repo

Robbie's ask (2026-10-22, not today): a small public repo that serves the Qwen3.6-27B metamodels
(agu18dec O-lens + AR, ceselder L42 NLA AV + AR, J-lens / J++, maybe the base model too) behind one
API: requests in (transcripts or activations), readouts out, vLLM speed, every version pin and
gotcha below baked in. Every agent on this sprint has rebuilt pieces of this (pod/server.py, the
O-lens vLLM reader, the NLA reader). Before building it, check whether
[ceselder/vllm-metamodels](https://github.com/ceselder/vllm-metamodels) plus a thin recipe covers
it; it may be enough to contribute the Qwen3.6 recipe there.

## What already exists

- [ceselder/vllm-metamodels](https://github.com/ceselder/vllm-metamodels) (Celeste, 2026): a fork of
  [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) built for metamodel workloads. It does
  per-request steering at 3-59x vllm-lens's speed, Karvonen injection `h + coeff·‖h‖·unit(v)` on the
  full residual (`norm_match=True`), embedding replacement, CUDA graphs, cheap hidden-state capture,
  LoRA merging and early exit. It is benchmarked on Qwen3.6-27B (~3.5-3.9k tok/s steered on a B200)
  and tested on vLLM 0.16-0.28. This is what we use for the NLA.
- [asherps/EasyNLA](https://github.com/asherps/EasyNLA) (2026): NLA training. Its RL rollouts run on
  vllm-lens with vLLM 0.19.0 and transformers 4.57.1 (`scripts/install_vllm_lens.sh`,
  `docs/vllm-lens-setup.md`), with a patch for the norm-match bug that vllm-metamodels also fixes.
- [kitft/nla-inference](https://github.com/kitft/nla-inference) (2026): the NLA paper's inference
  client. It uses SGLang `input_embeds` (embedding-replacement NLAs) and covers Qwen2.5 / Gemma 3 /
  Llama 3.3 only, not Qwen3.6.
- In this repo: `olens/src/olens/vllm_lens.py` (the O-lens on vLLM via prompt embeds, which works
  because the O-lens replaces an input embedding) and `olens/src/olens/vllm_nla.py` (the NLA AV on
  vLLM via vllm-metamodels, which it needs because the ceselder NLA injects at block 1's output).
  Searched agu18dec, camilablank, ceselder, asherps and kitft on GitHub: nobody has a served O-lens
  or Qwen3.6 NLA endpoint besides these.

## Gotchas

- **The vLLM version decides whether injection happens at all.** vllm-lens 1.1.0 silently stops
  injecting on vLLM ≥ 0.22 (V2 model runner); vllm-metamodels forces the V1 runner. vLLM 0.19 needs
  transformers < 5, but our HF code (olens, Qwen3.6's `qwen3_5` classes) is on transformers 5.x, so
  we run vLLM 0.27.1 (tested by vllm-metamodels on Qwen3.6-27B).
- **Upstream vllm-lens norm-matches against half the residual** (the MLP delta), injecting ~8-50x
  too weakly on Qwen3.6. vllm-metamodels and EasyNLA's patch fix it; plain vllm-lens is wrong.
- **vLLM's LoRA runtime is inert or crashes on Qwen3.6-27B**, so serve lenses as merged checkpoints:
  `olens/scripts/merge_lens.py`; repeat `--adapter` to stack the NLA's SFT + RL LoRAs.
- **Keep `max_num_seqs <= 1024`** with CUDA graphs on Qwen3.6 (GDN decode kernel grid limit).
- **Prompt positions repeat.** About a fifth of the stop tokens in an env's transcripts sit in the
  starting context, which is identical for every run of an arm. The NLA reader caches readouts by a
  hash of the full token prefix; the O-lens reader doesn't yet.
- **Install:** `uv pip install "vllm==0.27.1" "transformers==5.16.1"`, then
  `uv pip install --no-deps "vllm-lens @ git+https://github.com/ceselder/vllm-metamodels"` plus
  `zstandard`. One combined `uv pip install` sat idle for 25 min resolving. That gives torch 2.13 +
  cu130 (fine on RunPod's driver 595), vLLM 0.27.1, vllm-metamodels 1.1.0.post7.

## NLA readout throughput (2026-10-22, cfbench env transcripts, greedy, ~160 tokens per readout)

| setup                                                        | readouts/s |
| ------------------------------------------------------------ | ---------- |
| HF `generate`, one transcript per batch, 1x H200             | 1.7        |
| HF `generate`, batch 256 pooled across transcripts, 1x H200  | 6-7.5      |
| vLLM AV (GPU 0) + HF capture and AR (GPU 1), 2x H200, serial | 16         |
| same, AR scoring overlapped with the next decode             | 20         |

vLLM decodes at ~3.2k tok/s here, close to vllm-metamodels' own Qwen3.6-27B benchmark. Against the
HF path on the same 2,763 positions: mean FVE 0.309 vs 0.310 (correlation 0.98), the first 40
characters identical in 83% of readouts, the first sentence in 68% (greedy decodes drift late from
bf16 noise, as the O-lens vLLM path does). Prefix dedupe skips another ~5-20% of decodes depending
on how much of an env's transcripts is shared prompt.
