# DeepSeek V4 Flash O-lens reads

Reads our transcripts with Camila Blank's V4 Flash O-lens inverter
([camila-blank/deepseek-v4-flash-olens-inverter](https://huggingface.co/camila-blank/deepseek-v4-flash-olens-inverter),
SFT stage, PEFT LoRA, injection at the input of decoder layer 1). Carriers exist only for layers 15, 18, ..., 39, 42 (V4
Flash has 43 layers); 30 is about two-thirds deep.

1. `prep.py`: pick transcripts (one per arm × item for the alignment-faking runs) -> `requests.jsonl`.
2. `read.py` (HF transformers 5.19 FP8, `kernels==0.17`): capture block outputs at the read positions and score the
   first-reply logits; with `--states` it reuses a capture. Its lens stage is slow (2.4-3.6 reads/s on 2xH200 at batch
   128-512: HF MoE loops + pipeline across GPUs).
3. `vllm_read.py`: the lens stage on vLLM 0.32, 21 reads/s on 2xH200 (greedy + 2 samples each, CUDA graphs). Needs
   vllm-metamodels (`vllm-lens @ git+https://github.com/ceselder/vllm-metamodels`, `--no-deps`) for per-request prompt-row
   routing, the `hc_inject/` plugin (pip install it; it writes 16,384-wide vectors into V4's folded 4-stream state at
   layer 1, which the fork refuses to do), the adapter renamed by `convert_adapter.py`, and a CUDA 13.0 toolkit for the
   DeepGEMM / FlashInfer JIT (`apt-get install cuda-nvcc-13-0 cuda-libraries-dev-13-0 libcurand-dev-13-0`,
   `CUDA_HOME=/usr/local/cuda-13.0`, `VLLM_USE_FLASHINFER_SAMPLER=0`). vLLM 0.27.1 has no V4 LoRA. Against HF on 4,405
   layer-30 reads: first 10 characters equal 65%, 20 49% (bf16 greedy drift, as on the Qwen port).
4. `vllm_sample_check.py`: base-model sampling on vLLM (T 1.0) vs OpenRouter's rates.

Box: `sky.yaml` (RunPod 2xH200 or 2xB200, weights in /data). Use `sky exec <cluster> --gpus H200-SXM:2 "<cmd>"` for ad
hoc commands (without `--gpus` the job sees 0 GPUs).
