# olens

Reads the public oracle-lens checkpoints for `Qwen/Qwen3.6-27B` without the unreleased
`global_workspace` package: the
[agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar) lens (AO) and its activation
reconstructor (AR), whitened FVE, all served by one vLLM engine with
[ceselder/vllm-metamodels](https://github.com/ceselder/vllm-metamodels)
([docs/olens-one-engine-reader.md](../docs/olens-one-engine-reader.md)). The HF transformers path
(hook-LoRA, HF generate and AR, the ceselder L42 lens, NLA, J++, the merged-lens vLLM wrapper) was
deleted 2026-10-27; it is at PR #229's first commit.

```
src/olens/lens.py             the AO's read contract: carrier prompt, marker, alpha, bullet parsing, Sampling
src/olens/one_engine.py       OneEngine: capture (residual at positions), read_items (lens LoRA, prompt_embeds), ar
                              (AR LoRA + final norm + head), one vLLM 0.27.1 engine
src/olens/fve.py              whitener (published pooled moments) + NNLS whitened FVE
src/olens/provenance.py       reads.provenance.json next to a batch of reads: lens, AR, whitener, revisions, sampling
src/olens/vectors.py          probe / steering directions in an HF model's residual stream (no caller since 2026-10-26)
scripts/lora_for_vllm.py      rename a PEFT adapter's keys so vLLM's LoRA runtime applies it (lens and AR layouts)
scripts/download.sh           checkpoints into checkpoints/ (gitignored); EVAL=1 also fetches the whiteners
```

Conventions, verified against the published held-out joint FVE (0.486 here vs 0.487 reported, on the
old lens with the HF path): the AO's marker input embedding is overwritten with `16000 * h/|h|`; the
AR head predicts the **raw** activation at the token before the span; target and predictions are
whitened as `Σ^-1/2 (x − μ)` with the pooled whitener; a bullet goes to the AR as the text after
`- `, with a leading space for the space lens `olens_s3d_space_rl600` (`lens.bullet_prefix = " "`;
CLAUDE.md says never to use the old `olens_s3d_rl600`). Capture hook index L is the output of
decoder block L (HF's `hidden_states[L+1]`); the AR reads block 63's output through the final
RMSNorm (HF's `last_hidden_state`).

The whiteners come from the gated dataset
[agu18dec/local-workspace](https://huggingface.co/datasets/agu18dec/local-workspace) (needs an
`HF_TOKEN` with access). Needs vLLM 0.27.1 + vllm-metamodels + scipy on a GPU with 80 GB+
(`runpod/olens_reads.sh setup` installs them). Run with `PYTHONPATH=olens/src`; the batch reader
`scripts/olens_read_requests.py` is the only caller.
