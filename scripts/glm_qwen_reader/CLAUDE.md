# GLM-5.3 lens + Qwen reader of GLM-5.3 activations

Reads one set of GLM-5.3 prompt positions with two lenses and puts them side by side in expview:

- Celeste's GLM-native one-bullet lens
  ([ceselder/glm-5.3-olens-lens](https://huggingface.co/ceselder/glm-5.3-olens-lens)
  `rl1b_rl1b_2_step_0100`, block 51, `../glm_olens_dense/worker.py`), greedy + 4 T=1 bullets;
- the cross-model reader
  ([ceselder/glm-5.3-olens-qwen-reader](https://huggingface.co/ceselder/glm-5.3-olens-qwen-reader)
  `release/`): the same block-51 vectors mapped by its affine map (6144 → 5120) into Qwen3.6-27B's
  layer-42 space, added norm-matched at block 1 at the marker of the public Qwen L42 lens prompt
  ([olens_usage.py](https://huggingface.co/ceselder/olens-qwen3.6-27b-l42/blob/main/olens_usage.py)),
  decoded by the reader LoRA under that lens's bullet grammar: greedy + 4 T=1 reads of 4 bullets.

The two can't share an engine (different base models), so one box does them in turn: the GLM
worker captures block 51 and runs the lens (job flag `save_h51` keeps the vectors), stops, then four
HF-transformers Qwen shards (one per GPU) read the saved vectors.

| file             | where   | what                                                                                                          |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `make_job.py`    | GPU box | requests → worker job; worker output → `glm_reads.jsonl` + `h51.pt`                                           |
| `qwen_reader.py` | GPU box | `h51.pt` → `qwen_reads_<shard>.jsonl`                                                                         |
| `box.sh`         | GPU box | `setup` (weights, vLLM 0.31 + glm53 fork, reader) and `run` (both lenses, raw outputs to HF `RUN/raw`)        |
| `sky.yaml`       | bramble | one sky job running `box.sh` on RunPod 4xB200                                                                 |
| `to_reads.py`    | bramble | raw outputs → `RUN/reads.jsonl` + `reads.transcripts.jsonl` (labels `L51 GLM lens`, `L51→Qwen reader`), dups copied |

Requests come from `scripts/af_v8x_olens_requests.py glm` (GLM raw empty-think render, prompt ids,
token strings, positions with shared prefixes read once). Flow:

```bash
uv run python scripts/runs.py new alignment_faking olens-v8n-glm53 -d "..." --model z-ai/glm-5.3 --served-by "..."
uv run --with transformers --with jinja2 python scripts/af_v8x_olens_requests.py glm $RUN/requests.jsonl --prompts v8n-,v10
REQ=$RUN/requests.jsonl RUN=$(basename $RUN) sky launch ... scripts/glm_qwen_reader/sky.yaml   # see the yaml header
hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset --include "runs/alignment_faking/$(basename $RUN)/raw/*" --local-dir data
uv run python scripts/glm_qwen_reader/to_reads.py $RUN/requests.jsonl $RUN/raw $RUN && uv run python scripts/runs.py finish $RUN
```

Untested end to end (written 2026-10-29 before any box was up). The Qwen side follows the model
card's HF usage; Qwen3.6's linear-attention layers run on transformers' torch fallback unless
`flash-linear-attention` is installed.
