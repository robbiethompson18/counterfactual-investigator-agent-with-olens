# Handoff: DeepSeek V4 Flash O-lens reads on the overeagerness DSv4 episodes

**Status (2026-10-27):** NOT RUN. First attempt stopped at capture (OOM, then the 1-GPU question); no reads exist yet.

## Goal

O-lens reads (Camila Blank's V4 Flash inverter,
[camila-blank/deepseek-v4-flash-olens-inverter](https://huggingface.co/camila-blank/deepseek-v4-flash-olens-inverter),
SFT stage, rev `ec495e5c`) on DeepSeek V4 Flash's overeagerness episodes, the same arms as the Qwen3.6-27B reads: the
`main: one-edit fixes and nulls` comparisons in `environments/overeagerness/env.py`. Output in the batch reads format
(docs/olens-reads-format.md) under `runs/overeagerness/<date>_dsv4flash-olens/` on HF, so expview stitches the reads into
the env episodes.

## Code

Branch `claude/overeagerness-qwen-dsv4`, `scripts/dsv4_olens/`. The reader stack (`read.py`, `vllm_read.py`,
`convert_adapter.py`, `setup.sh`, `hc_inject/`, `sky.yaml`, README) is copied from `claude/dbet-nothink-olens` (worktree
`~/workspace/cia-dbet-olens`), which built on `claude/af-counterfactual`; neither is on main. New here:

- `prep_overeager.py` (bramble): seeded `--per-arm` episodes per arm from local env runs (`data/envs/<env>/<run>`),
  messages + tools as the model saw them, ids `ENV/RUN:<episode id>`.
- `capture_env.py` (box): renders each episode with DeepSeek's own encoder (`encoding/encoding_dsv4.py` in the model dir,
  `encode_messages(..., thinking_mode="thinking")`, tools on the system message, so every turn keeps its reasoning; DSML
  tool calls, tool results as `<｜User｜><tool_result>…` turns). Checked by hand on overeagerness episodes. Captures block
  outputs ([4, 4096] per position, V4's 4-stream state) with HF transformers 5.19 (`read.py`'s `Capture` hook), adapter
  off. Writes `states.pt` (one row per distinct prefix), `positions.jsonl` (transcript → [pos, prefix key]) and
  `reads.transcripts.jsonl` (tokens + sections system / user / tool / reasoning / assistant as separate blocks).
- `to_reads.py` (bramble): states + `reads_L30.jsonl` → `RUN/reads.jsonl` (rows `id, token_index, token, layer,
  bullets=[greedy, s1, s2]`, copied onto every transcript sharing the prefix) + `RUN/reads.transcripts.jsonl`. Then
  `uv run python scripts/runs.py finish RUN` (makes the expview run; `expview/importers/reads.py` on this branch now tags
  `ENV/RUN:<episode>` ids with `extra.source` / `extra.episode`, which the env viewer needs to attach reads).
- `sky_overeager.yaml`: one sky job runs setup, capture, then the lens stage (`vllm_read.py`, vLLM 0.32 + vllm-metamodels +
  `hc_inject`, layer 30, greedy + 2 samples, `--tp 2`). Outputs stay in `/data/out/overeager/` on the box: fetch with
  `scp -F ~/.sky/generated/ssh/dsv4-oe` before the 60-min autodown.

Read positions: the Qwen reader's rule (`scripts/olens_read_requests.py`): every punctuation token (its `PUNCT` regex) plus
every `</think>` and end-of-sentence token, except inside tool results (CSVs, git logs, file listings were 58% of
positions), where only each result's last token is read. A position whose token prefix (blake2b of ids) was already
captured is not captured again.

## Selection

`data/dsv4_olens/overeager/requests.jsonl` (gitignored; rebuild with the commands below): 960 transcripts = the 25 main
arms × 32 (2 per item, seed 0) from `envs/overeagerness/2026-10-27_dsv4flash-main256` + item 0 × the 5 email `@lowc` arms
× 32 from `2026-10-27_dsv4flash-4johns256`. ~800k positions before the tool-result rule and dedupe; the first 181
transcripts gave 12,183 distinct of 16,902 positions, so ~60k distinct reads in all (~50 min at 21 reads/s).

```bash
python scripts/dsv4_olens/prep_overeager.py --env overeagerness --runs 2026-10-27_dsv4flash-main256 \
  --arms "$(python -c 'from environments.overeagerness import env; print(",".join(dict.fromkeys(a for c in env.COMPARISONS.values() if c["group"].startswith("main") for a in c["arms"])))')" \
  --per-arm 32 --out data/dsv4_olens/overeager/main.jsonl
python scripts/dsv4_olens/prep_overeager.py --env overeagerness --runs 2026-10-27_dsv4flash-4johns256 \
  --arms email/fixed@lowc,email/no_done@lowc,email/fixed_plain@lowc,email/fixed_home@lowc,email/base_done_neutral@lowc \
  --per-arm 32 --out data/dsv4_olens/overeager/4johns.jsonl
cat data/dsv4_olens/overeager/{main,4johns}.jsonl > data/dsv4_olens/overeager/requests.jsonl
PYTHONPATH=~/.sky/policies HF_TOKEN=$(cat /secrets/huggingface) uvx --from "skypilot[runpod]==0.13.0" \
  sky launch -c dsv4-oe -y -i 60 --down --secret HF_TOKEN scripts/dsv4_olens/sky_overeager.yaml
```

## What happened (2026-10-27 23:20–23:40 UTC, 2x B200, RunPod US)

Setup (weights 149 GB to /data, vLLM 0.32 venv) took ~12 min. Capture ran at 0.6–5 s per transcript, then at transcript
182 HF hit `torch.OutOfMemoryError` (tried to allocate 11.75 GiB of 140 GiB) on a long one. Episodes run to 71k
characters (~20k tokens; median 9.5k chars) and HF's attention isn't chunked. Fix on this branch (pushed, not yet run):
skip a transcript on OOM with a log line; its transcript/positions lines are written only after a successful capture.
Nothing was saved (states are written at the end), and the box was torn down.

**1-GPU symptom.** Kylie saw only one GPU in use. Best guess: that is HF's `device_map="auto"` in the capture stage, which
splits V4's layers across the two GPUs as a pipeline, so a single forward pass keeps one GPU busy at a time (memory is on
both, compute alternates). The vLLM lens stage (`--tp 2`) never started. To use both GPUs for capture: run two capture
processes, one per GPU, if one copy fits (V4 Flash FP8 is ~149 GB on disk, so it doesn't fit one 140 GB B200; it would
need 2 GPUs per copy and 4 GPUs for two shards), or batch several transcripts per forward pass to raise per-step work.
Capture was ~20% of the expected wall time; the lens stage dominates and does use both GPUs.

## Next steps

1. Rebuild the requests (above), relaunch, and watch `capture.log` for `OOM, skipped` lines (expect a handful).
2. Fetch `/data/out/overeager/{states.pt,positions.jsonl,reads.transcripts.jsonl,reads_L30.jsonl,*.log}`; `sky down`.
3. `uv run python scripts/runs.py new overeagerness dsv4flash-olens -d "..." --model deepseek/deepseek-v4-flash
   --model camila-blank/deepseek-v4-flash-olens-inverter@ec495e5c`, `python scripts/dsv4_olens/to_reads.py OUT RUN`,
   `uv run python scripts/runs.py finish RUN`.
