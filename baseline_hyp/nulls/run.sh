#!/usr/bin/env bash
# usage: baseline_hyp/nulls/run.sh VIEW N TAG [extra nulls_run.py args]
#   -> baseline_hyp/nulls/rows/VIEW__TAG.jsonl (Qwen3.6-27B on OpenRouter, provider pinned)
set -u
V=$1; N=$2; TAG=$3
cd "$(dirname "$0")/../.."
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-$(cat /secrets/openrouter)} QWEN_VLLM_API_KEY=unused EVALCUE_MODEL=qwen/qwen3.6-27b \
  uv run python baseline_hyp/nulls_run.py --view "$V" --n "$N" --out "baseline_hyp/nulls/rows/${V}__${TAG}.jsonl" "${@:4}"
