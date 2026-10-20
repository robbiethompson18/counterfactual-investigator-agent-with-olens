#!/usr/bin/env bash
# Fetch the oracle-lens checkpoints into checkpoints/ (gitignored).
#   scripts/download.sh                 # agu18dec pair + ceselder steps 10 and 390
#   scripts/download.sh 100 200 300     # extra ceselder steps
#   EVAL=1 scripts/download.sh          # also the AR whiteners + AO eval sets from the gated
#                                       # agu18dec/local-workspace dataset (needs HF_TOKEN)
set -euo pipefail
cd "$(dirname "$0")/.."
HF=.venv/bin/hf
$HF download agu18dec/olens_and_ar --local-dir checkpoints/agu18dec_olens_and_ar >/dev/null
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(10 390)
inc=(--include README.md --include rl4gpu_fve.jsonl)
for s in "${steps[@]}"; do inc+=(--include "$(printf 'step_%05d/*' "$s")"); done
$HF download ceselder/olens-rl-grpo-trajectory-qwen36-27b --local-dir checkpoints/ceselder_olens_rl "${inc[@]}" >/dev/null
if [ "${EVAL:-0}" = 1 ]; then
  inc=(--include "data/rl/s3d/rl_gate_0.parquet" --include "data/rl/s3d/*.json" --include "data/rl/s3d/*.yaml"
       --include "data/ao/pool/eval_pool_*" --include "data/meta/*.json"
       --include "data/whiteners/pooled/whitening_iolens_pooled_provenance.json")
  for L in 20 24 28 32 36 40 44 48 52 56 60 63; do
    inc+=(--include "data/whiteners/pooled/whitening_iolens_pooled_L$L.safetensors"); done
  $HF download agu18dec/local-workspace --repo-type dataset --local-dir checkpoints/local_workspace "${inc[@]}" >/dev/null
fi
echo downloaded
