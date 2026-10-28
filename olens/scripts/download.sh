#!/usr/bin/env bash
# Fetch the oracle-lens checkpoints into checkpoints/ (gitignored). On a read pod, runpod/olens_reads.sh setup does this.
#   scripts/download.sh                 # the agu18dec lens + AR pair
#   EVAL=1 scripts/download.sh          # also the AR whiteners + AO eval sets from the gated
#                                       # agu18dec/local-workspace dataset (needs HF_TOKEN)
set -euo pipefail
cd "$(dirname "$0")/.."
HF=.venv/bin/hf
$HF download agu18dec/olens_and_ar --local-dir checkpoints/agu18dec_olens_and_ar >/dev/null
if [ "${EVAL:-0}" = 1 ]; then
  inc=(--include "data/rl/s3d/rl_gate_0.parquet" --include "data/rl/s3d/*.json" --include "data/rl/s3d/*.yaml"
       --include "data/ao/pool/eval_pool_*" --include "data/meta/*.json"
       --include "data/whiteners/pooled/whitening_iolens_pooled_provenance.json")
  for L in 20 24 28 32 36 40 44 48 52 56 60 63; do
    inc+=(--include "data/whiteners/pooled/whitening_iolens_pooled_L$L.safetensors"); done
  $HF download agu18dec/local-workspace --repo-type dataset --local-dir checkpoints/local_workspace "${inc[@]}" >/dev/null
fi
echo downloaded
