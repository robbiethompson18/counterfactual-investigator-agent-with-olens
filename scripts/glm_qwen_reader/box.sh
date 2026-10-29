#!/usr/bin/env bash
# GLM-5.3 O-lens + the Qwen reader of GLM's block 51 over one requests file, on a 4xB200 box (sky.yaml runs it; on a hand-rented
# vast box run the same two steps). Needs HF_TOKEN (write on the sprint dataset) for the upload.
# On a vast box (24 GB root disk, /dev/shm noexec): stop the image's own vLLM first (it holds ~169 GB/GPU), then D=/dev/shm/w
# for the weights and caches and BASE_PY=/venv/main/bin/python to reuse the image's torch; the venv stays in W on the root disk.
#   bash box.sh setup                    # weights, lens, reader, vLLM 0.31 + the glm53 fork (~464 GB GLM download)
#   RUN=<date>_<slug> [FREE_GLM=1] bash box.sh run    # GLM worker (capture + lens, h51 kept) -> 4 Qwen reader shards -> HF runs/alignment_faking/$RUN/raw
set -euo pipefail
W=${W:-/root/w}
D=${D:-$W} # weights, lens, reader, HF cache
export HF_HOME=$D/hf UV_CACHE_DIR=$D/uv-cache UV_LINK_MODE=copy
HERE=$(cd "$(dirname "$0")" && pwd)
LENS_REV=689c5dbae3d2f83dbf969e234a023958f1612185
READER_REV=f2f9d79e38d05a2adf0229e40f35b127a46edef2
L42_REV=1d8e6377f87bcb7e899c96b715ca594b6a5aa765
PY=$W/v31/bin/python
mkdir -p "$W" "$D"

if [ "$1" = setup ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
  if [ -n "${BASE_PY:-}" ]; then # a vast vLLM image: its torch 2.13 + deps, vllm 0.31 and transformers 5.17 on top (root disk is 24 GB)
    [ -x "$PY" ] || uv venv -q --system-site-packages --python "$BASE_PY" "$W/v31"
    uv pip install -q -p "$PY" --no-deps vllm==0.31.0 "transformers>=5.16.1,<5.18" peft hf_transfer
  else
    [ -x "$PY" ] || uv venv -q --python 3.12 "$W/v31"
    uv pip install -q -p "$PY" vllm==0.31.0 "transformers>=5.16.1,<5.18" peft pandas pyarrow hf_transfer "huggingface_hub[cli]"
  fi
  HF="$PY -m huggingface_hub.cli.hf download"
  $HF ceselder/glm-5.3-olens-lens --revision $LENS_REV --local-dir "$D/lens" \
    --include "rl1b_rl1b_2_step_0100/*" "olens_glm53_usage.py" "vllm_metamodels_glm53_b26f02d.tar.gz" "glm53_stack/*"
  uv pip install -q -p "$PY" --no-deps "$D/lens/vllm_metamodels_glm53_b26f02d.tar.gz"
  uv cache clean -q
  $HF ceselder/glm-5.3-olens-qwen-reader --revision $READER_REV --local-dir "$D/reader" --include "release/*"
  $HF ceselder/olens-qwen3.6-27b-l42 --revision $L42_REV --local-dir "$D/l42" --include olens_usage.py
  $HF nvidia/GLM-5.3-NVFP4 --local-dir "$D/GLM-5.3-NVFP4"
  echo SETUP_DONE
  exit
fi

: "${RUN:?set RUN=<date>_<slug>}" "${REQ:=$HERE/requests.jsonl}"
J=$W/jobs O=$W/out
mkdir -p "$J" "$O"
cp "$D/lens/olens_glm53_usage.py" "$HERE/../glm_olens_dense/worker.py" "$HERE/../glm_olens_dense/grammar_lp.py" "$W/"
[ -f "$O/h51.pt" ] || {
  $PY "$HERE/make_job.py" job "$REQ" "$J/000_af.job.pt"
  (cd "$W" && $PY worker.py --model-dir "$D/GLM-5.3-NVFP4" --rl-lens "$D/lens/rl1b_rl1b_2_step_0100" --jobs "$J" --prefix-cache \
    >"$O/glm_worker.log" 2>&1) &
  until [ -f "$J/000_af.out.pt" ]; do
    [ -f "$J/000_af.job.pt.failed" ] && { echo "GLM job failed, see $O/glm_worker.log"; exit 1; }
    kill -0 $! 2>/dev/null || { echo "GLM worker died, see $O/glm_worker.log"; exit 1; }
    sleep 20
  done
  touch "$J/STOP"
  wait
  $PY "$HERE/make_job.py" collect "$REQ" "$J/000_af.out.pt" "$O"
}
# Qwen after GLM: a 440 GiB /dev/shm holds GLM (432 GiB) or Qwen (50 GiB), not both; FREE_GLM=1 deletes GLM's weights first
[ "${FREE_GLM:-0}" = 1 ] && rm -rf "$D/GLM-5.3-NVFP4"
HF_HUB_ENABLE_HF_TRANSFER=1 $PY -m huggingface_hub.cli.hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 >/dev/null
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $PY "$HERE/qwen_reader.py" "$O/h51.pt" "$D/reader/release" "$D/l42/olens_usage.py" "$O/qwen_reads_$i.jsonl" \
    --shard $i/4 >"$O/qwen_reader_$i.log" 2>&1 &
done
wait
cp "$REQ" "$O/requests.jsonl"
$PY -m huggingface_hub.cli.hf upload Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens "$O" "runs/alignment_faking/$RUN/raw" \
  --repo-type dataset --commit-message "raw GLM lens + Qwen reader reads: $RUN"
echo READS_DONE
