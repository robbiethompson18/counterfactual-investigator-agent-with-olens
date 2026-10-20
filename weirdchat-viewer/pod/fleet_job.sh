#!/usr/bin/env bash
# One fleet node: the same setup as sky/pod.yaml, then one read server per GPU (ports 8802+g),
# started one after another so host RAM only holds one model load at a time. Run as a detached
# sky job; each server exits after 15 idle minutes and the node then autodowns.
set -euo pipefail
unset CUDA_VISIBLE_DEVICES   # sky exec without --gpus hides the GPUs from the job
export HF_HOME=/data/hf HF_XET_HIGH_PERFORMANCE=1 MODEL_REVISION=${MODEL_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /data/hf
cd ~/olens
test -x .venv/bin/python || ~/.local/bin/uv venv -q --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv ~/.local/bin/uv pip install -q --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1+cu128" \
  -r requirements.txt kernels flash-linear-attention
.venv/bin/hf download Qwen/Qwen3.6-27B --revision "$MODEL_REVISION" >/dev/null
EVAL=1 bash scripts/download.sh
n=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=src TRITON_CACHE_AUTOTUNING=1 .venv/bin/python ~/sky_workdir/pod/server.py \
    --port $((8802 + g)) --idle-min 15 --max-hours 3.75 --no-warmup --batch "$( [ $(nvidia-smi -i $g --query-gpu=memory.total --format=csv,noheader,nounits) -gt 90000 ] && echo 192 || echo 24 )" &
  until curl -s localhost:$((8802 + g))/status | grep -q '"ready": true'; do
    kill -0 $! 2>/dev/null || { echo "server on GPU $g died"; break; }
    sleep 10
  done
  echo "GPU $g ready"
done
wait
