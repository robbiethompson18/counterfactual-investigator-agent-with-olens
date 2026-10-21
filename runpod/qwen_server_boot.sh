#!/usr/bin/env bash
# Runs ON the shared Qwen pod; started by `runpod/qwen_server.sh up|restart`, which passes API_KEY. Installs vLLM 0.25.1,
# downloads Qwen3.6-27B, and serves it on 0.0.0.0:8000 with one replica per GPU (data parallel) behind one endpoint.
# Served as both "qwen" and "Qwen/Qwen3.6-27B" so clients written for either name work. Re-running restarts the server.
set -e
: "${API_KEY:?}"
cd /workspace
pip install -q --break-system-packages uv huggingface_hub hf_transfer hf_xet
hf download Qwen/Qwen3.6-27B >/dev/null &  # weights (52 GB, ~1 min) download alongside the install
if [ ! -x vllm-env/bin/vllm ]; then
  uv venv -q --clear --python 3.12 vllm-env
  # A uv download once stalled for 26 min on a dead connection; cap each attempt and retry (the uv cache keeps progress).
  for attempt in 1 2 3; do
    VIRTUAL_ENV=/workspace/vllm-env UV_HTTP_TIMEOUT=60 timeout 900 uv pip install -q vllm==0.25.1 ninja \
      --extra-index-url https://wheels.vllm.ai/0.25.1/cu129 --torch-backend=cu129 --index-strategy unsafe-best-match && break
    echo "uv install attempt $attempt failed or timed out" >&2
  done
fi
wait
for pid in $(pgrep -f "bin/vllm serve"); do kill "$pid"; done
sleep 3
# FlashInfer and Qwen3.6's linear-attention kernels JIT-compile at startup (~9 min the first time): needs ninja + nvcc on PATH
export PATH=/workspace/vllm-env/bin:/usr/local/cuda/bin:$PATH
nohup vllm serve Qwen/Qwen3.6-27B --served-model-name qwen Qwen/Qwen3.6-27B --host 0.0.0.0 --port 8000 --api-key "$API_KEY" \
  --data-parallel-size "$(nvidia-smi -L | wc -l)" --max-model-len 262144 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --generation-config auto --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching > /workspace/vllm.log 2>&1 < /dev/null &
echo "vllm starting; see /workspace/vllm.log"
