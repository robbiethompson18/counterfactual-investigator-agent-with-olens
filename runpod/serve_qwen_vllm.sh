#!/usr/bin/env bash
# Install vLLM 0.25.1 and serve Qwen3.6-27B (model name "qwen") on the pod's 127.0.0.1:8000, full 262k context.
# Run ON the pod:  runpod/pod.sh ssh NAME 'bash -s' < runpod/serve_qwen_vllm.sh
# Re-running restarts the server. Tunnel from your machine with
#   ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 <args from runpod/pod.sh up>   (8000/8001 are taken locally)
# Gotchas (docs/eval-cue-reward-hacking.md): vLLM JIT-compiles FlashInfer kernels at startup, so it needs ninja and
# nvcc on PATH (first start spends ~9 min compiling); don't `pkill -f "vllm serve"` from a command line that itself
# contains that string, it kills your own shell.
set -e
cd /workspace
if [ ! -x vllm-env/bin/vllm ]; then
  pip install -q --break-system-packages uv huggingface_hub hf_transfer hf_xet
  uv venv -q --python 3.12 vllm-env
  VIRTUAL_ENV=/workspace/vllm-env uv pip install -q vllm==0.25.1 ninja \
    --extra-index-url https://wheels.vllm.ai/0.25.1/cu129 --torch-backend=cu129 --index-strategy unsafe-best-match
fi
hf download Qwen/Qwen3.6-27B >/dev/null
for pid in $(pgrep -f "bin/vllm serve"); do kill "$pid"; done
sleep 3
export PATH=/workspace/vllm-env/bin:/usr/local/cuda/bin:$PATH
nohup vllm serve Qwen/Qwen3.6-27B --served-model-name qwen --host 127.0.0.1 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --enable-prefix-caching > /workspace/vllm.log 2>&1 < /dev/null &
echo "vllm starting; tail /workspace/vllm.log"
