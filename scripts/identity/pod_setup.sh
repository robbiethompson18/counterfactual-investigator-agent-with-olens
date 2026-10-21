#!/usr/bin/env bash
# On a 2-GPU RunPod pod (after `runpod/pod.sh sync NAME`): vLLM serving Qwen3.6-27B on GPU0
# (127.0.0.1:8000, model name "qwen") and the O-lens read server on GPU1 (127.0.0.1:8802,
# transcript-viewer/pod/server.py). Both bind localhost; reach them with scripts/identity/tunnel.sh.
#   runpod/pod.sh ssh NAME 'nohup bash /workspace/repo/scripts/identity/pod_setup.sh > /workspace/setup.log 2>&1 &'
set -x
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
cd /workspace/repo
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

# vLLM venv, installed in the background while the weights download
uv venv -q /workspace/vllm --python 3.12
( uv pip install -q --python /workspace/vllm/bin/python vllm 'huggingface_hub[cli]'; echo VLLM_PIP_DONE ) &

# olens venv (same pins as transcript-viewer/sky/pod.yaml)
cd olens
uv venv -q --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -q --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1+cu128" -r requirements.txt kernels flash-linear-attention
.venv/bin/hf download Qwen/Qwen3.6-27B >/dev/null
EVAL=1 bash scripts/download.sh || bash scripts/download.sh
wait

PATH=/workspace/vllm/bin:$PATH CUDA_VISIBLE_DEVICES=0 nohup /workspace/vllm/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-27B \
  --served-model-name qwen --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --port 8000 --host 127.0.0.1 --generation-config auto --reasoning-parser qwen3 > /workspace/vllm.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TRITON_CACHE_AUTOTUNING=1 nohup .venv/bin/python ../transcript-viewer/pod/server.py \
  --idle-min 0 --max-hours 0 --batch 192 > /workspace/lens.log 2>&1 &
echo SETUP_DONE
