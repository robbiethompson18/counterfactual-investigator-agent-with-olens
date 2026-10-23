#!/usr/bin/env bash
# On a RunPod pod (after `runpod/pod.sh sync NAME`): the O-lens read server for scripts/identity/qwen.py's read()
# (127.0.0.1:8802, transcript-viewer/pod/server.py). It binds localhost; reach it with scripts/identity/tunnel.sh.
# Qwen samples come from the shared server ($QWEN_VLLM_URL). The 2026-10-21 identity runs also served vLLM on GPU0 of
# this pod, with the lens on GPU1; that vLLM half was removed on 2026-10-22 (it's in git history).
#   runpod/pod.sh ssh NAME 'nohup bash /workspace/repo/scripts/identity/pod_setup.sh > /workspace/setup.log 2>&1 &'
set -x
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
cd /workspace/repo
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }

# olens venv (same pins as transcript-viewer/sky/pod.yaml)
cd olens
uv venv -q --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -q --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1+cu128" -r requirements.txt kernels flash-linear-attention
.venv/bin/hf download Qwen/Qwen3.6-27B >/dev/null
EVAL=1 bash scripts/download.sh || bash scripts/download.sh

PYTHONPATH=src TRITON_CACHE_AUTOTUNING=1 nohup .venv/bin/python ../transcript-viewer/pod/server.py \
  --idle-min 0 --max-hours 0 --batch 192 > /workspace/lens.log 2>&1 &
echo SETUP_DONE
