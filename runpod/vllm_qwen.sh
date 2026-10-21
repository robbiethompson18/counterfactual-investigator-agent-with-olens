#!/usr/bin/env bash
# Serve Qwen3.6-27B (bf16) with vLLM on a RunPod pod, OpenAI-compatible on port 8000.
#   runpod/vllm_qwen.sh up NAME        # create a pod with 22/tcp + 8000/http, install vLLM, download, serve
#   runpod/vllm_qwen.sh url NAME       # print the proxy URL for QWEN_VLLM_URL
#   runpod/vllm_qwen.sh log NAME       # tail the server log
# Needs RUNPOD_API_KEY and QWEN_VLLM_API_KEY (.envrc.local). Delete the pod with runpod/pod.sh down NAME.
# One H100 80GB ($3.99/hr, 2026-10) takes ~15 min from create to serving: ~5 min vLLM install, ~5 min
# 52 GB download, ~4 min load. Throughput ~3k output tok/s at 256 concurrent sequences.
set -euo pipefail

IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
cmd=$1 name=${2:?pod name}
pod_id() { runpodctl pod list | python3 -c "import json,sys; print(next(p['id'] for p in json.load(sys.stdin) if p['name'] == sys.argv[1]))" "$1"; }
ssh_args() {
  runpodctl pod get "$(pod_id "$1")" | python3 -c "
import json, sys
s = json.load(sys.stdin).get('ssh') or {}
if s.get('ip') and s.get('port'): print(f\"-i {s['ssh_key']['path']} -p {s['port']} root@{s['ip']}\")"
}

case $cmd in
  up)
    runpodctl pod create --name "$name" --image "$IMAGE" --gpu-id "${3:-NVIDIA H100 80GB HBM3}" --container-disk-in-gb 200 \
      --ports 22/tcp,8000/http --env "{\"HF_TOKEN\": \"$(hf auth token 2>/dev/null)\"}" >/dev/null
    until args=$(ssh_args "$name") && [ -n "$args" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $args true 2>/dev/null; do sleep 10; done
    ssh -o StrictHostKeyChecking=no $args "cat > /workspace/boot.sh <<'EOF'
set -x
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
uv venv /workspace/venv --python 3.12 -q
uv pip install --python /workspace/venv/bin/python -q 'huggingface_hub[cli]' hf_transfer
( uv pip install --python /workspace/venv/bin/python -q vllm; echo PIPDONE ) &
/workspace/venv/bin/hf download Qwen/Qwen3.6-27B --exclude '*.gguf' 2>&1 | tail -2
wait
nohup /workspace/venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-27B --served-model-name qwen \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.92 --max-num-seqs 256 --port 8000 --host 0.0.0.0 \
  --api-key $QWEN_VLLM_API_KEY --generation-config auto > /workspace/vllm.log 2>&1 &
EOF
nohup bash /workspace/boot.sh > /workspace/boot.log 2>&1 &"
    echo "booting; poll with: $0 log $name" ;;
  url) echo "https://$(pod_id "$name")-8000.proxy.runpod.net/v1" ;;
  log) ssh -o StrictHostKeyChecking=no $(ssh_args "$name") 'tail -5 /workspace/boot.log; tail -5 /workspace/vllm.log 2>/dev/null' ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
