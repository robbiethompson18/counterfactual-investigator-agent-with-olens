#!/usr/bin/env bash
# Start command of the shared Qwen pod (vllm/vllm-openai image, network volume at /workspace), set by `runpod/qwen_server.sh up`;
# `restart` re-runs it over ssh. Needs API_KEY in the pod's env. Starts sshd, then serves Qwen3.6-27B on 0.0.0.0:8000 with one
# replica per GPU (data parallel) behind one endpoint, as both "qwen" and "Qwen/Qwen3.6-27B". Re-running restarts the server.
set -e
# `restart` runs this over ssh, whose env lacks the image's ENV and the pod's (API_KEY, HF_TOKEN). Take PID 1's: a different env
# also changes vLLM's compile-cache key, and the restart recompiles (76 s).
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ
: "${API_KEY:?}"
echo "$(date -u +%T) boot start"
# The vllm image has no sshd; RunPod puts our key in $PUBLIC_KEY. Its .debs are cached on the volume, per Ubuntu release
# (apt-get update alone takes ~30 s; v0.25 images are 22.04, v0.32 is 24.04).
debs=/workspace/debs-$(. /etc/os-release && echo "$VERSION_ID")
if ! command -v sshd >/dev/null; then
  ls $debs/*.deb >/dev/null 2>&1 || { mkdir -p $debs/partial && apt-get update -qq &&
    apt-get install -y -qq --download-only -o Dir::Cache::archives=$debs openssh-server; }
  dpkg -i $debs/*.deb >/dev/null || { apt-get update -qq && apt-get install -y -qq -f openssh-server >/dev/null; }
fi
mkdir -p ~/.ssh /run/sshd && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
pgrep -x sshd >/dev/null || /usr/sbin/sshd
echo "$(date -u +%T) sshd up"
# Weights and every compile cache live on the network volume, so only the first boot on a fresh volume downloads and compiles.
# The caches (vllm's torch.compile + inductor, flashinfer, triton, ...) all live under ~, so point HOME at the volume.
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home
mkdir -p "$HOME"
hf download Qwen/Qwen3.6-27B >/dev/null
echo "$(date -u +%T) weights present"
# [b]in: PID 1 is `bash -c <this script>`, whose cmdline must not match
for pid in $(pgrep -f "[b]in/vllm serve"); do kill "$pid"; done
sleep 3
nohup vllm serve Qwen/Qwen3.6-27B --served-model-name qwen Qwen/Qwen3.6-27B --host 0.0.0.0 --port 8000 --api-key "$API_KEY" \
  --data-parallel-size "$(nvidia-smi -L | wc -l)" --max-model-len 262144 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --generation-config auto --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching > /workspace/vllm.log 2>&1 < /dev/null &
echo "$(date -u +%T) vllm starting; see /workspace/vllm.log"
