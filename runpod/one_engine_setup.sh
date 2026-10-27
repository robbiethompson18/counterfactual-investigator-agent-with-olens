#!/usr/bin/env bash
# Set up a pod for the one-engine O-lens reader (scripts/olens_read_requests.py --one-engine; olens/src/olens/one_engine.py):
# vLLM 0.27.1 + ceselder/vllm-metamodels, the lens and AR adapters with vLLM-renamed keys, whiteners. Boot the pod with
#   IMAGE=vllm/vllm-openai:v0.27.1 VOLUMES=cdde9ia5iu NAME=one-engine runpod/qwen_server.sh up 1
# then
#   runpod/one_engine_setup.sh one-engine
#   runpod/olens_reads.sh run one-engine REQ.jsonl TAG --layers 32,40,48 --one-engine \
#       --lens-lora /workspace/lora_vllm/olens_space_rl600 --ar-lora /workspace/lora_vllm/ar_ptag_pooled
# (`run`'s positional MERGED_DIR is ignored with --one-engine; the base checkpoint comes from --base.)
set -euo pipefail
NAME=${1:?pod name}
ssh_pod() { runpod/pod.sh ssh "$NAME" "$@"; }

# the boot script's vLLM server (and a possibly orphaned engine) hold the GPU; [b]in keeps pgrep off PID 1's command line
ssh_pod 'for p in $(pgrep -f "[b]in/vllm serve"); do kill $p; done; sleep 10
  for p in $(pgrep -f "[b]in/vllm serve"); do kill -9 $p; done; pkill -9 -f "[V]LLM::EngineCore" || true; mkdir -p /root/repo /workspace/reads'
COPYFILE_DISABLE=1 tar czf - --exclude .git --exclude .venv --exclude data --exclude vendor --exclude checkpoints \
  --exclude node_modules --exclude __pycache__ --exclude .envrc.local --exclude transcript-viewer olens scripts runpod |
  ssh_pod 'tar xzf - -C /root/repo 2>/dev/null'
ssh_pod 'bash -s' <<'EOF'
set -e
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ   # HF_TOKEN etc. (ssh sessions lack the pod's env)
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home
python3 -c "import vllm; print('vllm', vllm.__version__)"
# vllm-metamodels as a zip (the image may lack git); --no-deps so pip never touches the image's torch/vllm pins
pip install -q --no-deps "https://github.com/ceselder/vllm-metamodels/archive/refs/heads/main.zip" zstandard 2>&1 | grep -v -e "running pip as" -e notice || true
pip install -q scipy pyyaml 2>&1 | grep -v -e "running pip as" -e notice || true
python3 -c "import vllm_lens; print('vllm_lens', getattr(vllm_lens, '__version__', '?'))"
C=/workspace/olens_ckpt   # checkpoints + whiteners on the volume; olens/ expects them at olens/checkpoints
hf download agu18dec/olens_and_ar --local-dir $C/agu18dec_olens_and_ar >/dev/null
inc=(); for L in 20 24 28 32 36 40 44 48 52 56 60; do inc+=(--include "data/whiteners/pooled/whitening_iolens_pooled_L$L.safetensors"); done
hf download agu18dec/local-workspace --repo-type dataset --local-dir $C/local_workspace "${inc[@]}" >/dev/null
ln -sfn $C /root/repo/olens/checkpoints
hf download Qwen/Qwen3.6-27B >/dev/null
cd /root/repo
test -f /workspace/lora_vllm/olens_space_rl600/adapter_model.safetensors || python3 olens/scripts/lora_for_vllm.py $C/agu18dec_olens_and_ar/olens_s3d_space_rl600 /workspace/lora_vllm/olens_space_rl600
test -f /workspace/lora_vllm/ar_ptag_pooled/adapter_model.safetensors || python3 olens/scripts/lora_for_vllm.py $C/agu18dec_olens_and_ar/ar_ptag_pooled/lora /workspace/lora_vllm/ar_ptag_pooled
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo SETUP_DONE
EOF
