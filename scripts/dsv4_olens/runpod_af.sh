#!/usr/bin/env bash
# Pod-side job for DSV4 Flash O-lens reads on one of Robbie's RunPod pods: the port of Kylie's sky_af_v4.yaml setup + run
# steps (usage, the pod recipe and the local driver commands are in CLAUDE.md here). Idempotent: rerun after a crash and it
# skips the stages whose outputs exist. Writes $OUT/{SETUP_DONE,CAPTURE_DONE,READS_DONE} as stages finish and $OUT/FAILED on
# any error, which runpod/watch_pod.sh treats as "delete the pod".
#
# Everything expensive lives on the network volume mounted at /workspace (volume dsv4-cache, sujswd59wj, US-CA-2): HOME is
# moved there, so the weights ($W), both uv venvs (~/v4, ~/vl), uv's cache, and the HF / vLLM / DeepGEMM / FlashInfer JIT
# caches under ~/.cache all survive the pod. Only the CUDA 13.0 apt packages are reinstalled per pod (~2 min).
#   nohup bash /workspace/code/runpod_af.sh > /workspace/dsv4/out/<run>/job.log 2>&1 < /dev/null &
set -uo pipefail
export HOME=${DSV4_HOME:-/workspace/dsv4/home} W=${W:-/workspace/dsv4/w} OUT=${OUT:-/workspace/dsv4/out/af-v7} CODE=${CODE:-/workspace/code}
export PATH=$HOME/.local/bin:$PATH
REQ=${REQ:-$CODE/requests.jsonl}
CAPTURE_LAYERS=${CAPTURE_LAYERS:-24,30,36}
LENS_LAYERS=${LENS_LAYERS:-30}
READ_ARMS=${READ_ARMS:-preview,public}
mkdir -p "$OUT" "$HOME"
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND"; touch $OUT/FAILED' ERR
set -e
cd "$CODE"
stamp() { echo "=== $(date -u '+%F %TZ') $*"; }

if [ ! -f "$OUT/SETUP_DONE" ]; then
  stamp "setup: weights + HF venv (setup.sh, W=$W, HOME=$HOME)"
  bash setup.sh
  stamp "setup: vLLM 0.32 venv + vllm-metamodels + hc_inject"
  mkdir -p ~/vl && cd ~/vl && { [ -d .venv ] || uv venv -q --python 3.12 .venv; }
  VIRTUAL_ENV=~/vl/.venv uv pip install -q vllm==0.32.0
  VIRTUAL_ENV=~/vl/.venv uv pip install -q --no-deps "vllm-lens @ git+https://github.com/ceselder/vllm-metamodels" zstandard "$CODE/hc_inject"
  cd "$CODE" && [ -f "$W/v4lens/lora_vllm/adapter_model.safetensors" ] || ~/v4/.venv/bin/python convert_adapter.py "$W/v4lens/lora" "$W/v4lens/lora_vllm"
  touch "$OUT/SETUP_DONE"
  stamp "setup done"
fi

if [ ! -d /usr/local/cuda-13.0/bin ]; then  # container disk: redone on every pod
  stamp "cuda: CUDA 13.0 toolkit for vLLM's DeepGEMM / FlashInfer JIT"
  . /etc/os-release
  distro=ubuntu${VERSION_ID//./}  # ubuntu2404 on runpod/pytorch:*-ubuntu2404
  # runpod/pytorch already carries the NVIDIA repo (/etc/apt/sources.list.d/cuda.list); adding cuda-keyring's copy on top makes
  # apt refuse every source ("Conflicting values set for option Signed-By", 2026-10-27). Add the keyring only on an image without it.
  if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list /etc/apt/sources.list.d/; then
    cd /tmp && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
  fi
  apt-get update > "$OUT/apt.log" 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-crt-13-0 libcublas-dev-13-0 \
    cuda-nvrtc-dev-13-0 libcurand-dev-13-0 cuda-libraries-dev-13-0 ninja-build rsync >> "$OUT/apt.log" 2>&1
  cd "$CODE"
  stamp "cuda done"
fi

if [ ! -f "$OUT/CAPTURE_DONE" ]; then
  stamp "capture: HF transformers, layers $CAPTURE_LAYERS, first-reply logits, arms $READ_ARMS"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/v4/.venv/bin/python read.py \
    --model "$W/v4flash" --lens "$W/v4lens" --requests "$REQ" --out "$OUT" --layers "$CAPTURE_LAYERS" \
    --read-arms "$READ_ARMS" --dense --actual-reply --skip-lens 2>&1 | tee -a "$OUT/capture.log"
  touch "$OUT/CAPTURE_DONE"
  stamp "capture done"
fi

stamp "lens: vLLM 0.32, layers $LENS_LAYERS"
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LENS_CUDA_GRAPHS=1
~/vl/.venv/bin/python vllm_read.py --model "$W/v4flash" --lens "$W/v4lens" --states "$OUT/states.pt" \
  --layers "$LENS_LAYERS" --public-first --out "$OUT/reads_L$LENS_LAYERS.jsonl" 2>&1 | tee -a "$OUT/lens.log"
touch "$OUT/READS_DONE"
stamp "READS_DONE"
