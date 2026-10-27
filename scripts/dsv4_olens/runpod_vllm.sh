#!/usr/bin/env bash
# Pod-side job for the one-engine DSV4 Flash O-lens reader (vllm_capture_read.py) on one of Robbie's RunPod pods attached to
# the dsv4-cache volume (sujswd59wj, US-CA-2, mounted at /workspace), which runpod_af.sh's simple run filled: weights at $W,
# the vLLM venv at ~/vl (HOME on the volume), the converted adapter. Nothing is downloaded here: a missing cache is FAILED.
# Stages, each skipped when its marker exists: CUDA 13.0 toolkit (container disk, per pod), reinstall hc_inject (adds the
# hc_capture plugin), a --limit 20 smoke read compared with the simple run's reads ($REF), the full file compared the same
# way. Writes $OUT/READS_DONE or $OUT/FAILED for runpod/watch_pod.sh.
#   nohup bash /workspace/code_vllm/runpod_vllm.sh > /workspace/dsv4/out/vllm-capture/job.log 2>&1 < /dev/null &
set -uo pipefail
export HOME=${DSV4_HOME:-/workspace/dsv4/home} W=${W:-/workspace/dsv4/w} OUT=${OUT:-/workspace/dsv4/out/vllm-capture} CODE=${CODE:-/workspace/code_vllm}
export PATH=$HOME/.local/bin:$PATH
REQ=${REQ:-$CODE/requests.jsonl}
REF=${REF:-/workspace/dsv4/out/af-v7}
LAYERS=${LAYERS:-30}
READ_ARMS=${READ_ARMS:-preview,public}
mkdir -p "$OUT"
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND"; touch $OUT/FAILED' ERR
set -e
cd "$CODE"
stamp() { echo "=== $(date -u '+%F %TZ') $*"; }

stamp "cache check"
test -f "$W/v4flash/config.json" && test -f "$W/v4lens/lora_vllm/adapter_model.safetensors" && test -x ~/vl/.venv/bin/python

if [ ! -d /usr/local/cuda-13.0/bin ]; then  # container disk: redone on every pod
  stamp "cuda: CUDA 13.0 toolkit for vLLM's DeepGEMM / FlashInfer JIT"
  . /etc/os-release
  distro=ubuntu${VERSION_ID//./}
  cd /tmp && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && apt-get update -qq >/dev/null 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-crt-13-0 libcublas-dev-13-0 \
    cuda-nvrtc-dev-13-0 libcurand-dev-13-0 cuda-libraries-dev-13-0 ninja-build rsync >/dev/null 2>&1
  cd "$CODE"
  stamp "cuda done"
fi

stamp "hc_inject + hc_capture plugins into ~/vl"
VIRTUAL_ENV=~/vl/.venv uv pip install -q --no-deps --reinstall-package hc-inject "$CODE/hc_inject"
~/vl/.venv/bin/python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='vllm.general_plugins')])"

export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LENS_CUDA_GRAPHS=1
run() {  # run NAME ARGS...
  local name=$1; shift
  [ -f "$OUT/$name/DONE" ] && return 0
  stamp "$name: vllm_capture_read.py layers $LAYERS arms $READ_ARMS $*"
  ~/vl/.venv/bin/python vllm_capture_read.py --model "$W/v4flash" --lens "$W/v4lens" --requests "$REQ" --out "$OUT/$name" \
    --layers "$LAYERS" --read-arms "$READ_ARMS" --dense --actual-reply --public-first "$@" 2>&1 | tee -a "$OUT/$name/log.txt"
  [ -f "$REF/reads_L$LAYERS.jsonl" ] && ~/vl/.venv/bin/python compare_reads.py "$OUT/$name" "$REF" --reads "reads_L$LAYERS.jsonl" | tee "$OUT/$name/parity.txt"
  touch "$OUT/$name/DONE"
}
mkdir -p "$OUT/smoke" "$OUT/full"
run smoke --limit 20
run full
touch "$OUT/READS_DONE"
stamp "READS_DONE"
