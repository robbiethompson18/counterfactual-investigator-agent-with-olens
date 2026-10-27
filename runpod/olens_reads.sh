#!/usr/bin/env bash
# Batch O-lens reads (scripts/olens_read_requests.py: capture, lens and AR on one vLLM, olens/src/olens/one_engine.py) on a pod
# booted with the vLLM 0.27.1 image and a cache volume:
#   IMAGE=vllm/vllm-openai:v0.27.1 VOLUMES=cdde9ia5iu NAME=X runpod/qwen_server.sh up 1
# See runpod/CLAUDE.md, docs/runpod-qwen-server.md ("O-lens reads") and docs/olens-one-engine-reader.md.
#   runpod/olens_reads.sh setup NAME                    # stop the pod's vLLM server, copy the repo, vllm-metamodels, checkpoints, adapters
#   runpod/olens_reads.sh run NAME REQUESTS.jsonl TAG [ARGS...]   # start reading in the background -> /workspace/reads/TAG.reads.jsonl
#                                                       # READER=scripts/jpp_read_requests.py for J++ reads (same request format)
#                                                       # ARGS: --layers 32,40,48 [--max-model-len 65536 for transcripts over 32k tokens]
#   runpod/olens_reads.sh status NAME TAG               # last progress lines
#   runpod/olens_reads.sh fetch NAME TAG LOCAL.jsonl    # copy the reads so far, and LOCAL.provenance.json
#   runpod/olens_reads.sh stop NAME                     # kill the reader (a rerun of `run` with the same TAG resumes)
set -euo pipefail
NAME=${2:?pod name}
ssh_pod() { runpod/pod.sh ssh "$NAME" "$@"; }

case ${1:?command} in
  setup)
    # [b]in: keeps pgrep from matching PID 1, whose command line holds the boot script
    # a vLLM still loading weights can ignore SIGTERM (one kept 130 GB of the GPU), so wait and escalate
    # the boot script's vLLM server has an EngineCore child that survives its parent and keeps ~130 GB of the GPU; the
    # reader's own vLLM then fails at init_device (2026-10-26, both honest-lens pods). Kill both, then check the GPU is free.
    ssh_pod 'for p in $(pgrep -f "[b]in/vllm serve"); do kill $p; done; sleep 10
      for p in $(pgrep -f "[b]in/vllm serve"); do kill -9 $p; done; pkill -9 -f "[V]LLM::EngineCore" || true; sleep 5
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      if [ "$used" -gt 2000 ]; then echo "GPU still holds ${used} MiB after killing vLLM:"; nvidia-smi; exit 1; fi
      mkdir -p /root/repo /workspace/reads'
    COPYFILE_DISABLE=1 tar czf - --exclude .git --exclude .venv --exclude data --exclude vendor --exclude checkpoints \
      --exclude node_modules --exclude __pycache__ --exclude .envrc.local --exclude transcript-viewer olens scripts runpod src |
      ssh_pod 'tar xzf - -C /root/repo 2>/dev/null'
    ssh_pod 'bash -s' <<'EOF'
set -e
while IFS= read -r -d '' kv; do export "$kv"; done < /proc/1/environ   # HF_TOKEN etc. (ssh sessions lack the pod's env)
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home
python3 -c "import vllm; print('vllm', vllm.__version__)"
# vllm-metamodels as a zip (the image may lack git); --no-deps so pip never touches the image's torch/vllm pins. scipy (NNLS)
# isn't in the vllm image.
pip install -q --no-deps "https://github.com/ceselder/vllm-metamodels/archive/refs/heads/main.zip" zstandard 2>&1 | grep -v -e "running pip as" -e notice || true
pip install -q scipy 2>&1 | grep -v -e "running pip as" -e notice || true
python3 -c "import vllm_lens; print('vllm_lens', getattr(vllm_lens, '__version__', '?'))"
C=/workspace/olens_ckpt   # checkpoints + whiteners on the volume; olens/ expects them at olens/checkpoints
hf download agu18dec/olens_and_ar --local-dir $C/agu18dec_olens_and_ar >/dev/null
inc=(); for L in 20 24 28 32 36 40 44 48 52 56 60; do inc+=(--include "data/whiteners/pooled/whitening_iolens_pooled_L$L.safetensors"); done
hf download agu18dec/local-workspace --repo-type dataset --local-dir $C/local_workspace "${inc[@]}" >/dev/null
ln -sfn $C /root/repo/olens/checkpoints
hf download Qwen/Qwen3.6-27B >/dev/null
# the space lens (CLAUDE.md) and the AR, keys renamed for vLLM's LoRA runtime (docs/metamodel-serving.md)
cd /root/repo
test -f /workspace/lora_vllm/olens_space_rl600/adapter_model.safetensors || python3 olens/scripts/lora_for_vllm.py $C/agu18dec_olens_and_ar/olens_s3d_space_rl600 /workspace/lora_vllm/olens_space_rl600
test -f /workspace/lora_vllm/ar_ptag_pooled/adapter_model.safetensors || python3 olens/scripts/lora_for_vllm.py $C/agu18dec_olens_and_ar/ar_ptag_pooled/lora /workspace/lora_vllm/ar_ptag_pooled
echo SETUP_DONE
EOF
    ;;
  run)
    req=${3:?requests.jsonl}; tag=${4:?tag}; shift 4
    ssh_pod "cat > /workspace/reads/$tag.jsonl" < "$req"
    ssh_pod "cat > /workspace/reads/$tag.sh" <<EOF
while IFS= read -r -d '' kv; do export "\$kv"; done < /proc/1/environ
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home OMP_NUM_THREADS=10 PYTHONDONTWRITEBYTECODE=1
cd /root/repo && PYTHONPATH=olens/src:src exec python3 ${READER:-scripts/olens_read_requests.py} /workspace/reads/$tag.jsonl \
  /workspace/reads/$tag.reads.jsonl $*
EOF
    ssh_pod "nohup bash /workspace/reads/$tag.sh > /workspace/reads/$tag.log 2>&1 < /dev/null & echo started" ;;
  status)
    ssh_pod "test -f /workspace/reads/${3:?tag}.log || echo 'no log yet (reader not started)'
      grep -E '^\[[0-9]+/|Traceback|Error|to read' /workspace/reads/$3.log 2>/dev/null | tail -3
      echo \"reader processes: \$(pgrep -fc '[_]read_requests' || true)\"" ;;
  fetch)
    ssh_pod "cat /workspace/reads/${3:?tag}.reads.jsonl" > "${4:?local path}.tmp" && mv "$4.tmp" "$4" && wc -l "$4"
    ssh_pod "cat /workspace/reads/$3.reads.provenance.json" > "${4%.jsonl}.provenance.json" ;;  # olens.provenance
  stop)
    # killing the reader orphans vLLM's engine process, which keeps its GPU memory: kill both
    ssh_pod 'pkill -f "[_]read_requests"; sleep 5; pkill -9 -f "[V]LLM::EngineCore"; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader' ;;
  *) echo "unknown command: $1" >&2; exit 1 ;;
esac
