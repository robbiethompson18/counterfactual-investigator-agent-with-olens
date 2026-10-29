#!/usr/bin/env bash
# NLA readouts on a pod (scripts/olens_read_requests.py --readout nla: the all-layer NLA's AV on the one engine,
# olens/src/olens/nla.py), then the critic's FVE as a separate job (scripts/nla_score.py). Same pod recipe as the O-lens reads:
#   IMAGE=vllm/vllm-openai:v0.27.1 VOLUMES=cdde9ia5iu NAME=X runpod/qwen_server.sh up 1
# See runpod/CLAUDE.md and docs/nla-reader.md.
#   runpod/nla_reads.sh setup NAME                 # runpod/olens_reads.sh setup, then the AV adapter (vLLM-renamed) and nla_meta.yaml
#   runpod/nla_reads.sh run NAME REQUESTS.jsonl TAG [ARGS...]   # -> /workspace/reads/TAG.reads.jsonl + TAG.reads.acts.bf16
#   runpod/nla_reads.sh status NAME TAG            # last progress lines
#   runpod/nla_reads.sh fetch NAME TAG LOCAL.jsonl # reads, activations and provenance
#   runpod/nla_reads.sh stop NAME                  # kill the reader (a rerun with the same TAG resumes)
#   runpod/nla_reads.sh score NAME TAG             # the critic job on the pod's reads (downloads the 51 GB critic once) -> TAG.score.log
#   runpod/nla_reads.sh fetch NAME TAG LOCAL.jsonl # again, for the scored rows
set -euo pipefail
NAME=${2:?pod name}
ssh_pod() { runpod/pod.sh ssh "$NAME" "$@"; }
C=/workspace/olens_ckpt/nla64  # not mlnla/, the 11-layer camilablank NLA's dir: setup skips an AV already converted there
NLA_REPO=agu18dec/qwen3.6-27b-all-layer-nla-rl2000  # olens.nla.NLA_REPO

case ${1:?command} in
  setup)
    runpod/olens_reads.sh setup "$NAME"
    ssh_pod 'bash -s' <<EOF
set -e
while IFS= read -r -d '' kv; do export "\$kv"; done < /proc/1/environ
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home
# named files, not --include 'av/*': the glob form left av/ out on 2026-10-27
hf download $NLA_REPO av/adapter_config.json av/adapter_model.safetensors nla_meta.yaml run_config.yaml --local-dir $C >/dev/null
cd /root/repo
test -f /workspace/lora_vllm/nla64_av/adapter_model.safetensors || python3 olens/scripts/lora_for_vllm.py $C/av /workspace/lora_vllm/nla64_av
echo NLA_SETUP_DONE
EOF
    ;;
  run)
    req=${3:?requests.jsonl}; tag=${4:?tag}; shift 4
    runpod/olens_reads.sh run "$NAME" "$req" "$tag" --readout nla "$@" ;;
  status) runpod/olens_reads.sh status "$NAME" "${3:?tag}" ;;
  fetch)
    runpod/olens_reads.sh fetch "$NAME" "${3:?tag}" "${4:?local path}"
    ssh_pod "cat /workspace/reads/$3.reads.acts.bf16" > "${4%.jsonl}.acts.bf16" && ls -la "${4%.jsonl}.acts.bf16" ;;
  stop) runpod/olens_reads.sh stop "$NAME" ;;
  score)
    tag=${3:?tag}
    ssh_pod "cat > /workspace/reads/$tag.score.sh" <<EOF
while IFS= read -r -d '' kv; do export "\$kv"; done < /proc/1/environ
export HF_HOME=/workspace/.cache/huggingface HOME=/workspace/home PYTHONDONTWRITEBYTECODE=1
hf download $NLA_REPO --include 'ar/*' --local-dir $C >/dev/null
cd /root/repo && PYTHONPATH=olens/src:src exec python3 scripts/nla_score.py /workspace/reads/$tag.reads.jsonl --critic $C/ar --meta $C/nla_meta.yaml
EOF
    ssh_pod "nohup bash /workspace/reads/$tag.score.sh > /workspace/reads/$tag.score.log 2>&1 < /dev/null & echo score started" ;;
  *) echo "unknown command: $1" >&2; exit 1 ;;
esac
