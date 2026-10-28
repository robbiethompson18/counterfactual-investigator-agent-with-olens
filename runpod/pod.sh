#!/usr/bin/env bash
# Robbie's RunPod pods, by name, on the RunPod account in .envrc.local. See runpod/CLAUDE.md.
#   runpod/pod.sh up NAME [GPU_ID] [GPU_COUNT]   # create, wait for ssh (default: 1x H200)
#   runpod/pod.sh ssh NAME [CMD...]              # shell, or run CMD on the pod
#   runpod/pod.sh sync NAME                      # rsync this repo to /workspace/repo (no secrets, .venv or data)
#   runpod/pod.sh down NAME                      # delete the pod
#   runpod/pod.sh list
# GPU ids come from `runpodctl gpu list`. HF_TOKEN (from `hf auth token`) is set in the pod's env.
# VOLUME=<network volume id> mounts that volume at /workspace (pins the pod to its datacenter); IMAGE= overrides the image.
set -euo pipefail

IMAGE=${IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}

pod_id() { runpodctl pod list | python3 -c "import json,sys; print(next(p['id'] for p in json.load(sys.stdin) if p['name'] == sys.argv[1]))" "$1"; }

ssh_args() {  # "-i KEY -p PORT root@IP", or nothing while the pod is still booting
  runpodctl pod get "$(pod_id "$1")" | python3 -c "
import json, sys
s = json.load(sys.stdin).get('ssh') or {}
if s.get('ip') and s.get('port'): print(f\"-i {s['ssh_key']['path']} -p {s['port']} root@{s['ip']}\")"
}

cmd=$1
[ "$cmd" = list ] && exec runpodctl pod list
name=${2:?pod name}
case $cmd in
  up)
    runpodctl pod create --name "$name" --image "$IMAGE" --gpu-id "${3:-NVIDIA H200}" --gpu-count "${4:-1}" \
      --container-disk-in-gb 200 --ports "${PORTS:-22/tcp}" ${VOLUME:+--network-volume-id "$VOLUME"} --env "{\"HF_TOKEN\": \"$(hf auth token 2>/dev/null)\", \"HF_XET_HIGH_PERFORMANCE\": \"1\"}" >/dev/null
    until args=$(ssh_args "$name") && [ -n "$args" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 $args true 2>/dev/null; do sleep 10; done
    # --env only reaches PID 1; ssh sessions get it via pam_env
    ssh -o StrictHostKeyChecking=no $args 'tr "\0" "\n" < /proc/1/environ | grep "^HF_" >> /etc/environment'
    echo "ssh $args" ;;
  ssh) shift 2; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $(ssh_args "$name") "$@" ;;
  sync)
    rsync -az --exclude .envrc.local --exclude .venv --exclude data --exclude checkpoints --exclude node_modules \
      -e "ssh -o StrictHostKeyChecking=no $(ssh_args "$name" | sed 's/ root@.*//')" ./ "root@$(ssh_args "$name" | sed 's/.*root@//'):/workspace/repo/" ;;
  down) runpodctl pod delete "$(pod_id "$name")" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
