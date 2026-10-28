#!/usr/bin/env bash
# The shared Qwen3.6-27B vLLM server, pod "qwen-shared". Use it instead of starting your own; see runpod/CLAUDE.md.
#   runpod/qwen_server.sh up [GPU_COUNT]   # create (default 2x H200) on the cache volume; ~3 min to serving
#   runpod/qwen_server.sh url              # print QWEN_VLLM_URL (raw TCP, not RunPod's HTTP proxy)
#   runpod/qwen_server.sh status           # one progress line; run every 3 min while booting
#   runpod/qwen_server.sh restart          # re-upload and re-run the boot script on the running pod
#   runpod/pod.sh down qwen-shared         # stop paying for it
# Needs RUNPOD_API_KEY and QWEN_VLLM_API_KEY (.envrc.local); the latter is the server's API key.
set -euo pipefail
NAME=${NAME:-qwen-shared}
# Network volumes holding weights + compile caches, tried in order until one's datacenter has the GPUs free:
# "qwen-fastboot" (US-CA-2), "qwen-fastboot-nc1" (US-NC-1). VOLUMES=id overrides.
VOLUMES=${VOLUMES:-wwwce2mejb cdde9ia5iu}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.32.0}  # as fast as 0.25.1 on a healthy host (docs/runpod-qwen-server.md); IMAGE=vllm/vllm-openai:v0.25.1 for the old one

case ${1:?command} in
  up)
    # runpodctl can't set a start command, so create through the REST API. The image's entrypoint is vllm; run the boot script
    # instead, in a subshell so the container (and sshd) stays up if it fails.
    for vol in $VOLUMES; do
      jq -n --rawfile boot runpod/qwen_server_boot.sh --arg name "$NAME" --arg image "$IMAGE" --argjson n "${2:-2}" --arg vol $vol \
        --arg key "$QWEN_VLLM_API_KEY" --arg hf "$(hf auth token 2>/dev/null)" '{
          name: $name, imageName: $image, gpuTypeIds: ["NVIDIA H200"], gpuCount: $n,
          networkVolumeId: $vol, volumeMountPath: "/workspace", containerDiskInGb: 50, ports: ["22/tcp", "8000/tcp"],
          env: {API_KEY: $key, HF_TOKEN: $hf, HF_XET_HIGH_PERFORMANCE: "1"},
          dockerEntrypoint: ["bash", "-c"], dockerStartCmd: ["mountpoint -q /workspace || { echo no network volume at /workspace; sleep infinity; }\nexec > /workspace/boot.log 2>&1\n(" + $boot + ")\nsleep infinity"]}' |
        curl -sS --fail-with-body https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY" \
          -H "Content-Type: application/json" -d @- | jq -r '.id // error(tostring) | "created \(.) on volume '$vol'"' && break
      echo "no ${2:-2}x H200 free with volume $vol"
      [ "$vol" = "${VOLUMES##* }" ] && exit 1
    done
    echo "booting; run '$0 status' every 3 min until it says READY" ;;
  url)
    runpod/pod.sh ssh $NAME 'tr "\0" "\n" < /proc/1/environ' |
      awk -F= '/^RUNPOD_PUBLIC_IP=/{ip=$2} /^RUNPOD_TCP_PORT_8000=/{port=$2} END{print "http://" ip ":" port "/v1"}' ;;
  status)
    # Per GPU "SM MHz, temp C, throttle bits". Under load a healthy H200 sits at 1500-1980 MHz, <80 C, bits 0x4 (power cap) or 0;
    # 0x20/0x40 is thermal throttling, which once ran decode 2.5-3x slow (docs/runpod-qwen-server.md). Idle reads 345 MHz, 0x1: no signal.
    if curl -sf -m 5 -H "Authorization: Bearer $QWEN_VLLM_API_KEY" "$($0 url)/models" >/dev/null; then
      echo "READY | gpu: $(runpod/pod.sh ssh $NAME nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits | paste -sd '|' -)"; exit 0
    fi
    # [x]yz patterns keep pgrep from matching this command's own shell
    runpod/pod.sh ssh $NAME 'echo "boot: $(tail -1 /workspace/boot.log) | vllm=$(pgrep -fc "[b]in/vllm serve")" \
      "log: $(tail -c 3000 /workspace/vllm.log 2>/dev/null | tr "\r" "\n" | grep -v "^\s*$" | tail -1 | cut -c1-160)"
      grep -hE "Traceback|Error|error:|out of memory|timed out" /workspace/boot.log /workspace/vllm.log 2>/dev/null | tail -3' ;;
  restart)
    runpod/pod.sh ssh $NAME 'cat > /workspace/qwen_server_boot.sh' < runpod/qwen_server_boot.sh
    runpod/pod.sh ssh $NAME 'nohup bash /workspace/qwen_server_boot.sh > /workspace/boot.log 2>&1 < /dev/null &' ;;
  *) echo "unknown command: $1" >&2; exit 1 ;;
esac
