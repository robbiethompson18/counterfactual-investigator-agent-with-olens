#!/usr/bin/env bash
# The shared Qwen3.6-27B vLLM server, pod "qwen-shared". Use it instead of starting your own; see runpod/CLAUDE.md.
#   runpod/qwen_server.sh up [GPU_COUNT]   # create (default 2x H200) and boot; ~15 min to serving
#   runpod/qwen_server.sh url              # print QWEN_VLLM_URL (raw TCP, not RunPod's HTTP proxy)
#   runpod/qwen_server.sh status           # one progress line; run every 3 min while booting
#   runpod/qwen_server.sh restart          # re-upload and re-run the boot script (keeps the venv and weights)
#   runpod/pod.sh down qwen-shared         # stop paying for it
# Needs RUNPOD_API_KEY and QWEN_VLLM_API_KEY (.envrc.local); the latter is the server's API key.
set -euo pipefail
NAME=qwen-shared

boot() {
  runpod/pod.sh ssh $NAME 'cat > /workspace/qwen_server_boot.sh' < runpod/qwen_server_boot.sh
  runpod/pod.sh ssh $NAME "API_KEY=$QWEN_VLLM_API_KEY nohup bash /workspace/qwen_server_boot.sh > /workspace/boot.log 2>&1 < /dev/null &"
}

case ${1:?command} in
  up)
    PORTS=22/tcp,8000/tcp runpod/pod.sh up $NAME "NVIDIA H200" "${2:-2}"
    boot
    echo "booting; run '$0 status' every 3 min until it says READY" ;;
  url)
    runpod/pod.sh ssh $NAME 'tr "\0" "\n" < /proc/1/environ' |
      awk -F= '/^RUNPOD_PUBLIC_IP=/{ip=$2} /^RUNPOD_TCP_PORT_8000=/{port=$2} END{print "http://" ip ":" port "/v1"}' ;;
  status)
    if curl -sf -m 5 -H "Authorization: Bearer $QWEN_VLLM_API_KEY" "$($0 url)/models" >/dev/null; then echo READY; exit 0; fi
    # [x]yz patterns keep pgrep from matching this command's own shell
    runpod/pod.sh ssh $NAME 'echo "uv_cache=$(du -sm ~/.cache/uv 2>/dev/null | cut -f1)M weights=$(du -sm /workspace/.cache/huggingface 2>/dev/null | cut -f1)M" \
      "installing=$(pgrep -fc "[u]v pip install") vllm=$(pgrep -fc "[b]in/vllm serve")" \
      "log: $(tail -c 3000 /workspace/vllm.log 2>/dev/null | tr "\r" "\n" | grep -v "^\s*$" | tail -1 | cut -c1-160)"
      grep -hE "Traceback|Error|error:|out of memory|timed out" /workspace/boot.log /workspace/vllm.log 2>/dev/null | tail -3' ;;
  restart) boot ;;
  *) echo "unknown command: $1" >&2; exit 1 ;;
esac
