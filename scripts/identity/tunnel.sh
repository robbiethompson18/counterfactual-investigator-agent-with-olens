#!/usr/bin/env bash
# Forward a pod's vLLM (8000) and O-lens read server (8802) to localhost:28000 and :28802. Runs in the foreground.
#   scripts/identity/tunnel.sh NAME            # the main 2-GPU pod
#   scripts/identity/tunnel.sh NAME 28001      # a vLLM-only replica: forward just its 8000 to 28001
set -euo pipefail
id=$(runpodctl pod list | python3 -c "import json,sys; print(next(p['id'] for p in json.load(sys.stdin) if p['name'] == sys.argv[1]))" "$1")
args=$(runpodctl pod get "$id" | python3 -c "
import json, sys
s = json.load(sys.stdin)['ssh']
print(f\"-i {s['ssh_key']['path']} -p {s['port']} root@{s['ip']}\")")
fwd=(-L 127.0.0.1:28000:127.0.0.1:8000 -L 127.0.0.1:28802:127.0.0.1:8802)
[ $# -ge 2 ] && fwd=(-L "127.0.0.1:$2:127.0.0.1:8000")
exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N "${fwd[@]}" $args
