#!/usr/bin/env bash
# Forward a pod's O-lens read server (8802, scripts/identity/pod_setup.sh) to localhost:28802. Runs in the foreground.
#   scripts/identity/tunnel.sh NAME
set -euo pipefail
id=$(runpodctl pod list | python3 -c "import json,sys; print(next(p['id'] for p in json.load(sys.stdin) if p['name'] == sys.argv[1]))" "$1")
args=$(runpodctl pod get "$id" | python3 -c "
import json, sys
s = json.load(sys.stdin)['ssh']
print(f\"-i {s['ssh_key']['path']} -p {s['port']} root@{s['ip']}\")")
exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L 127.0.0.1:28802:127.0.0.1:8802 $args
