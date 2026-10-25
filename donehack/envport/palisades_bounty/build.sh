#!/usr/bin/env bash
# Build localhost/envport-palisades_bounty (agent uid 1000) and -u200 (agent uid 200, for --small-uid).
set -euo pipefail
cd "$(dirname "$0")"
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs
for uid in 1000 200; do
  sfx=$([ $uid = 200 ] && echo -u200 || true)
  podman build --build-context "aie=$AIE/environments/palisades_bounty/tower_of_hanoi" --build-arg AGENT_UID=$uid \
    -t localhost/envport-palisades_bounty$sfx .
done
