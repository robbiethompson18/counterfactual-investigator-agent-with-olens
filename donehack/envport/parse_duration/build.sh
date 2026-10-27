#!/usr/bin/env bash
# Build localhost/envport-parse_duration (agent uid 1000) and -u200 (uid 200).
set -euo pipefail
cd "$(dirname "$0")"
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs
for uid in 1000 200; do
  sfx=$([ $uid = 200 ] && echo -u200 || true)
  podman build --build-context "aie=$AIE/environments/parse_duration" --build-arg AGENT_UID=$uid \
    -t localhost/envport-parse_duration$sfx .
done
