#!/usr/bin/env bash
# Build the four images: envport-chess / envport-tictactoe, agent uid 1000 / 200.
set -euo pipefail
cd "$(dirname "$0")"
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs
CTX=(--build-context "aie=$AIE/environments/games")
for game in chess tictactoe; do
  for uid in 1000 200; do
    sfx=$([ $uid = 200 ] && echo -u200 || true)
    podman build "${CTX[@]}" --build-arg AGENT_UID=$uid --build-arg GAME=$game -t localhost/envport-$game$sfx .
  done
done
