#!/usr/bin/env bash
# Serve expview on bramble's tailnet addresses: http://bramble:8830/
#   expview/run.sh
# Env: EXPVIEW_PORT (8830), EXPVIEW_HOSTS (default: tailscale ip -4 and -6), EXPVIEW_ROOTS (see expview.roots()).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${EXPVIEW_PORT:-8830}"
HOSTS=(${EXPVIEW_HOSTS:-$(tailscale ip -4) $(tailscale ip -6)})
args=()
for h in "${HOSTS[@]}"; do args+=(--host "$h"); done
exec uv run --script "$REPO/expview/server.py" --port "$PORT" "${args[@]}"
