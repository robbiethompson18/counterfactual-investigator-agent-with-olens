#!/usr/bin/env bash
# Build localhost/envport-revert_or_refactor (dev uid 1000) and ...-u200 (dev uid 200, for --small-uid).
# Network is used at build time only (apt, pip).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs @ 310c03f
SRC="$AIE/environments/revert_or_refactor"
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Containerfile" "$HERE/ror_apply.py" "$SRC/apply_patch.py" "$CTX/"
cp -r "$SRC/task" "$CTX/task"
find "$CTX/task" -name __pycache__ -prune -exec rm -rf {} +
podman build -t localhost/envport-revert_or_refactor "$CTX"
podman build --build-arg DEV_UID=200 -t localhost/envport-revert_or_refactor-u200 "$CTX"
