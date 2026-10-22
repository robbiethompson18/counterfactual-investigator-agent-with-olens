#!/usr/bin/env bash
# Serve the file browser. Default binds the tailscale addresses (Kylie's setup); pass --bind ADDR to override.
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then set -- --bind "$(tailscale ip -4)" --bind "$(tailscale ip -6)"; fi
exec python3 server.py --port "${PORT:-8812}" "$@"
