#!/usr/bin/env bash
# Build localhost/envport-tables_coverage_tests (dev uid 1000) and ...-u200 (dev uid 200, for --small-uid).
# Network is used at build time only (apt, pip, git clone of Python-Markdown).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/Containerfile" "$CTX/"
cp "$AIE/environments/tables_coverage_tests/grading_leak" "$CTX/"
cp -r "$AIE/environments/tables_coverage/grading_runner" "$CTX/"
find "$CTX/grading_runner" -name __pycache__ -prune -exec rm -rf {} +
cp "$AIE/environments/tables_coverage/build_tasks.py" "$CTX/"
podman build -t localhost/envport-tables_coverage_tests "$CTX"
podman build --build-arg DEV_UID=200 -t localhost/envport-tables_coverage_tests-u200 "$CTX"
