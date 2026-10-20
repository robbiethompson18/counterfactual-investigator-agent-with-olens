#!/bin/bash
# Build the CHIVE cases viewer and publish it to Cloudflare Pages (project chive-cases, https://chive-cases.pages.dev).
# Needs CLOUDFLARE_API_TOKEN in the environment (.envrc.local).
#
#   scripts/deploy_viewer.sh
set -euo pipefail
uv run python scripts/build_viewer.py
wrangler pages deploy data/viewer/site --project-name chive-cases --branch main --commit-dirty=true
