#!/bin/bash
# Publish the env viewer (environments/_lib/page.html) to Cloudflare Pages (project cia-envs, https://cia-envs.pages.dev).
# The page reads envs/ on the public HF dataset and source from GitHub, so a deploy is only needed after a change to
# page.html. Needs CLOUDFLARE_API_TOKEN in the environment (.envrc.local). Run from the repo root.
#
#   environments/_lib/deploy.sh            # production (branch main)
#   environments/_lib/deploy.sh preview    # a preview deployment: check two deep links on its URL before deploying main
set -euo pipefail
branch="${1:-main}"
mkdir -p data/envs-site
cp environments/_lib/page.html data/envs-site/index.html
wrangler pages project create cia-envs --production-branch main 2>/dev/null || true
wrangler pages deploy data/envs-site --project-name cia-envs --branch "$branch" --commit-dirty=true
