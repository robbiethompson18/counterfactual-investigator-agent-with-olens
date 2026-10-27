#!/bin/bash
# Publish expview's page to Cloudflare Pages (project expview, https://expview.pages.dev). The page reads every run
# straight from the public HF dataset, so a deploy is only needed after a change to expview/static/index.html.
# Needs CLOUDFLARE_API_TOKEN in the environment (.envrc.local). Run from the repo root.
#
#   expview/deploy.sh            # production (branch main)
#   expview/deploy.sh preview    # a preview deployment: check two deep links on its URL before deploying main
set -euo pipefail
branch="${1:-main}"
mkdir -p data/expview/site
sed -e 's/{{TITLE}}/expview/' -e 's/{{DATA}}//' expview/static/index.html > data/expview/site/index.html
wrangler pages project create expview --production-branch main 2>/dev/null || true
wrangler pages deploy data/expview/site --project-name expview --branch "$branch" --commit-dirty=true
