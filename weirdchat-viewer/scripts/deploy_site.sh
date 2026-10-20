#!/bin/bash
# Deploy the read-only viewer (web/index.html in static mode) to Cloudflare Pages: https://weirdchat-olens.pages.dev
# Its data is fetched from the HF dataset (scripts/publish.py), so rerun this only when web/index.html changes.
# Needs CLOUDFLARE_API_TOKEN (.envrc.local).
#
#   weirdchat-viewer/scripts/deploy_site.sh
set -euo pipefail
base=https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/resolve/main/weirdchat/site/
out=$(mktemp -d)
sed "s|</head>|<script>window.WCO_STATIC = '$base';</script></head>|" weirdchat-viewer/web/index.html > "$out/index.html"
wrangler pages deploy "$out" --project-name weirdchat-olens --branch main --commit-dirty=true
