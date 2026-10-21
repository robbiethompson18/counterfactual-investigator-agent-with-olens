#!/bin/bash
# Deploy the read-only viewer (web/index.html in static mode) to Cloudflare Pages: https://olens-viewer.pages.dev. Its old name
# https://weirdchat-olens.pages.dev gets only a 301 to it; browsers carry the #p=…&t=… fragment across, so old links keep working.
# Its data is fetched from the HF dataset (scripts/publish.py), so rerun this only when web/index.html changes.
# Needs CLOUDFLARE_API_TOKEN (.envrc.local).
#
#   transcript-viewer/scripts/deploy_site.sh
set -euo pipefail
base=https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/resolve/main/weirdchat/site/
out=$(mktemp -d)
sed "s|</head>|<script>window.WCO_STATIC = '$base';</script></head>|" transcript-viewer/web/index.html > "$out/index.html"
wrangler pages deploy "$out" --project-name olens-viewer --branch main --commit-dirty=true
old=$(mktemp -d)
echo '/* https://olens-viewer.pages.dev/:splat 301' > "$old/_redirects"
wrangler pages deploy "$old" --project-name weirdchat-olens --branch main --commit-dirty=true
