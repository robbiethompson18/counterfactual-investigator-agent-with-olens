#!/bin/bash
# Deploy the viewer page (web/index.html) to Cloudflare Pages: https://olens-viewer.pages.dev. Its old name
# https://weirdchat-olens.pages.dev gets only a 301 to it; browsers carry the #p=…&t=… fragment across, so old links keep working.
# The page fetches its data from the HF dataset (scripts/publish.py), so rerun this only when web/index.html changes.
# Needs CLOUDFLARE_API_TOKEN (.envrc.local).
#
#   transcript-viewer/scripts/deploy_site.sh                     # production (branch main)
#   transcript-viewer/scripts/deploy_site.sh --branch preview    # https://preview.olens-viewer.pages.dev, to check a change first
set -euo pipefail
branch=main
[[ ${1:-} == --branch ]] && branch=$2
out=$(mktemp -d)
cp transcript-viewer/web/index.html "$out/index.html"
wrangler pages deploy "$out" --project-name olens-viewer --branch "$branch" --commit-dirty=true
[[ $branch == main ]] || exit 0
old=$(mktemp -d)
echo '/* https://olens-viewer.pages.dev/:splat 301' > "$old/_redirects"
wrangler pages deploy "$old" --project-name weirdchat-olens --branch main --commit-dirty=true
