#!/bin/sh
# docs/final-writeup.md -> docs/final-writeup.html, one file with every chart embedded. Run from the repo root once the charts exist
# (a missing PNG fails the build). Links to other docs become GitHub links, so they work from a downloaded copy. The HTML is a
# deploy artifact (gitignored since 2026-10-31): scripts/deploy_writeup.sh runs this in the main checkout, then publishes it to
# Cloudflare Pages. Don't build it in a PR. PNGs are gitignored, so images are also looked up in the checkout that data/ links to.
sed -E 's#\]\(([A-Za-z0-9_/.-]+\.md)\)#](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/main/docs/\1)#g' docs/final-writeup.md |
  pandoc -f gfm -t html5 -s --lua-filter=scripts/writeup_heading_links.lua --embed-resources --resource-path="docs:$(dirname "$(readlink -f data)")/docs" --fail-if-warnings --metadata pagetitle="Final write-up" -V maxwidth=50em -o docs/final-writeup.html
