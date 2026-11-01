#!/bin/sh
# Builds the write-up HTML and the talk deck, then publishes both to Cloudflare Pages, by hand (no CI since 2026-10-31):
#   https://nn-sprint-writeup.pages.dev/           the write-up (docs/final-writeup.html, charts embedded)
#   https://nn-sprint-writeup.pages.dev/talk.pdf   the deck (slides/talk/build/talk.pdf)
# Run from the repo root. Needs CLOUDFLARE_API_TOKEN (.envrc.local), pandoc, and the slides toolchain (slides/CLAUDE.md).
set -eu
PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin" # TinyTeX's default install on macOS
sh scripts/build_writeup.sh
make -C slides
site=$(mktemp -d)
cp docs/final-writeup.html "$site/index.html"
cp slides/talk/build/talk.pdf "$site/"
wrangler pages deploy "$site" --project-name nn-sprint-writeup --branch main --commit-dirty=true
