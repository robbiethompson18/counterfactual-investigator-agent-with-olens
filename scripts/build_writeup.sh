#!/bin/sh
# docs/final-writeup.md -> docs/final-writeup.html, one file with every chart embedded. Run from the repo root once the charts exist
# (a missing PNG fails the build). Links to other docs become GitHub links, so they work from a downloaded copy. The HTML is in LFS
# (git-lfs's own pre-push hook uploads it). .githooks/pre-commit runs this whenever the Markdown is staged. PNGs are gitignored, so images
# are also looked up in the checkout that data/ links to: a pool checkout borrows the main checkout's charts.
sed -E 's#\]\(([A-Za-z0-9_/.-]+\.md)\)#](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/main/docs/\1)#g' docs/final-writeup.md |
  pandoc -f gfm -t html5 -s --embed-resources --resource-path="docs:$(dirname "$(readlink -f data)")/docs" --fail-if-warnings --metadata pagetitle="Final write-up" -V maxwidth=50em -o docs/final-writeup.html
