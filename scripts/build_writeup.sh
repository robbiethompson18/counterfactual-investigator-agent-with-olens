#!/bin/sh
# docs/final-writeup.md -> docs/final-writeup.html, one file with every chart embedded. Run from the repo root once the charts exist
# (a missing PNG fails the build). Links to other docs become GitHub links, so they work from a downloaded copy. The HTML is in LFS;
# .githooks has no pre-push hook, so upload it with `git lfs push origin BRANCH` before `git push`.
sed -E 's#\]\(([A-Za-z0-9_/.-]+\.md)\)#](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/main/docs/\1)#g' docs/final-writeup.md |
  pandoc -f gfm -t html5 -s --embed-resources --resource-path=docs --fail-if-warnings --metadata pagetitle="Final write-up" -V maxwidth=50em -o docs/final-writeup.html
