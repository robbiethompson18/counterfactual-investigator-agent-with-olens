#!/usr/bin/env bash
# judge one batch file with a headless Sonnet (Read/Write only); retries once if no answers file appears
f="$1"; ans="${f%.md}.answers.json"
for t in 1 2; do
  [ -s "$ans" ] && exit 0
  timeout 600 claude -p --model sonnet --allowedTools Read Write --disallowedTools Bash Agent WebFetch WebSearch -- \
    "Read the file $f and follow its instructions exactly: judge each pair yourself (no subagents, only Read and Write), write the JSON answers to the answers file it names, then reply only \"done <n>\"." \
    < /dev/null > "${f%.md}.log" 2>&1
done
[ -s "$ans" ]
