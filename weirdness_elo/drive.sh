#!/usr/bin/env bash
# run rounds FROM..TO: pair (unless batches exist), judge all unanswered batches 20-wide, ingest (re-judge problem batches), fit
set -u; cd "$(dirname "$0")"; PY=.venv/bin/python; FROM=$1; TO=$2
for r in $(seq $FROM $TO); do
  rr=$(printf %02d $r)
  ls run/batches/r${rr}_k*.md >/dev/null 2>&1 || $PY elo.py pair run --round $r --both-orders --batch-size 20 --neighbours 12 ${FOCUS:+--focus $FOCUS} > /dev/null
  for attempt in 1 2 3; do
    ls run/batches/r${rr}_k*.md | while read f; do [ -s "${f%.md}.answers.json" ] || echo "$PWD/$f"; done | xargs -r -P 20 -n 1 ./judge_one.sh
    out=$($PY elo.py ingest run); echo "round $r ingest: $out"
    probs=$(echo "$out" | $PY -c "import json,sys;print(' '.join(sorted({p.split(':')[0].split(' ')[0] for p in json.load(sys.stdin)['problems']})))")
    [ -z "$probs" ] && break
    # drop answers + already-ingested rows of problem batches, then re-judge them
    for b in $probs; do rm -f run/batches/$b.answers.json; done
    $PY - $probs <<'PYEOF'
import json,sys; bad=set(sys.argv[1:]); p='run/comparisons.jsonl'
rows=[l for l in open(p) if l.strip() and json.loads(l)['batch'] not in bad]; open(p,'w').writelines(rows)
PYEOF
  done
  echo "round $r fit: $($PY elo.py fit run)"
done
echo ALLDONE
