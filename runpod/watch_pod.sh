#!/usr/bin/env bash
# Detached watcher for one of Robbie's pods (runpod/CLAUDE.md: no pod without a live watcher). Every minute it
#   - deletes the pod when the job's done marker exists on it (ssh, hard 25 s timeout), after rsyncing the job's
#     small outputs (logs and jsonl, never *.pt) to FETCH_DIR;
#   - deletes the pod HOURS after the watcher started;
#   - deletes the pod when the account balance (GraphQL clientBalance) falls under FLOOR dollars.
# The pod is looked up by name on every tick, so the watcher can be started before `pod create`.
#   nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' runpod/watch_pod.sh NAME REMOTE_OUT FETCH_DIR [HOURS=5] [FLOOR=15] \
#     > /tmp/watch_NAME.log 2>&1 < /dev/null &   # macOS has no setsid; perl detaches it from the session
#   pgrep -fl "watch_pod.sh NAME"      # verify it is running before creating the pod
# Done markers: REMOTE_OUT/READS_DONE or REMOTE_OUT/FAILED. Needs RUNPOD_API_KEY and runpodctl.
set -uo pipefail
NAME=$1 REMOTE_OUT=$2 FETCH_DIR=$3 HOURS=${4:-5} FLOOR=${5:-15}
T0=$(date +%s)
KEY=~/.runpod/ssh/RunPod-Key-Go

log() { echo "$(date '+%F %T') $*"; }
pod_id() { runpodctl pod list 2>/dev/null | python3 -c "import json,sys; print(next((p['id'] for p in json.load(sys.stdin) if p['name'] == sys.argv[1]), ''))" "$1"; }
ssh_target() {  # "PORT IP" once the host has a public ip and port 22 mapped, else nothing
  curl -sS -m 20 "https://rest.runpod.io/v1/pods/$1" -H "Authorization: Bearer $RUNPOD_API_KEY" |
    python3 -c "import json,sys; p=json.load(sys.stdin); ip=p.get('publicIp'); port=(p.get('portMappings') or {}).get('22'); print(f'{port} {ip}' if ip and port else '')" 2>/dev/null
}
balance() {
  curl -sS -m 20 https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d '{"query":"{ myself { clientBalance } }"}' | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['myself']['clientBalance'])" 2>/dev/null
}
remote() { perl -e 'alarm 25; exec @ARGV' ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes -p $1 root@$2 "${@:3}"; }
fetch() {
  mkdir -p "$FETCH_DIR"
  perl -e 'alarm 600; exec @ARGV' rsync -az --exclude '*.pt' -e "ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes -p $1" \
    "root@$2:$REMOTE_OUT/" "$FETCH_DIR/" && log "fetched $REMOTE_OUT -> $FETCH_DIR"
}
delete() {
  log "deleting pod $NAME ($1): $2"
  runpodctl pod delete "$1" && log "deleted" || log "delete failed; retrying next tick"
}

log "watching $NAME: done marker $REMOTE_OUT/{READS_DONE,FAILED}, deadline ${HOURS}h, balance floor \$$FLOOR"
while true; do
  id=$(pod_id "$NAME")
  bal=$(balance)
  if [ -n "$id" ]; then
    if [ -n "$bal" ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$bal" "$FLOOR"; then
      delete "$id" "balance \$$bal under floor \$$FLOOR"
    elif [ $(( $(date +%s) - T0 )) -gt $(( HOURS * 3600 )) ]; then
      delete "$id" "deadline ${HOURS}h passed"
    else
      read -r port ip <<< "$(ssh_target "$id")"
      if [ -n "${port:-}" ]; then
        marker=$(remote "$port" "$ip" "ls $REMOTE_OUT/READS_DONE $REMOTE_OUT/FAILED 2>/dev/null" 2>/dev/null)
        if [ -n "$marker" ]; then
          log "done marker: $marker"
          fetch "$port" "$ip"
          delete "$id" "job done ($marker)"
        fi
      fi
    fi
  else
    [ -n "${seen:-}" ] && { log "pod $NAME is gone; exiting"; exit 0; }
  fi
  [ -n "$id" ] && seen=1
  echo "$(date '+%T') pod=${id:-none} balance=\$${bal:-?} elapsed=$(( ($(date +%s) - T0) / 60 ))m" >> "${WATCH_STATUS:-/tmp/watch_${NAME}.status}"
  sleep 60
done
