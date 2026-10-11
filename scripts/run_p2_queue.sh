#!/bin/bash
# Run every (arm, seed) of ${Q:-scripts/p2_queue.txt} that has no result yet, P jobs at a time (env nckh).
#   bash scripts/run_p2_queue.sh [P]      (Q=<queue file> to override)
cd "$(dirname "$0")/.."
P=${1:-2}; mkdir -p papers/p2/logs
while read -r arm rest; do for s in 42 43; do
  [ -f papers/p2/runs/${arm}_s$s/result.json ] || echo "$arm $s $rest"; done; done < ${Q:-scripts/p2_queue.txt} |
xargs -P "$P" -L 1 bash -c 'arm=$0; s=$1; shift 1; python -m mh.p2 --arm $arm --seed $s "$@" > papers/p2/logs/${arm}_s$s.log 2>&1; rc=$?; echo "$(date -u +%H:%M) $arm s$s exit $rc"'
