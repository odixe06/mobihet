#!/bin/bash
# Run every (arm, seed) of scripts/pilot_queue.txt that has no result yet, P jobs at a time (env nckh).
#   bash scripts/run_pilot_queue.sh [P]
cd "$(dirname "$0")/.."
P=${1:-2}; mkdir -p papers/pilot/logs
while read -r arm rest; do for s in 42 43; do
  [ -f papers/pilot/runs/${arm}_s$s/result.json ] || echo "$arm $s $rest"; done; done < scripts/pilot_queue.txt |
xargs -P "$P" -L 1 bash -c 'arm=$0; s=$1; shift 1; python -m mh.pilot --arm $arm --seed $s "$@" > papers/pilot/logs/${arm}_s$s.log 2>&1; rc=$?; echo "$(date -u +%H:%M) $arm s$s exit $rc"'
