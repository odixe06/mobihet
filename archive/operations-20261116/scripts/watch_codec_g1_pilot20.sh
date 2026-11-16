#!/bin/bash
# Watcher for codec-pilot20 (10 arms) and g1-pilot20 (2 arms).
# Poll Kaggle every 20 minutes (1200s).
# Pull terminal kernels, audit and run analyse_codec_pilot20.py when all are pulled or terminal.
set -u
cd /home/odixe/project/mobihet
PY=/home/odixe/Tools/miniconda3/envs/nckh/bin/python
export PATH=/home/odixe/Tools/miniconda3/envs/nckh/bin:$PATH

echo "=== WATCHER START $(date -Iseconds) ==="

for cycle in $(seq 1 45); do
  echo "--- Cycle $cycle $(date -Iseconds) ---"

  # 1. codec-pilot20 collect
  n_codec=$(ls artifacts/codec-pilot20/*/pulled.ok 2>/dev/null | wc -l)
  echo "codec-pilot20 pulled: $n_codec / 10"
  if [ "$n_codec" -lt 10 ]; then
    timeout 900 $PY scripts/run_codec_pilot20.py collect > runs/codec-pilot20/collect-last.log 2>&1 || true
    cat runs/codec-pilot20/collect-last.log
    n_codec=$(ls artifacts/codec-pilot20/*/pulled.ok 2>/dev/null | wc -l)
    echo "codec-pilot20 pulled after collect: $n_codec / 10"
  fi

  # 2. g1-pilot20 collect
  n_g1=$(ls artifacts/g1-pilot20/*/pulled.ok 2>/dev/null | wc -l)
  echo "g1-pilot20 pulled: $n_g1 / 2"
  if [ "$n_g1" -lt 2 ]; then
    timeout 900 $PY scripts/run_g1_pilot20.py collect > runs/g1-pilot20/collect-last.log 2>&1 || true
    cat runs/g1-pilot20/collect-last.log
    n_g1=$(ls artifacts/g1-pilot20/*/pulled.ok 2>/dev/null | wc -l)
    echo "g1-pilot20 pulled after collect: $n_g1 / 2"
  fi

  # Run incremental audit / analysis
  $PY scripts/analyse_codec_pilot20.py 20 || true
  $PY scripts/analyse_codec_pilot20.py 20 g1-pilot20 || true

  # Check termination
  all_codec_term=$($PY -c 'import json; s=json.load(open("runs/codec-pilot20/status.json")); print(all(x.get("terminal") for x in s))' 2>/dev/null || echo "False")
  all_g1_term=$($PY -c 'import json; s=json.load(open("runs/g1-pilot20/status.json")); print(all(x.get("terminal") for x in s))' 2>/dev/null || echo "False")

  if { [ "$n_codec" -ge 10 ] || [ "$all_codec_term" = "True" ]; } && \
     { [ "$n_g1" -ge 2 ] || [ "$all_g1_term" = "True" ]; }; then
    echo "All target arms pulled or terminal! Finishing watch loop."
    break
  fi

  echo "Sleeping 1200s until next check..."
  sleep 1200
done

echo "=== FINAL AUDIT & ANALYSIS $(date -Iseconds) ==="
$PY scripts/analyse_codec_pilot20.py 20 || true
$PY scripts/analyse_codec_pilot20.py 20 g1-pilot20 || true

date > runs/codec-pilot20/DONE
date > runs/g1-pilot20/DONE

echo "ALL_PILOTS_DONE"
