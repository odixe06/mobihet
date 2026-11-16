#!/bin/bash
# Detached simulator-batch watcher: poll Kaggle every 20 min, pull finished arms, then audit + locked analysis.
#   setsid nohup bash scripts/watch_sim_batch.sh <batch> <n_arms> > runs/sim-kaggle-<batch>/watch.log 2>&1 &
# Idempotent: collect skips arms with pulled.ok. Writes runs/sim-kaggle-<batch>/{watch.pid,DONE}. Never pushes or restarts kernels.
B=$1; N=$2
cd /home/odixe/project/mobihet
RUN=runs/sim-kaggle-$B
PY=/home/odixe/Tools/miniconda3/envs/nckh/bin/python
export SIM_BATCH=$B
echo $$ > $RUN/watch.pid
rm -f $RUN/DONE
for i in $(seq 1 36); do
  timeout 900 $PY scripts/run_sim_kaggle.py collect > $RUN/collect-last.log 2>&1
  date; grep -v "^  pulled" $RUN/collect-last.log
  n=$(ls artifacts/sim-kaggle-$B/*/pulled.ok 2>/dev/null | wc -l)
  [ "$n" -ge "$N" ] && break
  sleep 1200
done
$PY scripts/audit_sim_batch.py $B 20; echo "audit rc $?"
for r in 16 18 20; do KM_BATCH=$B $PY scripts/analyse_km_lite.py $r || true; done
date > $RUN/DONE
