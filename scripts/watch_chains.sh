#!/bin/bash
# Background chain watcher (owner 14/11): every 1200 s pull finished Kaggle sessions and push the next same-owner session,
# for mobihet B21 (transfer + chain + seed43) and fl_veremi_bert (run_batch collect). No model polling; stop: kill the PID in the log.
# A tick waits while another collect of the same repo runs (no duplicate pushes); push intents guard re-pushes.
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh && conda activate nckh
MH=/home/odixe/project/mobihet; FB=/home/odixe/project/fl_veremi_bert
LOG=$MH/runs/chain-watch; mkdir -p $LOG
echo "WATCHER_START pid $$ $(date -Iseconds)"
for tick in $(seq 1 120); do
  while pgrep -f "run_batch.py collect|run_b21_chain.py|run_b21_transfer.py" >/dev/null; do sleep 60; done
  echo "TICK $tick $(date -Iseconds)"
  ( cd $MH && python -u scripts/run_b21_transfer.py collect && python -u scripts/run_b21_chain.py collect \
      && python -u scripts/run_b21_chain.py launch ) > $LOG/mobihet-$tick.log 2>&1; echo "mobihet rc $?"
  grep -E "PUSHED|DEFER|SKIP|COMPLETE|ERROR" $LOG/mobihet-$tick.log | tail -12
  for sd in 42 41 43; do [ $sd = 42 ] || [ -f $FB/runs/batch-s$sd/plan.json ] || continue
    ( cd $FB && FLB_SEED=$sd timeout 3000 python -u scripts/run_batch.py collect ) >> $LOG/bert-$tick.log 2>&1; echo "bert s$sd rc $?"; done
  grep -E "^PUSHED|^[A-Z]+ +r[0-9]|REMAINING|ALL_TERMINAL|Traceback" $LOG/bert-$tick.log | tail -20
  if [ "$(grep -c ALL_TERMINAL $LOG/bert-$tick.log)" -ge 3 ] && ! grep -q "B21CHAIN_PUSHED\|RUNNING" $LOG/mobihet-$tick.log; then
    echo "WATCHER_IDLE_EXIT $(date -Iseconds)"; exit 0; fi
  sleep 1200
done
echo "WATCHER_MAX_TICKS $(date -Iseconds)"
