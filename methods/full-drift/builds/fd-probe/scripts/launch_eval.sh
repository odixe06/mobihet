#!/bin/bash
# Fast evaluation push (owner 21/10): gen_eval.py + upload/verify source r26 and checkpoint + push + PUSHED marker.
# usage: IDS=<poll list> launch_eval.sh <train session rid> <a-b rounds> <owner>
set -euo pipefail
B=/home/odixe/nckh/veremi/mobihet/methods/full-drift/builds/fd-probe
H=/home/odixe/nckh/veremi/mobihet/.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py

export PATH=$(conda run -n nckh python -c 'import sys,os;print(os.path.dirname(sys.executable))'):$PATH
cd $B/scripts
out=$(python gen_eval.py --session "$1" --rounds "$2" --owner "$3" | tail -1)
echo "$out"
kid=$(echo "$out" | awk '{print $1}'); rid=${kid#*/mobihet-}
python upload_stage.py $B/artifacts/input/fd-src-20261015-r26 "$3/mobihet-fulldrift-src-20261015-r26" | tail -1
python upload_stage.py $B/artifacts/input/$rid | tail -1
push=$(python $H "$3" -- kaggle kernels push -p $B/private_push/$rid 2>&1 | tail -1); echo "$push"
echo "$push" | grep -q "successfully pushed" || { echo "PUSH NOT CONFIRMED $kid"; exit 1; }
echo "$kid v1 pushed $(date -u +%Y-%m-%dT%H:%MZ)" > $B/notebooks/$rid/PUSHED
[ -n "${IDS:-}" ] && echo "$kid" >> "$IDS"   # optional status-poll list
echo "PUSHED $kid"
