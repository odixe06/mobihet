#!/bin/bash
# Continuation push (final seed42, ~11.5h session, fixture cache attached): cache marker + gen_continue.py + upload/verify
# source r25 and the parent bridge + push + PUSHED marker.
# usage: IDS=<poll list> launch_continue.sh <owner> <clients> <parent session rid> <new session code>
set -euo pipefail
B=/home/odixe/nckh/veremi/mobihet/methods/full-drift/builds/fd-probe
H=/home/odixe/nckh/veremi/mobihet/.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py
export PATH=$(conda run -n nckh python -c 'import sys,os;print(os.path.dirname(sys.executable))'):$PATH
cd $B/scripts
[ -f $B/artifacts/fixture-cache/$1-k$2.json ] || python gen_fixture.py --owner "$1" --clients "$2" --check
python gen_continue.py --parent "$3" --owner "$1" --session "$4" | tail -1
rid=fd-s42$4-20261015-r25-v1
grep -q "\"$1/mobihet-fd-fixture-k$2-20261015-r25\"" $B/notebooks/$rid/kernel-metadata.json || { echo "fixture cache not attached $rid"; exit 1; }
python upload_stage.py $B/artifacts/input/fd-src-20261015-r25 "$1/mobihet-fulldrift-src-20261015-r25" | tail -1
python upload_stage.py $B/artifacts/input/$rid-parent | tail -1
push=$(python $H "$1" -- kaggle kernels push -p $B/private_push/$rid 2>&1 | tail -1); echo "$push"
echo "$push" | grep -q "successfully pushed" || { echo "PUSH NOT CONFIRMED $1/mobihet-$rid"; exit 1; }
echo "$1/mobihet-$rid v1 pushed $(date -u +%Y-%m-%dT%H:%MZ)" > $B/notebooks/$rid/PUSHED
[ -n "${IDS:-}" ] && echo "$1/mobihet-$rid" >> "$IDS"   # optional status-poll list
echo "PUSHED $1/mobihet-$rid"
