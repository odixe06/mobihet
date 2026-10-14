"""Budget and push the authorized 14-Oct batch once; no automatic push retry."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
sys.path.insert(0,str(HELPER))
from kaggle_account import account_quota, active_user


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true')
    parser.add_argument('--exclude',action='append',default=[]);args=parser.parse_args()
    rows=[]
    for build,tag,hours in [('full-validation','20261014-cont02',[3.1,5.,5.,3.1]),
                            ('hybrid-full','20261014-cont01',[5.,5.,4.8,4.8]),
                            ('boundary-cells','20261014-stage01',[2.2,2.5])]:
        folders=sorted((ROOT/'papers'/build/'notebook').glob(f'*{tag}*'))
        assert len(folders)==len(hours)
        for folder,h in zip(folders,hours):
            meta=json.loads((folder/'kernel-metadata.json').read_text())
            assert (folder/'input-access.json').exists(), ('missing owner-scoped input proof',meta['id'])
            if (folder/'PUSHED').exists() or meta['id'] in args.exclude:continue
            rows.append(dict(folder=str(folder.relative_to(ROOT)),kernel=meta['id'],owner=meta['id'].split('/')[0],
                             session_hours=h,reserved_with_margin=1.15*h))
    need=collections.defaultdict(float);sessions=collections.Counter()
    for row in rows:need[row['owner']]+=row['reserved_with_margin'];sessions[row['owner']]+=1
    assert max(sessions.values())<=2
    quotas={}
    for owner in need:
        for attempt in range(3):
            try:quotas[owner]=account_quota(owner);break
            except Exception:
                if attempt==2:raise RuntimeError(f'Quota unreadable: {owner}; nothing pushed') from None
                time.sleep(2)
        assert quotas[owner]['gpu_left']>=need[owner], ('quota shortfall',owner,quotas[owner]['gpu_left'],need[owner])
    record=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                active_account=active_user(),account_switch=False,quotas=quotas,rows=rows,
                session_hours=sum(r['session_hours'] for r in rows),
                reserved_with_15pct=sum(need.values()),
                limits='Staged batch only; fifty-round targets require later quota. No GPU reservation until service accepts push.')
    proof=ROOT/'docs/operations/launch-batch-2026-10-14.json'
    if proof.exists():
        previous=json.loads(proof.read_text());record['previous_batches']=previous.get('previous_batches',[])+[
            {k:v for k,v in previous.items() if k!='previous_batches'}]
    proof.write_text(json.dumps(record,indent=2)+'\n')
    print('BATCH budget PASS',len(rows),'sessions;',record['session_hours'],'h +15%; active',record['active_account'],flush=True)
    if not args.push:return
    for row in rows:
        public=ROOT/row['folder'];private=public.parents[1]/'private_push'/public.name
        print('PUSH',row['kernel'],'account-scoped;',row['session_hours'],'h boundary',flush=True)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/kaggle/push_nb.py'),str(private)],capture_output=True,text=True)
        print(result.stdout[-2000:],flush=True)
        if result.returncode:
            if (private/'PUSHED').exists(): (public/'PUSHED').write_bytes((private/'PUSHED').read_bytes())
            row['push_failed']=True;proof.write_text(json.dumps(record,indent=2)+'\n')
            raise RuntimeError('Push response failed/refused input; inspect exact status before retry')
        (public/'PUSHED').write_bytes((private/'PUSHED').read_bytes())
        row['pushed']=True;proof.write_text(json.dumps(record,indent=2)+'\n')
    assert active_user()==record['active_account']
    print('Prepared subset pushed:',len(rows),'; remote preparation/training not yet verified',flush=True)


if __name__=='__main__':main()
