"""One read-only startup pass: exact batch status, W&B records, both-rank compile/T4 log evidence."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.metrics import TEN
from scripts.watch_kaggle_batch import query


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--entity',required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    batch=json.loads(a.batch.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:statuses=list(pool.map(query,batch))
    import wandb
    api=wandb.Api(timeout=30);rows=[]
    for spec,status in zip(batch,statuses):
        record={'arm':spec['arm'],'ref':status['ref'],'kaggle_status':status['status'],'run_path':f'{a.entity}/mobihet-v4/{spec["slug"]}','ready':False}
        try:
            run=api.run(record['run_path']);summary=dict(run.summary)
            metrics={k:summary.get('private/'+k) for k in TEN}
            record.update(wandb_state=run.state,round=summary.get('round'),world=run.config.get('world'),backend=run.config.get('backend'),clients=run.config.get('clients'),metrics=metrics,skip=summary.get('skipped_frac'),last_timestamp=summary.get('_timestamp'))
            valid=all(isinstance(v,(int,float)) and np.isfinite(v) for v in metrics.values())
            if valid and record['world']==2 and record['backend']=='compiled' and record['clients']==5343:
                log=a.out/(spec['slug']+'.startup.log')
                if not log.exists():
                    cmd=['timeout','--foreground','--kill-after=5s','30s',sys.executable,str(ROOT/'.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'),spec['account'],'--',sys.executable,str(ROOT/'scripts/kaggle_cli_bounded.py'),'kernels','logs',f'{spec["account"]}/{spec["slug"]}/{spec["version"]}','--follow']
                    with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                    log.chmod(0o600)
                text=log.read_text(errors='replace')
                record['two_T4_logged']="['Tesla T4', 'Tesla T4']" in text
                record['compile_rank0']='compile gate rank0: PASS' in text
                record['compile_rank1']='compile gate rank1: PASS' in text
                record['ready']=record['two_T4_logged'] and record['compile_rank0'] and record['compile_rank1']
                if not record['ready']:log.unlink() # retry a bounded read next local-worker cycle, never model polling
        except Exception as exc:record['error_type']=type(exc).__name__
        rows.append(record)
    result={'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'all_ready':all(r['ready'] for r in rows),'runs':rows,'scope':'startup health only; final source/CM/snapshot audit still required'}
    p=a.out/'startup.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
    print('STARTUP_PASS' if result['all_ready'] else 'STARTUP_PENDING',sum(r['ready'] for r in rows),'/',len(rows),flush=True)


if __name__=='__main__':main()
