"""Bounded smoke for every immutable BoundaryCells control branch; no utility claim."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
PROJ=ROOT/'papers/boundary-cells/proj'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROJ))
from mh import pilot as P
from boundary_method import ARMS,Method,private_epoch


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    args=SimpleNamespace(seed=42,private='dagsnet',smoke=True,lr=.001,lr_min=.00001,
        rounds=20,batch=512,aux_batch=64,arm='bc_single')
    run=P.Run(args);run.X=[x[:1024] for x in run.X];run.Y=[y[:1024] for y in run.Y]
    run.n=[len(y) for y in run.Y];run.counts=torch.stack([torch.bincount(y,minlength=16) for y in run.Y]).double()
    run.log_pi=((run.counts+1)/(run.counts+1).sum(1,keepdim=True)).log().float()
    results=[]
    for arm in ARMS:
        args.arm=arm;run.bytes=[]
        for model in run.priv:model.load_state_dict(run.init.state_dict(),strict=True)
        method=Method(run);method.prior();method.start_round()
        row=private_epoch(run,method,0,1)
        assert row['applied']>0 and row['skipped']<=16 and row['eligible_rows']<=1024
        assert torch.isfinite(method.mass).all() and torch.isfinite(run.log_pi).all()
        results.append(dict(arm=arm,client=0,rows=1024,applied=row['applied'],skipped=row['skipped'],
                            loss=row['loss'],eligible_rows=row['eligible_rows'],wire_events=len(run.bytes)))
        print('BRANCH PASS',arm,flush=True)
        del method;torch.cuda.empty_cache()
    output=dict(stage='LOCAL_BOUNDED_BRANCH_SMOKE',branches=results,cohort_clients=20,
        trained_clients_per_branch=1,rounds=1,rows_per_client=1024,utility_claim=False,T4_cost_claim=False,
        source_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROJ.glob('*.py')})
    (ROOT/'papers/boundary-cells/branch-proof-initial2.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with (ROOT/'papers/.artifact-verification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
