"""Bounded new-method smoke, corrupted-packet guards and checkpoint resume."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
PROJ = ROOT/'papers/boundary-cells/proj'
sys.path.insert(0,str(PROJ))
sys.path.insert(0,str(ROOT))
from cells import packet, decode
from mh.dagsnet import build_model, CFG


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='initial1');args=parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1)
    token = bytes(range(16))
    good = packet(1,2,b'payload',token)
    assert decode(good,1,2,token) == b'payload'
    refusal = 0
    for bad,identity in [(good[:-1],token), (good[:-1]+b'X',token), (good,bytes(16)), (b'',token)]:
        try:
            decode(bad,1,2,identity)
        except AssertionError:
            refusal += 1
        else:
            raise AssertionError('invalid packet accepted')
    out = ROOT/'papers/boundary-cells/smoke'/args.tag
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    arms=['bc_single','bc_direct2_fl','bc_direct4_fl','bc_teacher2_fl','bc_teacher4_fl']
    for arm in arms:
        command=[sys.executable,str(PROJ/'boundary_driver.py'),'--arm',arm,'--smoke','--out',str(out)]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        run=out/f'{arm}_s42'
        result=json.loads((run/'result.json').read_text())
        assert result['complete'] and len(result['history'])==3
        assert all(len(row['clients'])==20 for row in result['history'])
        weights=torch.load(run/'weights/round_003.pt',map_location='cpu',weights_only=True)
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).eval()
        model.load_state_dict(weights['private'][0],strict=True)
        assert sum(p.numel() for p in model.parameters())==395024
        x=np.load(ROOT/'papers/pilot/fixture/group_X.npy',mmap_mode='r')[:32].astype('float32')
        with torch.no_grad():
            assert torch.isfinite(model(torch.from_numpy(x))[0]).all()
    # Full-vs-resume uses the real driver and its completed-round state.
    arm='bc_teacher4_fl'
    split=ROOT/'papers/boundary-cells/smoke'/f'{args.tag}-resume'
    base=[sys.executable,str(PROJ/'boundary_driver.py'),'--arm',arm,'--smoke','--out',str(split)]
    subprocess.run(base+['--stop_after','1'],cwd=ROOT,env=env,check=True)
    subprocess.run(base+['--resume'],cwd=ROOT,env=env,check=True)
    left=out/f'{arm}_s42'
    right=split/f'{arm}_s42'
    checks=[]
    for kind in ('weights','resume'):
        a=torch.load(left/kind/'round_003.pt',map_location='cpu',weights_only=True)
        b=torch.load(right/kind/'round_003.pt',map_location='cpu',weights_only=True)
        from scripts.check_v3_driver import equal
        assert equal(a,b), ('exact resume mismatch',kind)
        checks.append(kind)
    with np.load(left/'predictions/round_003.npz') as a, np.load(right/'predictions/round_003.npz') as b:
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    report=dict(stage='LOCAL_SMOKE_ONLY',arms=arms,rounds=3,clients=20,rows_per_client=256,
                packet_refusals=refusal,exact_resume=checks+['predictions'],
                utility_claim=False,T4_cost_claim=False,
                source_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROJ.glob('*.py')})
    (ROOT/'papers/boundary-cells'/f'local-proof-{args.tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BoundaryCells smoke/rebuild/exact-resume PASS',flush=True)


if __name__=='__main__':
    main()
