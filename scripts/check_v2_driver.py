"""Exact V2 codepath smoke, interrupted resume equivalence, and artifact tamper rejection."""
import json
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run(arm,out,extra=()):
    cmd=[sys.executable,'-m','mh.v2_run','--arm',arm,'--smoke','--out',str(out),*extra]
    subprocess.run(cmd,cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',default='deterministic')
    args=p.parse_args()
    base=ROOT/'papers/v2/smoke'/args.tag; reference=ROOT/'papers/v2/resume_reference'/args.tag
    # All branches; small/medium and no-EF have identical driver contracts but different compressors.
    arms=['local','diag','a_own','a_dense','a_small_noef','a_small_ef','a_medium_noef','a_medium_ef',
          'b_dense','b_own512','b_seed512','b_own1024','b_seed1024','c_own','c_fl','c_shuffle']
    for arm in arms:
        if (base/f'{arm}_s42/result.json').exists():
            raise SystemExit('smoke outputs already exist; use a fresh directory, do not overwrite')
        run(arm,base)
    for arm in ('a_small_ef','b_seed512','c_fl'):
        run(arm,reference,('--stop_after','1'))
        run(arm,reference,('--resume',))
    # Import torch only AFTER subprocess smoke so parent holds no CUDA context/model memory.
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
    import torch
    from verify_v2 import verify
    for arm in arms: verify(base/f'{arm}_s42',infer=True)
    for arm in ('a_small_ef','b_seed512','c_fl'):
        for r in range(1,4):
            left=torch.load(base/f'{arm}_s42/weights/round_{r:03d}.pt',weights_only=True)
            right=torch.load(reference/f'{arm}_s42/weights/round_{r:03d}.pt',weights_only=True)
            for role in ('private','proxy'):
                for a,b in zip(left[role],right[role]):
                    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),(arm,r,role)
            l=torch.load(base/f'{arm}_s42/resume/round_{r:03d}.pt',weights_only=True)['algorithm']
            q=torch.load(reference/f'{arm}_s42/resume/round_{r:03d}.pt',weights_only=True)['algorithm']
            for key in l:
                if isinstance(l[key],torch.Tensor): assert torch.equal(l[key],q[key]),(arm,r,key)
        verify(reference/f'{arm}_s42',infer=True)
    target=base/'a_small_ef_s42/result.json'; original=target.read_text(); changed=json.loads(original)
    changed['history'][-1]['eval']['private']['group']['gp']['f1_macro']+=.1
    target.write_text(json.dumps(changed))
    try:
        try: verify(target.parent)
        except AssertionError: pass
        else: raise AssertionError('tampered metrics accepted')
    finally: target.write_text(original)
    print('PASS V2: all16 arms3rounds + exact A/B/C interrupted resume + decisive inference + tamper rejection')


if __name__=='__main__': main()
