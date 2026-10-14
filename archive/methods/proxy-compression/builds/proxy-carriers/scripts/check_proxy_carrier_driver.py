"""All V3 small driver paths; nested tensor exact-resume and actual artifact verification."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT))
from mh.v3_math import ARMS


def run(arm,out,extra=()):
    subprocess.run([sys.executable,'-m','mh.v3_run','--arm',arm,'--smoke','--out',str(out),*extra],cwd=ROOT,check=True)


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',default='final')
    p.add_argument('--arms',nargs='+',choices=ARMS); args=p.parse_args(); arms=args.arms or ARMS
    base=ROOT/'papers/v3/smoke'/args.tag; resume=ROOT/'papers/v3/resume_reference'/args.tag
    for arm in arms:
        assert not (base/f'{arm}_s42').exists(),'use fresh smoke tag'
        run(arm,base)
    representatives=['a_r_small_ef','a_t_medium_train','a_o_medium_ef','a_p_medium_ef']
    for arm in representatives:
        run(arm,resume,('--stop_after','1')); run(arm,resume,('--resume',))
    import torch; torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'scripts')); from verify_v3 import verify
    for arm in arms: verify(base/f'{arm}_s42',infer=True)
    for arm in representatives:
        for r in range(1,4):
            for kind in ('weights','resume'):
                left=torch.load(base/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                right=torch.load(resume/f'{arm}_s42/{kind}/round_{r:03d}.pt',weights_only=True)
                assert equal(left,right),(arm,r,kind)
        verify(resume/f'{arm}_s42',infer=True)
    target=base/'a_r_small_ef_s42/knowledge/round_003.bin'; original=target.read_bytes()
    target.write_bytes(original[:-1]+bytes([original[-1]^1]))
    try:
        try: verify(target.parent.parent)
        except AssertionError: pass
        else: raise AssertionError('tampered payload accepted')
    finally: target.write_bytes(original)
    print('PASS V3',len(arms),'arms x3rounds; exact4-carrier resume; inference; tamper rejection')


if __name__=='__main__': main()
