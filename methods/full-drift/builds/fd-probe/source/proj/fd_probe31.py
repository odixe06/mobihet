"""Local observability diagnostic, not generation/training: private logits inside teacher-feature nullspace."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fd_probe23 import C,P,basis,fd,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--fixture',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();start=time.monotonic();fd.deterministic();torch.set_num_threads(2);dev=torch.device('cuda')
    # Tiny finite differences require full-mantissa convolution, not a TF32 forward with an FP32 analytic derivative.
    torch.backends.cudnn.allow_tf32=False
    fx=Path(a.fixture);manifest=json.loads((fx/'manifest.json').read_text())
    assert manifest['train_frac']==.025 and manifest['group_overlap']==0
    zero=[manifest['features'].index(n) for n in fd.BLOCK+fd.POSITION]
    active=torch.tensor([j for j in range(66) if j not in zero],device=dev);assert len(active)==53
    weights=torch.load(a.weights,map_location=dev,weights_only=True)
    def model(state):
        m=P.make('dagsnet').to(dev);m.load_state_dict(state,strict=True);return m.eval().requires_grad_(False)
    teacher=model(weights['proxy'][0]);private=[model(s) for s in weights['private']];assert len(private)==20
    projection=basis(teacher.head[5].weight,48).float()
    def expand(x):
        full=torch.zeros(1,66,device=dev);full[:,active]=x;return full
    def latent(x):return (teacher.head[:4](teacher.combined(expand(x)))@projection)[0]
    def logits(m,x):
        z=m.head(m.combined(expand(x)))[0];return z-z.mean()
    out=dict(scope='16 class-stratified train points, local FP32 Jacobians/finite perturbations; not population/F1 or R1/R2 proof',
             weights_sha=sha(a.weights),fixture_sha=sha(fx/'manifest.json'),source_sha=sha(__file__),
             protocol=dict(rank_relative_threshold=1e-5,rank_absolute_floor=1e-8,epsilons=[.01,.001],seed=31,
                           matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),points=[])
    rng=np.random.default_rng(31)
    for c in range(C):
        key=f'train_{c}';xp=fx/f'{key}_X.npy';yp=fx/f'{key}_y.npy'
        assert sha(xp)==manifest['files'][xp.name] and sha(yp)==manifest['files'][yp.name]
        y=np.load(yp,mmap_mode='r');index=int(rng.choice(np.where(y==c)[0]));row=np.array(np.load(xp,mmap_mode='r')[index],dtype='float32',copy=True)
        x=torch.from_numpy(row).to(dev)[active]
        jt=torch.autograd.functional.jacobian(latent,x,vectorize=True).double()
        _,s,vh=torch.linalg.svd(jt,full_matrices=True)
        threshold=max(1e-8,1e-5*float(s.max()));rank=int((s>threshold).sum())
        null=torch.eye(53,device=dev,dtype=torch.float64)-vh[:rank].T@vh[:rank]
        point=dict(true_class=c,source_client=c,row=index,teacher_rank=rank,teacher_null_dimension=53-rank,clients={})
        for k,m in enumerate(private):
            jp=torch.autograd.functional.jacobian(lambda z:logits(m,z),x,vectorize=True).double()
            sensitive=jp@null;_,_,v=torch.linalg.svd(sensitive,full_matrices=False)
            direction=null@v[0];direction=direction/direction.norm().clamp_min(1e-30)
            linear_teacher=float((jt@direction).norm()/jt.norm().clamp_min(1e-30))
            assert linear_teacher<=2e-5,linear_teacher
            result=dict(private_null_energy_fraction=float(sensitive.square().sum()/jp.square().sum().clamp_min(1e-30)),
                        teacher_linear_relative_response=linear_teacher,finite={})
            with torch.no_grad():
                for epsilon in (.01,.001):
                    plus=x+epsilon*direction.float();minus=x-epsilon*direction.float()
                    dt=(latent(plus)-latent(minus)).double();dp=(logits(m,plus)-logits(m,minus)).double()
                    result['finite'][str(epsilon)]=dict(teacher_normalized_response=float(dt.norm()/(2*epsilon*jt.norm()).clamp_min(1e-30)),
                                                      private_normalized_response=float(dp.norm()/(2*epsilon*jp.norm()).clamp_min(1e-30)))
            point['clients'][k]=result
        out['points'].append(point);out['seconds']=time.monotonic()-start
        Path(a.out).write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
        print(round(out['seconds'],1),'class',c,'teacher rank',rank,'null',53-rank,flush=True)


if __name__=='__main__':main()
