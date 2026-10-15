"""Actual rank autograd/NNLS/wire, bounded synthetic DAGSNet SGD and exact next-round state replay."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from rank_math import C,gradient,basis
from rank_method import Method,rank_loss,nnls_class,frame,unframe,private_epoch
from mh.dagsnet import build_model,CFG
from scripts.check_v3_driver import equal


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(1729);z=torch.tensor(rng.normal(size=(160,C)),requires_grad=True)
    y=np.arange(160)%C;p=z.softmax(1);means=rng.random((2,C));counts=np.full((2,C),100.)
    coef=.25*1600/C/counts
    loss=rank_loss(p,torch.as_tensor(y),torch.tensor(means),torch.tensor(coef));loss.backward()
    expected=gradient(p.detach().numpy(),y,means,counts,1600,np.ones(C,dtype=bool))/len(y)
    error=float(np.max(np.abs(expected-z.grad.numpy())));assert error<1e-12
    target=rng.normal(size=(160,C));design=basis(p.detach().numpy(),y,z.detach().numpy())[:,:,:C].reshape(-1,C)
    beta=nnls_class(p.detach().numpy(),y,target)
    assert np.allclose(beta,np.clip(nnls(design,target.ravel())[0],0,4),rtol=1e-10,atol=1e-10)
    key=b'x'*16;wire=frame(b'payload',3,key);assert unframe(wire,3,key)==b'payload'
    for w,r,k in [(wire,4,key),(wire,3,b'y'*16),(wire[:-1]+b'x',3,key)]:
        try:unframe(w,r,k)
        except AssertionError:pass
        else:raise AssertionError('stale/wrong-identity/corrupt frame accepted')
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def make():
        torch.manual_seed(1729);models=[build_model(dict(CFG,n_features=66,num_classes=C),latent=True).cuda() for _ in range(5)]
        x=[torch.randn(512,66,device='cuda') for _ in range(5)];ys=[torch.arange(512,device='cuda')%C for _ in range(5)]
        run=SimpleNamespace(K=5,priv=models,X=x,Y=ys,n=[512]*5,dev=torch.device('cuda'),bytes=[],
            counts=torch.full((5,C),32.,dtype=torch.float64,device='cuda'),fingerprint=hashlib.sha256(b'fixture').hexdigest(),
            a=SimpleNamespace(arm='cr_fl',seed=42,batch=256,clip=1.),lr=lambda r:1e-4,
            gen=lambda *k:torch.Generator(device='cuda').manual_seed(int(np.random.SeedSequence([42,*k]).generate_state(1)[0])))
        run.true_log_pi=torch.full((5,C),1/C,device='cuda').log();run.log_pi=run.true_log_pi.clone()
        return run
    run=make();m=Method(run);m.prior();m.scan(2)
    assert m.support.all()
    for k in range(run.K):private_epoch(run,m,k,3)
    m.aggregate(3)
    state=copy.deepcopy(m.state());weights=copy.deepcopy([v.state_dict() for v in run.priv]);prior=run.log_pi.clone()
    buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);restored=torch.load(buffer,weights_only=True)
    for k in range(run.K):private_epoch(run,m,k,4)
    m.aggregate(4);expected_weights=[copy.deepcopy(v.state_dict()) for v in run.priv];expected_state=copy.deepcopy(m.state())
    replay=make()
    for model,sd in zip(replay.priv,weights):model.load_state_dict(sd,strict=True)
    other=Method(replay,initialize=False)
    for k,v in restored.items():setattr(other,k,v)
    for k in range(replay.K):private_epoch(replay,other,k,4)
    other.aggregate(4)
    assert equal(expected_weights,[v.state_dict() for v in replay.priv]) and equal(expected_state,other.state())
    assert torch.equal(prior,run.log_pi),'rank changed fixed deployment prior'
    proof=dict(rank_gradient_error=error,nnls_matches_independent_solver=True,frame_refusals=True,
        bounded_synthetic_clients=5,rows_per_client=512,actual_dagsnet_sgd=True,exact_next_round_state=True,
        fixed_prior=True,pass_check=True,scope='Local synthetic fixture only; not production training, T4 cost or utility')
    (BASE/'probes/local-preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print('Rank utility local PASS',proof)


if __name__=='__main__':main()
