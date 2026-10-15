"""Explicit pairwise gradients, malformed wire, and held-out calibration projection."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'source/proj'))
from rank_math import C,gradient,encode_up,decode_up,aggregate,decode_down,projection

torch.set_num_threads(1);torch.manual_seed(1729)
z=torch.randn(9,3,dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,1,2,0,2,2,0,1])
p=z.softmax(1);risk=z.sum()*0
for c in range(3):
    risk+=((1-p[y==c,c,None]+p[None,y!=c,c])**2).mean()/3*.25
exact=torch.autograd.grad(risk,z)[0].numpy()
means=np.stack([[float(p[y==c,c].detach().mean()) for c in range(3)],
                [float(p[y!=c,c].detach().mean()) for c in range(3)]])
counts=np.array([[3]*3,[6]*3])
actual=gradient(p.detach().numpy(),y.numpy(),means,counts,9,np.ones(3,dtype=bool))/9
error=float(np.max(np.abs(exact-actual)));assert error<1e-12
for start,stop in ((0,2),(2,5),(5,9)):
    block=z.detach()[start:stop].clone().requires_grad_(True);pp=block.softmax(1)
    positive=torch.arange(3)[None,:]==y[start:stop,None]
    mu=torch.from_numpy(means);nn=torch.from_numpy(counts)
    surrogate=.25*9/3*torch.where(positive,(pp-1-mu[1])**2/nn[0],(pp+1-mu[0])**2/nn[1]).sum(1).mean()
    block_gradient=torch.autograd.grad(surrogate,block)[0].numpy()
    assert np.allclose(block_gradient,exact[start:stop]*9/(stop-start),atol=1e-12,rtol=1e-8)
eps=1e-5;numeric=np.zeros_like(exact)
for i in range(9):
    for c in range(3):
        plus=z.detach().clone();minus=plus.clone();plus[i,c]+=eps;minus[i,c]-=eps
        def value(logits):
            q=logits.softmax(1)
            return sum(((1-q[y==j,j,None]+q[None,y!=j,j])**2).mean()/3*.25 for j in range(3))
        numeric[i,c]=float((value(plus)-value(minus))/(2*eps))
assert np.max(np.abs(numeric-exact))<1e-8
missing=gradient(np.array([[1.,0.,0.]]),np.array([0]),np.zeros((2,3)),np.zeros((2,3)),1,np.zeros(3,dtype=bool))
assert np.array_equal(missing,np.zeros_like(missing))
identity=bytes(range(16));n=np.stack([np.full(C,128,dtype=np.uint64),np.full(C,128*15,dtype=np.uint64)])
sums=np.stack([n[0]*.2,n[1]*.1,n[0]*.04,n[1]*.01])
wire=encode_up(n,sums,identity);assert len(wire)==928
restored=decode_up(wire,identity);assert np.array_equal(restored[0],n)
down,cohort=aggregate([wire]*5,identity);assert len(down)==746
mu,counts,eligible,total=decode_down(down,identity);assert eligible.all() and total==5*128*C
broken=bytearray(wire);broken[-1]^=1
try:decode_up(bytes(broken),identity)
except AssertionError:pass
else:raise AssertionError('corrupt payload accepted')
rng=np.random.default_rng(19);logits=rng.normal(size=(400,3));prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
labels=np.arange(400)%3;fit=np.arange(400)%2==0
delta=np.broadcast_to(np.array([.2,-.1,-.1]),prob.shape).copy()
residual,info=projection(prob,labels,logits,delta,fit)
assert np.mean(residual**2)<1e-20 and info['columns']==6
from rank_math import gates
clients=[dict(scan_rows=100,valid_scan_rows=100,classes=[dict(own_energy=1.,delta_energy=.01,residual_energy=0.) for _ in range(C)]) for _ in range(5)]
assert not gates(clients,np.ones(C,dtype=bool))['pass_probe']
for client in clients:
    for item in client['classes']:item['residual_energy']=.005
assert gates(clients,np.ones(C,dtype=bool))['pass_probe']
proof=dict(pass_check=True,gradient_max_abs_error=error,finite_difference_max_abs_error=float(np.max(np.abs(numeric-exact))),
    wire_sizes=[len(wire),len(down)],missing_support=True,corruption_refused=True,calibration_projection_refused=True,
    scope='generated algebra/projection only; no SGD or production/utility evidence')
(BASE/'probes/local-preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
print('CrossClientRank algebra/wire/projection PASS',error)
