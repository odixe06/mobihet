"""V3 projection arithmetic on saved real dense update; no classifier training."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG
from mh.v2_math import Layout
from mh.v3_math import coordinates,basis,matrix,unpad,orthogonal,pack,decode,ContextSum


def main():
    torch.set_num_threads(1); torch.manual_seed(901)
    layout=Layout(build_model(dict(CFG,n_features=66,num_classes=16),latent=True))
    path=ROOT/'papers/v2/runs/a_dense_s42/resume/round_001.pt'
    real=torch.load(path,map_location='cpu',weights_only=True)['algorithm']['last_mean']
    assert real.numel()==395024 and torch.isfinite(real).all()
    records=[]
    token=bytes(range(16))
    packet=pack(real[:512],token,count=1)
    for bad in (packet[:-1],bytes([packet[0]^1])+packet[1:]):
        try: decode(bad,expected_context=token)
        except AssertionError: pass
        else: raise AssertionError('framecut/context mismatch accepted')
    for k in (512,24576):
        for rolling in (False,True):
            idx=coordinates(layout,k,1,rolling); assert len(torch.unique(idx))==k
            packet=pack(real[idx],token,count=1)
            decoded=decode(packet,expected_context=token)['values']; restored=torch.zeros_like(real); restored[idx]=decoded
            error=real-restored
            assert error.norm()<=real.norm()*(1+1e-5)+2e-5
            assert torch.equal(restored[idx],real[idx]) and torch.count_nonzero(error[idx])==0
            records.append(dict(kind='rolling' if rolling else 'random',k=k,up=len(packet),down=48+k*4,
                                retained_squared=float(restored.square().sum()/real.square().sum())))
    candidates=[real+.001*torch.randn_like(real) for _ in range(5)]
    counts=[3,7,11,17,19]
    average=sum(v*n for v,n in zip(candidates,counts))/sum(counts)
    for rank in (1,8,32):
        q=basis(rank,1,'cpu'); first=ContextSum(token,0,512*rank,'cpu')
        for k,(v,n) in enumerate(zip(candidates,counts)):
            first.add(k,pack((matrix(v)@q).reshape(-1)*n,token,count=n))
        total,n=first.finish(); mean=total/n
        assert torch.allclose(mean.reshape(512,rank),matrix(average)@q,atol=2e-5,rtol=1e-5)
        reconstructed=unpad(mean.reshape(512,rank)@q.T)
        assert (average-reconstructed).norm()<=average.norm()*(1+1e-5)+2e-5
        left=orthogonal(mean.reshape(512,rank)); second=ContextSum(token,0,772*rank,'cpu')
        residuals=[]
        for k,(v,n) in enumerate(zip(candidates,counts)):
            packet=pack((matrix(v).T@left).reshape(-1)*n,token,count=n)
            decoded=decode(packet,expected_context=token)['values']/n
            reconstructed_i=unpad(left@decoded.reshape(772,rank).T)
            residuals.append(v-reconstructed_i)
            assert residuals[-1].norm()<=v.norm()*(1+1e-5)+2e-5
            second.add(k,packet)
        total,n=second.finish(); projected=unpad(left@(total/n).reshape(772,rank).T)
        assert torch.allclose(projected,unpad(left@left.T@matrix(average)),atol=2e-5,rtol=1e-5)
        assert torch.allclose(sum(e*n for e,n in zip(residuals,counts))/sum(counts),average-projected,atol=2e-5,rtol=1e-5)
        records.append(dict(kind='orthogonal',rank=rank,up=56+512*rank*4,down=48+512*rank*4,
            residual_fraction=float((average-reconstructed).norm()/average.norm())))
        records.append(dict(kind='power',rank=rank,application_bytes=208+8*(512+772)*rank,
            residual_fraction=float((average-projected).norm()/average.norm())))
    out=ROOT/'papers/v3'; out.mkdir(exist_ok=True)
    (out/'math-probe.json').write_text(json.dumps(dict(source=str(path.relative_to(ROOT)),
        evidence='real saved dense aggregate update; arithmetic only, no F1 or client training',checks=records),indent=2))
    print('PASS V3 real-update linearity, decoded residual, orthogonality, two-stage projection and real bytes')


if __name__=='__main__': main()
