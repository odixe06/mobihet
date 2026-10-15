"""Reproduce frozen probe projection; emit aggregate fit/test diagnostics, never records/gradients."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from rank_probe import load_model,predict
from rank_math import C,decode_up,decode_down,gradient,basis,projection


def measure(prob,y,logits,delta,fit):
    design=basis(prob,y,logits);x=design[fit].reshape(-1,32);target=delta[fit].ravel()
    scale=np.sqrt((x*x).mean(0));keep=scale>1e-12
    coeff,_,rank,singular=np.linalg.lstsq(x[:,keep]/scale[keep],target,rcond=1e-10)
    predicted=np.einsum('nkd,d->nk',design[:,:,keep]/scale[keep],coeff)
    residual=delta-predicted
    frozen,_=projection(prob,y,logits,delta,fit)
    assert np.allclose(residual[~fit],frozen,rtol=1e-10,atol=1e-12)
    def energy(mask):
        return dict(rows=int(mask.sum()),delta=float(np.mean(delta[mask]**2)),
                    predicted=float(np.mean(predicted[mask]**2)),residual=float(np.mean(residual[mask]**2)))
    classes=[]
    for c in range(C):
        if not np.any(y==c):classes.append(None);continue
        item={name:energy(mask&(y==c)) for name,mask in [('fit',fit),('test',~fit)]}
        e=np.sort((residual[(~fit)&(y==c)]**2).sum(1))[::-1]
        item['test_residual_top1_fraction']=float(e[0]/e.sum()) if e.sum()>0 else 0.
        item['test_residual_top5_fraction']=float(e[:5].sum()/e.sum()) if e.sum()>0 else 0.
        classes.append(item)
    original=np.zeros(32);original[keep]=coeff/scale[keep]
    return dict(rank=int(rank),columns=int(keep.sum()),fit_scale=scale.tolist(),
        coefficients_normalized=coeff.tolist(),coefficients_original=original.tolist(),
        singular_values=singular.tolist(),condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        fit=energy(fit),test=energy(~fit),classes=classes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    import hashlib
    identity=hashlib.sha256((a.input/'protocol.json').read_bytes()).digest()[:16]
    replay=json.loads((a.out/'replay.json').read_text());pooled,ng,eligible,total=decode_down(bytes.fromhex(replay['cohort']),identity)
    assert torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    weights=torch.load(a.input/'warm_weights.pt',map_location='cpu',weights_only=True)
    assert weights['round']==2 and len(weights['private'])==20 and not weights['proxy']
    for k in range(a.shard,20,2):
        n,s,_=decode_up(bytes.fromhex(replay['up'][k]),identity);own=np.divide(s[:2],n,out=np.zeros((2,C)),where=n>0)
        support=eligible&(n>=32).all(0);x=np.load(a.input/f'train_{k}_X.npy',mmap_mode='r');y=np.load(a.input/f'train_{k}_y.npy')
        indexes=[];fit=[]
        for c in np.flatnonzero(support):
            available=np.flatnonzero(y==c);rng=np.random.default_rng(np.random.SeedSequence([1729,k,int(c)]))
            chosen=rng.choice(available,min(256,len(available)),replace=False);indexes.extend(chosen.tolist())
            fit.extend([True]*(len(chosen)//2)+[False]*(len(chosen)-len(chosen)//2))
        assert indexes,'original probe had no common support'
        indexes=np.asarray(indexes);fit=np.asarray(fit,dtype=bool);yy=y[indexes]
        model=load_model(weights,k,'cuda:0');z,prob=predict(model,x[indexes],'cuda:0')
        go=gradient(prob,yy,own,n,int(n[0].sum()),support);delta=gradient(prob,yy,pooled,ng,total,support)-go
        diag=measure(prob,yy,z.astype('float64'),delta,fit)
        for c,item in enumerate(diag['classes']):
            reference=replay['clients'][k]['classes'][c]
            assert (item is None)==(reference is None)
            if item is not None:
                selected=(~fit)&(yy==c)
                for name,value in [('own_energy',float(np.mean(go[selected]**2))),('delta_energy',item['test']['delta']),('residual_energy',item['test']['residual'])]:
                    assert np.isclose(value,reference[name],rtol=1e-6,atol=1e-18),('original energy mismatch',k,c,name,value,reference[name])
        diag.update(client=k,seed=a.seed,original_energies_reproduced=True)
        (a.out/f'diagnostic_{k:02d}.json').write_text(json.dumps(diag,indent=2,allow_nan=False)+'\n')
        print('DIAGNOSTIC',a.seed,k,'condition',diag['condition'],'rank',diag['rank'],flush=True)
        del model;torch.cuda.empty_cache()


if __name__=='__main__':main()
