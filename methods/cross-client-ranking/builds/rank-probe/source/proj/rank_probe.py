"""Train-only probes on audited CE-r2 weights. Never write per-record logits or gradients."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from rank_math import C,encode_up,decode_up,aggregate,decode_down,gradient,projection,gates,packet

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def load_model(weights,k,device):
    model=build_model(dict(CFG,n_features=66,num_classes=C),latent=True).to(device).eval()
    model.load_state_dict(weights['private'][k],strict=True)
    assert sum(p.numel() for p in model.parameters())==395024
    return model


@torch.no_grad()
def predict(model,x,device):
    z=[];p=[]
    for start in range(0,len(x),4096):
        xx=torch.from_numpy(np.array(x[start:start+4096],dtype=np.float32,copy=True)).to(device)
        logits=model(xx)[0].float();prob=logits.softmax(1)
        assert torch.isfinite(logits).all() and torch.isfinite(prob).all()
        z.append(logits.cpu().numpy());p.append(prob.cpu().numpy())
    return np.concatenate(z),np.concatenate(p).astype('float64')


def scan(a,identity,weights):
    for k in range(a.shard,20,2):
        start=time.monotonic();model=load_model(weights,k,'cuda:0')
        x=np.load(a.input/f'train_{k}_X.npy',mmap_mode='r');y=np.load(a.input/f'train_{k}_y.npy')
        n=np.bincount(y,minlength=C).astype('uint64');counts=np.stack([n,len(y)-n])
        sums=np.zeros((4,C),dtype='float64')
        # Bounded batches; no per-record activation escapes this simulated client.
        for offset in range(0,len(y),4096):
            _,prob=predict(model,x[offset:offset+4096],'cuda:0')
            yy=y[offset:offset+4096];positive=np.arange(C)[None,:]==yy[:,None]
            sums[0]+=(prob*positive).sum(0);sums[1]+=(prob*~positive).sum(0)
            sums[2]+=((prob*prob)*positive).sum(0)
            sums[3]+=((prob*prob)*~positive).sum(0)
        (a.out/f'client_{k:02d}.bin').write_bytes(encode_up(counts,sums,identity))
        print('SCAN',a.seed,k,len(y),time.monotonic()-start,flush=True)
        del model;torch.cuda.empty_cache()


def probe(a,identity,weights):
    pooled,n_global,eligible,total=decode_down((a.out/'cohort.bin').read_bytes(),identity)
    for k in range(a.shard,20,2):
        n,sums,_=decode_up((a.out/f'client_{k:02d}.bin').read_bytes(),identity)
        own=np.divide(sums[:2],n,out=np.zeros((2,C)),where=n>0)
        support=eligible&(n>=32).all(0)
        x=np.load(a.input/f'train_{k}_X.npy',mmap_mode='r');y=np.load(a.input/f'train_{k}_y.npy')
        indexes=[];fit=[]
        for c in range(C):
            if not support[c]:continue
            available=np.flatnonzero(y==c)
            rng=np.random.default_rng(np.random.SeedSequence([1729,k,c]))
            chosen=rng.choice(available,min(256,len(available)),replace=False)
            indexes.extend(chosen.tolist());fit.extend([True]*(len(chosen)//2)+[False]*(len(chosen)-len(chosen)//2))
        classes=[None]*C
        if indexes:
            indexes=np.asarray(indexes);fit=np.asarray(fit,dtype=bool)
            model=load_model(weights,k,'cuda:0');logits,prob=predict(model,x[indexes],'cuda:0');yy=y[indexes]
            # Both rank terms use COMMON support, so missing-own classes cannot manufacture peer signal.
            gp=gradient(prob,yy,pooled,n_global,total,support)
            go=gradient(prob,yy,own,n,int(n[0].sum()),support)
            delta=gp-go;assert np.isfinite(delta).all()
            residual,fit_info=projection(prob,yy,logits.astype('float64'),delta,fit)
            for c in np.flatnonzero(support):
                selected=yy[~fit]==c
                classes[c]=dict(rows=int(selected.sum()),own_energy=float(np.mean(go[~fit][selected]**2)),
                    delta_energy=float(np.mean(delta[~fit][selected]**2)),residual_energy=float(np.mean(residual[selected]**2)))
            del model;torch.cuda.empty_cache()
        else:fit_info=dict(columns=32,retained_columns=0,fit_rows=0,test_rows=0)
        result=dict(client=k,classes=classes,projection=fit_info,scan_rows=len(y),
                    valid_scan_rows=int(n[0,eligible].sum()),support=support.tolist())
        save(a.out/f'client_{k:02d}_probe.json',result)
        print('PROBE',a.seed,k,'common classes',int(support.sum()),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['scan','aggregate','probe','finish'],required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((a.input/'protocol.json').read_text());assert protocol['seed']==a.seed
    identity=hashlib.sha256((a.input/'protocol.json').read_bytes()).digest()[:16]
    if a.phase=='aggregate':
        wire,record=aggregate([(a.out/f'client_{k:02d}.bin').read_bytes() for k in range(20)],identity)
        (a.out/'cohort.bin').write_bytes(wire)
        (a.out/'bootstrap.bin').write_bytes(packet((a.input/'protocol.json').read_bytes(),3,identity))
        save(a.out/'cohort.json',record);return
    if a.phase=='finish':
        _,_,eligible,_=decode_down((a.out/'cohort.bin').read_bytes(),identity)
        clients=[json.loads((a.out/f'client_{k:02d}_probe.json').read_text()) for k in range(20)]
        result=gates(clients,eligible)
        result.update(seed=a.seed,protocol=protocol,weights_sha=sha(a.input/'warm_weights.pt'),
            original_fixture_sha=sha(a.input/'manifest.json'),model_training=False,holdout_read=False,
            bytes=dict(up=sum((a.out/f'client_{k:02d}.bin').stat().st_size for k in range(20)),
                down=20*(a.out/'cohort.bin').stat().st_size,bootstrap=20*(a.out/'bootstrap.bin').stat().st_size))
        save(a.out/'probe_result.json',result);print('PROBE RESULT',a.seed,result['pass_probe'],flush=True);return
    assert torch.cuda.device_count()==1
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    weights=torch.load(a.input/'warm_weights.pt',map_location='cpu',weights_only=True)
    assert weights['round']==2 and len(weights['private'])==20 and len(weights['proxy'])==0
    assert weights['fingerprint']==protocol['warm_fingerprint']
    if a.phase=='scan':scan(a,identity,weights)
    else:probe(a,identity,weights)


if __name__=='__main__':main()
