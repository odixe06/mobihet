"""Two full 80-slot local books on at most 512 producer rows, no private training."""
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG
from mh.v5_books import Method


def main():
    torch.set_num_threads(1); torch.manual_seed(42); dev=torch.device('cuda')
    fx=ROOT/'papers/pilot/fixture'; X=np.load(fx/'train_0_X.npy',mmap_mode='r'); Y=np.load(fx/'train_0_y.npy',mmap_mode='r')
    indices=np.concatenate([np.flatnonzero(Y==c)[:32] for c in range(16)])
    x=torch.tensor(np.array(X[indices]),device=dev).float(); y=torch.tensor(np.array(Y[indices]),device=dev).long()
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(dev)
    mean=torch.stack([x[y==c].mean(0) for c in range(16)])
    sd=torch.stack([x[y==c].var(0,correction=0).clamp_min(1e-4).sqrt() for c in range(16)])
    checks=[]
    for ipc in (1,4):
        run=SimpleNamespace(init=model,priv=[copy.deepcopy(model)],dev=dev,K=1,bytes=[],X=[x],Y=[y],n=[len(y)],
            a=SimpleNamespace(arm=f'g{ipc}_grad_trained',smoke=True,seed=42))
        run.gen=lambda *keys: torch.Generator(device=dev).manual_seed(int(np.random.SeedSequence([42,*keys]).generate_state(1)[0]))
        method=Method(run,initialize=False); method.mu=mean.half().float(); method.sd=sd.half().float()
        torch.cuda.synchronize(); start=time.perf_counter(); book,diag=method.condense(0,1,steps=80); torch.cuda.synchronize()
        assert diag['slots']==diag['applied']==80 and torch.isfinite(book).all()
        checks.append(dict(ipc=ipc,rows=len(y),seconds=time.perf_counter()-start,fit_first16=diag['fit'][:16],
            fit_last16=diag['fit'][-16:],parameter_hash=diag['encoder_parameter_sha256']))
        del method,run; torch.cuda.empty_cache()
    (ROOT/'papers/v5/book80-probe.json').write_text(json.dumps(dict(evidence='at most 32 real producer rows/class, DAGSNet init, local class Gaussian, FP32 80-slot books; no F1/current-trained/T4 timing claim',checks=checks),indent=2))
    print('PASS two full 80-slot FP32 books IPC1/4 on bounded producer records')


if __name__=='__main__': main()
