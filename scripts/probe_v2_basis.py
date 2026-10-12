"""Four local batches at actual approved K; shape/numeric/runtime probe, never pilot F1."""
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import CFG,build_model
from mh.v2_math import SeedBasis


def main():
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    fx=ROOT/'papers/pilot/fixture'
    x=torch.from_numpy(np.load(fx/'train_0_X.npy')[:32].astype('float32')).cuda()
    y=torch.from_numpy(np.load(fx/'train_0_y.npy')[:32].astype('int64')).cuda(); rows=[]
    for k in (512,1024):
        torch.manual_seed(42); model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).cuda().eval()
        torch.cuda.reset_peak_memory_stats(); t0=time.time(); basis=SeedBasis(model,k,1901,'cuda')
        torch.cuda.synchronize(); setup=time.time()-t0
        a=torch.zeros(k,device='cuda',requires_grad=True); opt=torch.optim.AdamW([a],lr=.1,weight_decay=0.)
        losses=[]; t1=time.time()
        for _ in range(4):
            z=basis.forward(model,a,x)[0]; loss=F.cross_entropy(z.float(),y)
            assert torch.isfinite(loss); losses.append(float(loss.detach())); loss.backward()
            assert torch.isfinite(a.grad).all()
            torch.nn.utils.clip_grad_norm_([a],1); opt.step(); opt.zero_grad(set_to_none=True)
        final=float(F.cross_entropy(basis.forward(model,a,x)[0].float(),y).detach())
        assert final<losses[0],(k,losses,final)
        torch.cuda.synchronize()
        row=dict(k=k,rows=32,steps=4,setup_sec=setup,step_sec=(time.time()-t1)/4,
                 initial_loss=losses[0],final_loss=final,coefficient_norm=float(a.detach().norm()),
                 peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        rows.append(row); print(json.dumps(row),flush=True)
        del a,opt,basis,model; gc.collect(); torch.cuda.empty_cache()
    target=ROOT/'papers/v2/basis-probe.json'; target.write_text(json.dumps(rows,indent=2))
    print('PASS production-basis local numeric probe; no performance/gate claim')


if __name__=='__main__': main()
