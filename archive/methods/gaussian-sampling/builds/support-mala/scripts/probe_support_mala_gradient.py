"""Small FP32 DAGSNet mixed-derivative feasibility probe; no pilot training."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG


def main():
    torch.set_num_threads(1); torch.manual_seed(4201)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device('cuda'); model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(dev).eval()
    fx=ROOT/'papers/pilot/fixture'
    xx=np.load(fx/'train_0_X.npy',mmap_mode='r'); yy=np.load(fx/'train_0_y.npy',mmap_mode='r')
    idx=np.flatnonzero(yy==0)[:32]; x=torch.tensor(np.array(xx[idx]),device=dev,dtype=torch.float32)
    y=torch.zeros(len(x),device=dev,dtype=torch.long); params=list(model.parameters())
    real=torch.autograd.grad(F.cross_entropy(model(x)[0],y),params)
    results=[]
    for ipc in (1,4):
        u=torch.randn(ipc,66,device=dev,requires_grad=True); opt=torch.optim.Adam([u],lr=.03)
        times=[]; norms=[]; losses=[]
        for step in range(4):
            torch.cuda.synchronize(); start=time.perf_counter()
            syn=torch.autograd.grad(F.cross_entropy(model(u)[0],y[:ipc]),params,create_graph=True)
            terms=[1-F.cosine_similarity(a.flatten(),b.detach().flatten(),dim=0,eps=1e-8)
                   for a,b in zip(syn,real) if b.ndim>=2]
            loss=torch.stack(terms).mean(); grad,=torch.autograd.grad(loss,u)
            assert torch.isfinite(loss) and torch.isfinite(grad).all()
            opt.zero_grad(set_to_none=True); u.grad=grad; opt.step()
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            norms.append(float(grad.norm())); losses.append(float(loss.detach()))
        results.append(dict(ipc=ipc,steps=4,loss=losses,input_gradient_norm=norms,
                            step_seconds=times,mean_warm_seconds=float(np.mean(times[1:]))))
    out=ROOT/'papers/v4'; out.mkdir(exist_ok=True)
    (out/'gradient-probe.json').write_text(json.dumps(dict(evidence='32 local train rows, FP32 eval DAGSNet, four mixed-derivative steps per size; no F1 or T4 speed claim',
        torch=torch.__version__,device=torch.cuda.get_device_name(),checks=results),indent=2))
    print('PASS mixed second derivatives, IPC1/4; cost gate still requires production T4 timing')


if __name__=='__main__': main()
