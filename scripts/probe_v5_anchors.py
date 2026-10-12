"""Small train-only kernel coverage for both anchor budgets/all proposed bandwidths."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.v2_math import pack_vector,unpack


def main():
    torch.set_num_threads(1); dev='cuda'; base=ROOT/'papers/v5'; base.mkdir(exist_ok=True)
    resume=torch.load(ROOT/'papers/v2/runs/c_fl_s42/resume/round_020.pt',map_location=dev,weights_only=True)
    mu=resume['algorithm']['mu']; sd=resume['algorithm']['sd']
    mean=mu.mean(0); scale=((sd.square()+mu.square()).mean(0)-mean.square()).clamp_min(1e-4).sqrt()
    data=[]; labels=[]; fx=ROOT/'papers/pilot/fixture'
    for k in range(20):
        X=np.load(fx/f'train_{k}_X.npy',mmap_mode='r'); Y=np.load(fx/f'train_{k}_y.npy',mmap_mode='r')
        idx=np.random.default_rng(4901+k).choice(len(Y),min(2048,len(Y)),replace=False)
        data.append(torch.tensor(np.array(X[idx]),device=dev).float()); labels.append(torch.tensor(np.array(Y[idx]),device=dev).long())
    x=torch.cat(data); y=torch.cat(labels); n=torch.bincount(y,minlength=16)
    assert (n>0).all(); a=x/scale; results=[]
    for m in (32,512):
        ay=torch.arange(m,device=dev)%16
        noise=torch.from_numpy(np.random.default_rng(4902+m).standard_normal((m,66)).astype('float32')).to(dev)
        packet=pack_vector('dense',(mu[ay]+sd[ay]*noise).flatten(),half=True)
        anchors=unpack(packet,dev)['values'].reshape(m,66); b=anchors/scale
        d2=(a.square().sum(1)[:,None]+b.square().sum(1)[None]-2*a@b.T).clamp_min(0)
        for h in (1.,2.,4.,66**.5):
            kernels=(-d2/(2*h*h)).exp(); covered=kernels.max(1).values>=1e-3
            perclass=[float(covered[y==c].float().mean()) for c in range(16)]
            # Class-conditional aggregate only; no row, kernel-per-record or producer table exported.
            H=torch.stack([kernels[y==c].sum(0)/n[c] for c in range(16)],dim=1)
            total=H.sum(1,keepdim=True); target=H/total.clamp_min(1e-30)
            entropy=-(target*target.clamp_min(1e-30).log()).sum(1)/np.log(16)
            results.append(dict(anchors=m,h=h,rows=len(y),coverage=float(covered.float().mean()),
                coverage_per_class=perclass,coverage_gate=bool(covered.float().mean()>=.90 and min(perclass)>=.80),
                anchor_packet_bytes=len(packet),zero_target_rows=int((total==0).sum()),
                mean_normalized_target_entropy=float(entropy.mean()),mean_max_target=float(target.max(1).values.mean())))
    record=dict(evidence='2048 train rows/client sample, verified C Gaussian bootstrap, noise-init fp16 anchors; coverage only, no selected bandwidth/F1/full scan',
        scale_rule='class-balanced mixture standard deviation from decoded Gaussian mu/sd, floor0.01',checks=results)
    (base/'anchor-probe.json').write_text(json.dumps(record,indent=2))
    print('PASS bounded train-only anchor coverage probe; full-train preflight and bandwidth decision still pending')


if __name__=='__main__': main()
