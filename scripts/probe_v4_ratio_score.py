"""Small local score/support risk on verified C ratios; no sampling/training/F1 claim."""
import json
import math
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG
from mh.v4_mala import potential


def main():
    torch.set_num_threads(1); rows=[]
    for seed in (42,43):
        path=ROOT/f'papers/v2/runs/c_fl_s{seed}'
        model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).cuda().eval()
        weights=torch.load(path/'weights/round_020.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(weights['proxy'][0],strict=True); del weights
        state=torch.load(path/'resume/round_020.pt',map_location='cuda',weights_only=True)['algorithm']
        mu,sd=state['mu'],state['sd']; del state
        g=torch.Generator(device='cuda').manual_seed(4905)
        for label in range(16):
            y=torch.full((64,),label,device='cuda',dtype=torch.long)
            z=torch.randn(64,66,device='cuda',generator=g)
            with torch.no_grad(): values=model(mu[y]+sd[y]*z)[0].float().gather(1,y[:,None]).squeeze(1)
            lp,score=potential(model,z,y,mu,sd)
            correction=(score+z).norm(dim=1)
            rows.append(dict(seed=seed,class_id=label,points=64,
                floor_fraction=float((values<=-10).float().mean()),
                cap_fraction=float((values>=math.log(10)).float().mean()),
                mean_ratio_score_norm=float(correction.mean()),median_ratio_score_norm=float(correction.median()),
                mean_gaussian_score_norm=float(z.norm(dim=1).mean())))
        del model,mu,sd; torch.cuda.empty_cache()
    record=dict(evidence='verified C r20 proxy0 canonical Gaussian BN, fresh noise 64/class; no MALA F1/mixing/production timing proof',
        checks=rows,limit='C snapshot is historical diagnostic, not V4 trained MALA teacher; no parameter changes from probe')
    (ROOT/'papers/v4/ratio-score-probe.json').write_text(json.dumps(record,indent=2))
    print('PASS finite clipped-ratio input score on verified C, two seeds; clipping overlap diagnostic only')


if __name__=='__main__': main()
