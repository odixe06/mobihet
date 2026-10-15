"""Bounded saved-target forensics; no SGD, official test or new hyperparameter selection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.metrics import metrics_from_confusion


def sample(hold):
    fx=ROOT/'papers/pilot/fixture'
    if hold=='train':
        x=np.concatenate([np.load(fx/f'train_{k}_X.npy') for k in range(20)])
        y=np.concatenate([np.load(fx/f'train_{k}_y.npy') for k in range(20)])
    else:
        x,y=np.load(fx/f'{hold}_X.npy'),np.load(fx/f'{hold}_y.npy')
    rng=np.random.default_rng(1729)
    index=np.concatenate([rng.choice(np.flatnonzero(y==c),min(128,int((y==c).sum())),replace=False) for c in range(16)])
    return torch.from_numpy(x[index].astype('float32')),torch.from_numpy(y[index].astype('int64'))


def main():
    torch.set_num_threads(1)
    samples={h:sample(h) for h in ('train','group')}
    rows=[]
    proof=json.loads((ROOT/'docs/operations/restored-v9-forensics-2026-10-15.json').read_text())
    for item in proof['files']:
        path=ROOT/item['path']
        if path.suffix!='.pt':continue
        identity=json.loads((path.parent.parent/'config.json').read_text())
        for name,digest in identity['source'].items():
            assert hashlib.sha256((ROOT/'mh'/name).read_bytes()).hexdigest()==digest
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['round']==20 and state['fingerprint']==identity['fingerprint']
        alg=state['algorithm']
        initial,target=alg['initial_target'].float(),alg['targets'][0].float()
        labels=alg['labels'].long()
        delta=target-initial
        q0,qt=initial.softmax(1),target.softmax(1)
        normalized=qt.clamp_min(1e-12).log()
        record=dict(run=path.parent.parent.name,bandwidth=alg['h'],anchors=len(labels),
            anchor_argmax_changed=float((target.argmax(1)!=initial.argmax(1)).float().mean()),
            mean_target_delta_norm=float(delta.norm(dim=1).mean()),max_target_delta_norm=float(delta.norm(dim=1).max()),
            target_anchor_label_probability=float(qt[torch.arange(len(labels)),labels].mean()),
            mean_target_entropy_fraction=float((-(qt*normalized).sum(1)/np.log(16)).mean()),
            delta_centered_energy_fraction=float((delta-delta.mean(0)).square().sum()/delta.square().sum().clamp_min(1e-12)),
            diagnostics=alg['diag'],holds={})
        anchor=alg['anchors'].float()/alg['scale'].float()
        for hold,(x,y) in samples.items():
            xx=x/alg['scale'].float()
            distance=(xx.square().sum(1)[:,None]+anchor.square().sum(1)[None]-2*xx@anchor.T).clamp_min(0)
            kernel=torch.exp(-distance/(2*alg['h']**2))
            weight=kernel/kernel.sum(1,keepdim=True).clamp_min(1e-30)
            square_mass=weight.double().square().sum(1)
            effective=torch.where(square_mass>0,weight.double().sum(1).square()/square_mass/len(labels),0.)
            ent=-(weight*weight.clamp_min(1e-30).log()).sum(1)/np.log(len(labels))
            same=labels[None,:]==y[:,None]
            nearest_same=distance.masked_fill(~same,float('inf')).min(1).values
            cutoff=-2*alg['h']**2*np.log(1e-3)
            scores={}
            for name,q in [('bootstrap',q0),('final',qt)]:
                pred=(weight@q).argmax(1)
                cm=torch.bincount(y*16+pred,minlength=256).view(16,16).numpy()
                scores[name]=metrics_from_confusion(cm)
            record['holds'][hold]=dict(rows=len(y),mean_kernel_entropy=float(ent.mean()),
                mean_effective_anchor_fraction=float(effective.mean()),
                zero_kernel_rows=int((kernel.sum(1)==0).sum()),
                nearest_anchor_label_accuracy=float((labels[distance.argmin(1)]==y).float().mean()),
                mean_true_class_anchor_mass=float((weight*same).sum(1).mean()),
                any_anchor_coverage=float((distance.min(1).values<=cutoff).float().mean()),
                same_class_anchor_coverage=float((nearest_same<=cutoff).float().mean()),
                final_vs_bootstrap_prediction_changed=float(((weight@q0).argmax(1)!=(weight@qt).argmax(1)).float().mean()),
                metrics=scores)
        rows.append(record)
    result=dict(scope='two saved M512 r20 targets; balanced128/class T and group; kernel diagnostic is not DAGSNet utility or causal isolation',
                source_fingerprints_verified=True,records=rows)
    out=BASE/'probes/forensics-20261015.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
