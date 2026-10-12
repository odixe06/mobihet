"""Small DAGSNet head-bound proof for a proposed unbased JEM density, no training."""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.dagsnet import build_model,CFG


def main():
    torch.set_num_threads(1)
    checkpoint=ROOT/'papers/v2/runs/a_dense_s42/weights/round_020.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).eval()
    model.load_state_dict(state['proxy'][0],strict=True)
    model.requires_grad_(False)
    norm,first,last=model.head[0],model.head[2],model.head[5]
    assert isinstance(norm,torch.nn.LayerNorm)
    assert isinstance(first,torch.nn.Linear) and isinstance(last,torch.nn.Linear)
    n=norm.weight.numel()
    normalized_bound=float(norm.weight.abs().max()*n**.5+norm.bias.norm())
    hidden_bound=float(first.weight.norm()*normalized_bound+first.bias.norm())
    logit_bound=last.weight.norm(dim=1)*hidden_bound+last.bias.abs()
    assert torch.isfinite(logit_bound).all()
    # Arbitrary combined vectors exercise the learned head; theorem is algebraic, not this sample.
    g=torch.Generator().manual_seed(4601); samples=[]
    for scale in (1,10,100,1000):
        with torch.no_grad(): logits=model.head(torch.randn(32,n,generator=g)*scale)
        assert (logits.abs()<=logit_bound[None]+1e-3).all()
        samples.append(dict(scale=scale,max_logit=float(logits.abs().max())))
    record=dict(evidence='head bound on verified dense checkpoint; no generative training',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        normalized_norm_bound=normalized_bound,hidden_norm_bound=hidden_bound,
        per_logit_absolute_bound=logit_bound.tolist(),sample_checks=samples,
        conclusion='exp(logsumexp(DAGSNet logits)) has positive global lower bound over R^66; integral infinite without base measure or bounded domain')
    out=ROOT/'papers/v3/energy-probe.json'; out.write_text(json.dumps(record,indent=2))
    print('PASS DAGSNet finite head bound; unbased JEM over R^66 FAIL normalizability')


if __name__=='__main__': main()
